"""Run one episode: compile -> settle at home -> execute policy -> settle -> evaluate. Optional split-view video.

Rollouts are deterministic: same task + policy => bit-identical telemetry, with or without video.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import imageio
import mujoco
import numpy as np

from core.controller import ArmController
from core.locked_evaluator import evaluate
from core.mjcf_compiler import SceneIndex, compile_scene, load_adapter, load_task, save_scene
from core.policy import PolicyExecutor
from core.schemas import PolicyConfig, RobotAdapter, TaskSpec
from core.telemetry import FPS, Telemetry

SETTLE_STEPS = 150
VIDEO_W, VIDEO_H = 640, 480


@dataclass
class Sim:
    m: mujoco.MjModel
    d: mujoco.MjData
    ix: SceneIndex
    ctrls: dict[str, ArmController]
    xml: str


@dataclass
class Episode:
    success: bool
    evaluation: dict
    trace: dict
    resolved: list[dict]
    wall_s: float
    sim_s: float
    video: str | None = None
    arrays: dict = field(default_factory=dict, repr=False)

    def summary(self) -> dict:
        return {"success": self.success, "evaluation": self.evaluation, "trace": self.trace,
                "resolved": self.resolved, "wall_s": round(self.wall_s, 2), "sim_s": round(self.sim_s, 3),
                "video": self.video}


def build_sim(task: TaskSpec, adapter: RobotAdapter | None = None) -> Sim:
    adapter = adapter or load_adapter(task.robot)
    xml, m = compile_scene(task, adapter)
    d = mujoco.MjData(m)
    ix = SceneIndex.build(m, task, adapter)
    ctrls = {}
    for arm in adapter.arms:
        c = ArmController(m, d, arm)
        c.set_joints(arm.home_qpos)
        d.qpos[c.g_qadr] = arm.gripper.max_width_m / 2
        c.open(arm.gripper.max_width_m)
        ctrls[arm.name] = c
    mujoco.mj_forward(m, d)
    for c in ctrls.values():
        c.reset_cmd()
    for _ in range(SETTLE_STEPS):
        for c in ctrls.values():
            c.apply()
        mujoco.mj_step(m, d)
    d.time = 0.0
    for c in ctrls.values():
        c.q_prev = c.q_cmd.copy()
    return Sim(m, d, ix, ctrls, xml)


class SplitRenderer:
    """1280x480: perspective + side clearance camera, contact-force arrows, time/frame/phase overlay,
    red border while an obstacle is being touched."""

    def __init__(self, m: mujoco.MjModel, cams=("front", "side")):
        self.r = mujoco.Renderer(m, VIDEO_H, VIDEO_W)
        self.cams = cams
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    def frame(self, d: mujoco.MjData, phase: str, obj: str | None, obstacle_hits: int, touching: bool) -> np.ndarray:
        views = []
        for cam in self.cams:
            self.r.update_scene(d, camera=cam, scene_option=self.opt)
            views.append(self.r.render())
        img = np.ascontiguousarray(np.concatenate(views, axis=1))
        label = f"t={d.time:5.2f}s  f{round(d.time * FPS):04d}  {phase}{' ' + obj if obj else ''}"
        cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if touching:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (230, 30, 30), 8)
            cv2.putText(img, f"OBSTACLE CONTACT x {obstacle_hits}", (12, img.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 30, 30), 2, cv2.LINE_AA)
        return img

    def close(self):
        self.r.close()


def _obstacle_touch(tel: Telemetry) -> tuple[bool, int]:
    t = tel.t[-1] if tel.t else -1
    hits, touching = 0, False
    for (etype, *_), rows in tel.rows.items():
        if etype != "ILLEGAL_OBSTACLE_COLLISION":
            continue
        prev = None
        for r in rows:
            if prev is None or r["t"] - prev > 0.04:
                hits += 1
            prev = r["t"]
        if rows and rows[-1]["t"] >= t - 1e-9:
            touching = True
    return touching, hits


def run_episode(task: TaskSpec, policy: PolicyConfig, adapter: RobotAdapter | None = None, *,
                video: str | Path | None = None, on_frame: Callable[[Sim], None] | None = None,
                out_dir: str | Path | None = None, max_time: float = 60.0) -> Episode:
    wall0 = time.time()
    sim = build_sim(task, adapter)
    m, d, ix = sim.m, sim.d, sim.ix
    init_pos = {o.id: d.xpos[ix.obj_body[o.id]].copy().tolist() for o in task.objects}
    ex = PolicyExecutor(ix, d, sim.ctrls, policy)
    tel = Telemetry(ix, d, sim.ctrls, ex)
    writer = rend = None
    if video:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video), fps=FPS, codec="libx264", quality=7, macro_block_size=8)
        rend = SplitRenderer(m)
    next_frame = 0.0
    t_end = None

    while True:
        if t_end is None and (ex.done or d.time >= max_time):
            t_end = d.time + task.settle_s
        if t_end is not None and d.time >= t_end:
            break
        ex.step()
        mujoco.mj_step(m, d)
        tel.record()
        if d.time >= next_frame - 1e-9:
            next_frame += 1.0 / FPS
            if writer is not None:
                phase, obj = ex.phase()
                touching, hits = _obstacle_touch(tel)
                writer.append_data(rend.frame(d, phase, obj, hits, touching))
            if on_frame is not None:
                on_frame(sim)
    if writer is not None:
        writer.close()
        rend.close()

    trace = tel.finish()
    trace["timed_out"] = not ex.done
    evaluation = evaluate(m, d, task, init_pos, list(zip(tel.t, tel.max_obj_speed)),
                          trace["metrics"]["unwanted_collision_count"], wipe_contacts=tel.wipe_contacts())
    resolved = [{"object_id": r.p.object_id, "arm": r.p.arm, **r.resolved} for r in ex.runners]
    ep = Episode(success=evaluation["success"] and not trace["timed_out"], evaluation=evaluation, trace=trace,
                 resolved=resolved, wall_s=time.time() - wall0, sim_s=float(d.time),
                 video=str(video) if video else None, arrays=tel.arrays())
    if out_dir:
        out = Path(out_dir)
        save_scene(out, task, sim.xml)
        (out / "policy.json").write_text(policy.model_dump_json(indent=1))
        (out / "result.json").write_text(json.dumps(ep.summary(), indent=1, default=float))
        np.savez_compressed(out / "telemetry.npz", **ep.arrays)
    return ep


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one WorldGym episode")
    ap.add_argument("task")
    ap.add_argument("policy")
    ap.add_argument("--video")
    ap.add_argument("--out")
    a = ap.parse_args()
    task = load_task(a.task)
    pol = PolicyConfig.model_validate_json(Path(a.policy).read_text())
    ep = run_episode(task, pol, video=a.video, out_dir=a.out)
    s = ep.summary()
    print(json.dumps({"success": s["success"], "metrics": s["trace"]["metrics"],
                      "predicates": [(p["kind"], p.get("object_id"), p["ok"]) for p in s["evaluation"]["predicates"]],
                      "events": [(e["type"], e["t"], e.get("pair") or e.get("object")) for e in s["trace"]["events"]],
                      "wall_s": s["wall_s"], "sim_s": s["sim_s"]}, indent=1))


if __name__ == "__main__":
    main()

"""Render a synthetic robot video with full ground truth (camera, object poses, TCP track, gripper state).

It is the oracle for the camera solve, Loop 1 and motion extraction before any real clip is used.

    uv run python scripts/make_synthetic_clip.py [--policy policies/panda_sort_v3.json] [--out demo/synthetic_panda_sort]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.mjcf_compiler import load_task, look_at_xyaxes  # noqa: E402
from core.rollout import run_episode  # noqa: E402
from core.schemas import CameraSpec, PolicyConfig  # noqa: E402


def camera_from_sphere(az_deg, el_deg, dist, look=(0.35, 0.0, 0.25)) -> tuple[list[float], list[float]]:
    a, e = math.radians(az_deg), math.radians(el_deg)
    pos = [look[0] + dist * math.cos(e) * math.cos(a), look[1] + dist * math.cos(e) * math.sin(a),
           look[2] + dist * math.sin(e)]
    return pos, look_at_xyaxes(pos, look)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tasks/panda_sort.json")
    ap.add_argument("--policy", default="policies/panda_sort_v3.json")
    ap.add_argument("--out", default="demo/synthetic_panda_sort")
    ap.add_argument("--az", type=float, default=25.0)
    ap.add_argument("--el", type=float, default=28.0)
    ap.add_argument("--dist", type=float, default=1.6)
    ap.add_argument("--fovy", type=float, default=48.0)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--rest", type=float, default=1.0, help="seconds of robot at rest before the motion")
    a = ap.parse_args()

    task = load_task(a.task)
    pos, xyaxes = camera_from_sphere(a.az, a.el, a.dist)
    cam = CameraSpec(name="video", pos=pos, xyaxes=xyaxes, fovy=a.fovy, w=a.w, h=a.h)
    task = task.model_copy(update={"cameras": [*task.cameras, cam]})
    pol = PolicyConfig.model_validate_json(Path(a.policy).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(f"{out}.mp4", fps=30, codec="libx264", quality=8, macro_block_size=8)
    state = {"r": None, "frames": [], "rest_written": False}

    def on_frame(sim):
        if state["r"] is None:
            state["r"] = mujoco.Renderer(sim.m, a.h, a.w)
        r = state["r"]
        r.update_scene(sim.d, camera="video")
        img = r.render()
        if not state["rest_written"]:  # hold the first frame: the robot at rest
            for _ in range(int(a.rest * 30)):
                writer.append_data(img)
            state["rest_written"] = True
        writer.append_data(img)
        ix, d = sim.ix, sim.d
        ctrl = next(iter(sim.ctrls.values()))
        state["frames"].append({"t": round(float(d.time) + a.rest, 4), "tcp": ctrl.tcp_pose()[0].round(5).tolist(),
                                "width": round(ctrl.gripper_width(), 5),
                                "objects": {o: d.xpos[b].round(5).tolist() for o, b in ix.obj_body.items()}})

    ep = run_episode(task, pol, on_frame=on_frame)
    writer.close()
    state["r"].close()
    gt = {"camera": cam.model_dump(), "camera_sphere": {"azimuth_deg": a.az, "elevation_deg": a.el,
                                                        "distance_m": a.dist, "look_at": [0.35, 0.0, 0.25]},
          "rest_s": a.rest, "fps": 30, "task": task.model_dump(), "policy": a.policy, "success": ep.success,
          "phase_timeline": [{**p, "t0": round(p["t0"] + a.rest, 3), "t1": round(p["t1"] + a.rest, 3)}
                             for p in ep.trace["phase_timeline"]],
          "frames": state["frames"]}
    Path(f"{out}.gt.json").write_text(json.dumps(gt))
    print(json.dumps({"video": f"{out}.mp4", "frames": len(state["frames"]), "success": ep.success,
                      "camera": cam.model_dump()}, indent=1))


if __name__ == "__main__":
    main()

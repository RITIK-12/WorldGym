"""Twin half of the pipeline: video -> verified MuJoCo twin + Draft 1, one run-registry stage per step.

qa -> video_architect -> robot_onboard -> camera_solve -> loop1_static (Scene Author + Twin Critic loop)
   -> motion_extract (parallel Robotics-ER labellers -> contact-constrained waypoints -> Draft 1 -> dynamic fidelity)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from core.argon import Argon
from core.camera_solve import add_video_camera, apply_solution, render_body_mask, solve_camera
from core.fidelity import measure_footprint, run_loop1
from core.motion_extract import extract
from core.robot_onboard import onboard
from core.rollout import build_sim, run_episode
from core.runs import Refused, Run
from core.scene_build import author_scene, fallback_scene, make_task, mask_overlay, measure_objects
from core.schemas import TaskSpec, VideoUnderstanding
from core.segment import grabcut_mask, masks_from_boxes, norm_box_to_px
from core.video_architect import dense_labels, understand
from core.video_probe import probe, read_frame

WORK_W = 1280


def _frame(video: str, t: float) -> np.ndarray:
    f = read_frame(video, t)
    if f.shape[1] > WORK_W:
        f = cv2.resize(f, (WORK_W, round(f.shape[0] * WORK_W / f.shape[1])), interpolation=cv2.INTER_AREA)
    return f


def _nearest_kf(vu: VideoUnderstanding, t: float):
    return min(vu.keyframes, key=lambda k: abs(k.t - t)) if vu.keyframes else None


def _fit_overlay(frame, real_mask, sim_mask, path: Path) -> str:
    img = (frame * 0.6).astype(np.uint8)
    for mk, col in ((real_mask, (0, 220, 255)), (sim_mask, (255, 220, 0))):
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, col, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return str(path)


def build_twin(run: Run, video: str, prompt: str, argon: Argon, force: bool = False) -> dict:
    with run.stage("qa") as s:
        pr = probe(video, run.path("qa"))
        s.headline = pr["headline"] + ("" if pr["accepted"] else " · " + "; ".join(pr["reasons"]))
        s.data = {k: v for k, v in pr.items() if k != "curve"}
        s.artifacts += ["qa/probe.json", "qa/motion.png"]
        if not pr["accepted"] and not force:
            raise Refused("; ".join(pr["reasons"]))

    with run.stage("video_architect") as s:
        vu = understand(video, pr, prompt, argon, run.path("architect"))
        run.write_json("architect/vu.json", vu.model_dump())
        s.headline = f"{vu.robot.make_model_guess} · {len(vu.objects)} objects · {len(vu.phases)} phases"
        s.data = {"robot": vu.robot.model_dump(), "objects": [o.model_dump() for o in vu.objects],
                  "phases": [p.model_dump() for p in vu.phases], "success_intent": vu.success_intent}
        s.artifacts.append("architect/vu.json")

    with run.stage("robot_onboard") as s:
        if vu.robot.arms != 1:
            raise Refused(f"{vu.robot.arms}-arm robots are not supported by this build yet (single-arm tier)")
        ob = onboard(vu.robot.menagerie_dir or "franka_emika_panda", argon, log_dir=run.path("onboard"))
        adapter = ob["adapter"]
        n_ok = sum(c["ok"] for c in ob["self_test"]["checks"].values())
        s.headline = f"{adapter.robot_id} ({ob['source']}) · self-test {n_ok}/{len(ob['self_test']['checks'])}"
        s.data = {"self_test": ob["self_test"], "source": ob["source"]}
        if not ob["self_test"]["ok"]:
            raise RuntimeError(f"robot self-test failed: {ob['self_test']['checks']}")

    with run.stage("camera_solve") as s:
        rest = _frame(video, vu.rest_frame_t)
        h, w = rest.shape[:2]
        kf = _nearest_kf(vu, vu.rest_frame_t)
        if kf is None or not kf.robot_box:
            raise RuntimeError("no robot box near the rest frame")
        robot_mask = grabcut_mask(rest, norm_box_to_px(kf.robot_box, w, h))
        base = add_video_camera(TaskSpec(name=run.run_id, robot=adapter.robot_id, objects=[]), w, h)
        sim = build_sim(base, adapter)
        init = {k: v for k, v in (vu.camera_guess or {}).items()}
        sol = solve_camera(sim.m, sim.d, robot_mask, sim.ix.robot_bodies, "video", init=init)
        cam_task = apply_solution(base, sol, w, h)
        cv2.imwrite(str(run.path("camera/rest_frame.jpg")), rest)
        sim_mask = render_body_mask(sim.m, sim.d, "video", sim.ix.robot_bodies, w, h)
        art = _fit_overlay(rest, robot_mask, sim_mask, run.path("camera/robot_fit.jpg"))
        s.headline = f"robot silhouette IoU {sol['iou']:.2f} · fovy {sol['fovy']:.0f}°"
        s.data = {k: v for k, v in sol.items() if k in ("pos", "xyaxes", "fovy", "params", "iou", "accepted",
                                                         "seconds")}
        s.artifacts += ["camera/rest_frame.jpg", run.rel(art)]
        if not sol["accepted"]:
            raise RuntimeError(f"camera solve rejected: silhouette IoU {sol['iou']:.2f} < 0.75")

    with run.stage("loop1_static") as s:
        boxes = {oid: b for oid, b in (kf.object_boxes or {}).items()}
        masks = masks_from_boxes(rest, boxes)
        sim = build_sim(cam_task, adapter)
        measured = measure_objects(sim.ix, sim.d, vu, masks)
        overlay = mask_overlay(rest, masks, run.path("loop1/masks.jpg"))
        try:
            draft = author_scene(argon, vu, measured, run.path("camera/rest_frame.jpg"), overlay, prompt)
        except Exception as e:  # noqa: BLE001 - the twin must still be built from the measurements
            draft = fallback_scene(vu, measured)
            s.data["author_error"] = str(e)[:300]
        task = make_task(run.run_id.split("_", 1)[-1] or "twin", adapter.robot_id, draft, cam_task, vu)
        context = f"Task: {vu.success_intent}. Robot: {vu.robot.make_model_guess}."
        res = run_loop1(task, rest, masks, run.path("loop1"), argon=argon, context=context, robot_mask=robot_mask)
        task = res["task"]
        run.write_json("loop1/task.json", task.model_dump())
        start = res["history"][0]["alignment"]
        s.headline = f"alignment {start * 100:.0f} % → {res['alignment'] * 100:.0f} % in {len(res['history']) - 1} iteration(s)"
        s.data = {"alignment": res["alignment"], "objects": res["objects"], "measured": measured,
                  "history": [{k: v for k, v in r.items() if k != "task"} for r in res["history"]],
                  "author_notes": draft.notes}
        s.artifacts += ["loop1/masks.jpg", "loop1/task.json", "loop1/history.json",
                        run.rel(res["final_views"]["matrix"])]

    with run.stage("motion_extract") as s:
        active = [p for p in vu.phases if p.action not in ("idle",)]
        t0 = min((p.t0 for p in active), default=0.0)
        t1 = max((p.t1 for p in active), default=float(pr["duration_s"]))
        times = [round(t, 2) for t in np.arange(t0, t1 + 1e-6, 0.25)][:96]
        labels = dense_labels(video, times, vu, argon, run.path("motion"))
        run.write_json("motion/labels.json", [k.model_dump() for k in labels])
        sim = build_sim(task, adapter)
        draft1, log = extract(task, labels, sim, w, h)
        if not draft1.primitives:
            raise RuntimeError("no grasp -> release segment found in the labels")
        run.write_json("motion/motion_extract.json", log)
        run.write_json("motion/draft1.json", draft1.model_dump())
        ep = run_episode(task, draft1)
        dyn = dynamic_fidelity(video, vu, task, ep, labels, w, h, cam_task)
        s.headline = (f"Draft 1 from {len(labels)} labelled frames · replay "
                      f"{'succeeds' if ep.success else 'fails'}"
                      + (f" · end error {dyn['end_pose_err_mm']:.0f} mm" if dyn.get("end_pose_err_mm") is not None else ""))
        s.data = {"segments": log["segments"], "dynamic": dyn, "n_labels": len(labels)}
        s.artifacts += ["motion/labels.json", "motion/motion_extract.json", "motion/draft1.json"]
    return {"task": task, "draft1": draft1, "adapter": adapter, "dynamic_ref": dyn, "vu": vu}


def dynamic_fidelity(video: str, vu: VideoUnderstanding, task: TaskSpec, ep, labels, w: int, h: int,
                     cam_task: TaskSpec) -> dict:
    """End-pose error of the moved objects (sim end state vs the last video frame) + replay success."""
    last = max((k for k in labels if k.object_boxes), key=lambda k: k.t, default=None)
    out = {"replay_success": ep.success, "end_pose_err_mm": None, "per_object_mm": {}}
    if last is None:
        return out
    frame = _frame(video, last.t)
    masks = masks_from_boxes(frame, last.object_boxes)
    probe_task = task.model_copy(update={"objects": []})
    sim = build_sim(probe_task)
    end_sim = build_sim(task)
    end_sim.d.qpos[:] = ep.arrays["qpos"][-1]
    mujoco.mj_forward(end_sim.m, end_sim.d)
    moved = {p.object_id for p in (task.phases or []) if p.object} or {o.id for o in task.objects if not o.is_fixed}
    errs = {}
    for o in task.objects:
        if o.id not in moved or o.id not in masks:
            continue
        fp = measure_footprint(sim.ix, sim.d, "video", masks[o.id], max(o.dims_m[2], 0.02))
        if fp is None:
            continue
        sim_xy = end_sim.d.xpos[end_sim.ix.obj_body[o.id]][:2]
        errs[o.id] = round(float(np.linalg.norm(np.asarray(fp["xy"]) - sim_xy)) * 1000, 1)
    if errs:
        out["per_object_mm"] = errs
        out["end_pose_err_mm"] = float(np.mean(list(errs.values())))
    return out

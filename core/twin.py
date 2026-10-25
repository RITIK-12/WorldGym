"""Twin half of the pipeline: video -> verified MuJoCo twin + Draft 1, one run-registry stage per step.

qa (static, or moving-camera mode for handheld clips)
 -> video_architect (Argon on the clip + keyframes, then Argon reviews its own labels)
 -> robot_onboard (Argon picks the robot by comparing rendered Menagerie candidates; adapter + self-test)
 -> camera_solve (Robotics-ER boxes per arm -> robot mask -> silhouette solve at a rest frame; moving camera: tracked
    through the clip on the table plane)
 -> loop1_static (Scene Author + Twin Critic loop on the 4-view matrix)
 -> motion_extract (parallel Robotics-ER labellers -> per-arm hold / wipe / pick-place segments -> Draft 1)
 -> replay_critic (Argon watches real | sim side by side and patches Draft 1 toward the demonstration)
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mujoco
import numpy as np
from pydantic import BaseModel

from core.argon import ROBOTICS_ER, Argon
from core.camera_solve import add_video_camera, apply_solution, render_body_mask, set_camera, solve_camera
from core.camera_track import camera_at, track_camera, track_overlay
from core.fidelity import measure_footprint, run_loop1
from core.mjcf_compiler import ROBOTS, scene_table
from core.motion_extract import extract, extract_multi
from core.replay_critic import run_replay_critic
from core.robot_onboard import onboard, self_test, verify_robot
from core.rollout import build_sim, run_episode
from core.runs import Refused, Run
from core.scene_build import author_scene, fallback_scene, make_task, mask_overlay, measure_objects
from core.schemas import RobotAdapter, TaskSpec, VideoUnderstanding
from core.segment import grabcut_mask, masks_from_boxes, norm_box_to_px
from core.video_architect import dense_labels, understand
from core.video_probe import probe, read_frame

WORK_W = 1280
LABEL_DT = 0.25


class ArmBox(BaseModel):
    name: str
    box: list[float]


class CalibBoxes(BaseModel):
    robot_arms: list[ArmBox]
    objects: dict[str, list[float]] = {}


def _frame(video: str, t: float) -> np.ndarray:
    f = read_frame(video, t)
    if f.shape[1] > WORK_W:
        f = cv2.resize(f, (WORK_W, round(f.shape[0] * WORK_W / f.shape[1])), interpolation=cv2.INTER_AREA)
    return f


def _fit_overlay(frame, real_mask, sim_mask, path: Path) -> str:
    img = (frame * 0.6).astype(np.uint8)
    for mk, col in ((real_mask, (0, 220, 255)), (sim_mask, (255, 220, 0))):
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, col, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return str(path)


def find_adapter(menagerie_dir: str, n_arms: int) -> RobotAdapter | None:
    for p in sorted(ROBOTS.glob("*.json")):
        a = RobotAdapter.model_validate_json(p.read_text())
        if a.menagerie_dir == menagerie_dir and len(a.arms) == n_arms:
            return a
    return None


def posed_sim(task: TaskSpec, adapter: RobotAdapter, pose: str):
    sim = build_sim(task, adapter)
    for arm in adapter.arms:
        sim.ctrls[arm.name].set_joints(arm.home_qpos if pose == "home" else (arm.rest_qpos or arm.home_qpos))
    mujoco.mj_forward(sim.m, sim.d)
    return sim


def calib_boxes(argon: Argon, frame_path: Path, vu: VideoUnderstanding, n_arms: int, tag: str) -> CalibBoxes:
    arms = "the robot arm" if n_arms == 1 else "each robot arm separately (name them by side: left, right)"
    ids = ", ".join(o.id for o in vu.objects)
    boxes, _ = argon.ask_json(
        f"calib_boxes_{tag}", "Precise spatial labeller for robot scenes.",
        f"Give tight boxes [ymin, xmin, ymax, xmax] (0..1000 of the image) for {arms} - every visible part from where "
        f"it enters the frame to its gripper - and for each of these objects that is visible: {ids}. Use exactly "
        "these object ids.", CalibBoxes, [frame_path], model=ROBOTICS_ER, thinking="high", max_output_tokens=8192)
    return boxes


def publish_gen(run: Run, n: int, title: str, headline: str, task: TaskSpec, adapter: RobotAdapter,
                policy=None) -> None:
    """Checkpoint a viewable result (gens/genN): the twin, plus its replay when a policy exists. The demo UI shows
    the newest generation while the next one is still being critiqued."""
    d = run.path(f"gens/gen{n}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(task.model_dump_json(indent=1))
    if policy is not None:
        (d / "policy.json").write_text(policy.model_dump_json(indent=1))
        run_episode(task, policy, adapter, out_dir=d / "nominal")
    idx = run.path("gens/gens.json")
    gens = json.loads(idx.read_text()) if idx.exists() else []
    gens = [g for g in gens if g["n"] != n] + [{"n": n, "title": title, "headline": headline,
                                                 "replay": policy is not None, "t": time.time()}]
    idx.write_text(json.dumps(sorted(gens, key=lambda g: g["n"]), indent=1))


def build_twin(run: Run, video: str, prompt: str, argon: Argon, force: bool = False) -> dict:
    with run.stage("qa") as s:
        pr = probe(video, run.path("qa"))
        moving = not pr["accepted"] and pr.get("moving_camera_ok", False)
        s.headline = pr["headline"] + ("" if pr["accepted"] else " · " + "; ".join(pr["reasons"]))
        if moving:
            s.headline += " → moving-camera mode"
        s.data = {k: v for k, v in pr.items() if k != "curve"}
        s.data["moving_camera"] = moving
        s.artifacts += ["qa/probe.json", "qa/motion.png"]
        if not pr["accepted"] and not moving and not force:
            raise Refused("; ".join(pr["reasons"]))

    with run.stage("video_architect") as s:
        vu = understand(video, pr, prompt, argon, run.path("architect"))
        run.write_json("architect/vu.json", vu.model_dump())
        s.headline = f"{vu.robot.make_model_guess} · {len(vu.objects)} objects · {len(vu.phases)} phases"
        s.data = {"robot": vu.robot.model_dump(), "objects": [o.model_dump() for o in vu.objects],
                  "phases": [p.model_dump() for p in vu.phases], "success_intent": vu.success_intent}
        s.artifacts.append("architect/vu.json")

    active = [p for p in vu.phases if p.action != "idle"]
    t_act0 = min((p.t0 for p in active), default=0.0)
    t_act1 = max((p.t1 for p in active), default=float(pr["duration_s"]))
    n_arms = max(1, vu.robot.arms)
    # dense labelling needs only the video and the phases: run the labelling swarm alongside the twin stages
    label_times = [round(t, 2) for t in np.arange(t_act0, t_act1 + 1e-6, LABEL_DT)][:160]
    pool = ThreadPoolExecutor(max_workers=1)
    labels_job = pool.submit(dense_labels, video, label_times, vu, argon, run.path("motion"))

    with run.stage("robot_onboard") as s:
        t_full = min((k.t for k in vu.keyframes), default=0.0)   # robot at rest, most of it in view
        fpath = run.path("onboard/real_frame.jpg")
        cv2.imwrite(str(fpath), _frame(video, t_full))
        fmid = run.path("onboard/real_frame_mid.jpg")
        cv2.imwrite(str(fmid), _frame(video, (t_act0 + t_act1) / 2))
        guess = vu.robot.menagerie_dir or "franka_emika_panda"
        try:
            pick = verify_robot(argon, [fpath, fmid], guess, run.path("onboard"))
            mdir = pick.menagerie_dir
            s.data["verify"] = pick.model_dump()
        except Exception as e:  # noqa: BLE001 - keep the first-pass guess if the check fails
            mdir = guess
            s.data["verify_error"] = str(e)[:300]
        adapter = find_adapter(mdir, n_arms)
        source = f"robots/{adapter.robot_id}.json" if adapter else None
        if adapter is None and n_arms == 1:
            ob = onboard(mdir, argon, log_dir=run.path("onboard"))
            adapter, source = ob["adapter"], ob["source"]
        if adapter is None:
            raise Refused(f"no {n_arms}-arm adapter for {mdir}")
        st = self_test(adapter)
        n_ok = sum(c["ok"] for c in st["checks"].values())
        changed = mdir != guess
        s.headline = (f"{adapter.robot_id}" + (f" (Argon corrected {guess} → {mdir} by comparing renders)" if changed
                                              else "") + f" · self-test {n_ok}/{len(st['checks'])}")
        s.data.update(self_test=st, source=source, adapter=adapter.robot_id)
        s.artifacts += ["onboard/real_frame.jpg", "onboard/candidates.jpg"]
        if not st["ok"]:
            raise RuntimeError(f"robot self-test failed: { {k: v['ok'] for k, v in st['checks'].items()} }")

    with run.stage("camera_solve") as s:
        cands = sorted({round(min((k.t for k in vu.keyframes), default=0.0), 2), round(vu.rest_frame_t, 2)})
        best = None
        for t in cands:
            frame = _frame(video, t)
            h, w = frame.shape[:2]
            fp = run.path(f"camera/frame_{t:06.2f}.jpg")
            cv2.imwrite(str(fp), frame)
            try:
                boxes = calib_boxes(argon, fp, vu, n_arms, f"{t:06.2f}")
                arm_boxes = [b.box for b in boxes.robot_arms]
                obj_boxes = boxes.objects
            except Exception:  # noqa: BLE001 - fall back to the architect's keyframe labels
                kf = min(vu.keyframes, key=lambda k: abs(k.t - t))
                arm_boxes, obj_boxes = ([kf.robot_box] if kf.robot_box else []), kf.object_boxes
            if not arm_boxes:
                continue
            mask = np.zeros((h, w), bool)
            for b in arm_boxes:
                mask |= grabcut_mask(frame, norm_box_to_px(b, w, h))
            base = add_video_camera(TaskSpec(name=run.run_id, robot=adapter.robot_id, objects=[]), w, h)
            fwd = float(np.mean([bp[2] for bp in (adapter.build or {}).get("arms", {}).values()] or [0.0]))
            init = dict(vu.camera_guess or {})
            if "azimuth_deg" in init:   # the architect measures azimuth from the robot's forward direction
                init["azimuth_deg"] = init["azimuth_deg"] + float(np.degrees(fwd))
            for pose in ("rest", "home"):
                sim = posed_sim(base, adapter, pose)
                sol = solve_camera(sim.m, sim.d, mask, sim.ix.robot_bodies, "video", init=init)
                if best is None or sol["iou"] > best["sol"]["iou"]:
                    best = {"t": t, "pose": pose, "sol": sol, "mask": mask, "frame": frame, "base": base,
                            "obj_boxes": obj_boxes, "sim": sim}
        if best is None:
            raise RuntimeError("no robot boxes to calibrate the camera")
        sol, frame, t_ref = best["sol"], best["frame"], best["t"]
        h, w = frame.shape[:2]
        cam_task = apply_solution(best["base"], sol, w, h)
        cv2.imwrite(str(run.path("camera/rest_frame.jpg")), frame)
        sim_mask = render_body_mask(best["sim"].m, best["sim"].d, "video", best["sim"].ix.robot_bodies, w, h)
        art = _fit_overlay(frame, best["mask"], sim_mask, run.path("camera/robot_fit.jpg"))
        s.artifacts += ["camera/rest_frame.jpg", run.rel(art)]
        track = None
        if moving:
            times = sorted({round(t, 2) for t in np.arange(t_act0, t_act1 + 1e-6, LABEL_DT)}
                           | {round(k.t, 2) for k in vu.keyframes})
            tb = scene_table(cam_task, adapter)
            rect = (*tb.center_xy, *tb.size_xy)
            track = track_camera(video, t_ref, sol["pos"], sol["xyaxes"], sol["fovy"], times, table_rect=rect)
            run.write_json("camera/track.json", track)
            for tt in (t_act0, (t_act0 + t_act1) / 2, t_act1):
                c = camera_at(track, tt)
                if c:
                    p = track_overlay(video, c, tt, run.path(f"camera/track_{tt:06.2f}.jpg"), rect)
                    s.artifacts.append(run.rel(p))
            s.data["tracked"] = f"{sum(c['ok'] for c in track)}/{len(track)}"
        s.headline = (f"robot silhouette IoU {sol['iou']:.2f} · fovy {sol['fovy']:.0f}° · rest pose '{best['pose']}'"
                      + (f" · tracked {s.data['tracked']} frames" if track else ""))
        s.data.update({k: v for k, v in sol.items() if k in ("pos", "xyaxes", "fovy", "params", "iou", "accepted",
                                                             "seconds")}, ref_t=t_ref, pose=best["pose"])
        if sol["iou"] < 0.6:
            raise RuntimeError(f"camera solve rejected: silhouette IoU {sol['iou']:.2f} < 0.6")
    cam_at = (lambda t: camera_at(track, t)) if track else None
    pose = best["pose"]

    with run.stage("loop1_static") as s:
        masks = masks_from_boxes(frame, best["obj_boxes"])
        sim = posed_sim(cam_task, adapter, pose)
        measured = measure_objects(sim.ix, sim.d, vu, masks)
        overlay = mask_overlay(frame, masks, run.path("loop1/masks.jpg"))
        try:
            draft = author_scene(argon, vu, measured, run.path("camera/rest_frame.jpg"), overlay, prompt)
        except Exception as e:  # noqa: BLE001 - the twin must still be built from the measurements
            draft = fallback_scene(vu, measured)
            s.data["author_error"] = str(e)[:300]
        task = make_task(run.run_id.split("_", 1)[-1] or "twin", adapter.robot_id, draft, cam_task, vu)
        try:   # Gen 1 already replays the demonstration: the labelling swarm ran alongside the stages above
            sim1 = build_sim(task, adapter)
            pol1 = (extract_multi(task, labels_job.result(), sim1, w, h, cam_at) if n_arms > 1 or moving
                    else extract(task, labels_job.result(), sim1, w, h))[0]
        except Exception as e:  # noqa: BLE001 - show the static world rather than nothing
            pol1 = None
            s.data["gen1_motion_error"] = str(e)[:300]
        publish_gen(run, 1, "First world", f"{len(task.objects)} objects placed by the Scene Author", task, adapter,
                    pol1 if pol1 is not None and pol1.primitives else None)
        context = f"Task: {vu.success_intent}. Robot: {adapter.robot_id} ({n_arms} arm(s))."
        res = run_loop1(task, frame, masks, run.path("loop1"), argon=argon, context=context,
                        robot_mask=best["mask"], build=lambda t: posed_sim(t, adapter, pose))
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
        labels = labels_job.result()
        pool.shutdown()
        run.write_json("motion/labels.json", [k.model_dump() for k in labels])
        sim = build_sim(task, adapter)
        if n_arms > 1 or moving:
            draft1, log = extract_multi(task, labels, sim, w, h, cam_at)
        else:
            draft1, log = extract(task, labels, sim, w, h)
        if not draft1.primitives:
            raise RuntimeError("no grasp -> release segment found in the labels")
        run.write_json("motion/motion_extract.json", log)
        run.write_json("motion/draft1.json", draft1.model_dump())
        ep = run_episode(task, draft1, adapter)
        real_end = real_end_positions(video, task, labels, cam_at)
        err, per = end_pose_error(task, ep, real_end)
        dyn = {"replay_success": ep.success, "end_pose_err_mm": err, "per_object_mm": per, "real_end_xy": real_end}
        kinds = ", ".join(f"{p.arm}:{p.kind}" for p in draft1.primitives)
        s.headline = (f"Draft 1 [{kinds}] from {len(labels)} labelled frames · replay "
                      f"{'succeeds' if ep.success else 'fails'}"
                      + (f" · end error {err:.0f} mm" if err is not None else ""))
        s.data = {"segments": log["segments"], "dynamic": dyn, "n_labels": len(labels)}
        s.artifacts += ["motion/labels.json", "motion/motion_extract.json", "motion/draft1.json"]
        publish_gen(run, 2, "Twin + demonstrated motion",
                    f"Twin Critic alignment {res['alignment'] * 100:.0f} % · motion from {len(labels)} labelled frames",
                    task, adapter, draft1)

    with run.stage("replay_critic") as s:
        grasp_t = log["segments"][0]["grasp_t"]
        rc = run_replay_critic(video, task, draft1, adapter, argon, grasp_t, log,
                               lambda e: end_pose_error(task, e, real_end)[0], run.path("replay_critic"),
                               cam_at=cam_at)
        draft1 = rc["draft"].model_copy(update={"name": "draft1"})
        run.write_json("motion/draft1.json", draft1.model_dump())
        dyn = {**dyn, "replay_success": rc["episode"].success, "end_pose_err_mm": rc["end_pose_err_mm"],
               "before_critic": rc["start"]}
        e0, e1 = rc["start"]["end_pose_err_mm"], rc["end_pose_err_mm"]
        s.headline = ("replay " + ("succeeds" if rc["episode"].success else "fails")
                      + (f" · end error {e0:.0f} → {e1:.0f} mm" if e0 is not None and e1 is not None else ""))
        s.data = {"history": rc["history"], "dynamic": dyn}
        publish_gen(run, 3, "Motion refined by the Replay Critic",
                    f"end-pose error {e0:.0f} → {e1:.0f} mm" if e0 is not None and e1 is not None else s.headline,
                    task, adapter, draft1)
        s.artifacts += [run.rel(rc["video"])] + [run.rel(h["video"]) for h in rc["history"] if h.get("video")]
    return {"task": task, "draft1": draft1, "adapter": adapter, "dynamic_ref": dyn, "vu": vu}


def real_end_positions(video: str, task: TaskSpec, labels, cam_at=None) -> dict[str, list[float]]:
    """Table positions of the moved objects in the last labelled frame (metric footprints)."""
    last = max((k for k in labels if k.object_boxes), key=lambda k: k.t, default=None)
    if last is None:
        return {}
    frame = _frame(video, last.t)
    masks = masks_from_boxes(frame, last.object_boxes)
    sim = build_sim(task.model_copy(update={"objects": []}))
    c = cam_at(last.t) if cam_at else None
    if c is not None:
        set_camera(sim.m, sim.d, "video", c["pos"], c["xyaxes"], c["fovy"])
    moved = {p.object_id for p in task.immutable_success_predicates if p.target_id or p.kind == "pushed_to_region"} or \
            {o.id for o in task.objects if not o.is_fixed and not o.is_obstacle}
    out = {}
    for o in task.objects:
        if o.id in moved and o.id in masks and not o.is_fixed:
            fp = measure_footprint(sim.ix, sim.d, "video", masks[o.id], max(o.dims_m[2], 0.02))
            if fp:
                out[o.id] = fp["xy"]
    return out


def end_pose_error(task: TaskSpec, ep, real_end: dict) -> tuple[float | None, dict]:
    if not real_end:
        return None, {}
    sim = build_sim(task)
    sim.d.qpos[:] = ep.arrays["qpos"][-1]
    mujoco.mj_forward(sim.m, sim.d)
    per = {oid: round(float(np.linalg.norm(np.asarray(xy) - sim.d.xpos[sim.ix.obj_body[oid]][:2])) * 1000, 1)
           for oid, xy in real_end.items()}
    return float(np.mean(list(per.values()))), per

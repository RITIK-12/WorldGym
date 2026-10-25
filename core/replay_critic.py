"""Replay Critic: the dynamic half of the PS2 loop. Render the real clip and the sim replay of the current draft side
by side (time-aligned at the first grasp), let Argon watch both, and patch the draft toward the demonstration.

A patch is kept only if it improves the verification signal (replay success first, then mean end-pose error).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np

from core.argon import ARGON, Argon
from core.critic import DSL_DOC, PatchError, apply_patch, param_diff, robot_facts
from core.rollout import Episode, build_sim, run_episode
from core.schemas import PolicyConfig, PolicyPatch, RobotAdapter, TaskSpec

FPS = 10
PANEL_W = 640

SYSTEM = """You are the Replay Critic of WorldGym. A robot demonstration video was converted into a simulated replay
(Draft policy in a closed parametric DSL). You watch the REAL video (left) and the SIM replay (right) side by side,
time-aligned at the first grasp, and compare the motion: where the gripper grasps each object, how high it carries
it and along which arc, where and how it places it, the order and timing of the steps, and how the objects end up.
Propose the smallest parameter patch that makes the replay reproduce the demonstration more faithfully. Do not try
to generalise to other scenes here (a later loop does that): match THIS video. Only DSL fields may change; never
object_id, target_id or arm. Cite what you saw with timestamps (seconds of the side-by-side video).

Output: ONE JSON object {"from_iteration": int, "critique": [{"timestamp_sec": float, "video_frame_idx": int,
"observation": str}], "root_causes": [str], "patch": [{"object_id": "<id or *>", "set": {...}}]}"""


def _first_grasp(ep: Episode) -> float:
    ph = ep.arrays["phase"]
    idx = np.nonzero(ph == "close")[0]
    return float(ep.arrays["t"][idx[0]]) if len(idx) else 0.0


def side_by_side(video: str, task: TaskSpec, ep: Episode, video_grasp_t: float, out: Path, label: str,
                 cam_at=None) -> Path:
    """Real (left) | sim replay from the solved video camera (right), at FPS, aligned at the first grasp.
    cam_at(t) -> tracked camera for a moving camera."""
    from core.camera_solve import set_camera

    sim = build_sim(task)
    m, d = sim.m, sim.d
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "video")
    vw, vh = int(task.cameras[[c.name for c in task.cameras].index("video")].w), 0
    cap = cv2.VideoCapture(video)
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    vh = round(PANEL_W * f0.shape[0] / f0.shape[1]) if ok else 360
    r = mujoco.Renderer(m, vh, PANEL_W)
    offset = video_grasp_t - _first_grasp(ep)          # video time = sim time + offset
    t_sim, qpos = ep.arrays["t"], ep.arrays["qpos"]
    t_start = max(0.0, min(offset, n_v / fps_v - 1))
    t_end = min(n_v / fps_v, float(t_sim[-1]) + offset)
    writer = imageio.get_writer(str(out), fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    try:
        for t in np.arange(t_start, t_end, 1.0 / FPS):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps_v)))
            ok, fr = cap.read()
            real = cv2.resize(fr, (PANEL_W, vh)) if ok else np.zeros((vh, PANEL_W, 3), np.uint8)
            k = int(np.clip(np.searchsorted(t_sim, t - offset), 0, len(t_sim) - 1))
            d.qpos[:] = qpos[k]
            c = cam_at(t) if cam_at else None
            if c is not None:
                set_camera(m, d, "video", c["pos"], c["xyaxes"], c["fovy"])
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            r.update_scene(d, camera=cam)
            simf = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            frame = np.concatenate([real, simf], 1)
            for x, txt in ((10, f"REAL  t={t:5.2f}s"), (PANEL_W + 10, f"SIM {label}  phase={ep.arrays['phase'][k]}")):
                cv2.putText(frame, txt, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, txt, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
        cap.release()
        r.close()
    return out


def _score(ep: Episode, end_err: float | None) -> tuple:
    return (ep.success, -(end_err if end_err is not None else 1e3))


def run_replay_critic(video: str, task: TaskSpec, draft: PolicyConfig, adapter: RobotAdapter, argon: Argon,
                      video_grasp_t: float, motion_log: dict, end_error, out_dir: Path, max_iters: int = 2,
                      good_enough_mm: float = 15.0, cam_at=None) -> dict:
    """end_error(ep) -> mean end-pose error in mm (or None). Returns the best draft and the critique history."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = run_episode(task, draft)
    err = end_error(ep)
    best = (draft, ep, err)
    history, prev_id = [], None
    for it in range(1, max_iters + 1):
        clip = side_by_side(video, task, best[1], video_grasp_t, out_dir / f"side_by_side_iter{it}.mp4",
                            best[0].name, cam_at)
        s = best[1].summary()
        prompt = f"""{robot_facts(adapter)}

{DSL_DOC}

Current draft ({best[0].name}):
{best[0].model_dump_json(indent=1)}

Resolved values in the replay: {json.dumps(s['resolved'])}
Replay outcome: success={s['success']}, end-pose error vs the last video frame = {best[2]} mm,
predicates={json.dumps([(p['kind'], p.get('object_id'), p['ok']) for p in s['evaluation']['predicates']])},
events={json.dumps([(e['type'], e['t'], e.get('pair') or e.get('object')) for e in s['trace']['events']][:25])}
Waypoints measured from the video (metres, video time): {json.dumps(motion_log['segments'])[:6000]}

The attached side-by-side video is iteration {it}. Return the patch (from_iteration = {it})."""
        try:
            patch, resp = argon.ask_json(f"replay_critic_iter{it}", SYSTEM, prompt, PolicyPatch, [clip],
                                         model=ARGON, thinking="high", previous_id=prev_id)
            prev_id = resp.get("interaction_id") or prev_id
            cand = apply_patch(best[0], patch, name=f"draft1.{it}")
        except PatchError as e:
            history.append({"iteration": it, "error": str(e)})
            continue
        ep2 = run_episode(task, cand)
        err2 = end_error(ep2)
        kept = _score(ep2, err2) > _score(best[1], best[2])
        history.append({"iteration": it, "critique": patch.critique, "root_causes": patch.root_causes,
                        "diff": param_diff(best[0], cand), "success": ep2.success, "end_pose_err_mm": err2,
                        "kept": kept, "video": str(clip)})
        if kept:
            best = (cand, ep2, err2)
        if best[1].success and best[2] is not None and best[2] <= good_enough_mm:
            break
    final_clip = side_by_side(video, task, best[1], video_grasp_t, out_dir / "side_by_side_final.mp4", best[0].name,
                              cam_at)
    return {"draft": best[0], "episode": best[1], "end_pose_err_mm": best[2], "history": history,
            "video": str(final_clip), "start": {"success": ep.success, "end_pose_err_mm": err}}

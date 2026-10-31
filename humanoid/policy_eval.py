"""L3 and the referee on the physics rollout: run the trained policy (deterministic, no root assist) from the first
frame of the padded reference, record the G1, and score it.

- tracking (GT-free): mean joint error (rad), key-body error (cm), whether it fell, fraction of the motion completed.
- referee (eval only): the rollout's G1 joints projected into the clip at the clip's own frames, scored like the
  kinematic reference (limb cos, PCK, contact and plant frame errors), with the kinematic reference's numbers alongside.
- video: the source clip with the projected rollout skeleton | the rollout rendered from the side.

    uv run python -m humanoid.policy_eval --ref runs/<id>/clips/0278/reference.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import torch

from humanoid import penn, referee
from humanoid import skeleton as sk
from humanoid.render import _skeleton
from humanoid.retarget import g1_points
from humanoid.track_env import CTRL_HZ, PAD_START_S, TrackEnv
from humanoid.train import ActorCritic


def penn2d(m, qpos_frames: np.ndarray, ref) -> np.ndarray:
    d = mujoco.MjData(m)
    X = np.zeros((len(qpos_frames), 13, 3))
    for i, q in enumerate(qpos_frames):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        X[i] = g1_points(m, d, sk.G1_PENN)
    Xh = X / float(ref["scale"]) + ref["pin_offset"][:len(X), None, :]
    Pc = Xh @ ref["world_from_cam"] + ref["cam_origin"]
    return ref["focal"][:len(X), None, None] * Pc[..., :2] / Pc[..., 2:3] + ref["principal"]


def evaluate(ref_path: Path, policy_path: Path, out: Path, video: bool = True) -> dict:
    ref = np.load(ref_path)
    clip = str(ref["clip"])
    env = TrackEnv(ref["qpos"], float(ref["fps"]), assist=0.0, bat=str(ref.get("bat", "none")),
                   top_hand=str(ref.get("top_hand", "right")))
    ac = ActorCritic(env.obs_dim, env.nu)
    ac.load_state_dict(torch.load(policy_path, map_location="cpu"))
    ac.eval()
    obs = env.reset()
    env.t = 0
    mujoco.mj_resetData(env.m, env.d)
    env.d.qpos[:], env.d.qvel[:] = env.q_ref[0], env.v_ref[0]
    mujoco.mj_forward(env.m, env.d)
    obs = env._obs()
    qs, eq, ek, fell = [env.d.qpos.copy()], [], [], False
    while True:
        with torch.no_grad():
            a = ac.actor(ac.norm(torch.as_tensor(obs)[None]))[0].numpy()
        obs, _, done, info = env.step(a)
        qs.append(env.d.qpos.copy())
        eq.append(np.sqrt(info["e_q"] / env.nu))
        ek.append(info["e_k_cm"])
        if done:
            fell = bool(info["fell"])
            break
    qs = np.array(qs)
    n_clip = len(ref["qpos"])
    idx = [int(round(PAD_START_S * CTRL_HZ + i / float(ref["fps"]) * CTRL_HZ)) for i in range(n_clip)]
    reached = sum(i < len(qs) for i in idx)
    frames_q = qs[np.minimum(idx, len(qs) - 1)]
    rollout2d = penn2d(env.m, frames_q, ref)
    kin2d = penn2d(env.m, ref["qpos"], ref)
    res = {"clip": clip, "fell": fell, "steps": len(qs) - 1, "of": env.T - 1, "clip_frames_reached": int(reached),
           "completed": round(reached / n_clip, 3), "joint_err_rad": round(float(np.mean(eq)), 4),
           "key_err_cm": round(float(np.mean(ek)), 2),
           "referee_rollout": referee.score(clip, None, rollout2d), "referee_kinematic": referee.score(clip, None, kin2d)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    if video:
        _video(env.m, clip, qs, idx, rollout2d, out / "rollout.mp4")
    return res


def _video(m, clip: str, qs: np.ndarray, idx: list[int], rollout2d: np.ndarray, path: Path, h: int = 360) -> None:
    r = mujoco.Renderer(m, h, h)
    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -10, 90
    frames = []
    for i, k in enumerate(idx):
        img = penn.frame(clip, i).copy()
        _skeleton(img, rollout2d[i], (255, 140, 0))
        img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        d.qpos[:] = qs[min(k, len(qs) - 1)]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = d.qpos[:3] + np.array([0, 0, -0.2])
        r.update_scene(d, cam)
        frames.append(np.hstack([img, r.render()]))
    imageio.mimsave(path, frames, fps=15, macro_block_size=None)
    r.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--policy", default="")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    ref = Path(a.ref)
    pol = Path(a.policy) if a.policy else ref.parent / "rl" / "policy.pt"
    res = evaluate(ref, pol, pol.parent, video=not a.no_video)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("referee")}, indent=1))


if __name__ == "__main__":
    main()

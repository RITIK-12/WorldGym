"""Human keypoints (camera space) -> world frame -> Unitree G1 joint trajectory (the reference motion).

World frame: z up is the normal of a plane fitted to the foot keypoints over the clip (iteratively, keeping the points
within plane_band_m of each frame's lowest), x is the athlete's facing direction at frame 1, the origin is the ground
under the ankles at frame 1. SAM 3D Body points are camera space (OpenCV: x right, y down, z forward) plus cam_t, so a
static camera gives the root's travel directly; `pinned` removes the pelvis's horizontal travel instead.

Retargeting: bone_directions keeps each human bone's direction and gives it the G1's length, walking out from the hips,
so every target is reachable by construction; uniform scales every point by the leg-length ratio.
IK: damped least squares on weighted position targets plus pelvis and torso orientation, warm-started from the previous
frame, joints clamped to their ranges. Then the root is shifted so the lower sole touches the floor.
Bat (humanoid.bat): rigid in the top hand. Its axis target is the striker's knuckle line (index to pinky knuckle,
averaged over both hands, smoothed): the bat runs through the fist along it and both thumbs point to the barrel, so
the hand farther along that axis is the top hand. While the hands are close the bottom palm is held on the handle
(an IK task that replaces the bottom wrist's own target); when they separate (follow-through) it lets go.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from humanoid import skeleton as sk
from humanoid.config import HarnessConfig

REPO = Path(__file__).resolve().parents[1]
G1_DIR = REPO / "assets" / "menagerie" / "unitree_g1"
G1_SCENE = G1_DIR / "scene_mjx.xml"        # simplified colliders: the training model
G1_FULL = G1_DIR / "scene.xml"             # full collision meshes: self-collision checks
IK_POINTS = [n for n in sk.G1_POINTS if n != "mid_hip"]


@dataclass
class Reference:
    clip: str
    qpos: np.ndarray                      # (T, nq) G1 joint trajectory
    fps: float                            # robot frames per second (demo fps / time_scale)
    targets: np.ndarray                   # (T, K, 3) retarget targets, order IK_POINTS
    reached: np.ndarray                   # (T, K, 3) G1 points after IK
    human: np.ndarray                     # (T, 70, 3) human keypoints in the world frame (smoothed, human scale)
    human_raw: np.ndarray                 # (T, 70, 3) same, unsmoothed
    at_limit: np.ndarray                  # (T, n_hinge) joint clamped at a range end
    plane_rms_m: float
    world_from_cam: np.ndarray            # (3, 3) rotation
    cam_origin: np.ndarray                # (3,) world origin in camera coordinates
    scale: float                          # robot metres per human metre (root and projection)
    pin_offset: np.ndarray                # (T, 3) horizontal pelvis travel removed by `pinned` (human scale)
    root_shift: np.ndarray                # (T,) vertical ground-snap shift applied to the root (robot scale)
    focal: np.ndarray                     # (T,)
    principal: np.ndarray                 # (2,)
    info: dict = field(default_factory=dict)


_MODELS: dict[str, mujoco.MjModel] = {}


def g1_model(path: Path = G1_SCENE, bat: str = "none", top_hand: str = "right", grip: bool = False) -> mujoco.MjModel:
    key = f"{path}|{bat}|{top_hand}|{grip}"
    if key not in _MODELS:
        if bat == "none":
            _MODELS[key] = mujoco.MjModel.from_xml_path(str(path))
        else:
            from humanoid import bat as batlib
            spec = mujoco.MjSpec.from_file(str(path))
            batlib.attach(spec, top_hand, bat)
            if grip:
                batlib.add_grip_constraint(spec, top_hand)
            _MODELS[key] = spec.compile()
    return _MODELS[key]


def ref_model(ref: "Reference", full: bool = False) -> mujoco.MjModel:
    return g1_model(G1_FULL if full else G1_SCENE, ref.info.get("bat", "none"), ref.info.get("top_hand", "right"))


def g1_points(m: mujoco.MjModel, d: mujoco.MjData, names=IK_POINTS) -> np.ndarray:
    out = []
    for n in names:
        if n == "mid_hip":
            out.append((g1_points(m, d, ["l_hip"])[0] + g1_points(m, d, ["r_hip"])[0]) / 2)
            continue
        body, off = sk.G1_POINTS[n]
        b = m.body(body).id
        out.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(off))
    return np.array(out)


def stand_pose(m: mujoco.MjModel) -> np.ndarray:
    k = next((i for i in range(m.nkey) if m.key(i).name == "stand"), 0 if m.nkey else None)
    return m.key_qpos[k].copy() if k is not None else m.qpos0.copy()


def g1_bone_lengths(m: mujoco.MjModel) -> dict[tuple[str, str], float]:
    d = mujoco.MjData(m)
    d.qpos[:] = stand_pose(m)
    mujoco.mj_kinematics(m, d)
    names = sorted({n for b in sk.BONES for n in b})
    pos = dict(zip(names, g1_points(m, d, names)))
    return {b: float(np.linalg.norm(pos[b[1]] - pos[b[0]])) for b in sk.BONES}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def world_frame(P: np.ndarray, cfg: HarnessConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """(T, 70, 3) camera-space points -> (world_from_cam R, origin in camera coords, plane RMS in m)."""
    up = np.array([0.0, -1.0, 0.0])
    rms = float("nan")
    if cfg.up_mode == "feet_plane":
        F = P[:, sk.FOOT]                                  # (T, 6, 3)
        for _ in range(4):
            hgt = F @ up
            keep = hgt <= hgt.min(axis=1, keepdims=True) + cfg.plane_band_m
            pts = F[keep]
            if len(pts) < 8:
                break
            c = pts.mean(0)
            n = np.linalg.svd(pts - c)[2][-1]
            head = sk.named(P, "head").mean(0) - c
            up = n if n @ head > 0 else -n
            rms = float(np.sqrt(np.mean(((pts - c) @ up) ** 2)))
    left = (sk.named(P[:3], "l_hip") - sk.named(P[:3], "r_hip")).mean(0)
    y = _unit(left - (left @ up) * up)
    x = np.cross(y, up)
    R = np.stack([x, y, up])                               # rows: world axes in camera coordinates
    F0 = P[0, sk.FOOT]
    ground = F0[np.argsort(F0 @ up)[:3]].mean(0)           # the three lowest foot points at frame 1
    ankles = (sk.named(P[0], "l_ankle") + sk.named(P[0], "r_ankle")) / 2
    origin = ankles - ((ankles - ground) @ up) * up
    return R, origin, rms


def human_world(per: dict, cfg: HarnessConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    P = per["keypoints_3d"] + per["cam_t"][:, None, :]
    R, o, rms = world_frame(P, cfg)
    raw = (P - o) @ R.T
    pin = np.zeros((len(raw), 3))
    if cfg.root_motion == "pinned":
        hip = sk.named(raw, "mid_hip")
        pin[:, :2] = hip[:, :2] - hip[:1, :2]
        raw = raw - pin[:, None, :]
    smooth = gaussian_filter1d(raw, cfg.smooth_sigma, axis=0, mode="nearest") if cfg.smooth_sigma > 0 else raw.copy()
    return raw, smooth, R, o, pin, rms


def bat_track(per: dict, R: np.ndarray, cfg: HarnessConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """Bat axis per frame (world, toward the barrel), grip frames (hands close), and the top hand."""
    P = per["keypoints_3d"]
    knuckles = _unit(P[:, 28] - P[:, 40]) + _unit(P[:, 49] - P[:, 61])      # index - pinky knuckle, right + left
    u = _unit(gaussian_filter1d(_unit(knuckles), max(cfg.smooth_sigma, 0.5), axis=0, mode="nearest"))
    rc, lc = P[:, 21:42].mean(1), P[:, 42:63].mean(1)                      # hand centres
    grip = gaussian_filter1d(np.linalg.norm(rc - lc, axis=1), 1.0, mode="nearest") < cfg.grip_release_m
    along = ((rc - lc) * u).sum(1)
    top = "right" if not grip.any() or along[grip].mean() > 0 else "left"
    return u @ R.T, grip, top


def make_targets(H: np.ndarray, cfg: HarnessConfig, bones: dict) -> tuple[np.ndarray, float]:
    """Human world points (T, 70, 3) -> G1 targets (T, K, 3) in IK_POINTS order, plus the root scale."""
    def length(a, b):
        return np.linalg.norm(sk.named(H, b) - sk.named(H, a), axis=-1)
    human_leg = np.median((length("l_hip", "l_knee") + length("l_knee", "l_ankle") + length("r_hip", "r_knee")
                           + length("r_knee", "r_ankle")) / 2)
    robot_leg = (bones[("l_hip", "l_knee")] + bones[("l_knee", "l_ankle")] + bones[("r_hip", "r_knee")]
                 + bones[("r_knee", "r_ankle")]) / 2
    s = float(robot_leg / human_leg)
    pts = {n: sk.named(H, n) * s for n in sk.G1_POINTS}
    if cfg.retarget_mode == "bone_directions":
        out = {"mid_hip": pts["mid_hip"]}
        for a, b in sk.BONES:
            out[b] = out[a] + _unit(sk.named(H, b) - sk.named(H, a)) * bones[(a, b)]
        pts = out
    return np.stack([pts[n] for n in IK_POINTS], axis=1), s


def _frame_rot(left: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    y = _unit(left)
    z = _unit(up_hint - (up_hint @ y) * y)
    return np.stack([np.cross(y, z), y, z], axis=1)       # columns: body x, y, z in world


def _ik(m, d, target: np.ndarray, w: np.ndarray, rots: dict, cfg: HarnessConfig, q_prev: np.ndarray, iters: int,
        hinge_qadr: np.ndarray, lo: np.ndarray, hi: np.ndarray, bat: dict | None = None) -> None:
    nv = m.nv
    jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
    bodies = [m.body(sk.G1_POINTS[n][0]).id for n in IK_POINTS]
    offs = [np.asarray(sk.G1_POINTS[n][1]) for n in IK_POINTS]
    reg = np.zeros(nv)
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rows, errs = [], []
        for k, (b, off) in enumerate(zip(bodies, offs)):
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ off
            mujoco.mj_jac(m, d, jacp, None, p, b)
            rows.append(w[k] * jacp.copy())
            errs.append(w[k] * (target[k] - p))
        for body, Rdes in rots.items():
            b = m.body(body).id
            qd, qc, qe, ve = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_mat2Quat(qd, Rdes.ravel())
            mujoco.mju_mat2Quat(qc, d.xmat[b])
            mujoco.mju_negQuat(qc, qc)
            mujoco.mju_mulQuat(qe, qd, qc)
            mujoco.mju_quat2Vel(ve, qe, 1.0)
            mujoco.mj_jacBody(m, d, None, jacr, b)
            rows.append(cfg.w_orient * jacr.copy())
            errs.append(cfg.w_orient * ve)
        if bat is not None:
            b = bat["body"]
            Rb = d.xmat[b].reshape(3, 3)
            mujoco.mj_jacBody(m, d, None, jacr, b)
            rows.append(cfg.w_bat * jacr.copy())                          # rotate the bat axis onto the target
            errs.append(cfg.w_bat * np.cross(Rb[:, 2], bat["axis"]))
            if bat["grip"]:                                                # bottom palm onto the handle
                hb = bat["hand"]
                palm = d.xpos[hb] + d.xmat[hb].reshape(3, 3) @ bat["palm"]
                handle = d.xpos[b] + Rb @ np.array([0.0, 0.0, bat["s"]])
                jp, jh = np.zeros((3, nv)), np.zeros((3, nv))
                mujoco.mj_jac(m, d, jp, None, palm, hb)
                mujoco.mj_jac(m, d, jh, None, handle, b)
                rows.append(cfg.w_grip * (jp - jh))
                errs.append(cfg.w_grip * (handle - palm))
        J, e = np.vstack(rows), np.concatenate(errs)
        mujoco.mj_differentiatePos(m, reg, 1.0, d.qpos, q_prev)
        A = J.T @ J + (cfg.ik_damping ** 2 + cfg.w_smooth) * np.eye(nv)
        dq = np.linalg.solve(A, J.T @ e + cfg.w_smooth * reg)
        mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
        d.qpos[hinge_qadr] = np.clip(d.qpos[hinge_qadr], lo, hi)


def sole_points(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """(2 feet, 4 corners, 3) sole corners of the foot boxes (toe/heel x left/right edge)."""
    out = []
    for side in ("left", "right"):
        b = m.body(f"{side}_ankle_roll_link").id
        R = d.xmat[b].reshape(3, 3)
        out.append([d.xpos[b] + R @ np.array([x, y, -0.037]) for x in (0.13, -0.05) for y in (0.03, -0.03)])
    return np.array(out)


def retarget(clip: str, per: dict, cfg: HarnessConfig, fps: float) -> Reference:
    if cfg.mirror:
        from humanoid.perception import mirror
        per = mirror(per)
    raw, H, R, o, pin, rms = human_world(per, cfg)
    axis, grip, top = bat_track(per, R, cfg)
    m = g1_model(G1_SCENE, cfg.bat, top)
    d = mujoco.MjData(m)
    bones = g1_bone_lengths(m)
    bat = None
    if cfg.bat != "none":
        from humanoid import bat as batlib
        low = batlib.bottom_hand(top)
        bat = {"body": m.body("bat").id, "hand": m.body(f"{low}_wrist_yaw_link").id, "palm": batlib.PALM,
               "s": batlib.PRESETS[cfg.bat].bottom_grip}
        low_k = IK_POINTS.index(f"{low[0]}_wrist")
    targets, s = make_targets(H, cfg, bones)
    gw = {"legs": cfg.w_legs, "feet": cfg.w_feet, "arms": cfg.w_arms, "head": cfg.w_head}
    w = np.array([gw[sk.GROUP[n]] for n in IK_POINTS])
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = np.array([m.jnt_qposadr[j] for j in hinge])
    lo, hi = m.jnt_range[hinge, 0], m.jnt_range[hinge, 1]
    T = len(H)
    qpos = np.zeros((T, m.nq))
    reached = np.zeros_like(targets)
    d.qpos[:] = stand_pose(m)
    hip0 = targets[0, IK_POINTS.index("l_hip")] / 2 + targets[0, IK_POINTS.index("r_hip")] / 2
    d.qpos[:2] = hip0[:2]
    for t in range(T):
        tg = {n: targets[t, k] for k, n in enumerate(IK_POINTS)}
        left_hip = tg["l_hip"] - tg["r_hip"]
        spine = (tg["l_shoulder"] + tg["r_shoulder"]) / 2 - (tg["l_hip"] + tg["r_hip"]) / 2
        rots = {"pelvis": _frame_rot(left_hip, np.array([0.0, 0.0, 1.0])),
                "torso_link": _frame_rot(tg["l_shoulder"] - tg["r_shoulder"], spine)}
        q_prev = d.qpos.copy()
        wt = w
        if bat is not None:
            bat.update(axis=axis[t], grip=bool(grip[t]))
            if grip[t]:
                wt = w.copy()
                wt[low_k] = 0.0          # the grip task places the bottom hand
        _ik(m, d, targets[t], wt, rots, cfg, q_prev, cfg.ik_iters * (4 if t == 0 else 1), qadr, lo, hi, bat)
        mujoco.mj_kinematics(m, d)
        qpos[t] = d.qpos
        reached[t] = g1_points(m, d)
    shift = np.zeros(T)
    if cfg.ground_snap == "smoothed":
        low = np.zeros(T)
        for t in range(T):
            d.qpos[:] = qpos[t]
            mujoco.mj_kinematics(m, d)
            low[t] = sole_points(m, d)[..., 2].min()
        shift = -gaussian_filter1d(low, 1.0, mode="nearest")
        qpos[:, 2] += shift
        reached[..., 2] += shift[:, None]
    at_limit = (np.abs(qpos[:, qadr] - lo) < 1e-3) | (np.abs(qpos[:, qadr] - hi) < 1e-3)
    return Reference(clip=clip, qpos=qpos, fps=fps / cfg.time_scale, targets=targets, reached=reached, human=H,
                     human_raw=raw, at_limit=at_limit, plane_rms_m=rms, world_from_cam=R, cam_origin=o, scale=s,
                     pin_offset=pin, root_shift=shift, focal=np.ravel(per["focal_length"]).astype(float),
                     principal=np.asarray(per["principal"], float),
                     info={"redetect_frac": float(np.mean(per["redetect"])), "frames": T, "bat": cfg.bat,
                           "top_hand": top, "grip_frames": int(grip.sum()), "bat_axis": axis, "mirror": cfg.mirror})


def project(ref: Reference, X: np.ndarray) -> np.ndarray:
    """Robot-world points (T, K, 3) -> image pixels (T, K, 2) through the clip's camera."""
    Xh = X / ref.scale + ref.pin_offset[:, None, :]
    Pc = Xh @ ref.world_from_cam + ref.cam_origin
    return ref.focal[:, None, None] * Pc[..., :2] / Pc[..., 2:3] + ref.principal

"""The RL-usefulness ladder, GT-free: is this reference motion something a policy can learn to track on a real G1?

L0 perception sanity  - ground-plane fit residual, torso jitter (high-frequency residual, so real swing rotation is
                        not noise), over-smoothing (on the points retargeting uses), bone-length consistency, re-detects.
L1 kinematic          - IK residual, joints pinned at range ends, joint speed over the datasheet, foot skate, floor
                        penetration, self-collision (full collision model with the bat; pairs already touching in
                        the stand pose are collision-model overlaps and the bat in the hands is the grip, so both are
                        ignored; the bat hitting the robot's own body counts).
L2 dynamic            - inverse dynamics with contacts and constraints off gives the generalized force the motion
                        needs; the feet in contact (friction pyramids, NNLS) explain what they can; the rest of the
                        root wrench is the ghost force/torque nobody can supply. Joint torques with those contact
                        forces are compared with the actuator limits.

Each metric has a tolerance (goodness = exp(-value / tol)) and a pass bar. A rung passes when every metric is under its
bar; the ladder score is the number of rungs passed plus the mean goodness of the first rung that fails, so it climbs
from 0 to 3. Every metric comes with frame-level facts (what, where, when) for the critic; never causes.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls

from humanoid import skeleton as sk
from humanoid.config import HarnessConfig
from humanoid.retarget import Reference, ref_model, sole_points, stand_pose

# (tolerance, pass bar, unit)
SPEC = {
    "L0": {"plane_rms_cm": (1.5, 3.0, "cm"), "torso_jitter_ms2": (6.0, 15.0, "m/s^2"),
           "oversmooth_cm": (1.0, 2.5, "cm"), "bone_cv": (0.04, 0.08, ""), "redetect_frac": (0.05, 0.15, "")},
    "L1": {"ik_err_cm": (2.0, 4.0, "cm"), "limit_frac": (0.02, 0.05, ""), "speed_over_frac": (0.01, 0.03, ""),
           "skate_cm_s": (5.0, 12.0, "cm/s"), "penetration_cm": (0.5, 1.5, "cm"), "self_collision_frac": (0.02, 0.05, "")},
    "L2": {"ghost_force_mg": (0.08, 0.2, "x body weight"), "ghost_torque_m": (0.04, 0.1, "m (N·m per N of weight)"),
           "torque_over_frac": (0.01, 0.03, "")},
}
MU = 0.8
# Joint speed limits (rad/s), approximate Unitree G1 datasheet values: big leg/waist motors vs arm and wrist motors.
SPEED = {"hip": 20.0, "knee": 20.0, "ankle": 20.0, "waist": 20.0, "shoulder": 30.0, "elbow": 30.0, "wrist": 30.0}


def _hinges(m):
    return [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def derivatives(m: mujoco.MjModel, qpos: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    T = len(qpos)
    qvel = np.zeros((T, m.nv))
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        mujoco.mj_differentiatePos(m, qvel[t], (b - a) * dt, qpos[a], qpos[b])
    qacc = np.gradient(qvel, dt, axis=0)
    return qvel, qacc


def l0(ref: Reference) -> tuple[dict, list[str]]:
    dt = 1.0 / ref.fps
    torso_n = ("mid_hip", "l_shoulder", "r_shoulder")
    torso = np.stack([sk.named(ref.human, n) for n in torso_n], 1)
    slow = gaussian_filter1d(np.stack([sk.named(ref.human_raw, n) for n in torso_n], 1), 3.0, axis=0, mode="nearest")
    acc = np.linalg.norm(np.diff(torso - slow, 2, axis=0), axis=-1) / dt ** 2
    used = [n for n in sk.G1_POINTS]
    over = float(np.mean([np.linalg.norm(sk.named(ref.human, n) - sk.named(ref.human_raw, n), axis=-1).mean()
                          for n in used]) * 100)
    lengths = np.stack([np.linalg.norm(sk.named(ref.human_raw, b) - sk.named(ref.human_raw, a), axis=-1)
                        for a, b in sk.BONES if a != "mid_hip"], 1)
    cv = float(np.mean(lengths.std(0) / lengths.mean(0)))
    met = {"plane_rms_cm": ref.plane_rms_m * 100 if np.isfinite(ref.plane_rms_m) else 10.0,
           "torso_jitter_ms2": float(acc.mean()), "oversmooth_cm": float(over), "bone_cv": cv,
           "redetect_frac": ref.info.get("redetect_frac", 0.0)}
    facts = []
    if len(acc):
        t = int(acc.max(1).argmax())
        facts.append(f"torso high-frequency acceleration peaks at {acc.max():.0f} m/s^2 around frame {t + 1}")
    return met, facts


def l1(ref: Reference, cfg: HarnessConfig) -> tuple[dict, list[str]]:
    m = ref_model(ref)
    d = mujoco.MjData(m)
    hinge = _hinges(m)
    names = [m.joint(j).name for j in hinge]
    dt = 1.0 / ref.fps
    qvel, _ = derivatives(m, ref.qpos, dt)
    vmax = np.array([next(v for k, v in SPEED.items() if k in n) for n in names])
    speed = np.abs(qvel[:, [m.jnt_dofadr[j] for j in hinge]])
    over = speed > vmax
    err = np.linalg.norm(ref.reached - ref.targets, axis=-1)
    soles = np.zeros((len(ref.qpos), 2, 4, 3))
    for t, q in enumerate(ref.qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        soles[t] = sole_points(m, d)
    centre = soles.mean(2)
    contact = soles[..., 2].min(2) < cfg.contact_height_m
    both = contact[1:] & contact[:-1]
    slide = np.linalg.norm(np.diff(centre[..., :2], axis=0), axis=-1) / dt
    skate = float(slide[both].mean() * 100) if both.any() else 0.0
    pen = float(max(0.0, -soles[..., 2].min()) * 100)
    selfc, pairs = _self_collisions(ref)
    met = {"ik_err_cm": float(err.mean() * 100), "limit_frac": float(ref.at_limit.mean()),
           "speed_over_frac": float(over.mean()), "skate_cm_s": skate, "penetration_cm": pen,
           "self_collision_frac": float(np.mean(selfc > 0))}
    facts = []
    k = err.mean(0).argmax()
    facts.append(f"largest IK residual: {_ik_name(k)} {err.mean(0)[k] * 100:.1f} cm mean, {err[:, k].max() * 100:.1f} cm max")
    for j in np.argsort(-ref.at_limit.mean(0))[:3]:
        if ref.at_limit[:, j].any():
            fr = np.flatnonzero(ref.at_limit[:, j])
            facts.append(f"{names[j]} at its range end in {len(fr)} frames ({fr[0] + 1}-{fr[-1] + 1})")
    for j in np.argsort(-over.sum(0))[:3]:
        if over[:, j].any():
            fr = np.flatnonzero(over[:, j])
            facts.append(f"{names[j]} over {vmax[j]:.0f} rad/s in frames {fr[0] + 1}-{fr[-1] + 1}, peak {speed[:, j].max():.1f} rad/s")
    if both.any():
        t, f = np.unravel_index(np.argmax(np.where(both, slide, 0)), slide.shape)
        facts.append(f"{'left' if f == 0 else 'right'} foot slides {slide[t, f] * 100:.0f} cm/s while on the floor at frame {t + 2}")
    if selfc.any():
        top = ", ".join(f"{a}/{b} x{n}" for (a, b), n in pairs.most_common(3))
        facts.append(f"self-collision in {int((selfc > 0).sum())} frames, first at frame {int(np.flatnonzero(selfc)[0]) + 1}: {top}")
    return met, facts


def _ik_name(k: int) -> str:
    from humanoid.retarget import IK_POINTS
    return IK_POINTS[k]


def _body_pairs(m, d, depth: float = -0.01) -> list[tuple[int, int]]:
    out = []
    for c in d.contact[:d.ncon]:
        b1, b2 = sorted((int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])))
        if c.dist < depth and b1 != 0:
            out.append((b1, b2))
    return out


def _self_collisions(ref: Reference):
    import collections
    m = ref_model(ref, full=True)
    d = mujoco.MjData(m)
    d.qpos[:] = stand_pose(m)
    mujoco.mj_forward(m, d)
    ignore = set(_body_pairs(m, d, 0.0))                       # overlapping in the stand pose: model artefacts
    if ref.info.get("bat", "none") != "none":
        bat = m.body("bat").id
        for side in ("left", "right"):
            for link in ("wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link"):
                ignore.add(tuple(sorted((bat, m.body(f"{side}_{link}").id))))
    out = np.zeros(len(ref.qpos), int)
    pairs = collections.Counter()
    for t, q in enumerate(ref.qpos):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        hits = [p for p in _body_pairs(m, d) if p not in ignore]
        out[t] = len(hits)
        pairs.update((m.body(a).name, m.body(b).name) for a, b in hits)
    return out, pairs


def l2(ref: Reference, cfg: HarnessConfig) -> tuple[dict, list[str]]:
    m = ref_model(ref)
    d = mujoco.MjData(m)
    dt = 1.0 / ref.fps
    qvel, qacc = derivatives(m, ref.qpos, dt)
    flags = m.opt.disableflags
    m.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    weight = float(m.body_subtreemass[1] * 9.81)
    hinge = _hinges(m)
    dofs = [m.jnt_dofadr[j] for j in hinge]
    names = [m.joint(j).name for j in hinge]
    frc = m.jnt_actfrcrange[hinge, 1]
    dirs = [_unit(np.array([sx * MU, sy * MU, 1.0])) for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    jac = np.zeros((3, m.nv))
    T = len(ref.qpos)
    gf, gt, ratio = np.zeros(T), np.zeros(T), np.zeros((T, len(hinge)))
    try:
        for t in range(T):
            d.qpos[:], d.qvel[:], d.qacc[:] = ref.qpos[t], qvel[t], qacc[t]
            mujoco.mj_inverse(m, d)
            need = d.qfrc_inverse.copy()
            cols = []
            for f, side in enumerate(("left", "right")):
                b = m.body(f"{side}_ankle_roll_link").id
                for p in sole_points(m, d)[f]:
                    if p[2] < cfg.contact_height_m:
                        mujoco.mj_jac(m, d, jac, None, p, b)
                        cols += [jac.T @ u for u in dirs]
            if cols:
                A = np.stack(cols, 1)
                c, _ = nnls(A[:6], need[:6])
                supplied = A @ c
            else:
                supplied = np.zeros(m.nv)
            ghost = need[:6] - supplied[:6]
            gf[t], gt[t] = np.linalg.norm(ghost[:3]) / weight, np.linalg.norm(ghost[3:6]) / weight
            ratio[t] = np.abs(need[dofs] - supplied[dofs]) / frc
    finally:
        m.opt.disableflags = flags
    met = {"ghost_force_mg": float(gf.mean()), "ghost_torque_m": float(gt.mean()),
           "torque_over_frac": float((ratio > 1).mean())}
    facts = [f"ghost force peaks at {gf.max():.2f} x body weight at frame {int(gf.argmax()) + 1}; "
             f"ghost torque peaks at {gt.max():.3f} m at frame {int(gt.argmax()) + 1}"]
    for j in np.argsort(-ratio.max(0))[:3]:
        if ratio[:, j].max() > 1:
            facts.append(f"{names[j]} needs {ratio[:, j].max():.1f}x its torque limit (frame {int(ratio[:, j].argmax()) + 1})")
    return met, facts


def _unit(v):
    return v / np.linalg.norm(v)


def evaluate(ref: Reference, cfg: HarnessConfig) -> dict:
    rungs = {}
    facts = []
    for name, fn in (("L0", lambda: l0(ref)), ("L1", lambda: l1(ref, cfg)), ("L2", lambda: l2(ref, cfg))):
        met, f = fn()
        good = {k: float(np.exp(-max(v, 0.0) / SPEC[name][k][0])) for k, v in met.items()}
        passed = all(met[k] <= SPEC[name][k][1] for k in met)
        rungs[name] = {"metrics": met, "goodness": good, "passed": passed, "score": float(np.mean(list(good.values())))}
        facts += [f"[{name}] {x}" for x in f]
    score = 0.0
    for name in ("L0", "L1", "L2"):
        if rungs[name]["passed"]:
            score += 1.0
        else:
            score += rungs[name]["score"]
            break
    return {"score": round(score, 4), "rungs": rungs, "facts": facts}


def table(ev: dict) -> str:
    lines = [f"ladder score {ev['score']:.3f} / 3"]
    for name, r in ev["rungs"].items():
        lines.append(f"{name} {'PASS' if r['passed'] else 'fail'} (mean goodness {r['score']:.2f})")
        for k, v in r["metrics"].items():
            tol, bar, unit = SPEC[name][k]
            lines.append(f"  {k:22s} {v:9.4f} {unit:24s} bar {bar}{'  <-- over' if v > bar else ''}")
    return "\n".join(lines)

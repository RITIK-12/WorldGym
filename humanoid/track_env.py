"""G1 motion-tracking environment on CPU MuJoCo, for one reference motion from the harness.

Control at 50 Hz (4 ms physics x 5). Action: 29 joint offsets added to the reference joint angles, used as the
position actuators' targets, so the policy learns corrections, not the motion. Observation: deployable signals only,
from the pelvis IMU and the joint encoders (projected gravity, angular velocity, joint positions and velocities, last
action) plus the command (phase, reference joints now and next step).
Reward (DeepMimic-style): joint pose, joint velocity, key-body positions relative to the root (feet, wrists, head),
root height and orientation, minus an action-rate penalty.
Episodes start at a random phase from the reference state (reference state initialisation) and end when the root
height is off by more than 0.25 m or the torso tilt by more than 0.8 rad (relaxed tolerances, the VideoMimic lesson),
or at the end of the padded reference: 0.5 s holding the first frame, the motion, 1.0 s holding the last frame.
A root assist (a PD wrench on the pelvis toward the reference, scaled by `assist`) can be annealed to zero during
training (residual force control); evaluation always runs with assist 0.
"""
from __future__ import annotations

import multiprocessing as mp

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from humanoid.retarget import G1_SCENE, g1_model

CTRL_HZ, DECIM = 50, 5
ACTION_SCALE = 0.25
KEY_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
PAD_START_S, PAD_END_S = 0.5, 1.0


def resample(qpos: np.ndarray, fps: float, m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Reference at its own fps -> padded 50 Hz qpos and qvel."""
    T = len(qpos)
    t_src = np.arange(T) / fps
    t_new = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / CTRL_HZ)
    out = np.empty((len(t_new), m.nq))
    for j in range(m.nq):
        if 3 <= j < 7:
            continue
        out[:, j] = np.interp(t_new, t_src, qpos[:, j])
    quat_xyzw = qpos[:, [4, 5, 6, 3]]
    out[:, [4, 5, 6, 3]] = Slerp(t_src, Rotation.from_quat(quat_xyzw))(t_new).as_quat()
    pad0 = np.repeat(out[:1], int(PAD_START_S * CTRL_HZ), 0)
    pad1 = np.repeat(out[-1:], int(PAD_END_S * CTRL_HZ), 0)
    out = np.vstack([pad0, out, pad1])
    vel = np.zeros((len(out), m.nv))
    for t in range(len(out) - 1):
        mujoco.mj_differentiatePos(m, vel[t], 1.0 / CTRL_HZ, out[t], out[t + 1])
    vel[-1] = vel[-2] * 0
    return out, vel


def _heading_inv(quat_wxyz: np.ndarray) -> np.ndarray:
    yaw = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler("ZYX")[0]
    return Rotation.from_euler("Z", -yaw).as_matrix()


class TrackEnv:
    def __init__(self, ref_qpos: np.ndarray, ref_fps: float, seed: int = 0, assist: float = 0.0, bat: str = "none",
                 top_hand: str = "right"):
        # the bat rides in the hands during training (its mass and the bottom hand's grip constraint), as in the demo
        self.m = g1_model(G1_SCENE, bat, top_hand, grip=bat != "none") if bat != "none" else \
            mujoco.MjModel.from_xml_path(str(G1_SCENE))
        self.m.opt.timestep = 1.0 / (CTRL_HZ * DECIM)
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.q_ref, self.v_ref = resample(ref_qpos, ref_fps, self.m)
        self.T = len(self.q_ref)
        self.assist = assist
        self.nu = self.m.nu
        self.lo, self.hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        self.jadr = np.array([self.m.jnt_qposadr[self.m.actuator_trnid[a, 0]] for a in range(self.nu)])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.actuator_trnid[a, 0]] for a in range(self.nu)])
        self.pelvis = self.m.body("pelvis").id
        self.torso = self.m.body("torso_link").id
        self.keys = [self.m.body(b).id for b in KEY_BODIES]
        self.key_ref = self._key_positions_ref()
        self.obs_dim = 3 + 3 + 2 * self.nu + self.nu + 2 + 2 * self.nu
        self.t = 0
        self.last_a = np.zeros(self.nu)

    def _key_positions_ref(self) -> np.ndarray:
        d = mujoco.MjData(self.m)
        out = np.zeros((self.T, len(self.keys), 3))
        for t in range(self.T):
            d.qpos[:] = self.q_ref[t]
            mujoco.mj_kinematics(self.m, d)
            Hinv = _heading_inv(self.q_ref[t, 3:7])
            out[t] = (d.xpos[self.keys] - d.xpos[self.pelvis]) @ Hinv.T
        return out

    def reset(self) -> np.ndarray:
        self.t = int(self.rng.integers(0, max(1, self.T - CTRL_HZ)))
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.q_ref[self.t]
        self.d.qvel[:] = self.v_ref[self.t]
        mujoco.mj_forward(self.m, self.d)
        self.last_a = np.zeros(self.nu)
        return self._obs()

    def _obs(self) -> np.ndarray:
        d = self.d
        R = d.xmat[self.pelvis].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        angvel = d.qvel[3:6]                         # free joint angular velocity is in the body frame
        q = d.qpos[self.jadr] - self.q_ref[self.t, self.jadr]
        qd = d.qvel[self.vadr] * 0.1
        phase = 2 * np.pi * self.t / self.T
        nxt = min(self.t + 1, self.T - 1)
        return np.concatenate([grav, angvel * 0.25, q, qd, self.last_a, [np.sin(phase), np.cos(phase)],
                               self.q_ref[self.t, self.jadr], self.q_ref[nxt, self.jadr]]).astype(np.float32)

    def step(self, a: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = np.clip(a, -3.0, 3.0)
        nxt = min(self.t + 1, self.T - 1)
        self.d.ctrl[:] = np.clip(self.q_ref[nxt, self.jadr] + ACTION_SCALE * a, self.lo, self.hi)
        for _ in range(DECIM):
            if self.assist > 0:
                self._assist(nxt)
            mujoco.mj_step(self.m, self.d)
        self.d.xfrc_applied[self.pelvis] = 0
        self.t = nxt
        r, info = self._reward(a)
        self.last_a = a
        d = self.d
        h_err = abs(d.qpos[2] - self.q_ref[self.t, 2])
        up = d.xmat[self.torso].reshape(3, 3)[:, 2]
        tilt = np.arccos(np.clip(up[2], -1, 1))
        fell = h_err > 0.25 or abs(tilt - self._ref_tilt(self.t)) > 0.8
        done = fell or self.t >= self.T - 1
        info.update(fell=fell, timeout=(not fell) and done)
        return self._obs(), r, done, info

    def _ref_tilt(self, t: int) -> float:
        if not hasattr(self, "_tilts"):
            dd = mujoco.MjData(self.m)
            self._tilts = np.zeros(self.T)
            for k in range(self.T):
                dd.qpos[:] = self.q_ref[k]
                mujoco.mj_kinematics(self.m, dd)
                self._tilts[k] = np.arccos(np.clip(dd.xmat[self.torso].reshape(3, 3)[2, 2], -1, 1))
        return self._tilts[t]

    def _assist(self, t: int) -> None:
        d = self.d
        f = 1500.0 * (self.q_ref[t, :3] - d.qpos[:3]) - 120.0 * d.qvel[:3]
        qe = np.zeros(4)
        qc = np.zeros(4)
        mujoco.mju_negQuat(qc, d.qpos[3:7])
        mujoco.mju_mulQuat(qe, self.q_ref[t, 3:7], qc)
        rv = np.zeros(3)
        mujoco.mju_quat2Vel(rv, qe, 1.0)
        R = d.xmat[self.pelvis].reshape(3, 3)
        tq = 300.0 * rv - 30.0 * (R @ d.qvel[3:6])
        d.xfrc_applied[self.pelvis] = self.assist * np.concatenate([f, tq])

    def _reward(self, a: np.ndarray) -> tuple[float, dict]:
        d, t = self.d, self.t
        e_q = np.sum((d.qpos[self.jadr] - self.q_ref[t, self.jadr]) ** 2)
        e_v = np.sum((d.qvel[self.vadr] - self.v_ref[t, self.vadr]) ** 2)
        Hinv = _heading_inv(d.qpos[3:7])
        keys = (d.xpos[self.keys] - d.xpos[self.pelvis]) @ Hinv.T
        e_k = np.sum((keys - self.key_ref[t]) ** 2)
        e_h = (d.qpos[2] - self.q_ref[t, 2]) ** 2
        qe = np.zeros(4)
        qc = np.zeros(4)
        mujoco.mju_negQuat(qc, d.qpos[3:7])
        mujoco.mju_mulQuat(qe, self.q_ref[t, 3:7], qc)
        e_o = 2 * np.arccos(np.clip(abs(qe[0]), -1, 1))
        e_a = np.sum((a - self.last_a) ** 2)
        r = (0.40 * np.exp(-2.0 * e_q) + 0.10 * np.exp(-0.1 * e_v) + 0.25 * np.exp(-20.0 * e_k)
             + 0.10 * np.exp(-40.0 * e_h) + 0.15 * np.exp(-4.0 * e_o ** 2) - 0.005 * e_a)
        return float(r), {"e_q": float(e_q), "e_k_cm": float(np.sqrt(e_k / len(self.keys)) * 100)}


def _worker(remote, ref_qpos, ref_fps, n, seed, bat, top_hand):
    envs = [TrackEnv(ref_qpos, ref_fps, seed * 1000 + i, bat=bat, top_hand=top_hand) for i in range(n)]
    while True:
        cmd, arg = remote.recv()
        if cmd == "reset":
            remote.send(np.stack([e.reset() for e in envs]))
        elif cmd == "assist":
            for e in envs:
                e.assist = arg
            remote.send(None)
        elif cmd == "step":
            obs, rew, done, infos = [], [], [], []
            for e, a in zip(envs, arg):
                o, r, dn, info = e.step(a)
                if dn:
                    info["ep_len"] = e.t
                    o = e.reset()
                obs.append(o)
                rew.append(r)
                done.append(dn)
                infos.append(info)
            remote.send((np.stack(obs), np.array(rew, np.float32), np.array(done), infos))
        elif cmd == "close":
            remote.close()
            return


class VecTrackEnv:
    """n_workers processes x envs_per_worker environments, stepped in lockstep."""

    def __init__(self, ref_qpos: np.ndarray, ref_fps: float, n_workers: int = 8, envs_per_worker: int = 32,
                 bat: str = "none", top_hand: str = "right"):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, ref_qpos, ref_fps, envs_per_worker, w, bat, top_hand), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.n = n_workers * envs_per_worker
        self.per = envs_per_worker
        probe = TrackEnv(ref_qpos, ref_fps, bat=bat, top_hand=top_hand)
        self.obs_dim, self.act_dim, self.T = probe.obs_dim, probe.nu, probe.T

    def reset(self) -> np.ndarray:
        for r in self.remotes:
            r.send(("reset", None))
        return np.concatenate([r.recv() for r in self.remotes])

    def set_assist(self, x: float) -> None:
        for r in self.remotes:
            r.send(("assist", x))
        for r in self.remotes:
            r.recv()

    def step(self, actions: np.ndarray):
        for k, r in enumerate(self.remotes):
            r.send(("step", actions[k * self.per:(k + 1) * self.per]))
        out = [r.recv() for r in self.remotes]
        return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                np.concatenate([o[2] for o in out]), [i for o in out for i in o[3]])

    def close(self) -> None:
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=2)

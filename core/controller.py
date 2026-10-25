"""Per-arm Cartesian controller.

- DLS IK runs one step per physics step on a scratch MjData holding the *commanded* joints (q_cmd), so contact
  disturbances never wind up the IK.
- Position servos get a velocity feed-forward: ctrl = q_cmd + tau * qd_cmd with tau = kv/kp = -biasprm[2]/gainprm[0].
  Without it Menagerie servos lag ~0.1 s x velocity.
- Per-joint datasheet velocity limits clip each IK step.
- Collision reflex: if the commanded TCP runs > 12 mm ahead of the actual TCP the path clock freezes; after 0.3 s
  blocked the segment aborts, q_cmd snaps to the measured joints and a MOTION_BLOCKED fact is emitted.
- Gripper: open = PD on finger width, close = constant squeeze force.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from core.schemas import ArmSpec

LAG_FREEZE_M = 0.012
BLOCK_ABORT_S = 0.3
SETTLE_S = 0.15


def top_down_R(yaw: float, closing_axis: str = "y") -> np.ndarray:
    """TCP rotation pointing straight down with the finger closing axis at world yaw `yaw`."""
    c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    if closing_axis == "y":
        y = c
        x = np.cross(y, z)
    else:
        x = c
        y = np.cross(z, x)
    return np.column_stack([x, y, z])


def min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau * tau)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class CartesianPath:
    """Piecewise-linear path through `points` with ONE min-jerk time scaling over its arc length."""
    points: list[np.ndarray]
    yaw0: float
    yaw1: float
    speed: float
    t0: float
    seg_len: np.ndarray = field(init=False)
    length: float = field(init=False)
    duration: float = field(init=False)

    def __post_init__(self):
        self.points = [np.asarray(p, float) for p in self.points]
        self.seg_len = np.array([np.linalg.norm(b - a) for a, b in zip(self.points[:-1], self.points[1:])])
        self.length = float(self.seg_len.sum())
        self.duration = max(0.4, self.length / max(self.speed, 1e-3) * 1.4, abs(self.yaw1 - self.yaw0) / 1.2)

    def sample(self, t: float) -> tuple[np.ndarray, float, bool]:
        s = min_jerk((t - self.t0) / self.duration)
        yaw = self.yaw0 + s * (self.yaw1 - self.yaw0)
        done = t - self.t0 >= self.duration + SETTLE_S
        if self.length < 1e-9:
            return self.points[-1].copy(), yaw, done
        d = s * self.length
        for k, L in enumerate(self.seg_len):
            if d <= L or k == len(self.seg_len) - 1:
                a, b = self.points[k], self.points[k + 1]
                return a + (b - a) * (min(d, L) / max(L, 1e-9)), yaw, done
            d -= L
        return self.points[-1].copy(), yaw, done


class ArmController:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, arm: ArmSpec):
        self.m, self.d, self.arm = m, d, arm
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm.joints]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in arm.actuators])
        self.tau = -m.actuator_biasprm[self.act, 2] / m.actuator_gainprm[self.act, 0]
        self.lo, self.hi = m.jnt_range[jid].T.copy()
        self.vel = np.array(arm.vel_limits)
        self.q_home = np.array(arm.home_qpos)
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{arm.name}")
        self.n = len(jid)
        self.scratch = mujoco.MjData(m)
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        self.dt = m.opt.timestep

        g = arm.gripper
        self.g_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, g.actuator)
        gj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in g.joints]
        self.g_qadr = np.array([m.jnt_qposadr[j] for j in gj])
        self.g_dadr = np.array([m.jnt_dofadr[j] for j in gj])
        self.grip_mode: tuple[str, float] = ("open", g.max_width_m)

        self.path: CartesianPath | None = None
        self.blocked_since: float | None = None
        self.events: list[dict] = []          # MOTION_BLOCKED facts, drained by telemetry
        self.reset_cmd()

    # ------------------------------------------------------------------------------------------------------------------
    def reset_cmd(self) -> None:
        self.q_cmd = self.d.qpos[self.qadr].copy()
        self.q_prev = self.q_cmd.copy()
        self.scratch.qpos[:] = self.d.qpos

    def set_joints(self, q) -> None:
        """Teleport the arm (before t=0 only) and align the command."""
        self.d.qpos[self.qadr] = q
        self.d.qvel[self.dadr] = 0
        self.d.ctrl[self.act] = q
        self.reset_cmd()

    def cmd_pose(self) -> tuple[np.ndarray, np.ndarray]:
        self.scratch.qpos[self.qadr] = self.q_cmd
        mujoco.mj_kinematics(self.m, self.scratch)
        mujoco.mj_comPos(self.m, self.scratch)
        return self.scratch.site_xpos[self.site].copy(), self.scratch.site_xmat[self.site].reshape(3, 3).copy()

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def tcp_yaw(self, R: np.ndarray | None = None) -> float:
        R = self.tcp_pose()[1] if R is None else R
        col = R[:, 1] if self.arm.closing_axis == "y" else R[:, 0]
        return math.atan2(col[1], col[0])

    def gripper_width(self) -> float:
        return float(self.d.qpos[self.g_qadr].sum())

    # ------------------------------------------------------------------------------------------------------------------
    def ik_step(self, p_des: np.ndarray, R_des: np.ndarray) -> float:
        p, R = self.cmd_pose()
        e = np.r_[p_des - p, 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))]
        mujoco.mj_jacSite(self.m, self.scratch, self.jacp, self.jacr, self.site)
        J = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])
        JJt = J @ J.T + 0.03 ** 2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)
        if self.n > 6:  # null-space pull to home only exists for redundant arms; on 6-DOF arms it causes drift
            N = np.eye(self.n) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (0.02 * (self.q_home - self.q_cmd))
        dq = np.clip(dq, -self.vel * self.dt, self.vel * self.dt)
        self.q_cmd = np.clip(self.q_cmd + dq, self.lo + 0.02, self.hi - 0.02)
        return float(np.linalg.norm(e[:3]))

    def solve_ik(self, p_des, R_des, iters: int = 400, tol: float = 1e-4) -> tuple[np.ndarray, float]:
        """Offline IK (used by self-tests / planning): iterate ik_step without velocity limits mattering much."""
        saved = self.q_cmd.copy()
        err = 1.0
        for _ in range(iters):
            p, R = self.cmd_pose()
            e = np.r_[p_des - p, 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))]
            err = float(np.linalg.norm(e[:3]))
            if err < tol and np.linalg.norm(e[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.scratch, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 ** 2 * np.eye(6), e)
            self.q_cmd = np.clip(self.q_cmd + dq, self.lo + 0.02, self.hi - 0.02)
        q = self.q_cmd.copy()
        self.q_cmd = saved
        return q, err

    def choose_yaw(self, yaw: float) -> float:
        """Pick yaw + k*pi nearest the current command yaw that keeps the last joint inside its range."""
        _, R = self.cmd_pose()
        cur = self.tcp_yaw(R)
        mujoco.mj_jacSite(self.m, self.scratch, self.jacp, self.jacr, self.site)
        jz = self.jacr[2, self.dadr[-1]]
        jz = jz if abs(jz) > 0.3 else -1.0
        best, best_cost = None, 1e9
        for k in range(-3, 4):
            cand = yaw + k * math.pi
            dth = cand - cur
            q_last = self.q_cmd[-1] + dth / jz
            if not (self.lo[-1] + 0.1 <= q_last <= self.hi[-1] - 0.1):
                continue
            if abs(dth) < best_cost:
                best, best_cost = cand, abs(dth)
        return cur + wrap(yaw - cur) if best is None else best

    # ------------------------------------------------------------------------------------------------------------------
    def move(self, points: list, yaw1: float, speed: float) -> CartesianPath:
        p, R = self.cmd_pose()
        self.path = CartesianPath([p, *points], self.tcp_yaw(R), yaw1, speed, self.d.time)
        self.blocked_since = None
        return self.path

    def step_motion(self) -> tuple[bool, float]:
        """Advance the active path one physics step. Returns (done, ik_err)."""
        if self.path is None:
            return True, 0.0
        t = self.d.time
        cmd_p, _ = self.cmd_pose()
        act_p, _ = self.tcp_pose()
        lag = float(np.linalg.norm(cmd_p - act_p))
        if lag > LAG_FREEZE_M:
            self.path.t0 += self.dt
            if self.blocked_since is None:
                self.blocked_since = t
            elif t - self.blocked_since > BLOCK_ABORT_S:
                self.events.append({"type": "MOTION_BLOCKED", "t": t, "lag_m": lag, "tcp_xyz": act_p.tolist()})
                self.q_cmd = self.d.qpos[self.qadr].copy()
                self.q_prev = self.q_cmd.copy()
                self.path = None
                self.blocked_since = None
                return True, lag
        else:
            self.blocked_since = None
        p_des, yaw, done = self.path.sample(t)
        err = self.ik_step(p_des, top_down_R(yaw, self.arm.closing_axis))
        if done:
            self.path = None
        return done, err

    def apply(self) -> None:
        qd_cmd = (self.q_cmd - self.q_prev) / self.dt
        self.q_prev = self.q_cmd.copy()
        self.d.ctrl[self.act] = self.q_cmd + self.tau * qd_cmd
        mode, val = self.grip_mode
        g = self.arm.gripper
        if mode == "close":
            ctrl = -val / g.finger_force_per_ctrl
        else:
            L = 0.5 * float(self.d.qpos[self.g_qadr].sum())
            Ld = 0.5 * float(self.d.qvel[self.g_dadr].sum())
            ctrl = 4000.0 * (val / 2 - L) - 60.0 * Ld
        self.d.ctrl[self.g_act] = float(np.clip(ctrl, -g.force_range, g.force_range))

    def open(self, width: float) -> None:
        self.grip_mode = ("open", float(min(width, self.arm.gripper.max_width_m)))

    def close(self, force_n: float) -> None:
        self.grip_mode = ("close", float(force_n))

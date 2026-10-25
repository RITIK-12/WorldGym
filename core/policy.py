"""Closed parametric primitive DSL and its executor.

pick_place: plan -> reach -> descend_grasp -> close -> lift_clear -> transit_arc -> lower_place -> release -> retract.
hold:       plan -> reach -> descend_grasp -> close -> hold (until every other arm is done) -> release -> retract.
wipe:       plan -> reach -> descend_grasp -> close -> lift_clear -> wipe_approach -> wipe_lower -> wipe (strokes along
            the target's long axis, wiper pressed onto its top) -> wipe_lift -> transit_arc -> lower_place (back where
            it was picked) -> release -> retract.
Any primitive may start with `wait` (sync_after: until that arm is holding or idle). Arms run concurrently.

Any numeric field may be a `Rel` ({"terms": [...], "offset": x}) = sum of measured scene features + offset, resolved
when the action starts. No generated code runs at execution time, so rollouts stay deterministic and model-free.

Place yaw semantics: `fixed_world` / `container_relative` set the *object's* yaw (TCP yaw is derived from the grasp),
`keep` keeps the TCP yaw from the grasp.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from core.controller import ArmController
from core.locked_evaluator import body_aabb
from core.mjcf_compiler import SceneIndex, object_height
from core.schemas import PolicyConfig, Primitive, Rel

CLOSE_S = 0.35
RELEASE_S = 0.35
MIN_TCP_Z = 0.008   # fingertips reach ~8 mm below the Panda TCP


def body_yaw(d: mujoco.MjData, b: int) -> float:
    R = d.xmat[b].reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def scene_features(ix: SceneIndex, d: mujoco.MjData, prim: Primitive, held_below_tcp: float | None) -> dict:
    task = ix.task
    obj = task.obj(prim.object_id)
    lo, hi = body_aabb(ix.m, d, ix.obj_body[obj.id])
    f = {"obj_top_z": float(hi[2]), "obj_height": float(hi[2] - lo[2]), "table_z": 0.0}
    tops_obst = [body_aabb(ix.m, d, ix.obj_body[o.id])[1][2] for o in task.objects if o.is_obstacle]
    tops_all = [body_aabb(ix.m, d, ix.obj_body[o.id])[1][2] for o in task.objects if o.id != obj.id]
    f["max_obstacle_top_z"] = float(max(tops_obst, default=0.0))
    f["max_scene_top_z"] = float(max(tops_all, default=0.0))
    if prim.target_id:
        t = task.obj(prim.target_id)
        tlo, thi = body_aabb(ix.m, d, ix.obj_body[t.id])
        f["target_rim_z"] = float(thi[2])
        f["target_floor_z"] = float(tlo[2] + t.wall_m) if t.shape == "container" else float(thi[2])
    else:
        f["target_rim_z"] = f["target_floor_z"] = 0.0
    depth = prim.grasp_z_depth_m if isinstance(prim.grasp_z_depth_m, float) else 0.018
    f["held_below_tcp"] = held_below_tcp if held_below_tcp is not None else max(object_height(obj) - depth, 0.0)
    return f


def resolve(v, feats: dict) -> float | None:
    if v is None:
        return None
    if isinstance(v, Rel):
        return float(sum(feats[t] for t in v.terms) + v.offset)
    return float(v)


class Action:
    zero = False

    def __init__(self, name: str, ex: "PrimitiveRunner"):
        self.name, self.ex = name, ex

    def start(self) -> None: ...

    def step(self) -> bool:
        return True


class PrimitiveRunner:
    """Executes one Primitive for one arm; owns the per-primitive context shared across its actions."""

    def __init__(self, ix: SceneIndex, d: mujoco.MjData, ctrl: ArmController, prim: Primitive,
                 executor: "PolicyExecutor | None" = None):
        self.ix, self.d, self.ctrl, self.p, self.exe = ix, d, ctrl, prim, executor
        self.ctx: dict = {"held_below_tcp": None}
        self.resolved: dict = {}
        acts = [_WaitSync("wait", self)] if prim.sync_after else []
        acts += [_Plan("plan", self), _Reach("reach", self), _Descend("descend_grasp", self), _Close("close", self)]
        if prim.kind == "hold":
            acts += [_HoldUntil("hold", self), _Release("release", self), _Retract("retract", self)]
        elif prim.kind == "wipe":
            acts += [_Lift("lift_clear", self), _WipeApproach("wipe_approach", self), _WipeLower("wipe_lower", self),
                     _Strokes("wipe", self), _WipeLift("wipe_lift", self), _Transit("transit_arc", self),
                     _Lower("lower_place", self), _Release("release", self), _Retract("retract", self)]
        else:
            acts += [_Lift("lift_clear", self), _Transit("transit_arc", self), _Lower("lower_place", self),
                     _Release("release", self), _Retract("retract", self)]
        self.actions = acts

    def feats(self) -> dict:
        return scene_features(self.ix, self.d, self.p, self.ctx["held_below_tcp"])

    def r(self, field: str) -> float | None:
        val = resolve(getattr(self.p, field), self.feats())
        if val is not None:
            self.resolved[field] = round(val, 4)
        return val

    def obj_pos(self, oid: str) -> np.ndarray:
        return self.d.xpos[self.ix.obj_body[oid]].copy()


def _xy(v) -> np.ndarray:
    return np.asarray(v, float)[:2]


class _Plan(Action):
    zero = True

    def start(self):
        ex, p, ctx = self.ex, self.ex.p, self.ex.ctx
        b = ex.ix.obj_body[p.object_id]
        oyaw = body_yaw(ex.d, b)
        c, s = math.cos(oyaw), math.sin(oyaw)
        off = np.array([c * p.grasp_xy_offset_m[0] - s * p.grasp_xy_offset_m[1],
                        s * p.grasp_xy_offset_m[0] + c * p.grasp_xy_offset_m[1]])
        ctx["grasp_xy"] = ex.obj_pos(p.object_id)[:2] + off
        ctx["pick_xy"] = ex.obj_pos(p.object_id)[:2].copy()
        yaw = p.grasp_yaw_rad + (oyaw if p.grasp_yaw_mode == "object_relative" else 0.0)
        ctx["grasp_yaw"] = ex.ctrl.choose_yaw(yaw)
        f = ex.feats()
        depth = ex.r("grasp_z_depth_m")
        ctx["grasp_z"] = max(f["obj_top_z"] - depth, f["table_z"] + MIN_TCP_Z)
        ex.resolved["grasp_z"] = round(ctx["grasp_z"], 4)
        ex.resolved["grasp_yaw"] = round(ctx["grasp_yaw"], 4)


class _Move(Action):
    def step(self):
        done, _ = self.ex.ctrl.step_motion()
        return done


class _Reach(_Move):
    def start(self):
        ex, p, ctx = self.ex, self.ex.p, self.ex.ctx
        ex.ctrl.open(p.pre_grasp_open_width_m)
        gx, gy = ctx["grasp_xy"]
        pre_z = ctx["grasp_z"] + ex.r("pre_grasp_clearance_m")
        pts = []
        clear = ex.r("reach_clearance_z_m")
        if clear is not None:
            cur, _ = ex.ctrl.cmd_pose()
            pts += [[cur[0], cur[1], max(clear, cur[2])], [gx, gy, max(clear, pre_z)]]
        pts.append([gx, gy, pre_z])
        ex.ctrl.move(pts, ctx["grasp_yaw"], p.speed_mps)


class _Descend(_Move):
    def start(self):
        ex, ctx = self.ex, self.ex.ctx
        gx, gy = ctx["grasp_xy"]
        ex.ctrl.move([[gx, gy, ctx["grasp_z"]]], ctx["grasp_yaw"], ex.p.speed_mps * 0.45)


class _Timed(Action):
    duration = 0.0

    def start(self):
        self.t_end = self.ex.d.time + self.duration
        self.begin()

    def begin(self): ...

    def step(self):
        self.ex.ctrl.ik_step(*self.ex.ctrl.cmd_pose())  # hold the command
        return self.ex.d.time >= self.t_end - 1e-9


class _Close(_Timed):
    duration = CLOSE_S

    def begin(self):
        self.ex.ctrl.close(self.ex.p.grip_force_n)

    def step(self):
        done = super().step()
        if done:
            ex = self.ex
            lo, _ = body_aabb(ex.ix.m, ex.d, ex.ix.obj_body[ex.p.object_id])
            tcp, R = ex.ctrl.tcp_pose()
            ex.ctx["held_below_tcp"] = float(max(tcp[2] - lo[2], 0.0))
            ex.ctx["grasp_rel"] = (ex.obj_pos(ex.p.object_id) - tcp).tolist()
            ex.ctx["obj_yaw_at_grasp"] = body_yaw(ex.d, ex.ix.obj_body[ex.p.object_id])
            ex.ctx["tcp_yaw_at_grasp"] = ex.ctrl.tcp_yaw(R)
            ex.resolved["held_below_tcp"] = round(ex.ctx["held_below_tcp"], 4)
        return done


class _Lift(_Move):
    def start(self):
        ex = self.ex
        cur, R = ex.ctrl.cmd_pose()
        ex.ctrl.move([[cur[0], cur[1], max(ex.r("post_grasp_lift_z_m"), cur[2])]], ex.ctrl.tcp_yaw(R),
                     ex.p.speed_mps * 0.7)


class _Transit(_Move):
    def start(self):
        ex, p, ctx = self.ex, self.ex.p, self.ex.ctx
        cur, R = ex.ctrl.cmd_pose()
        if ctx.get("place_override") is not None:
            tyaw = 0.0
            place = np.asarray(ctx["place_override"], float)
        elif p.target_id:
            tb = ex.ix.obj_body[p.target_id]
            tyaw = body_yaw(ex.d, tb)
            c, s = math.cos(tyaw), math.sin(tyaw)
            o = p.place_xy_offset_m
            place = ex.obj_pos(p.target_id)[:2] + np.array([c * o[0] - s * o[1], s * o[0] + c * o[1]])
        else:
            tyaw = 0.0
            place = _xy(p.place_xy_offset_m)
        ctx["place_xy"] = place
        tcp_yaw = ex.ctrl.tcp_yaw(R)
        if p.place_yaw_mode != "keep":
            want_obj = p.place_yaw_rad + (tyaw if p.place_yaw_mode == "container_relative" else 0.0)
            delta = ctx.get("tcp_yaw_at_grasp", tcp_yaw) - ctx.get("obj_yaw_at_grasp", 0.0)
            tcp_yaw = ex.ctrl.choose_yaw(want_obj + delta)
        ctx["place_yaw"] = tcp_yaw
        hover = ex.r("pre_place_hover_z_m")
        apex = max(ex.r("transit_apex_z_m"), hover)
        ctx["hover_z"] = hover
        if p.transit_via_mode == "none":
            pts = [[cur[0], cur[1], max(apex, cur[2])], [place[0], place[1], apex], [place[0], place[1], hover]]
        else:
            via = (cur[:2] + place) / 2 if p.transit_via_mode == "midpoint" else _xy(p.transit_via_point_xy_m)
            pts = [[via[0], via[1], apex], [place[0], place[1], hover]]
        ex.resolved["place_xy"] = [round(float(v), 4) for v in place]
        ex.resolved["place_yaw"] = round(tcp_yaw, 4)
        ex.ctrl.move(pts, tcp_yaw, p.speed_mps)


class _Lower(_Move):
    def start(self):
        ex, ctx = self.ex, self.ex.ctx
        f = ex.feats()
        drop = ex.r("release_drop_height_m")
        z = ctx.get("place_floor_z", f["target_floor_z"]) + drop + f["held_below_tcp"]
        z = min(z, ctx["hover_z"])
        ex.resolved["release_z"] = round(z, 4)
        px, py = ctx["place_xy"]
        ex.ctrl.move([[px, py, z]], ctx["place_yaw"], ex.p.speed_mps * 0.45)


class _Release(_Timed):
    duration = RELEASE_S

    def begin(self):
        self.ex.ctrl.open(self.ex.p.release_open_width_m)


class _Retract(_Move):
    def start(self):
        ex = self.ex
        cur, R = ex.ctrl.cmd_pose()
        ex.ctrl.move([[cur[0], cur[1], max(ex.r("retract_z_m"), cur[2])]], ex.ctrl.tcp_yaw(R), ex.p.speed_mps)


class _WaitSync(Action):
    def step(self):
        ex = self.ex
        ex.ctrl.ik_step(*ex.ctrl.cmd_pose())
        return ex.exe is None or ex.exe.phase(ex.p.sync_after)[0] in ("hold", "idle")


class _HoldUntil(_Move):
    """Keep the grasp (optionally lifted by hold_lift_m) until every other arm has finished."""

    def start(self):
        ex = self.ex
        if ex.p.hold_lift_m > 0:
            cur, R = ex.ctrl.cmd_pose()
            ex.ctrl.move([[cur[0], cur[1], cur[2] + ex.p.hold_lift_m]], ex.ctrl.tcp_yaw(R), ex.p.speed_mps * 0.5)

    def step(self):
        ex = self.ex
        if ex.ctrl.path is not None:
            ex.ctrl.step_motion()
            return False
        ex.ctrl.ik_step(*ex.ctrl.cmd_pose())
        exe = ex.exe
        return exe is None or all(exe.phase(a)[0] == "idle" for a in exe.ctrls if a != ex.p.arm)


def wipe_geometry(ex: "PrimitiveRunner") -> dict:
    """Stroke endpoints along the target's long horizontal axis, the surface top and the TCP height while pressing."""
    p = ex.p
    t = ex.ix.task.obj(p.target_id)
    b = ex.ix.obj_body[p.target_id]
    R = ex.d.xmat[b].reshape(3, 3)
    centre = ex.d.xpos[b][:2].copy()
    ax = R[:2, 0] if t.dims_m[0] >= t.dims_m[1] else R[:2, 1]
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    perp = np.array([-ax[1], ax[0]])
    f = ex.feats()
    surf = ex.r("wipe_surface_z_m")
    surf = f["target_rim_z"] if surf is None else surf
    z = surf - ex.r("wipe_press_m") + f["held_below_tcp"]
    start = centre + ax * p.wipe_start_m + perp * p.wipe_lateral_m
    end = centre + ax * p.wipe_end_m + perp * p.wipe_lateral_m
    ex.resolved.update(wipe_start_xy=[round(float(v), 4) for v in start], wipe_end_xy=[round(float(v), 4) for v in end],
                       wipe_surface_z=round(surf, 4), wipe_tcp_z=round(z, 4))
    return {"start": start, "end": end, "z": z, "surf": surf, "hover": surf + f["held_below_tcp"] + 0.04}


class _WipeApproach(_Move):
    def start(self):
        ex = self.ex
        g = ex.ctx["wipe"] = wipe_geometry(ex)
        cur, R = ex.ctrl.cmd_pose()
        hz = max(g["hover"], cur[2])
        ex.ctrl.move([[cur[0], cur[1], hz], [g["start"][0], g["start"][1], hz],
                      [g["start"][0], g["start"][1], g["hover"]]], ex.ctrl.tcp_yaw(R), ex.p.speed_mps)


class _WipeLower(_Move):
    def start(self):
        ex = self.ex
        g = ex.ctx["wipe"]
        _, R = ex.ctrl.cmd_pose()
        ex.ctrl.move([[g["start"][0], g["start"][1], g["z"]]], ex.ctrl.tcp_yaw(R), ex.p.speed_mps * 0.3)


class _Strokes(_Move):
    """wipe_strokes passes between the endpoints, each with its own min-jerk profile, keeping the press height."""

    def start(self):
        self.k = 0
        self._next()

    def _next(self):
        ex = self.ex
        g = ex.ctx["wipe"]
        pt = g["end"] if self.k % 2 == 0 else g["start"]
        _, R = ex.ctrl.cmd_pose()
        ex.ctrl.move([[pt[0], pt[1], g["z"]]], ex.ctrl.tcp_yaw(R), ex.p.wipe_speed_mps)

    def step(self):
        done, _ = self.ex.ctrl.step_motion()
        if not done:
            return False
        self.k += 1
        if self.k >= max(1, self.ex.p.wipe_strokes):
            return True
        self._next()
        return False


class _WipeLift(_Move):
    def start(self):
        ex, p, ctx = self.ex, self.ex.p, self.ex.ctx
        g = ctx["wipe"]
        cur, R = ex.ctrl.cmd_pose()
        ex.ctrl.move([[cur[0], cur[1], g["hover"] + 0.02]], ex.ctrl.tcp_yaw(R), p.speed_mps * 0.5)
        ctx["place_override"] = ctx["pick_xy"] if p.place_back else _xy(p.place_xy_offset_m)
        ctx["place_floor_z"] = 0.0


class PolicyExecutor:
    """Action queue per arm. Actions start lazily (parameters resolve from the current state); zero-duration actions
    and the start of the next action chain within the same tick so no idle tick separates phases."""

    def __init__(self, ix: SceneIndex, d: mujoco.MjData, ctrls: dict[str, ArmController], policy: PolicyConfig):
        self.ix, self.d, self.ctrls = ix, d, ctrls
        self.queues: dict[str, list[tuple[PrimitiveRunner, Action]]] = {a: [] for a in ctrls}
        self.runners: list[PrimitiveRunner] = []
        for prim in policy.primitives:
            run = PrimitiveRunner(ix, d, ctrls[prim.arm], prim, self)
            self.runners.append(run)
            self.queues[prim.arm] += [(run, a) for a in run.actions]
        self.current: dict[str, tuple[PrimitiveRunner, Action] | None] = {a: None for a in ctrls}

    @property
    def done(self) -> bool:
        return all(not q for q in self.queues.values()) and all(c is None for c in self.current.values())

    def phase(self, arm: str = "arm") -> tuple[str, str | None]:
        cur = self.current.get(arm)
        if cur is None:
            return ("idle", None) if not self.queues.get(arm) else ("pending", None)
        return cur[1].name, cur[0].p.object_id

    def step(self) -> None:
        for arm, ctrl in self.ctrls.items():
            stepped = False  # at most one IK step per tick
            for _ in range(16):  # bounded chaining within one tick
                if self.current[arm] is None:
                    if not self.queues[arm]:
                        if not stepped:
                            ctrl.ik_step(*ctrl.cmd_pose())
                        break
                    nxt = self.queues[arm].pop(0)
                    nxt[1].start()
                    if nxt[1].zero:
                        continue
                    self.current[arm] = nxt
                    if stepped:
                        break
                stepped = True
                if self.current[arm][1].step():
                    self.current[arm] = None
                    continue
                break
            ctrl.apply()

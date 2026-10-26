"""Gen 1: a runnable world straight from the Video Architect (objects, layout, SimReady assets, phases).

No measurement and no extra model call: the architect's robot-frame layout is mapped onto the onboarded robot, its
asset choices are fetched, and the phases become primitives (hold / wipe / pick_place). Gen 2 then corrects
positions with the solved camera and re-extracts the motion from the labelling swarm.
"""
from __future__ import annotations

import json
import math

import numpy as np

from core.mjcf_compiler import object_height
from core.motion_extract import DEPTH, _across
from core.schemas import (ArmMotion, AssetRef, ObjectSpec, Part, PolicyConfig, Predicate, Primitive, RobotAdapter,
                          TaskSpec, VideoUnderstanding)
from core.scene_build import SceneDraft, _finish_assets, colour_rgba

PRIMS = ("box", "cylinder", "sphere", "capsule", "tblock", "container")


def robot_frame(adapter: RobotAdapter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World xy of the architect's robot frame: origin (base, or midpoint of the arm bases), forward, left."""
    arms = (adapter.build or {}).get("arms") or {}
    if arms:
        bp = np.array(list(arms.values()), float)
        origin, yaw = bp[:, :2].mean(0), float(np.mean(bp[:, 2]))
    else:
        origin, yaw = np.zeros(2), 0.0
    f = np.array([math.cos(yaw), math.sin(yaw)])
    return origin, f, np.array([-f[1], f[0]])


def arm_bases(adapter: RobotAdapter) -> dict[str, np.ndarray]:
    arms = (adapter.build or {}).get("arms") or {}
    return {k: np.array(v[:2], float) for k, v in arms.items()} or {a.name: np.zeros(2) for a in adapter.arms}


def scene_from_vu(vu: VideoUnderstanding, adapter: RobotAdapter) -> SceneDraft:
    origin, f, l = robot_frame(adapter)
    base_yaw = math.atan2(f[1], f[0])
    tools = {p.object for p in vu.phases if p.object}
    surfaces = {p.target for p in vu.phases if p.target}
    contact = ("push", "wipe")
    wiped = ({p.target for p in vu.phases if p.target and p.action in contact}
             | {m.target for m in vu.motions if m.target and m.kind in contact})
    grasped = ({p.object for p in vu.phases if p.object and p.action not in ("reach", "idle", "retract")}
               | {m.object for m in vu.motions if m.object and m.kind != "idle"})
    tools |= {m.object for m in vu.motions if m.object}
    surfaces |= {m.target for m in vu.motions if m.target}
    fit = 0.7 * min(a.gripper.max_width_m for a in adapter.arms)   # a grasped part must fit between the fingers
    objs, preds = [], []
    for o in vu.objects:
        off = o.offset_m if o.offset_m and len(o.offset_m) >= 2 else [0.3, 0.0]
        xy = origin + f * off[0] + l * off[1]
        size = (list(o.size_guess_m) + [o.size_guess_m[-1]] * 3)[:3]
        if o.long_axis in ("x", "y") and (size[0] >= size[1]) != (o.long_axis == "x"):
            size[0], size[1] = size[1], size[0]          # the size guess must agree with the stated long axis
        if o.id in grasped and o.type not in ("sphere", "cylinder", "rail"):
            k = 0 if size[0] < size[1] else 1                 # a grasped sponge compresses between the fingers
            size[k] = min(size[k], fit)
        fixed = o.fixed or o.id in wiped
        kw = dict(id=o.id, colour=colour_rgba(o.colour), xy=[round(float(v), 4) for v in xy], is_fixed=fixed,
                  is_obstacle=o.id not in tools and o.id not in surfaces and o.type != "container")
        if o.type == "rail":
            objs.append(ObjectSpec(shape="composite", dims_m=[round(v, 4) for v in size], yaw=round(base_yaw, 4),
                                   parts=rail_parts(size, min(0.025, fit)), friction=0.6, **kw))
        elif o.asset:
            objs.append(ObjectSpec(shape="mesh", dims_m=[round(v, 4) for v in size], yaw=round(base_yaw, 4),
                                   asset=AssetRef(name=o.asset, url=o.asset), **kw))
        else:
            shape = o.type if o.type in PRIMS else "box"
            objs.append(ObjectSpec(shape=shape, dims_m=[round(v, 4) for v in size], yaw=round(base_yaw, 4),
                                   **{**kw, "is_fixed": fixed or shape == "container"}))
    wipers = ({p.object for p in vu.phases if p.object and p.target and p.action in contact}
              | {m.object for m in vu.motions if m.object and m.target and m.kind in contact})
    ids = {o.id for o in objs}
    for m in vu.motions:
        if m.kind in contact and m.object in ids and m.target in ids:
            preds.append(Predicate(kind="wiped_coverage", object_id=m.object, target_id=m.target, limit=0.5))
    for p in vu.phases:
        if p.object and p.target and p.action in contact:
            preds.append(Predicate(kind="wiped_coverage", object_id=p.object, target_id=p.target, limit=0.5))
        elif p.object and p.target and p.action in ("place", "release") and p.object not in wipers:
            t = next((x for x in objs if x.id == p.target), None)
            preds.append(Predicate(kind="inside_container" if t is not None and t.shape == "container"
                                   else "on_top_of", object_id=p.object, target_id=p.target))
    preds.append(Predicate(kind="terminal_rest"))
    uniq = {(p.kind, p.object_id, p.target_id): p for p in preds}
    draft = SceneDraft(objects=objs, immutable_success_predicates=list(uniq.values()), notes="from the Video Architect")
    return _finish_assets(draft)


def rail_parts(size: list[float], rod_d: float) -> list[Part]:
    """A bar / handle: two end posts, two uprights and a rod on top, filling exactly the given bounding box."""
    a = 0 if size[0] >= size[1] else 1            # long axis in the object frame
    L, W, H = size[a], size[1 - a], size[2]
    post_l, post_h = 0.13 * L, 0.45 * H
    up_h = H - post_h - rod_d

    def at(along: float, z: float) -> list[float]:
        v = [0.0, 0.0, round(z, 4)]
        v[a] = round(along, 4)
        return v

    def ext(along: float, across: float, h: float) -> list[float]:
        d = [0.0, 0.0, round(h, 4)]
        d[a], d[1 - a] = round(along, 4), round(across, 4)
        return d

    parts = []
    for sgn in (-1, 1):
        c = sgn * (L / 2 - post_l / 2)
        parts.append(Part(shape="box", dims_m=ext(post_l, W, post_h), pos=at(c, -H / 2 + post_h / 2)))
        parts.append(Part(shape="cylinder", dims_m=[rod_d, rod_d, round(up_h, 4)],
                          pos=at(c, -H / 2 + post_h + up_h / 2)))
    rod = Part(shape="capsule", dims_m=[rod_d, rod_d, round(L - post_l, 4)], pos=at(0.0, H / 2 - rod_d / 2),
               euler=[0.0, round(math.pi / 2, 4), 0.0] if a == 0 else [round(math.pi / 2, 4), 0.0, 0.0])
    return parts + [rod]


def _top(o: ObjectSpec) -> float:
    return float(object_height(o))


def _long_ax(o: ObjectSpec) -> np.ndarray:
    if o.dims_m[0] >= o.dims_m[1]:
        return np.array([math.cos(o.yaw), math.sin(o.yaw)])
    return np.array([-math.sin(o.yaw), math.cos(o.yaw)])


def frac_to_s(frac: float, t: ObjectSpec, adapter: RobotAdapter) -> float:
    """Architect fraction along the target (-1 robot's right end, +1 its left end) -> metres along its long axis."""
    _, _, left = robot_frame(adapter)
    sign = 1.0 if float(_long_ax(t) @ left) >= 0 else -1.0
    return float(np.clip(frac, -1.0, 1.0)) * sign * 0.42 * max(t.dims_m[0], t.dims_m[1])


def motion_fields(m: ArmMotion, t: ObjectSpec, adapter: RobotAdapter) -> dict:
    """Wipe fields that encode WHAT the demonstration does: where it starts and ends, how many passes, which way."""
    up: dict = {}
    if m.start_frac is not None and m.end_frac is not None:
        up.update(wipe_start_m=round(frac_to_s(m.start_frac, t, adapter), 4),
                  wipe_end_m=round(frac_to_s(m.end_frac, t, adapter), 4))
    one_way = m.pattern == "one_way"
    up["wipe_pattern"] = "one_way" if one_way else "back_and_forth"
    up["wipe_strokes"] = int(min(max(m.strokes or 1, 1), 6))
    if m.place_back is not None:
        up["place_back"] = bool(m.place_back)
    return up


def align_with_motions(policy: PolicyConfig, task: TaskSpec, vu: VideoUnderstanding, adapter: RobotAdapter,
                       keep_span: bool = False) -> PolicyConfig:
    """Keep the architect's structure (direction, passes, pattern, place back). With keep_span (Gen 2) the
    measured stroke span from the labels stays; only a span running the wrong way is flipped."""
    obj = {o.id: o for o in task.objects}
    by_arm = {m.arm: m for m in vu.motions if m.kind in ("wipe", "push") and m.target in obj}
    prims = []
    for p in policy.primitives:
        m = by_arm.get(p.arm)
        if p.kind == "wipe" and m is not None:
            up = motion_fields(m, obj[m.target], adapter)
            if keep_span and "wipe_start_m" in up:
                want = np.sign(up["wipe_end_m"] - up["wipe_start_m"])
                have = np.sign(p.wipe_end_m - p.wipe_start_m)
                up["wipe_start_m"], up["wipe_end_m"] = ((p.wipe_start_m, p.wipe_end_m) if want == have or not want
                                                        else (p.wipe_end_m, p.wipe_start_m))
            p = p.model_copy(update=up)
        prims.append(p)
    return policy.model_copy(update={"primitives": [_physical(p, prims, obj) for p in prims]})


def _physical(p: Primitive, prims: list[Primitive], obj: dict[str, ObjectSpec]) -> Primitive:
    """Keep a wipe physically possible: a thin tool is grasped above the table, and the strokes stop short of where
    another arm is holding the same target."""
    if p.kind != "wipe" or p.object_id not in obj or p.target_id not in obj:
        return p
    up = {"grasp_z_depth_m": round(min(p.grasp_z_depth_m or DEPTH, 0.45 * _top(obj[p.object_id])), 4)}
    t = obj[p.target_id]
    a = 0 if t.dims_m[0] >= t.dims_m[1] else 1
    for h in prims:
        if h.kind == "hold" and h.object_id == p.target_id and h.arm != p.arm:
            sh = float(h.grasp_xy_offset_m[a])
            lim = sh - 0.08 if sh > 0 else sh + 0.08
            clip = (lambda v: min(v, lim)) if sh > 0 else (lambda v: max(v, lim))
            up.update(wipe_start_m=round(clip(p.wipe_start_m), 4), wipe_end_m=round(clip(p.wipe_end_m), 4))
    return p.model_copy(update=up)



def policy_from_phases(task: TaskSpec, vu: VideoUnderstanding, adapter: RobotAdapter) -> PolicyConfig:
    """The architect's understanding as primitives: a held surface -> hold, a tool pressed on a target -> wipe,
    an object carried to a target -> pick_place. The per-arm motion structure sets direction, passes and pattern."""
    bases = arm_bases(adapter)
    names = [a.name for a in adapter.arms]
    obj = {o.id: o for o in task.objects}
    prims, holding = [], None
    wipes = {}     # arm -> (tool, target)
    for m in vu.motions:
        if m.kind in ("wipe", "push") and m.object in obj and m.target in obj:
            wipes.setdefault(m.arm if m.arm in names else names[0], (m.object, m.target))
    for p in vu.phases:
        arm = p.arm if p.arm in names else names[0]
        if p.action in ("push", "wipe") and p.object in obj and p.target in obj:
            wipes.setdefault(arm, (p.object, p.target))
    wiped = {t for _, t in wipes.values()}
    done = set()
    for p in vu.phases:     # holds first: the surface a wiping arm works on, held by another arm
        arm = p.arm if p.arm in names else names[0]
        if arm in wipes or arm in done or p.object not in obj or p.action not in ("grasp", "lift", "hold", "reach"):
            continue
        o = obj[p.object]
        if not (o.is_fixed or o.id in wiped):
            continue
        L = max(o.dims_m[0], o.dims_m[1])
        along_x = o.dims_m[0] >= o.dims_m[1]
        ax = np.array([math.cos(o.yaw), math.sin(o.yaw)]) if along_x else np.array([-math.sin(o.yaw), math.cos(o.yaw)])
        side = float(np.sign((bases.get(arm, np.zeros(2)) - np.array(o.xy)) @ ax) or -1.0)
        s = side * 0.4 * L
        top = _top(o)
        prims.append(Primitive(kind="hold", arm=arm, object_id=o.id,
                               grasp_xy_offset_m=[round(s, 4), 0.0] if along_x else [0.0, round(s, 4)],
                               grasp_z_depth_m=round(min(0.012, max(top - 0.02, 0.005)), 4),
                               grasp_yaw_mode="object_relative", grasp_yaw_rad=round(_across(o), 4)))
        holding = arm
        done.add(arm)
    for arm, (tool, tgt) in wipes.items():
        o, t = obj[tool], obj[tgt]
        L = max(t.dims_m[0], t.dims_m[1])
        along_x = t.dims_m[0] >= t.dims_m[1]
        ax = np.array([math.cos(t.yaw), math.sin(t.yaw)]) if along_x else np.array([-math.sin(t.yaw), math.cos(t.yaw)])
        side = float(np.sign((bases.get(arm, np.zeros(2)) - np.array(t.xy)) @ ax) or 1.0)
        a, b = sorted((side * 0.05 * L, side * 0.4 * L)) if holding else (-0.35 * L, 0.35 * L)
        ttop, held_below = _top(t), max(_top(o) - DEPTH, 0.0)
        prims.append(Primitive(kind="wipe", arm=arm, object_id=tool, target_id=tgt, grasp_yaw_mode="object_relative",
                               grasp_yaw_rad=round(_across(o), 4), wipe_start_m=round(a, 4), wipe_end_m=round(b, 4),
                               wipe_strokes=4, place_back=True, place_xy_offset_m=[0.0, 0.0],
                               post_grasp_lift_z_m=round(ttop + held_below + 0.08, 4),
                               transit_apex_z_m=round(ttop + held_below + 0.1, 4),
                               pre_place_hover_z_m=round(held_below + 0.06, 4), release_drop_height_m=0.005,
                               release_open_width_m=0.08, retract_z_m=round(ttop + 0.12, 4)))
        done.add(arm)
    for p in vu.phases:
        arm = p.arm if p.arm in names else names[0]
        if arm in done or p.action not in ("place", "release") or p.object not in obj or p.target not in obj:
            continue
        if obj[p.object].is_fixed:
            continue
        prims.append(Primitive(kind="pick_place", arm=arm, object_id=p.object, target_id=p.target))
        done.add(arm)
    if holding:
        prims = [x.model_copy(update={"sync_after": holding}) if x.arm != holding and x.kind != "hold" else x
                 for x in prims]
    return align_with_motions(PolicyConfig(name="gen1", primitives=prims), task, vu, adapter)


def refine_positions(task: TaskSpec, measured: dict, max_shift_m: float = 0.2) -> tuple[TaskSpec, dict]:
    """Gen 2: move each object to its measured footprint when the measurement agrees with the architect's layout
    (a far-off measurement is more likely a bad mask than a bad layout)."""
    objs, moved = [], {}
    for o in task.objects:
        fp = measured.get(o.id)
        if fp and fp.get("xy") is not None:
            d = float(np.linalg.norm(np.array(fp["xy"][:2]) - np.array(o.xy)))
            if d <= max_shift_m:
                objs.append(o.model_copy(update={"xy": [round(float(v), 4) for v in fp["xy"][:2]]}))
                moved[o.id] = round(d * 1000)
                continue
        objs.append(o)
    return task.model_copy(update={"objects": objs}), moved


def demonstrated_motion(vu: VideoUnderstanding) -> str:
    """The architect's per-arm structure, stated as fixed facts for the critic."""
    if not vu.motions:
        return ""
    rows = [m.model_dump(exclude_none=True) for m in vu.motions]
    return ("Demonstrated motion per arm (from the Video Architect). This is WHAT happens and is fixed: keep each "
            "arm's role, the stroke direction (start_frac -> end_frac, -1 = the robot's right end, +1 = its left "
            "end), the number of passes, the pattern (one_way / back_and_forth) and place_back; fix HOW it is done "
            "(grasp points, heights, speeds, positions):\n" + json.dumps(rows))

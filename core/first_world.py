"""Gen 1: a runnable world straight from the Video Architect (objects, layout, SimReady assets, phases).

No measurement and no extra model call: the architect's robot-frame layout is mapped onto the onboarded robot, its
asset choices are fetched, and the phases become primitives (hold / wipe / pick_place). Gen 2 then corrects
positions with the solved camera and re-extracts the motion from the labelling swarm.
"""
from __future__ import annotations

import math

import numpy as np

from core.mjcf_compiler import object_height
from core.motion_extract import DEPTH, _across
from core.schemas import (AssetRef, ObjectSpec, PolicyConfig, Predicate, Primitive, RobotAdapter, TaskSpec,
                          VideoUnderstanding)
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
    wiped = {p.target for p in vu.phases if p.target and p.action in ("push", "wipe")}
    grasped = {p.object for p in vu.phases if p.object and p.action in ("grasp", "lift", "hold", "transit", "push")}
    fit = 0.7 * min(a.gripper.max_width_m for a in adapter.arms)   # a grasped part must fit between the fingers
    objs, preds = [], []
    for o in vu.objects:
        off = o.offset_m if o.offset_m and len(o.offset_m) >= 2 else [0.3, 0.0]
        xy = origin + f * off[0] + l * off[1]
        size = (list(o.size_guess_m) + [o.size_guess_m[-1]] * 3)[:3]
        if o.long_axis in ("x", "y") and (size[0] >= size[1]) != (o.long_axis == "x"):
            size[0], size[1] = size[1], size[0]          # the size guess must agree with the stated long axis
        if o.id in grasped and o.type not in ("sphere", "cylinder"):
            k = 0 if size[0] < size[1] else 1                 # sponges compress, rails are thinner than their posts
            size[k] = min(size[k], fit)
        fixed = o.fixed or o.id in wiped
        kw = dict(id=o.id, colour=colour_rgba(o.colour), xy=[round(float(v), 4) for v in xy], is_fixed=fixed,
                  is_obstacle=o.id not in tools and o.id not in surfaces and o.type != "container")
        if o.asset:
            objs.append(ObjectSpec(shape="mesh", dims_m=[round(v, 4) for v in size], yaw=round(base_yaw, 4),
                                   asset=AssetRef(name=o.asset, url=o.asset), **kw))
        else:
            shape = o.type if o.type in PRIMS else "box"
            objs.append(ObjectSpec(shape=shape, dims_m=[round(v, 4) for v in size], yaw=round(base_yaw, 4),
                                   **{**kw, "is_fixed": fixed or shape == "container"}))
    wipers = {p.object for p in vu.phases if p.object and p.target and p.action in ("push", "wipe")}
    for p in vu.phases:
        if p.object and p.target and p.action in ("push", "wipe"):
            preds.append(Predicate(kind="wiped_coverage", object_id=p.object, target_id=p.target, limit=0.5))
        elif p.object and p.target and p.action in ("place", "release") and p.object not in wipers:
            t = next((x for x in objs if x.id == p.target), None)
            preds.append(Predicate(kind="inside_container" if t is not None and t.shape == "container"
                                   else "on_top_of", object_id=p.object, target_id=p.target))
    preds.append(Predicate(kind="terminal_rest"))
    uniq = {(p.kind, p.object_id, p.target_id): p for p in preds}
    draft = SceneDraft(objects=objs, immutable_success_predicates=list(uniq.values()), notes="from the Video Architect")
    return _finish_assets(draft)


def _top(o: ObjectSpec) -> float:
    return float(object_height(o))


def policy_from_phases(task: TaskSpec, vu: VideoUnderstanding, adapter: RobotAdapter) -> PolicyConfig:
    """The architect's phases as primitives: a held surface -> hold, a tool pressed on a target -> wipe,
    an object carried to a target -> pick_place."""
    bases = arm_bases(adapter)
    names = [a.name for a in adapter.arms]
    obj = {o.id: o for o in task.objects}
    prims, holding = [], None
    wipes = {}     # arm -> (tool, target)
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
        s = side * 0.38 * L
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
    return PolicyConfig(name="gen1", primitives=prims)


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

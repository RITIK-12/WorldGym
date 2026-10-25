"""Loop 1: static fidelity. Render the twin through the solved camera, watch it against the real frame, critique, fix.

Per object: IoU, size ratio and centroid offset (px) between the sim segmentation and the real mask; score =
mean over objects of 0.5*IoU + 0.5*exp(-offset_px/15). Real masks claim pixels smallest sim footprint first (so a
container never absorbs its contents) inside a 45 px window around the sim mask.

Each iteration: (1) an analytic patch measures every object's footprint on the table (real silhouette
back-projected at z=0 and at the object's top, intersected) and moves it there (xy; yaw and footprint extents for
box-like shapes); (2) the Twin Critic (Argon, multi-turn) watches the 4-view diagnostic matrix + the clean real
frame + the numbers and returns a ScenePatch (shape, height, stacking, composites, anything the geometry cannot
see). Stop at alignment >= 0.90 or after 3 iterations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict

from core.argon import ARGON, Argon
from core.camera_solve import close_renderer, seg_renderer
from core.diagnostics import backproject_mask, render_matrix, sim_object_masks
from core.mjcf_compiler import SceneIndex, object_height
from core.schemas import ObjectSpec, TaskSpec

WINDOW_PX = 45
OFFSET_SCALE_PX = 15.0
TARGET = 0.90
MAX_ITERS = 3
EDITABLE = {"xy", "yaw", "dims_m", "shape", "colour", "on_top_of", "z", "is_obstacle", "is_fixed", "parts", "wall_m",
            "mass", "friction", "stem_w_m", "flange_d_m"}


class Observation(BaseModel):
    view: str = ""
    object_id: str | None = None
    observation: str


class ObjectEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str
    set: dict


class ScenePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    iteration: int = 0
    observations: list[Observation] = []
    edits: list[ObjectEdit] = []
    add: list[ObjectSpec] = []
    remove: list[str] = []
    done: bool = False


class ScenePatchError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------

def _centroid(mk: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mk)
    return None if len(xs) == 0 else np.array([xs.mean(), ys.mean()])


def static_fidelity(sim_masks: dict[str, np.ndarray], real_raw: dict[str, np.ndarray]) -> dict:
    order = sorted(sim_masks, key=lambda k: int(sim_masks[k].sum()))
    claimed = np.zeros(next(iter(sim_masks.values())).shape, bool) if sim_masks else None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WINDOW_PX + 1, 2 * WINDOW_PX + 1))
    per, real_claimed = {}, {}
    for oid in order:
        sm = sim_masks[oid]
        raw = real_raw.get(oid)
        if raw is None:
            continue
        window = cv2.dilate(sm.astype(np.uint8), k).astype(bool) if sm.any() else np.ones_like(sm)
        rm = raw & window & ~claimed
        claimed |= rm
        real_claimed[oid] = rm
        inter, union = int((sm & rm).sum()), int((sm | rm).sum())
        iou = inter / union if union else 0.0
        cs, cr = _centroid(sm), _centroid(rm)
        off = float(np.linalg.norm(cs - cr)) if cs is not None and cr is not None else 999.0
        ratio = float(sm.sum() / rm.sum()) if rm.sum() else 0.0
        per[oid] = {"iou": round(iou, 3), "offset_px": round(off, 1), "size_ratio": round(ratio, 3),
                    "score": round(0.5 * iou + 0.5 * math.exp(-off / OFFSET_SCALE_PX), 3)}
    alignment = float(np.mean([v["score"] for v in per.values()])) if per else 0.0
    return {"objects": per, "alignment": round(alignment, 3), "real_claimed": real_claimed}


# ----------------------------------------------------------------------------------------------------------------------
# Analytic patch (metric footprints)
# ----------------------------------------------------------------------------------------------------------------------

def measure_footprint(ix: SceneIndex, d: mujoco.MjData, cam: str, mask: np.ndarray, height: float,
                      res_m: float = 0.001) -> dict | None:
    """Footprint of a real object on the table: silhouette back-projected at z=0 and z=height, intersected
    (rasterised at 1 mm). Returns centre xy, min-area-rect yaw and extents."""
    polys = [backproject_mask(ix, d, cam, mask, z) for z in (0.0, height)]
    if any(p is None or len(p) < 3 for p in polys):
        return None
    allp = np.concatenate(polys)
    lo = allp.min(0) - 0.01
    hi = allp.max(0) + 0.01
    size = np.ceil((hi - lo) / res_m).astype(int)
    if size.max() > 4000:
        return None
    regs = []
    for p in polys:
        r = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(r, [np.round((p - lo) / res_m).astype(np.int32)], 1)
        regs.append(r)
    fp = (regs[0] & regs[1]).astype(np.uint8)
    if fp.sum() < 10:
        return None
    cs, _ = cv2.findContours(fp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    ys, xs = np.nonzero(fp)
    centre = lo + (np.array([xs.mean(), ys.mean()]) + 0.5) * res_m
    yaw = math.radians(ang)
    ext = [w * res_m, h * res_m]
    return {"xy": [round(float(centre[0]), 4), round(float(centre[1]), 4)], "rect_yaw": round(yaw, 4),
            "rect_extents_m": [round(e, 4) for e in ext], "area_m2": round(float(fp.sum()) * res_m ** 2, 6)}


def _box_yaw_dims(o: ObjectSpec, fp: dict) -> tuple[float, list[float]]:
    """Map the min-area rect onto the object's x/y extents, choosing the yaw closest to the current one."""
    a, b = fp["rect_extents_m"]
    best = None
    for k in range(4):
        yaw = fp["rect_yaw"] + k * math.pi / 2
        ex, ey = (a, b) if k % 2 == 0 else (b, a)
        dyaw = abs((yaw - o.yaw + math.pi / 2) % math.pi - math.pi / 2)
        cost = dyaw + 2.0 * (abs(ex - o.dims_m[0]) + abs(ey - o.dims_m[1]))
        if best is None or cost < best[0]:
            best = (cost, yaw, [ex, ey])
    return best[1], best[2]


def analytic_patch(task: TaskSpec, ix: SceneIndex, d: mujoco.MjData, real_masks: dict[str, np.ndarray],
                   cam: str = "video", shape_from_footprint: bool = True) -> tuple[TaskSpec, dict]:
    data = task.model_dump()
    by_id = {o["id"]: o for o in data["objects"]}
    measured = {}
    for o in task.objects:
        mk = real_masks.get(o.id)
        if mk is None or mk.sum() < 30 or o.on_top_of:
            continue
        fp = measure_footprint(ix, d, cam, mk, object_height(o))
        if fp is None:
            continue
        measured[o.id] = fp
        by_id[o.id]["xy"] = fp["xy"]
        if shape_from_footprint and o.shape in ("box", "container"):
            yaw, ext = _box_yaw_dims(o, fp)
            by_id[o.id]["yaw"] = round(float(yaw), 4)
            if all(0.3 < e / max(g, 1e-6) < 3.0 for e, g in zip(ext, o.dims_m[:2])):
                by_id[o.id]["dims_m"] = [round(ext[0], 4), round(ext[1], 4), o.dims_m[2]]
    return TaskSpec.model_validate(data), measured


# ----------------------------------------------------------------------------------------------------------------------
# Twin Critic (Argon)
# ----------------------------------------------------------------------------------------------------------------------

CRITIC_SYSTEM = """You are the Twin Critic of WorldGym. A MuJoCo twin of a real robot scene was built from a video; you
compare what the simulator rendered against the real frame and fix the scene description so the twin matches
reality. You see four diagnostic views of the current twin plus the clean real frame:
1 GHOST: the real frame (dimmed) with sim object outlines in GREEN and real object outlines in MAGENTA; the sim robot
  silhouette in CYAN (the camera was solved from it, so a good robot fit means the camera is right).
2 TOP: orthographic top-down render of the sim objects (robot hidden) with a 5 cm grid; MAGENTA outlines are the real
  objects' measured footprints on the table. +x (away from the robot) points DOWN in the image, +y points RIGHT.
3 GROUND: table-level side view of the sim objects with height lines every 5 cm (check heights, floating, stacking).
4 WIREFRAME: the sim collision geometry over the real frame.

Coordinates: metres, table top z = 0, robot base at the origin, +x away from the robot, +y to the robot's left,
yaw in radians about +z. dims_m are FULL extents [x, y, z] in the object's frame (cylinder: [diameter, diameter,
height]). Shapes: box, cylinder, sphere, capsule, tblock, container (open box: dims = outer size, wall_m = wall),
composite (parts: [{shape, dims_m, pos, euler}] in the object frame, rescaled to dims_m).

Fix what the numbers and views show: wrong shape class, wrong height, wrong footprint size, residual offsets,
missing or spurious objects, stacking (on_top_of), composites for objects that are not a single primitive. Make
the minimal edits; do not undo analytic xy measurements unless a view clearly contradicts them. Set done=true when
the twin matches and further edits would not help."""


def _apply_edits(task: TaskSpec, patch: ScenePatch) -> TaskSpec:
    data = task.model_dump()
    by_id = {o["id"]: o for o in data["objects"]}
    errors = []
    for e in patch.edits:
        if e.object_id not in by_id:
            errors.append(f"unknown object {e.object_id!r}")
            continue
        bad = sorted(set(e.set) - EDITABLE)
        if bad:
            errors.append(f"{e.object_id}: fields {bad} cannot be edited")
            continue
        by_id[e.object_id].update(e.set)
    data["objects"] = [o for o in data["objects"] if o["id"] not in set(patch.remove)]
    data["objects"] += [o.model_dump() for o in patch.add if o.id not in by_id]
    if errors:
        raise ScenePatchError("; ".join(errors))
    try:
        return TaskSpec.model_validate(data)
    except Exception as ex:  # noqa: BLE001
        raise ScenePatchError(str(ex)) from ex


def _objects_json(task: TaskSpec) -> str:
    keep = ("id", "shape", "dims_m", "xy", "yaw", "on_top_of", "z", "is_fixed", "is_obstacle", "wall_m", "parts")
    return json.dumps([{k: v for k, v in o.model_dump().items() if k in keep and v not in (None, [], False)}
                       for o in task.objects], indent=1)


def critique_twin(argon: Argon, task: TaskSpec, views: dict[str, Path], real_frame: Path, metrics: dict,
                  measured: dict, iteration: int, previous_id: str | None, context: str) -> tuple[ScenePatch, dict]:
    m = {k: v for k, v in metrics.items() if k != "real_claimed"}
    prompt = f"""{context}
Iteration {iteration}. Current twin objects:
{_objects_json(task)}

Static fidelity (per object IoU, centroid offset in px, sim/real size ratio, score; alignment = mean score):
{json.dumps(m, indent=1)}

Analytic footprint measurements already applied this iteration (metres; rect_yaw/extents from the real footprint):
{json.dumps(measured, indent=1)}

Attachments in order: ghost.jpg, top.jpg, ground.jpg, wire.jpg, real_frame.jpg (clean).
Return a ScenePatch (iteration = {iteration})."""
    attachments = [views["ghost"], views["top"], views["ground"], views["wire"], real_frame]
    return argon.ask_json(f"twin_critic_iter{iteration}", CRITIC_SYSTEM, prompt, ScenePatch, attachments, model=ARGON,
                          thinking="high", previous_id=previous_id)


def run_loop1(task: TaskSpec, real_bgr: np.ndarray, real_masks: dict[str, np.ndarray], out_dir: str | Path,
              argon: Argon | None = None, context: str = "", max_iters: int = MAX_ITERS, target: float = TARGET,
              robot_mask: np.ndarray | None = None, build=None) -> dict:
    """Iterate analytic patch + Twin Critic until alignment >= target. `build(task)` -> Sim at the rest pose."""
    from core.rollout import build_sim

    build = build or build_sim
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h, w = real_bgr.shape[:2]
    real_path = out / "real_frame.jpg"
    cv2.imwrite(str(real_path), real_bgr)
    history, previous_id = [], None

    def score(t: TaskSpec, tag: str):
        sim = build(t)
        r = seg_renderer(sim.m, w, h)
        try:
            sm = sim_object_masks(sim.ix, sim.d, "video", w, h, r)
        finally:
            close_renderer(r)
        return sim, static_fidelity(sm, real_masks)

    sim, met = score(task, "start")
    history.append({"iteration": 0, "stage": "initial", "alignment": met["alignment"],
                    "objects": met["objects"], "task": task.model_dump()})
    for it in range(1, max_iters + 1):
        if met["alignment"] >= target and it > 1:
            break
        task, measured = analytic_patch(task, sim.ix, sim.d, met["real_claimed"])
        sim, met = score(task, "analytic")
        views = render_matrix(sim.ix, sim.d, real_bgr, met["real_claimed"], out / f"iter{it}", robot_mask=robot_mask,
                              note=f"iteration {it}: alignment {met['alignment'] * 100:.0f} %")
        rec = {"iteration": it, "stage": "analytic", "alignment": met["alignment"], "objects": met["objects"],
               "measured": measured, "views": {k: str(v) for k, v in views.items()}}
        if argon is not None and met["alignment"] < 0.995:
            try:
                patch, resp = critique_twin(argon, task, views, real_path, met, measured, it, previous_id, context)
                previous_id = resp.get("interaction_id") or previous_id
                task = _apply_edits(task, patch)
                sim, met = score(task, "critic")
                rec.update(stage="critic", alignment_after_critic=met["alignment"], objects=met["objects"],
                           critique=[o.model_dump() for o in patch.observations],
                           edits=[e.model_dump() for e in patch.edits], added=[o.id for o in patch.add],
                           removed=patch.remove, done=patch.done)
                if patch.done and met["alignment"] >= target:
                    rec["task"] = task.model_dump()
                    history.append(rec)
                    break
            except ScenePatchError as e:
                rec["critic_error"] = str(e)
        rec["task"] = task.model_dump()
        history.append(rec)
        if met["alignment"] >= target and argon is None:
            break
    final = render_matrix(sim.ix, sim.d, real_bgr, met["real_claimed"], out / "final", robot_mask=robot_mask,
                          note=f"final: alignment {met['alignment'] * 100:.0f} %")
    (out / "history.json").write_text(json.dumps(history, indent=1, default=float))
    return {"task": task, "alignment": met["alignment"], "objects": met["objects"], "history": history,
            "final_views": {k: str(v) for k, v in final.items()}}

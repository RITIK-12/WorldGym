"""Scene Author: VideoUnderstanding + solved camera + real masks -> first TaskSpec (Argon writes it, geometry grounds it).

Every object's footprint is measured on the table first (silhouette back-projected at the table and at the object's
guessed top, intersected), then Argon authors the scene: shape class (including composites), full extents,
fixed/obstacle/container roles, stacking, mass/friction guesses and the success predicates implied by the task.
Loop 1 then refines it against the real frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from pydantic import BaseModel

from core.argon import ARGON, Argon
from core.fidelity import measure_footprint
from core.mjcf_compiler import SceneIndex
from core.schemas import ObjectSpec, Predicate, TaskSpec, VideoUnderstanding

COLOURS = {"red": [0.85, 0.2, 0.15, 1], "blue": [0.15, 0.4, 0.85, 1], "green": [0.2, 0.7, 0.3, 1],
           "yellow": [0.95, 0.75, 0.1, 1], "orange": [0.95, 0.5, 0.1, 1], "purple": [0.55, 0.3, 0.75, 1],
           "white": [0.92, 0.92, 0.9, 1], "black": [0.12, 0.12, 0.13, 1], "grey": [0.4, 0.42, 0.45, 1],
           "gray": [0.4, 0.42, 0.45, 1], "silver": [0.75, 0.76, 0.78, 1], "brown": [0.5, 0.35, 0.2, 1],
           "pink": [0.95, 0.6, 0.7, 1], "beige": [0.85, 0.8, 0.65, 1]}


class SceneDraft(BaseModel):
    objects: list[ObjectSpec]
    immutable_success_predicates: list[Predicate]
    notes: str = ""


def colour_rgba(word: str) -> list[float]:
    w = (word or "").lower()
    for k, v in COLOURS.items():
        if k in w:
            return v
    return [0.6, 0.6, 0.6, 1]


def measure_objects(ix: SceneIndex, d: mujoco.MjData, vu: VideoUnderstanding, masks: dict[str, np.ndarray],
                    cam: str = "video") -> dict[str, dict]:
    out = {}
    for o in vu.objects:
        mk = masks.get(o.id)
        if mk is None or mk.sum() < 30:
            continue
        h = float(o.size_guess_m[2]) if len(o.size_guess_m) > 2 else float(o.size_guess_m[0])
        fp = measure_footprint(ix, d, cam, mk, h)
        if fp:
            out[o.id] = fp
    return out


def fallback_scene(vu: VideoUnderstanding, measured: dict) -> SceneDraft:
    """Deterministic scene from the measurements alone (used when no model is available)."""
    manipulated = {p.object for p in vu.phases if p.object}
    targets = {p.target for p in vu.phases if p.target}
    objs, preds = [], []
    for o in vu.objects:
        fp = measured.get(o.id, {"xy": [0.45, 0.0], "rect_yaw": 0.0})
        shape = o.type if o.type in ("box", "cylinder", "sphere", "capsule", "tblock", "container") else "box"
        size = list(o.size_guess_m) + [o.size_guess_m[-1]] * (3 - len(o.size_guess_m))
        objs.append(ObjectSpec(id=o.id, shape=shape, dims_m=size[:3], colour=colour_rgba(o.colour), xy=fp["xy"],
                               yaw=fp.get("rect_yaw", 0.0) if shape == "box" else 0.0,
                               is_fixed=shape == "container",
                               is_obstacle=o.id not in manipulated and o.id not in targets and shape != "container"))
    for p in vu.phases:
        if p.object and p.target and p.action in ("place", "release"):
            t = next((x for x in objs if x.id == p.target), None)
            kind = "inside_container" if t is not None and t.shape == "container" else "on_top_of"
            preds.append(Predicate(kind=kind, object_id=p.object, target_id=p.target))
    for o in objs:
        if o.is_obstacle:
            preds.append(Predicate(kind="max_obstacle_displacement", object_id=o.id, limit=0.01))
    preds.append(Predicate(kind="terminal_rest"))
    uniq = {(p.kind, p.object_id, p.target_id): p for p in preds}
    return SceneDraft(objects=objs, immutable_success_predicates=list(uniq.values()), notes="fallback")


AUTHOR_SYSTEM = """You are the Scene Author of WorldGym. From a robot video's understanding and metric measurements of each
object's footprint on the table, you write the MuJoCo scene description (a list of objects and the success
predicates of the task). Geometry is measured, so keep the measured xy unless an object is stacked on another;
use the measured footprint extents and yaw for box-like objects; choose heights from the video (the robot is the
scale reference). You decide each object's role: manipulated objects are free; containers are open boxes
(shape "container", is_fixed true, dims = OUTER size, wall_m ~0.005-0.01); things the robot must avoid are
obstacles (is_obstacle true); objects that stay put in the video while being held, pressed or wiped (a bolted bar,
a fixture) are is_fixed true. Use shape "composite" with parts when one primitive is a bad fit (e.g. a handle on two
blocks, an L-bracket). Predicates must be checkable at the END of the episode: inside_container(object, target),
on_top_of(object, target), max_obstacle_displacement(object, limit metres), pushed_to_region(object, region
[xmin, ymin, xmax, ymax]), wiped_coverage(object = the wiper, target = the wiped object, limit = required fraction of
the wiped length, region = optional [s_min, s_max] metres along the target's long axis from its centre),
terminal_rest. Frame: metres, table top z = 0, robot base at the origin, +x away from the
robot, +y to the robot's left, yaw radians about +z, dims_m FULL extents [x, y, z] in the object frame (cylinder:
[diameter, diameter, height]). Colours are RGBA in 0..1."""


def author_scene(argon: Argon, vu: VideoUnderstanding, measured: dict, rest_frame: Path, overlay: Path,
                 prompt: str) -> SceneDraft:
    text = f"""User prompt: "{prompt}"
Video understanding:
{vu.model_dump_json(indent=1, exclude={'keyframes'})}

Measured footprints on the table (metres; xy = footprint centre, rect_yaw/rect_extents_m = minimum-area rectangle):
{json.dumps(measured, indent=1)}

Attachments: rest_frame.jpg (clean) and masks.jpg (the real object masks the measurements came from, labelled).
Write the scene. Use exactly the object ids above."""
    draft, _ = argon.ask_json("scene_author", AUTHOR_SYSTEM, text, SceneDraft, [rest_frame, overlay], model=ARGON,
                              thinking="high")
    ids = {o.id for o in draft.objects}
    draft.immutable_success_predicates = [p for p in draft.immutable_success_predicates
                                          if (p.object_id is None or p.object_id in ids)
                                          and (p.target_id is None or p.target_id in ids)]
    return draft


def mask_overlay(frame: np.ndarray, masks: dict[str, np.ndarray], path: Path) -> Path:
    img = (frame * 0.6).astype(np.uint8)
    rng = np.random.default_rng(3)
    for oid, mk in masks.items():
        col = tuple(int(c) for c in rng.integers(80, 255, 3))
        img[mk] = (0.5 * img[mk] + 0.5 * np.array(col)).astype(np.uint8)
        ys, xs = np.nonzero(mk)
        if len(xs):
            cv2.putText(img, oid, (int(xs.mean()) - 30, int(ys.min()) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return path


def make_task(name: str, robot_id: str, draft: SceneDraft, cam_task: TaskSpec, vu: VideoUnderstanding) -> TaskSpec:
    return TaskSpec(name=name, robot=robot_id, objects=draft.objects, phases=vu.phases, cameras=cam_task.cameras,
                    immutable_success_predicates=draft.immutable_success_predicates)

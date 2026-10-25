"""Perturbation family -> scenarios (JSON overrides on a base TaskSpec), rejection-sampled and settle-gated.

A scenario is `{"scenario_id", "seed", "family", "overrides": {"objects": {"<id>": {field: value}}}}`, so agents can
propose scenarios as plain JSON. Sampling rejects layouts where objects overlap obstacle footprints or each other
(yawed-rectangle distance < 8 mm) or leave the reachable workspace; the settle gate rejects scenes whose free
objects drift > 2 mm in 300 steps with the robot holding home.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict

from core.schemas import ObjectSpec, TaskSpec

CLEARANCE_M = 0.008
SETTLE_STEPS = 300
SETTLE_DRIFT_M = 0.002
WORKSPACE = (0.25, 0.68, -0.32, 0.32)   # x0, x1, y0, y1 reachable with a top-down grasp


class ObjRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    xy_jitter_m: list[float] = [0.0, 0.0]          # +- dx, +- dy around the base pose
    yaw_jitter_rad: float = 0.0
    scale: list[float] = [1.0, 1.0]                # uniform scale of x/y extents
    height_m: list[float] | None = None            # absolute z extent range
    mass_scale: list[float] = [1.0, 1.0]
    friction: list[float] | None = None


class Family(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    objects: dict[str, ObjRange]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    seed: int | None = None
    family: str = "custom"
    overrides: dict


def default_family(task: TaskSpec, harder: bool = False) -> Family:
    """Rule-based ranges (Argon may refine them). `harder` goes beyond the dev ranges so 100 % stays meaningful."""
    k = 1.5 if harder else 1.0
    objs = {}
    for o in task.objects:
        if o.shape == "container":
            objs[o.id] = ObjRange(xy_jitter_m=[0.03 * k, 0.03 * k], yaw_jitter_rad=0.25 * k,
                                  scale=[0.85, 0.92] if harder else [0.92, 1.08],
                                  height_m=[o.dims_m[2] * (1.1 if harder else 0.8), o.dims_m[2] * (1.4 if harder else 1.2)])
        elif o.is_obstacle:
            h = o.dims_m[2]
            objs[o.id] = ObjRange(xy_jitter_m=[0.02 * k, 0.02 * k], yaw_jitter_rad=0.2 * k,
                                  height_m=[h * 1.55, h * 2.0] if harder else [h * 0.7, h * 1.5])
        elif o.is_fixed:   # fixtures (e.g. a wiped bar): small pose changes
            objs[o.id] = ObjRange(xy_jitter_m=[0.02 * k, 0.02 * k], yaw_jitter_rad=0.08 * k)
        elif not o.is_fixed:
            objs[o.id] = ObjRange(xy_jitter_m=[0.05 * k, 0.05 * k], yaw_jitter_rad=math.pi,
                                  scale=[1.1, 1.25] if harder else [0.85, 1.15], mass_scale=[0.7, 1.4],
                                  friction=[0.55, 0.7] if harder else [0.7, 1.0])
    return Family(name="harder" if harder else "dev", objects=objs)


def apply_overrides(task: TaskSpec, overrides: dict) -> TaskSpec:
    data = task.model_dump()
    by_id = {o["id"]: o for o in data["objects"]}
    for oid, fields in (overrides.get("objects") or {}).items():
        if oid not in by_id:
            raise KeyError(f"override for unknown object {oid!r}")
        by_id[oid].update(fields)
    return TaskSpec.model_validate(data)


def _footprint(o: ObjectSpec) -> tuple[np.ndarray, float, float, float]:
    """(centre xy, yaw, half-x, half-y) of the object's table footprint."""
    if o.shape in ("cylinder", "sphere", "capsule"):
        r = o.dims_m[0] / 2
        return np.asarray(o.xy, float), 0.0, r, r
    return np.asarray(o.xy, float), o.yaw, o.dims_m[0] / 2, o.dims_m[1] / 2


def _corners(c, yaw, hx, hy) -> np.ndarray:
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return c + np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]]) @ R.T


def rect_distance(a: ObjectSpec, b: ObjectSpec) -> float:
    """Separating-axis distance between two yawed footprints (negative = overlap depth)."""
    ca, cb = _corners(*_footprint(a)), _corners(*_footprint(b))
    best = -math.inf
    for poly in (ca, cb):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)
            pa, pb = ca @ n, cb @ n
            gap = max(pb.min() - pa.max(), pa.min() - pb.max())
            best = max(best, gap)
    return float(best)


def layout_ok(task: TaskSpec) -> tuple[bool, str]:
    x0, x1, y0, y1 = WORKSPACE
    ground = [o for o in task.objects if not o.on_top_of]
    for o in ground:
        corners = _corners(*_footprint(o))
        if corners[:, 0].min() < x0 or corners[:, 0].max() > x1 or corners[:, 1].min() < y0 or corners[:, 1].max() > y1:
            return False, f"{o.id} outside workspace"
    for i, a in enumerate(ground):
        for b in ground[i + 1:]:
            if rect_distance(a, b) < CLEARANCE_M:
                return False, f"{a.id} too close to {b.id}"
    return True, ""


def settle_drift(task: TaskSpec) -> float:
    from core.rollout import build_sim

    sim = build_sim(task)
    m, d, ix = sim.m, sim.d, sim.ix
    free = [o.id for o in task.objects if not o.is_fixed]
    # build_sim already settled for a few steps; measure drift from the nominal spawn pose
    spawn = {o.id: np.array([*o.xy, 0.0]) for o in task.objects if not o.is_fixed}
    for o in task.objects:
        if not o.is_fixed:
            spawn[o.id][2] = d.xpos[ix.obj_body[o.id]][2]
    start = {oid: d.xpos[ix.obj_body[oid]].copy() for oid in free}
    for _ in range(SETTLE_STEPS):
        for c in sim.ctrls.values():
            c.apply()
        mujoco.mj_step(m, d)
    drift = [np.linalg.norm(d.xpos[ix.obj_body[o]] - start[o]) for o in free]
    drift += [np.linalg.norm(start[o][:2] - spawn[o][:2]) for o in free]
    return float(max(drift, default=0.0))


def sample_overrides(task: TaskSpec, fam: Family, rng: np.random.Generator) -> dict:
    out: dict = {}
    for o in task.objects:
        r = fam.objects.get(o.id)
        if r is None:
            continue
        f: dict = {}
        if any(r.xy_jitter_m):
            f["xy"] = [round(o.xy[0] + rng.uniform(-r.xy_jitter_m[0], r.xy_jitter_m[0]), 4),
                       round(o.xy[1] + rng.uniform(-r.xy_jitter_m[1], r.xy_jitter_m[1]), 4)]
        if r.yaw_jitter_rad:
            f["yaw"] = round(o.yaw + rng.uniform(-r.yaw_jitter_rad, r.yaw_jitter_rad), 4)
        dims = list(o.dims_m)
        if r.scale != [1.0, 1.0]:
            s = rng.uniform(*r.scale)
            if o.shape in ("cylinder", "sphere", "capsule"):
                dims[0] = dims[1] = dims[0] * s
                if o.shape == "sphere":
                    dims[2] = dims[0]
            elif o.shape == "container" or o.is_obstacle:
                dims[0], dims[1] = dims[0] * s, dims[1] * s
            else:
                dims = [v * s for v in dims]
        if r.height_m and o.shape != "sphere":
            dims[2] = rng.uniform(*r.height_m)
        if dims != list(o.dims_m):
            f["dims_m"] = [round(v, 4) for v in dims]
        if r.mass_scale != [1.0, 1.0]:
            f["mass"] = round(o.mass * rng.uniform(*r.mass_scale), 4)
        if r.friction:
            f["friction"] = round(rng.uniform(*r.friction), 3)
        if f:
            out[o.id] = f
    return {"objects": out}


def sample_scenarios(task: TaskSpec, fam: Family, n: int, seed: int, prefix: str, max_tries: int = 400,
                     settle: bool = True) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    out: list[Scenario] = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        s_seed = int(rng.integers(0, 2 ** 31 - 1))
        ov = sample_overrides(task, fam, np.random.default_rng(s_seed))
        t = apply_overrides(task, ov)
        ok, _ = layout_ok(t)
        if not ok:
            continue
        if settle and settle_drift(t) > SETTLE_DRIFT_M:
            continue
        out.append(Scenario(scenario_id=f"{prefix}{len(out):02d}", seed=s_seed, family=fam.name, overrides=ov))
    if len(out) < n:
        raise RuntimeError(f"only {len(out)}/{n} scenarios passed layout + settle gates in {tries} tries")
    return out

"""Locked, read-only referee for success predicates.

Self-contained on purpose: it only reads MjModel/MjData + the TaskSpec and never imports policy or telemetry code.
After its tests pass it is chmod 444 and must not be edited (its sha256 is shown in the Studio). Version 2 adds
`wiped_coverage` for wiping tasks (re-locked before any wiping benchmark existed).

Exact AABBs: a cylinder's world extent along axis i is r*sqrt(1 - a_i^2) + (h/2)*|a_i| where a is its axis
(the third column of xmat). Treating a cylinder as an (r, r, h/2) box inflates a yawed cylinder by sqrt(2).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

INSIDE_TOL_M = 0.002
FLOOR_TOL_M = 0.004
REST_SPEED = 0.01
REST_WINDOW_S = 0.5
ON_TOP_TOL_M = 0.01


def geom_aabb(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> tuple[np.ndarray, np.ndarray]:
    c = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    s = m.geom_size[g]
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        ext = np.abs(R) @ s
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a = R[:, 2]
        ext = s[0] * np.sqrt(np.clip(1 - a * a, 0, 1)) + s[1] * np.abs(a)
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        ext = s[0] + s[1] * np.abs(R[:, 2])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        ext = np.full(3, s[0])
    else:  # meshes etc.: MuJoCo's own (conservative) box
        ext = np.abs(R) @ m.geom_aabb[g, 3:]
        c = c + R @ m.geom_aabb[g, :3]
    return c - ext, c + ext


def body_geoms(m: mujoco.MjModel, body: int) -> list[int]:
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]


def body_aabb(m: mujoco.MjModel, d: mujoco.MjData, body: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = [geom_aabb(m, d, g) for g in body_geoms(m, body)]
    return np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)


def _body(m, name: str) -> int:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise KeyError(name)
    return b


def _tilt_deg(R: np.ndarray, shape: str) -> float:
    if shape in ("box", "sphere"):  # any face down counts as upright for a box
        return math.degrees(math.acos(min(1.0, float(np.max(np.abs(R[2, :]))))))
    return math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))


def _obj(task, oid: str):
    for o in task.objects:
        if o.id == oid:
            return o
    raise KeyError(oid)


def inside_container(m, d, task, object_id: str, target_id: str, max_tilt_deg: float) -> dict:
    o, c = _obj(task, object_id), _obj(task, target_id)
    ob, cb = _body(m, object_id), _body(m, target_id)
    lo, hi = body_aabb(m, d, ob)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    Rc = d.xmat[cb].reshape(3, 3)
    local = (corners - d.xpos[cb]) @ Rc
    ihx, ihy = c.dims_m[0] / 2 - c.wall_m, c.dims_m[1] / 2 - c.wall_m
    c_bottom = d.xpos[cb][2] - c.dims_m[2] / 2
    floor_top = c_bottom + c.wall_m
    rim = c_bottom + c.dims_m[2]
    in_xy = bool(np.all(np.abs(local[:, 0]) <= ihx + INSIDE_TOL_M) and np.all(np.abs(local[:, 1]) <= ihy + INSIDE_TOL_M))
    bottom_ok = bool(floor_top - FLOOR_TOL_M <= lo[2] < rim)
    tilt = _tilt_deg(d.xmat[ob].reshape(3, 3), o.shape)
    ok = in_xy and bottom_ok and tilt <= max_tilt_deg
    return {"ok": ok, "in_xy": in_xy, "bottom_ok": bottom_ok, "bottom_z": float(lo[2]), "floor_top_z": float(floor_top),
            "tilt_deg": round(tilt, 2), "max_abs_local_xy": [float(np.abs(local[:, 0]).max()),
                                                             float(np.abs(local[:, 1]).max())],
            "inner_half_xy": [ihx, ihy]}


def on_top_of(m, d, task, object_id: str, target_id: str, max_tilt_deg: float) -> dict:
    o = _obj(task, object_id)
    ob, tb = _body(m, object_id), _body(m, target_id)
    olo, ohi = body_aabb(m, d, ob)
    tlo, thi = body_aabb(m, d, tb)
    cx, cy = (olo[:2] + ohi[:2]) / 2
    over = bool(tlo[0] <= cx <= thi[0] and tlo[1] <= cy <= thi[1])
    gap = float(olo[2] - thi[2])
    tilt = _tilt_deg(d.xmat[ob].reshape(3, 3), o.shape)
    ok = over and abs(gap) <= ON_TOP_TOL_M and tilt <= max_tilt_deg
    return {"ok": ok, "over_footprint": over, "gap_m": gap, "tilt_deg": round(tilt, 2)}


def wiped_coverage(m, d, task, object_id: str, target_id: str, limit: float | None, region,
                   points: list | None) -> dict:
    """Fraction of 1 cm bins along the target's long horizontal axis touched by the wiper (contact points recorded
    while wiping). Span = `region` [s_min, s_max] in metres from the target centre, else its central 80 %."""
    t = _obj(task, target_id)
    tb = _body(m, target_id)
    R = d.xmat[tb].reshape(3, 3)
    ax = R[:, 0] if t.dims_m[0] >= t.dims_m[1] else R[:, 1]
    L = max(t.dims_m[0], t.dims_m[1])
    s0, s1 = (region[0], region[1]) if region else (-0.4 * L, 0.4 * L)
    n = max(1, int(round((s1 - s0) / 0.01)))
    hit = np.zeros(n, bool)
    pts = np.asarray(points or [], float).reshape(-1, 3)
    if len(pts):
        s = (pts - d.xpos[tb]) @ ax
        k = np.floor((s - s0) / (s1 - s0) * n).astype(int)
        hit[k[(k >= 0) & (k < n)]] = True
    cov = float(hit.mean())
    need = 0.5 if limit is None else limit
    return {"ok": cov >= need, "coverage": round(cov, 3), "required": need, "span_m": [round(s0, 3), round(s1, 3)],
            "n_points": int(len(pts))}


def evaluate(m: mujoco.MjModel, d: mujoco.MjData, task, initial_pos: dict[str, list[float]],
             max_speed_hist: list[tuple[float, float]], unwanted_collisions: int = 0,
             wipe_contacts: dict[str, list] | None = None) -> dict:
    """Score every immutable predicate.

    initial_pos: object id -> body position at t = 0 (after settle).
    max_speed_hist: (t, max linear speed over all free objects) samples covering at least the last 0.5 s.
    wipe_contacts: "wiper|target" -> contact points recorded while wiping.
    """
    results = []
    for p in task.immutable_success_predicates:
        r: dict
        if p.kind == "inside_container":
            r = inside_container(m, d, task, p.object_id, p.target_id, p.max_tilt_deg)
        elif p.kind == "on_top_of":
            r = on_top_of(m, d, task, p.object_id, p.target_id, p.max_tilt_deg)
        elif p.kind == "max_obstacle_displacement":
            disp = float(np.linalg.norm(d.xpos[_body(m, p.object_id)] - np.asarray(initial_pos[p.object_id])))
            r = {"ok": disp <= p.limit, "displacement_m": disp}
        elif p.kind == "terminal_rest":
            if not max_speed_hist:
                r = {"ok": False, "max_speed": None}
            else:
                t_end = max_speed_hist[-1][0]
                window = [v for t, v in max_speed_hist if t >= t_end - REST_WINDOW_S]
                vmax = float(max(window))
                r = {"ok": vmax <= REST_SPEED, "max_speed": vmax}
        elif p.kind == "max_unwanted_collisions":
            r = {"ok": unwanted_collisions <= (p.limit or 0), "count": unwanted_collisions}
        elif p.kind == "wiped_coverage":
            r = wiped_coverage(m, d, task, p.object_id, p.target_id, p.limit, p.region,
                               (wipe_contacts or {}).get(f"{p.object_id}|{p.target_id}"))
        elif p.kind == "pushed_to_region":
            x, y = d.xpos[_body(m, p.object_id)][:2]
            x0, y0, x1, y1 = p.region
            r = {"ok": bool(x0 <= x <= x1 and y0 <= y <= y1), "xy": [float(x), float(y)]}
        else:
            raise ValueError(p.kind)
        r["ok"] = bool(r["ok"])
        results.append({"kind": p.kind, "object_id": p.object_id, "target_id": p.target_id, **r})
    return {"success": all(r["ok"] for r in results), "predicates": results}

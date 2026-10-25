"""Motion from video -> Draft 1 (replay of the demonstration, absolute values).

Per-frame labels give the fingertip pixel, open/closed and the held object. Contact constraints make them metric:
- grasp = the held object's twin pose (from Loop 1); release = the fingertip ray hitting the target's top plane;
- carry heights = the fingertip ray intersected with the vertical plane through the grasp -> place line,
  clamped to [object top + 2 cm, 0.5 m];
- the highest carry point becomes an absolute transit via point (the demonstrated arc).
"""
from __future__ import annotations

import math

import numpy as np

from core.camera_solve import pixel_ray
from core.locked_evaluator import body_aabb
from core.mjcf_compiler import object_height
from core.schemas import Keyframe, PolicyConfig, Primitive, TaskSpec

DEPTH = 0.018


def _uv_px(g, w: int, h: int) -> np.ndarray:
    y, x = g.uv
    return np.array([x / 1000 * w, y / 1000 * h])


def _smooth_held(seq: list[str | None], k: int = 1) -> list[str | None]:
    out = []
    for i in range(len(seq)):
        win = seq[max(0, i - k): i + k + 1]
        vals = [v for v in win if v is not None]
        out.append(max(set(vals), key=vals.count) if len(vals) * 2 > len(win) else None)
    return out


def segments(labels: list[Keyframe], arm: str = "arm") -> list[dict]:
    rows = []
    for kf in sorted(labels, key=lambda k: k.t):
        g = next((g for g in kf.gripper if g.arm == arm and g.uv), None)
        if g is not None:
            rows.append((kf.t, g))
    if not rows:
        return []
    held = _smooth_held([None if g.open else g.held for _, g in rows])
    segs, cur = [], None
    for i, ((t, g), hv) in enumerate(zip(rows, held)):
        if cur is None and hv:
            cur = {"object": hv, "grasp_i": i, "frames": []}
        elif cur is not None and hv != cur["object"]:
            cur["release_i"] = i
            segs.append(cur)
            cur = {"object": hv, "grasp_i": i, "frames": []} if hv else None
        if cur is not None:
            cur["frames"].append(i)
    for s in segs:
        s["rows"] = rows
    return segs


def _target_for(task: TaskSpec, ix, d, obj: str, hit_xy: np.ndarray, phases) -> str | None:
    """Phase label first; else the container / support whose footprint contains the release point."""
    for p in phases:
        if p.object == obj and p.target and p.action in ("place", "release", "transit"):
            return p.target
    best, bd = None, 1e9
    for o in task.objects:
        if o.id == obj or o.is_obstacle:
            continue
        lo, hi = body_aabb(ix.m, d, ix.obj_body[o.id])
        inside = lo[0] - 0.01 <= hit_xy[0] <= hi[0] + 0.01 and lo[1] - 0.01 <= hit_xy[1] <= hi[1] + 0.01
        dist = float(np.linalg.norm((lo[:2] + hi[:2]) / 2 - hit_xy)) - (1.0 if o.shape == "container" else 0.0)
        if inside and dist < bd:
            best, bd = o.id, dist
    return best


def extract(task: TaskSpec, labels: list[Keyframe], sim, w: int, h: int, cam: str = "video",
            arm: str = "arm") -> tuple[PolicyConfig, dict]:
    m, d, ix = sim.m, sim.d, sim.ix
    prims, log = [], []
    for seg in segments(labels, arm):
        rows = seg["rows"]
        obj = task.obj(seg["object"])
        lo, hi = body_aabb(m, d, ix.obj_body[obj.id])
        top = float(hi[2])
        grasp_xy = d.xpos[ix.obj_body[obj.id]][:2].copy()
        held_below = max(object_height(obj) - DEPTH, 0.0)
        rel = rows[seg["release_i"]] if "release_i" in seg else rows[seg["frames"][-1]]
        o_r, d_r = pixel_ray(m, d, cam, _uv_px(rel[1], w, h), w, h)
        t0r = (held_below + 0.01 - o_r[2]) / d_r[2] if abs(d_r[2]) > 1e-6 else 0.0
        target = _target_for(task, ix, d, obj.id, o_r[:2] + t0r * d_r[:2], task.phases)
        if target:
            tb = ix.obj_body[target]
            tlo, thi = body_aabb(m, d, tb)
            to = task.obj(target)
            floor = float(tlo[2] + to.wall_m) if to.shape == "container" else float(thi[2])
            z_plane = float(thi[2])
        else:
            floor, z_plane = 0.0, 0.0
        z_release_guess = floor + held_below + 0.01  # TCP height when the object is just above the floor
        tr = (z_release_guess - o_r[2]) / d_r[2] if abs(d_r[2]) > 1e-6 else 0.0
        hit = o_r[:2] + tr * d_r[:2]
        if target:
            t_xy = d.xpos[ix.obj_body[target]][:2]
            R = d.xmat[ix.obj_body[target]].reshape(3, 3)
            yaw = math.atan2(R[1, 0], R[0, 0])
            delta = hit - t_xy
            off = [math.cos(-yaw) * delta[0] - math.sin(-yaw) * delta[1],
                   math.sin(-yaw) * delta[0] + math.cos(-yaw) * delta[1]]
            lim = [max(to.dims_m[0] / 2 - to.wall_m - 0.02, 0.0), max(to.dims_m[1] / 2 - to.wall_m - 0.02, 0.0)]
            off = [float(np.clip(off[0], -lim[0], lim[0])), float(np.clip(off[1], -lim[1], lim[1]))]
            place_xy = t_xy + np.array([math.cos(yaw) * off[0] - math.sin(yaw) * off[1],
                                        math.sin(yaw) * off[0] + math.cos(yaw) * off[1]])
        else:
            off = [float(hit[0]), float(hit[1])]
            place_xy = hit
        # vertical plane through grasp -> place
        line = place_xy - grasp_xy
        n = np.array([-line[1], line[0], 0.0]) / (np.linalg.norm(line) + 1e-9)
        p0 = np.array([grasp_xy[0], grasp_xy[1], 0.0])
        def on_plane(g) -> np.ndarray | None:
            o, dr = pixel_ray(m, d, cam, _uv_px(g, w, h), w, h)
            den = float(n @ dr)
            return None if abs(den) < 1e-3 else o + (float(n @ (p0 - o)) / den) * dr

        carry = [(rows[i][0], p) for i in seg["frames"] if (p := on_plane(rows[i][1])) is not None]
        zmin, zmax = top + 0.02, 0.5
        zs = [float(np.clip(p[2], zmin, zmax)) for _, p in carry] or [top + 0.1]
        k3 = max(1, len(zs) // 3)
        apex_i = int(np.argmax(zs))
        lift, apex = max(zs[:k3]), max(zs)
        near = [z for (t, p), z in zip(carry, zs) if np.linalg.norm(p[:2] - place_xy) < 0.03]
        hover = max(near) if near else zs[-1]   # highest point once above the place, before lowering
        rel_z = float(np.clip(o_r[2] + tr * d_r[2], floor, zmax))
        drop = float(np.clip(rel_z - held_below - floor, 0.005, 0.1))
        after = rows[seg.get("release_i", len(rows) - 1):][:6]
        rz = [float(p[2]) for _, g in after if (p := on_plane(g)) is not None]
        retract = float(np.clip(max([top + 0.08] + rz), zmin, zmax))
        via = carry[apex_i][1][:2] if carry else (grasp_xy + place_xy) / 2
        t_carry = (carry[-1][0] - carry[0][0]) if len(carry) > 1 else 3.0
        path_len = float(np.linalg.norm(line)) + 2 * max(apex - top, 0.0)
        speed = float(np.clip(path_len / max(t_carry, 0.5) * 1.3, 0.15, 0.5))
        prims.append(Primitive(arm=arm, object_id=obj.id, target_id=target,
                               grasp_yaw_mode="object_relative", grasp_yaw_rad=0.0,
                               post_grasp_lift_z_m=round(max(lift, top + 0.05), 4),
                               transit_apex_z_m=round(apex, 4), transit_via_mode="absolute",
                               transit_via_point_xy_m=[round(float(v), 4) for v in via],
                               pre_place_hover_z_m=round(max(hover, z_plane + held_below + 0.02), 4),
                               place_xy_offset_m=[round(v, 4) for v in off], release_drop_height_m=round(drop, 4),
                               retract_z_m=round(retract, 4), speed_mps=round(speed, 3)))
        log.append({"object": obj.id, "target": target, "grasp_t": rows[seg["grasp_i"]][0], "release_t": rel[0],
                    "grasp_xy": [round(float(v), 4) for v in grasp_xy], "place_xy": [round(float(v), 4) for v in place_xy],
                    "carry": [{"t": round(t, 3), "xyz": [round(float(v), 4) for v in p]} for t, p in carry],
                    "lift_z": lift, "apex_z": apex, "hover_z": hover, "release_drop": drop, "speed_mps": speed})
    return PolicyConfig(name="draft1", primitives=prims), {"segments": log}

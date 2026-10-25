"""Camera pose, intrinsics and metric scale from the robot silhouette.

The robot model has its true size, so it is the calibration target: a dedicated world camera ("video") is moved until
the rendered robot silhouette matches the robot mask of the real frame. Parameters: (azimuth, elevation, distance,
lookat xyz, roll, fovy). Optimisation runs in internal coordinates where distance is replaced by the log image scale
of the robot (distance to the robot's depth * tan(fovy / 2)), so fovy alone moves along the weakly observed
dolly-zoom direction. Search, at 320 px: coarse (azimuth, elevation, distance) grid around a verbal estimate, each cell
moment-aligned (lookat shifted to match the mask centroid, distance scaled to match its area); Levenberg-Marquardt on
the Gaussian-blurred mask residual from the best cells and from fovy restarts along the dolly direction; Powell on a
soft IoU (min/max of blurred masks). The same LM + Powell refinement repeats at 640 px; IoU is reported at full
resolution. Pixel coordinates follow OpenCV: integer values at pixel centres, v down.
"""
from __future__ import annotations

import math
import time

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize

from core.mjcf_compiler import DEFAULT_CAMERAS, look_at_xyaxes
from core.schemas import CameraSpec, TaskSpec
from core.segment import mask_iou

PARAMS = ("azimuth_deg", "elevation_deg", "distance_m", "lookat_x", "lookat_y", "lookat_z", "roll_deg", "fovy_deg")
DEFAULT_INIT = dict(zip(PARAMS, (0.0, 30.0, 1.4, 0.35, 0.0, 0.15, 0.0, 45.0)))
STEP = np.array([10.0, 5.0, 0.05, 0.05, 0.05, 0.05, 2.0, 5.0])  # optimiser unit (index 2: log image scale)
LO = np.array([-np.inf, -15.0, 0.3, -0.4, -0.8, -0.2, -30.0, 15.0])
HI = np.array([np.inf, 85.0, 4.0, 1.2, 0.8, 0.8, 30.0, 100.0])
GRID = (np.arange(-45, 46, 15.0), np.arange(-20, 21, 10.0), np.arange(-0.6, 0.61, 0.3))
TOP_K = 6                          # grid cells refined at low resolution
DOLLY = (-6.0, -3.0, 3.0, 6.0)     # fovy offsets (deg, robot size kept) re-solved from the best low-res fit
SEARCH_W, REFINE_W = 320, 640
ACCEPT_IOU = 0.75
OBJ_WEIGHT = 0.5
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


# ----------------------------------------------------------------------------------------------------------------------
# Camera parameterisation
# ----------------------------------------------------------------------------------------------------------------------

def _vec(p) -> np.ndarray:
    if isinstance(p, dict):
        return np.array([float(p.get(k, DEFAULT_INIT[k])) for k in PARAMS])
    return np.asarray(p, float)


def params_to_pose(p) -> tuple[list[float], list[float], float]:
    """8 params (dict or array in PARAMS order) -> (pos, xyaxes, fovy). Azimuth from +x in the xy-plane."""
    az, el, dist, lx, ly, lz, roll, fovy = _vec(p)
    a, e, r = math.radians(az), math.radians(el), math.radians(roll)
    look = np.array([lx, ly, lz])
    pos = look + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    x, y = np.reshape(look_at_xyaxes(pos, look), (2, 3))
    xr, yr = math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y
    return pos.tolist(), [float(v) for v in (*xr, *yr)], float(fovy)


def xyaxes_to_quat(xyaxes) -> np.ndarray:
    x, y = np.reshape(np.asarray(xyaxes, float), (2, 3))
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)
    y /= np.linalg.norm(y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, np.cross(x, y)]).ravel())
    return q


def _cam_id(m: mujoco.MjModel, cam_name: str) -> int:
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise KeyError(f"camera {cam_name!r} not in model (compile with add_video_camera)")
    return cid


def set_camera(m: mujoco.MjModel, d: mujoco.MjData, cam_name: str, pos, xyaxes, fovy: float) -> int:
    """Move a world camera in place and refresh `d.cam_xpos` / `d.cam_xmat`."""
    cid = _cam_id(m, cam_name)
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = xyaxes_to_quat(xyaxes)
    m.cam_fovy[cid] = fovy
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    return cid


def with_camera(task: TaskSpec, cam: CameraSpec) -> TaskSpec:
    cams = [c for c in (task.cameras or DEFAULT_CAMERAS) if c.name != cam.name]
    return task.model_copy(update={"cameras": [*cams, cam]}, deep=True)


def add_video_camera(task: TaskSpec, w: int = 640, h: int = 480, name: str = "video") -> TaskSpec:
    pos, xy, fovy = params_to_pose(DEFAULT_INIT)
    return with_camera(task, CameraSpec(name=name, pos=pos, xyaxes=xy, fovy=fovy, w=w, h=h))


def apply_solution(task: TaskSpec, sol: dict, w: int, h: int, name: str = "video") -> TaskSpec:
    return with_camera(task, CameraSpec(name=name, pos=[float(v) for v in sol["pos"]],
                                        xyaxes=[float(v) for v in sol["xyaxes"]], fovy=float(sol["fovy"]), w=w, h=h))


# ----------------------------------------------------------------------------------------------------------------------
# Rendering and pinhole geometry
# ----------------------------------------------------------------------------------------------------------------------

def seg_renderer(m: mujoco.MjModel, w: int, h: int) -> mujoco.Renderer:
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    r = mujoco.Renderer(m, h, w)
    r.enable_segmentation_rendering()
    return r


def close_renderer(r: mujoco.Renderer) -> None:
    """Close with the renderer's own GL context current. `Renderer.close` frees GL objects in whatever context is
    current; if that belongs to another live renderer, its framebuffers (same object ids) are deleted and it then
    renders garbage."""
    if getattr(r, "_gl_context", None):
        r._gl_context.make_current()
    r.close()


def _body_lut(m: mujoco.MjModel, body_ids) -> np.ndarray:
    return np.append(np.isin(m.geom_bodyid, list(body_ids)), False)   # index -1 (background) -> False


def _seg_mask(seg: np.ndarray, lut: np.ndarray) -> np.ndarray:
    ids = np.where(seg[..., 1] == GEOM, seg[..., 0], -1)
    return lut[ids]


def render_body_mask(m: mujoco.MjModel, d: mujoco.MjData, cam_name: str, body_ids, w: int, h: int,
                     renderer: mujoco.Renderer | None = None) -> np.ndarray:
    """Visible pixels of geoms on `body_ids` (occlusion included). A passed renderer is switched to segmentation."""
    r = renderer or seg_renderer(m, w, h)
    try:
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam_name)
        return _seg_mask(r.render(), _body_lut(m, body_ids))
    finally:
        if renderer is None:
            close_renderer(r)


def _intrinsics(m: mujoco.MjModel, cid: int, w: int, h: int) -> tuple[float, float, float]:
    return 0.5 * h / math.tan(math.radians(m.cam_fovy[cid]) / 2), (w - 1) / 2, (h - 1) / 2


def project_points(m: mujoco.MjModel, d: mujoco.MjData, cam_name: str, pts_world, w: int, h: int) -> np.ndarray:
    """(N, 3) world points -> (N, 2) pixels (u, v); NaN behind the camera."""
    cid = _cam_id(m, cam_name)
    f, cx, cy = _intrinsics(m, cid, w, h)
    pc = (np.atleast_2d(np.asarray(pts_world, float)) - d.cam_xpos[cid]) @ d.cam_xmat[cid].reshape(3, 3)
    z = -pc[:, 2]
    z = np.where(z > 1e-9, z, np.nan)
    return np.stack([cx + f * pc[:, 0] / z, cy - f * pc[:, 1] / z], 1)


def pixel_ray(m: mujoco.MjModel, d: mujoco.MjData, cam_name: str, uv, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel(s) (..., 2) -> (camera origin (3,), unit world direction (..., 3))."""
    cid = _cam_id(m, cam_name)
    f, cx, cy = _intrinsics(m, cid, w, h)
    uv = np.asarray(uv, float)
    dc = np.stack([(uv[..., 0] - cx) / f, -(uv[..., 1] - cy) / f, -np.ones(uv.shape[:-1])], -1)
    dw = dc @ d.cam_xmat[cid].reshape(3, 3).T
    return d.cam_xpos[cid].copy(), dw / np.linalg.norm(dw, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------------------------------

def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma) if sigma > 0 else a.astype(np.float32)


def _soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = float(np.maximum(a, b).sum())
    return float(np.minimum(a, b).sum()) / union if union > 0 else 0.0


def _moment_align(p: np.ndarray, rendered: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shift lookat (and camera) parallel to the image plane to match centroids; scale distance to match areas."""
    ar, at = rendered.sum(), target.sum()
    if ar < 10 or at < 10:
        return p
    h, w = rendered.shape
    vr, ur = np.nonzero(rendered)
    du = (target * np.arange(w)).sum() / at - ur.mean()
    dv = (target * np.arange(h)[:, None]).sum() / at - vr.mean()
    _, xy, fovy = params_to_pose(p)
    x, y = np.reshape(xy, (2, 3))
    q = p.copy()
    q[3:6] += (-du * x + dv * y) * p[2] / (0.5 * h / math.tan(math.radians(fovy) / 2))
    q[2] = p[2] * math.sqrt(ar / at)
    return np.clip(q, LO, HI)


class _Objective:
    """Weighted soft IoU of blurred rendered vs real masks at one resolution; one segmentation render per call."""

    def __init__(self, m, d, cam_name, terms, w, h, sigma, renderer):
        self.m, self.d, self.cam, self.r, self.sigma = m, d, cam_name, renderer, sigma
        self.luts = [_body_lut(m, b) for b, _, _ in terms]
        self.weights = [wt for _, _, wt in terms]
        self.targets_raw = [cv2.resize(mk.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                            for _, mk, _ in terms]
        self.targets = [_blur(t, sigma) for t in self.targets_raw]
        self.centre = d.xpos[sorted(terms[0][0])].mean(0)
        self.evals = 0

    def masks(self, p) -> list[np.ndarray]:
        self.evals += 1
        set_camera(self.m, self.d, self.cam, *params_to_pose(p))
        self.r.update_scene(self.d, camera=self.cam)
        seg = self.r.render()
        return [_seg_mask(seg, lut) for lut in self.luts]

    def residual(self, p) -> np.ndarray:
        return np.concatenate([math.sqrt(wt) * (_blur(mk, self.sigma) - t).ravel()
                               for mk, t, wt in zip(self.masks(p), self.targets, self.weights)])

    def score(self, p, masks=None) -> float:
        masks = masks if masks is not None else self.masks(p)
        return sum(wt * _soft_iou(_blur(mk, self.sigma), t) for mk, t, wt in zip(masks, self.targets, self.weights))


def _to_internal(p: np.ndarray) -> np.ndarray:
    """distance -> log image scale (distance * tan(fovy / 2)): fovy then changes perspective only, not size."""
    c = p.copy()
    c[2] = math.log(p[2] * math.tan(math.radians(p[7]) / 2))
    return c


def _from_internal(c: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    p = np.clip(c, lo, hi)
    p[2] = np.clip(math.exp(c[2]) / math.tan(math.radians(p[7]) / 2), lo[2], hi[2])
    return p


def _recentre(p: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Slide lookat along the viewing ray to the depth of `centre` (same camera), so the scale-preserving fovy
    direction of the internal coordinates keeps the robot's size fixed."""
    pos = np.array(params_to_pose(p)[0])
    fwd = (p[3:6] - pos) / p[2]
    t = float((centre - pos) @ fwd)
    if not LO[2] <= t <= HI[2]:
        return p
    q = p.copy()
    q[2], q[3:6] = t, pos + t * fwd
    return q


def _jacobian(obj: _Objective, c: np.ndarray, r0: np.ndarray, step: np.ndarray, lo, hi, h: float) -> np.ndarray:
    return np.stack([(obj.residual(_from_internal(c + h * e * step, lo, hi)) - r0) / h for e in np.eye(len(c))], 1)


def _lm(obj: _Objective, p0: np.ndarray, step: np.ndarray, lo: np.ndarray, hi: np.ndarray, iters: int = 25,
        h: float = 0.3) -> np.ndarray:
    """Levenberg-Marquardt on the blurred-mask residual (forward-difference Jacobian, steps of h * step).
    Gauss-Newton steps move along the coupled fovy / distance / elevation valley directly."""
    c = _to_internal(_recentre(p0, obj.centre))
    r = obj.residual(_from_internal(c, lo, hi))
    cost, mu = r @ r, 1e-3
    for _ in range(iters):
        J = _jacobian(obj, c, r, step, lo, hi, h)
        A, g = J.T @ J, J.T @ r
        damp = np.diag(A) + 1e-9 * np.trace(A)
        while mu < 1e4:
            dz = -np.linalg.solve(A + mu * np.diag(damp), g)
            cn = c + dz * step
            rn = obj.residual(_from_internal(cn, lo, hi))
            if rn @ rn < cost:
                break
            mu *= 4
        else:
            break
        c, r, cost, mu = cn, rn, rn @ rn, max(mu / 3, 1e-6)
        if np.abs(dz).max() < 0.01:
            break
    return _from_internal(c, lo, hi)


def _powell(obj: _Objective, p0: np.ndarray, step: np.ndarray, lo: np.ndarray, hi: np.ndarray, maxfev: int,
            xtol: float = 0.01, h: float = 0.3) -> np.ndarray:
    """Powell on -soft IoU in step-scaled internal coordinates, started on Gauss-Newton conjugate directions
    (eigenvectors of J^T J scaled by 1/sqrt(eigenvalue), capped). Line searches are unbounded (bracketed from the
    current point, so they stay local); parameter limits are enforced by clipping in the mapping."""
    p0 = _recentre(p0, obj.centre)
    c0 = _to_internal(p0)
    lam, V = np.linalg.eigh((J := _jacobian(obj, c0, obj.residual(p0), step, lo, hi, h)).T @ J)
    direc = (V * (h * np.clip(np.sqrt(lam.max() / np.maximum(lam, 1e-12)), 1.0, 30.0))).T
    res = minimize(lambda z: -obj.score(_from_internal(c0 + z * step, lo, hi)), np.zeros(len(p0)), method="Powell",
                   options=dict(xtol=xtol, ftol=1e-6, maxfev=maxfev, direc=direc))
    return _from_internal(c0 + res.x * step, lo, hi)


def _dolly(p: np.ndarray, dfovy: float, centre: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    c = _to_internal(_recentre(p, centre))
    c[7] += dfovy
    return _from_internal(c, lo, hi)


def _grid(obj: _Objective, p0: np.ndarray, lo: np.ndarray, hi: np.ndarray, k: int) -> list[np.ndarray]:
    """Top-k moment-aligned cells of the (azimuth, elevation, distance) grid around p0."""
    scored = []
    for daz in GRID[0]:
        for de in GRID[1]:
            for dd in GRID[2]:
                p = np.clip(p0 + np.array([daz, de, dd, 0, 0, 0, 0, 0]), lo, hi)
                p = _moment_align(p, obj.masks(p)[0], obj.targets_raw[0])
                scored.append((obj.score(p), p))
    scored.sort(key=lambda t: -t[0])
    return [p for _, p in scored[:k]]


def solve_camera(m: mujoco.MjModel, d: mujoco.MjData, robot_mask: np.ndarray, robot_bodies: set[int],
                 cam_name: str = "video", init: dict | None = None, obj_terms: list | None = None) -> dict:
    """Fit the world camera `cam_name` so the rendered robot (+ optional `(body_ids, mask)` object terms, weight
    0.5 each; masks at the robot mask's resolution) matches the real masks. `init` takes any PARAMS keys (others are
    ignored). Leaves the camera at the solution."""
    t0 = time.time()
    H, W = robot_mask.shape
    sw = min(SEARCH_W, W)
    sh = max(1, round(sw * H / W))
    p0 = _vec({**DEFAULT_INIT, **{k: v for k, v in (init or {}).items() if k in PARAMS}})
    lo, hi = LO.copy(), HI.copy()
    lo[0], hi[0] = p0[0] - 180, p0[0] + 180
    p0 = np.clip(p0, lo, hi)
    terms = [(set(robot_bodies), robot_mask, 1.0)] + [(set(b), mk, OBJ_WEIGHT) for b, mk in (obj_terms or [])]
    rw = min(REFINE_W, W)
    rh = max(1, round(rw * H / W))
    r_lo, r_mid = seg_renderer(m, sw, sh), seg_renderer(m, rw, rh)
    try:
        coarse = _Objective(m, d, cam_name, terms, sw, sh, 4.0, r_lo)
        fine = _Objective(m, d, cam_name, terms, sw, sh, 2.0, r_lo)
        best = max((_lm(fine, p, STEP, lo, hi) for p in _grid(coarse, p0, lo, hi, TOP_K)), key=fine.score)
        best = max([best] + [_lm(fine, _dolly(best, df, fine.centre, lo, hi), STEP, lo, hi) for df in DOLLY],
                   key=fine.score)
        best = _powell(fine, best, STEP * 0.3, lo, hi, maxfev=600)
        mid = _Objective(m, d, cam_name, terms, rw, rh, 1.0, r_mid)
        best = max([_lm(mid, _dolly(best, df, mid.centre, lo, hi), STEP * 0.3, lo, hi) for df in (0.0, -1.5, 1.5)],
                   key=mid.score)
        best = _powell(mid, best, STEP * 0.1, lo, hi, maxfev=600)
        pos, xy, fovy = params_to_pose(best)
        set_camera(m, d, cam_name, pos, xy, fovy)
        iou = mask_iou(render_body_mask(m, d, cam_name, robot_bodies, W, H, r_mid if rw == W else None), robot_mask)
        evals = coarse.evals + fine.evals + mid.evals
    finally:
        close_renderer(r_lo)
        close_renderer(r_mid)
    return dict(pos=pos, xyaxes=xy, fovy=fovy, params=dict(zip(PARAMS, map(float, best))), iou=iou,
                accepted=iou >= ACCEPT_IOU, evals=evals, seconds=time.time() - t0)

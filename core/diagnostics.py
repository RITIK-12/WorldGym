"""4-view diagnostic matrix: what the Twin Critic watches each Loop 1 iteration.

1. GHOST     real frame (dimmed) with sim object outlines (green), real mask outlines (magenta), robot silhouette
             (cyan) through the solved video camera.
2. TOP       orthographic top-down sim render (robot hidden) with a 5 cm grid; each real mask is back-projected
             onto the table plane and onto its object's top plane and the two regions are intersected, which
             recovers the footprint of a prism-like object; drawn magenta, so offsets read in centimetres.
3. GROUND    table-level side view (robot hidden) with height lines every 5 cm: floating, sunk, wrongly stacked.
4. WIREFRAME collision geometry from the video camera blended over the real frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from core.camera_solve import close_renderer, pixel_ray, project_points, render_body_mask, seg_renderer
from core.mjcf_compiler import SceneIndex, object_height

GREEN, MAGENTA, CYAN, WHITE, YELLOW = (80, 255, 60), (255, 0, 255), (255, 220, 0), (255, 255, 255), (0, 220, 255)
GRID_M = 0.05


def _contours(img: np.ndarray, mask: np.ndarray, colour, thick: int = 2) -> None:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, colour, thick, cv2.LINE_AA)


def _label(img: np.ndarray, text: str, org, colour=WHITE, scale: float = 0.55) -> None:
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _title(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 1, cv2.LINE_AA)


def _rgb(r: mujoco.Renderer, d, cam: str, opt: mujoco.MjvOption | None = None, wire: bool = False) -> np.ndarray:
    r.disable_segmentation_rendering()
    r.update_scene(d, camera=cam, scene_option=opt)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = wire
    img = r.render().copy()
    r.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = False
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def sim_object_masks(ix: SceneIndex, d: mujoco.MjData, cam: str, w: int, h: int,
                     renderer: mujoco.Renderer | None = None) -> dict[str, np.ndarray]:
    return {oid: render_body_mask(ix.m, d, cam, {b}, w, h, renderer) for oid, b in ix.obj_body.items()}


def backproject_mask(ix: SceneIndex, d: mujoco.MjData, cam: str, mask: np.ndarray, z: float) -> np.ndarray | None:
    """Outline of a real mask intersected with the horizontal plane at height z -> (N, 2) world xy."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    h, w = mask.shape
    o, dirs = pixel_ray(ix.m, d, cam, c, w, h)
    t = (z - o[2]) / np.where(np.abs(dirs[:, 2]) > 1e-9, dirs[:, 2], np.nan)
    pts = o[None, :2] + t[:, None] * dirs[:, :2]
    return pts[np.isfinite(pts).all(1) & (t > 0)]


def footprint_px(ix: SceneIndex, d: mujoco.MjData, cam: str, mask: np.ndarray, height: float, to_px,
                 shape: tuple[int, int]) -> np.ndarray | None:
    """Real-object footprint in a top-view image: silhouette back-projected at z=0 AND at z=height."""
    regions = []
    for z in (0.0, height):
        pts = backproject_mask(ix, d, cam, mask, z)
        if pts is None or len(pts) < 3:
            return None
        poly = np.array([to_px(x, y) for x, y in pts], np.int32)
        r = np.zeros(shape, np.uint8)
        cv2.fillPoly(r, [poly], 1)
        regions.append(r.astype(bool))
    fp = regions[0] & regions[1]
    return fp if fp.any() else None


def render_matrix(ix: SceneIndex, d: mujoco.MjData, real_bgr: np.ndarray, real_masks: dict[str, np.ndarray],
                  out_dir: str | Path, cam: str = "video", robot_mask: np.ndarray | None = None,
                  note: str = "") -> dict[str, Path]:
    """Render the four views (sim at its current state, e.g. robot at the rest pose) and the 2x2 matrix."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m, task = ix.m, ix.task
    h, w = real_bgr.shape[:2]
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, w), max(m.vis.global_.offheight, h)
    seg = seg_renderer(m, w, h)
    rgb = mujoco.Renderer(m, h, w)
    try:
        sim_masks = sim_object_masks(ix, d, cam, w, h, seg)
        sim_robot = render_body_mask(m, d, cam, ix.robot_bodies, w, h, seg)

        ghost = (real_bgr * 0.55).astype(np.uint8)
        _contours(ghost, sim_robot, CYAN, 1)
        if robot_mask is not None:
            _contours(ghost, robot_mask, YELLOW, 1)
        for oid, mk in real_masks.items():
            _contours(ghost, mk, MAGENTA, 2)
        for oid, mk in sim_masks.items():
            if mk.any():
                _contours(ghost, mk, GREEN, 2)
                ys, xs = np.nonzero(mk)
                _label(ghost, oid, (xs.mean() - 20, ys.min() - 6), GREEN)
        _title(ghost, f"1 GHOST  sim=green  real=magenta  robot sim=cyan{'/real=yellow' if robot_mask is not None else ''}")

        no_robot = mujoco.MjvOption()
        no_robot.geomgroup[2] = no_robot.geomgroup[3] = 0
        top = _rgb(rgb, d, "top", no_robot)
        th, tw = top.shape[:2]
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "top")
        ppm = th / m.cam_fovy[cid]
        cxw, cyw = d.cam_xpos[cid][0], d.cam_xpos[cid][1]
        to_px = lambda x, y: (int(tw / 2 + (y - cyw) * ppm), int(th / 2 + (x - cxw) * ppm))  # noqa: E731
        for k in range(-20, 21):
            gx, gy = cxw + k * GRID_M, cyw + k * GRID_M
            col = (90, 90, 90) if k % 2 else (140, 140, 140)
            cv2.line(top, to_px(gx, cyw - 2), to_px(gx, cyw + 2), col, 1)
            cv2.line(top, to_px(cxw - 2, gy), to_px(cxw + 2, gy), col, 1)
        for o in task.objects:
            mk = real_masks.get(o.id)
            if mk is None or not mk.any():
                continue
            fp = footprint_px(ix, d, cam, mk, object_height(o), to_px, (th, tw))
            if fp is not None:
                _contours(top, fp, MAGENTA, 2)
        for o in task.objects:
            p = d.xpos[ix.obj_body[o.id]]
            _label(top, o.id, np.array(to_px(p[0], p[1])) + (6, -6), GREEN)
        _label(top, "+x (away from robot) is DOWN, +y is RIGHT", (10, th - 12), WHITE, 0.5)
        _title(top, "2 TOP (ortho)  grid 5 cm  real footprints=magenta")

        ground = _rgb(rgb, d, "ground", no_robot)
        gh, gw = ground.shape[:2]
        for zc in (0.0, 0.05, 0.10, 0.15, 0.20):
            uv = project_points(m, d, "ground", [[0.45, 0.0, zc]], gw, gh)[0]
            if np.isfinite(uv).all():
                cv2.line(ground, (0, int(uv[1])), (gw, int(uv[1])), (0, 200, 255), 1)
                _label(ground, f"z={zc * 100:.0f} cm", (8, int(uv[1]) - 4), (0, 200, 255), 0.45)
        _title(ground, "3 GROUND (table level, side)  height lines every 5 cm")

        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = opt.geomgroup[3] = 1   # objects (group 0) + robot collision meshes (group 3)
        wire = _rgb(rgb, d, cam, opt, wire=True)
        wire = cv2.addWeighted(real_bgr, 0.5, wire, 0.8, 0)
        _title(wire, "4 COLLISION WIREFRAME over the real frame")
    finally:
        close_renderer(rgb)
        close_renderer(seg)

    tiles = {"ghost": ghost, "top": cv2.resize(top, (w, h)), "ground": cv2.resize(ground, (w, h)), "wire": wire}
    paths = {}
    for k, img in tiles.items():
        paths[k] = out / f"{k}.jpg"
        cv2.imwrite(str(paths[k]), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    grid = np.concatenate([np.concatenate([tiles["ghost"], tiles["top"]], 1),
                           np.concatenate([tiles["ground"], tiles["wire"]], 1)], 0)
    if note:
        _label(grid, note, (12, grid.shape[0] - 14), YELLOW, 0.8)
    paths["matrix"] = out / "matrix.jpg"
    cv2.imwrite(str(paths["matrix"]), cv2.resize(grid, (min(grid.shape[1], 2560), int(grid.shape[0] * min(1, 2560 / grid.shape[1])))),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return paths

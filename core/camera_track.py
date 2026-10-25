"""Moving (handheld) camera: solve once at a reference frame, then track through the clip with the table plane.

The table top is the plane z = 0 and real tables/mats are textured, so each frame's camera follows from a
plane-induced homography: features matched between the reference frame and frame t (RANSAC) give H(ref -> t);
the plane's world -> image map is Hw_ref = K [r1 r2 t] from the solved reference camera, so Hw_t = H Hw_ref, and
K^-1 Hw_t = s [r1 r2 t] gives the camera at t (intrinsics fixed). Frames that do not match the reference directly
are chained through the nearest tracked frame.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from core.video_probe import read_frame

N_FEATURES = 5000
RATIO = 0.75
MIN_INLIERS = 40
RANSAC_PX = 3.0
WORK_W = 1280


def intrinsics(fovy_deg: float, w: int, h: int) -> np.ndarray:
    f = 0.5 * h / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def world_to_cv(pos, xyaxes) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo camera (x right, y up, looks along -z) -> OpenCV extrinsics R_cw, t (x right, y down, z forward)."""
    x = np.asarray(xyaxes[:3], float)
    y = np.asarray(xyaxes[3:], float)
    z = np.cross(x, y)
    R = np.stack([x, -y, -z])
    return R, -R @ np.asarray(pos, float)


def cv_to_world(R: np.ndarray, t: np.ndarray) -> tuple[list[float], list[float]]:
    pos = -R.T @ t
    return pos.tolist(), [*R[0].tolist(), *(-R[1]).tolist()]


def plane_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def pose_from_plane_h(K: np.ndarray, Hw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.linalg.solve(K, Hw)
    A = A * 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if A[2, 2] < 0:  # the plane must be in front of the camera
        A = -A
    R = np.column_stack([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, A[:, 2]


def _gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    s = WORK_W / frame.shape[1] if frame.shape[1] > WORK_W else 1.0
    if s != 1.0:
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), s


def _features(det, gray: np.ndarray, mask: np.ndarray | None):
    return det.detectAndCompute(gray, mask)


def _match_h(det, matcher, a, b) -> tuple[np.ndarray | None, int, float]:
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0, float("inf")
    pairs = matcher.knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None, len(good), float("inf")
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if H is None:
        return None, 0, float("inf")
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.median(np.linalg.norm(proj - dst[inl], axis=1))) if inl.any() else float("inf")
    return H, int(inl.sum()), err


def _table_mask(Hw: np.ndarray, rect, shape, scale: float, exclude) -> np.ndarray:
    cx, cy, sx, sy = rect
    corners = np.array([[cx - sx / 2, cy - sy / 2, 1], [cx + sx / 2, cy - sy / 2, 1], [cx + sx / 2, cy + sy / 2, 1],
                        [cx - sx / 2, cy + sy / 2, 1]], float)
    px = (Hw @ corners.T).T
    px = px[:, :2] / px[:, 2:3] * scale
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 255)
    for x0, y0, x1, y1 in exclude or []:
        cv2.rectangle(mask, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)), 0, -1)
    return mask


def track_camera(video: str | Path, ref_t: float, ref_pos, ref_xyaxes, fovy: float, times: list[float],
                 table_rect=(0.0, 0.0, 1.2, 0.8), exclude=None) -> list[dict]:
    """Camera pose (MuJoCo pos/xyaxes, fixed fovy) at each time. `exclude(t)` -> pixel boxes (robot, moving objects)
    to keep out of the reference feature mask."""
    ref = read_frame(video, ref_t)
    h, w = ref.shape[:2]
    K = intrinsics(fovy, w, h)
    R0, t0 = world_to_cv(ref_pos, ref_xyaxes)
    Hw_ref = plane_homography(K, R0, t0)
    det = cv2.SIFT_create(N_FEATURES) if hasattr(cv2, "SIFT_create") else cv2.ORB_create(N_FEATURES)
    norm = cv2.NORM_L2 if hasattr(cv2, "SIFT_create") else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    g_ref, s = _gray(ref)
    S = np.diag([s, s, 1.0])
    f_ref = _features(det, g_ref, _table_mask(Hw_ref, table_rect, g_ref.shape, s, exclude(ref_t) if exclude else None))
    out, prev = [], None   # prev = (features, Hw in work pixels) of the last tracked frame
    for t in sorted(times):
        g, _ = _gray(read_frame(video, t))
        f_t = _features(det, g, None)
        H, n, err = _match_h(det, matcher, f_ref, f_t)
        method, Hw_s = "direct", None
        if H is not None:
            Hw_s = H @ S @ Hw_ref
        elif prev is not None:
            H2, n, err = _match_h(det, matcher, prev[0], f_t)
            if H2 is not None:
                Hw_s, method = H2 @ prev[1], "chained"
        if Hw_s is None:
            out.append({"t": t, "ok": False, "n_inliers": n, "reproj_px": None, "method": "lost"})
            continue
        Hw = np.linalg.inv(S) @ Hw_s
        R, tt = pose_from_plane_h(K, Hw)
        pos, xy = cv_to_world(R, tt)
        out.append({"t": t, "ok": True, "pos": pos, "xyaxes": xy, "fovy": fovy, "n_inliers": n,
                    "reproj_px": round(err / s, 2), "method": method})
        Hw_tab = Hw_s
        prev = (_features(det, g, _table_mask(np.linalg.inv(S) @ Hw_tab, table_rect, g.shape, s, None)), Hw_tab)
    return out


def camera_at(track: list[dict], t: float) -> dict | None:
    ok = [c for c in track if c["ok"]]
    return min(ok, key=lambda c: abs(c["t"] - t)) if ok else None


def track_overlay(video: str | Path, cam: dict, t: float, out_path: str | Path, table_rect=(0.0, 0.0, 1.2, 0.8),
                  step: float = 0.05) -> Path:
    """Frame with the table grid (every `step` m) re-projected through the tracked camera."""
    frame = read_frame(video, t)
    h, w = frame.shape[:2]
    K = intrinsics(cam["fovy"], w, h)
    R, tt = world_to_cv(cam["pos"], cam["xyaxes"])
    cx, cy, sx, sy = table_rect
    for k in np.arange(-sx / 2, sx / 2 + 1e-9, step):
        for a, b in (((cx + k, cy - sy / 2), (cx + k, cy + sy / 2)),):
            _line(frame, K, R, tt, a, b)
    for k in np.arange(-sy / 2, sy / 2 + 1e-9, step):
        _line(frame, K, R, tt, (cx - sx / 2, cy + k), (cx + sx / 2, cy + k))
    cv2.imwrite(str(out_path), frame)
    return Path(out_path)


def _line(img, K, R, t, a, b) -> None:
    pts = np.array([[a[0], a[1], 0.0], [b[0], b[1], 0.0]])
    pc = (R @ pts.T).T + t
    if (pc[:, 2] <= 0.05).any():
        return
    px = (K @ pc.T).T
    px = px[:, :2] / px[:, 2:3]
    cv2.line(img, tuple(np.round(px[0]).astype(int)), tuple(np.round(px[1]).astype(int)), (0, 255, 255), 1, cv2.LINE_AA)

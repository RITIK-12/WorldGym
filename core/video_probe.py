"""Input-acceptance QA gate for a robot video: static camera, steady exposure, sharp frames, usable length.

A frame is sampled every 0.25 s at 640 px wide. Each consecutive pair is registered with ORB + RANSAC homography
(identity when >= 30 % of matches are stationary, so a large moving robot cannot fake camera motion):
camera shift = mean displacement of the image corners, residual motion = fraction of pixels that still differ after
warping the earlier frame onto the later one. Keyframes are peaks of the smoothed residual-motion curve.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SAMPLE_DT = 0.25
WORK_W = 640
ORB_FEATURES = 1500
TOP_MATCHES = 400
MIN_MATCHES = 10
RANSAC_PX = 3.0
DIFF_LEVEL = 25
STATIC_PX = 2.0
STILL_MATCH_FRACTION = 0.3
MIN_STATIC_FRACTION = 0.8
MAX_DURATION_S = 30.0
MAX_BRIGHTNESS_RANGE = 40.0
BLUR_VAR = 60.0
MAX_BLUR_FRACTION = 0.2
MIN_SIDE = 720
SMOOTH_K = 5
KEYFRAME_GAP_S = 0.5


# ----------------------------------------------------------------------------------------------------------------------
# Per-sample features and pair registration
# ----------------------------------------------------------------------------------------------------------------------

def _aruco_detector():
    try:
        return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    except Exception:
        return None


def _aruco_count(det, gray: np.ndarray) -> int:
    if det is None:
        return 0
    try:
        _, ids, _ = det.detectMarkers(gray)
        return 0 if ids is None else int(len(np.unique(np.asarray(ids).ravel())))
    except Exception:
        return 0


def _features(frame: np.ndarray, orb, aruco) -> dict:
    h, w = frame.shape[:2]
    if w != WORK_W:
        frame = cv2.resize(frame, (WORK_W, max(1, round(h * WORK_W / w))), interpolation=cv2.INTER_AREA)
    if frame.ndim == 2:
        g = frame
    else:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    kp, des = orb.detectAndCompute(g, None)
    return {"gray": g, "kp": kp, "des": des, "blur": float(cv2.Laplacian(g, cv2.CV_64F).var()),
            "brightness": float(g.mean()), "aruco": _aruco_count(aruco, g)}


def _homography(a: dict, b: dict, bf) -> np.ndarray | None:
    if a["des"] is None or b["des"] is None or len(a["kp"]) < MIN_MATCHES or len(b["kp"]) < MIN_MATCHES:
        return None
    ms = sorted(bf.match(a["des"], b["des"]), key=lambda m: m.distance)[:TOP_MATCHES]
    if len(ms) < MIN_MATCHES:
        return None
    src = np.float32([a["kp"][m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
    dst = np.float32([b["kp"][m.trainIdx].pt for m in ms]).reshape(-1, 1, 2)
    # A large moving robot can out-vote the background in RANSAC; if enough matches did not move at all, the camera
    # is static and the movers are the scene.
    if np.mean(np.linalg.norm((dst - src).reshape(-1, 2), axis=1) < STATIC_PX) >= STILL_MATCH_FRACTION:
        return np.eye(3)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    return H if H is not None and np.all(np.isfinite(H)) else None


def _register(a: dict, b: dict, bf) -> tuple[float, float]:
    """(camera shift px, residual motion fraction) from sample a to sample b; shift is inf when registration fails."""
    g0, g1 = a["gray"], b["gray"]
    H = _homography(a, b, bf)
    if H is None:
        return math.inf, float(np.mean(cv2.absdiff(g0, g1) > DIFF_LEVEL))
    h, w = g1.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    shift = float(np.linalg.norm(cv2.perspectiveTransform(corners, H) - corners, axis=2).mean())
    warped = cv2.warpPerspective(g0, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpPerspective(np.ones_like(g0), H, (w, h), flags=cv2.INTER_NEAREST) > 0
    diff = cv2.absdiff(g1, warped) > DIFF_LEVEL
    return shift, float(diff[valid].mean()) if valid.any() else 1.0


# ----------------------------------------------------------------------------------------------------------------------
# Curve analysis
# ----------------------------------------------------------------------------------------------------------------------

def _smooth(x: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    k = min(k, len(x) - (1 - len(x) % 2))
    if k < 3:
        return x.astype(float)
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def _keyframes(t: np.ndarray, residual: np.ndarray) -> list[float]:
    n = len(t)
    if n == 0:
        return []
    s = _smooth(residual)
    thr = np.percentile(s, 75)
    peaks = [i for i in range(1, n - 1) if s[i] > thr and s[i] >= s[i - 1] and s[i] > s[i + 1]]
    keep: list[int] = []
    for i in [0, n - 1] + sorted(peaks, key=lambda i: -s[i]):
        if all(abs(t[i] - t[j]) >= KEYFRAME_GAP_S for j in keep):
            keep.append(i)
    return sorted(round(float(t[i]), 2) for i in keep)


def _trim(t: np.ndarray, residual: np.ndarray, duration: float) -> list[float] | None:
    """<= MAX_DURATION_S window centred on the motion-energy centroid of the highest-energy window."""
    if duration <= MAX_DURATION_S:
        return None
    n_win = min(len(t), round(MAX_DURATION_S / SAMPLE_DT))
    c = np.concatenate([[0.0], np.cumsum(residual)])
    i = int(np.argmax(c[n_win:] - c[:-n_win]))
    tw, rw = t[i:i + n_win], residual[i:i + n_win]
    centre = float((tw * rw).sum() / rw.sum()) if rw.sum() > 0 else float(tw.mean())
    t0 = min(max(centre - MAX_DURATION_S / 2, 0.0), duration - MAX_DURATION_S)
    return [round(t0, 2), round(t0 + MAX_DURATION_S, 2)]


def _plot(path: Path, t: np.ndarray, residual: np.ndarray, moving: np.ndarray, keyframes: list[float]) -> None:
    """Residual-motion curve (smoothed in black), keyframe ticks in red, camera-moving samples as an orange strip."""
    W, H, L, R, T, B = 640, 200, 44, 12, 22, 26
    img = np.full((H, W, 3), 255, np.uint8)
    t_max = max(float(t[-1]) if len(t) else 0.0, 1e-6)
    y_max = max(float(residual.max()) if len(residual) else 0.0, 0.01)
    px = lambda tt: int(L + (W - L - R) * tt / t_max)
    py = lambda v: int(H - B - (H - B - T) * v / y_max)
    cv2.rectangle(img, (L, T), (W - R, H - B), (200, 200, 200), 1)
    for tt, mv in zip(t, moving):
        if mv:
            cv2.line(img, (px(tt), T - 6), (px(tt), T - 2), (0, 140, 255), 2)
    for k in keyframes:
        cv2.line(img, (px(k), T), (px(k), H - B), (60, 60, 230), 1)
    if len(t) > 1:
        for y, colour in ((residual, (230, 160, 90)), (_smooth(residual), (30, 30, 30))):
            pts = np.int32([[px(a), py(b)] for a, b in zip(t, y)]).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, colour, 1, cv2.LINE_AA)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.4
    cv2.putText(img, f"{y_max:.2f}", (4, T + 4), font, fs, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, "0", (28, H - B + 4), font, fs, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, f"{t_max:.1f} s", (W - R - 44, H - 8), font, fs, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, "residual motion, keyframes (red)", (L, H - 8), font, fs, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


# ----------------------------------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------------------------------

def _rejected(reason: str, warnings: list[str] | None = None) -> dict:
    return {"accepted": False, "moving_camera_ok": False, "reasons": [reason], "warnings": warnings or [],
            "headline": f"rejected: {reason}",
            "duration_s": 0.0, "fps": 0.0, "width": 0, "height": 0, "n_samples": 0, "static_fraction": None,
            "brightness_range": None, "blur_low_fraction": None, "aruco_detect_fraction": None, "keyframes_t": [],
            "trim": None, "curve": {"t": [], "shift_px": [], "residual": [], "blur": [], "brightness": []}}


def _fps(cap: cv2.VideoCapture) -> float | None:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if math.isfinite(fps) and 0 < fps <= 1000 else None


def _sample(video_path: Path) -> tuple[list[dict], list[tuple[float, float]], float, int, list[str], str | None]:
    """Decode sequentially, keep one sample per SAMPLE_DT. Returns samples, pair metrics, fps, frames read,
    warnings and a fatal error (or None)."""
    if not video_path.is_file():
        return [], [], 0.0, 0, [], f"file not found: {video_path}"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], 0.0, 0, [], f"cannot open video: {video_path.name}"
    warnings = []
    fps = _fps(cap)
    if fps is None:
        fps = 30.0
        warnings.append("frame rate unknown; assumed 30 fps")
    orb, aruco = cv2.ORB_create(nfeatures=ORB_FEATURES), _aruco_detector()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    samples, pairs, n_read, next_t = [], [], 0, 0.0
    try:
        while cap.grab():
            t = n_read / fps
            n_read += 1
            if t < next_t - 1e-6:
                continue
            next_t = (math.floor(t / SAMPLE_DT + 1e-6) + 1) * SAMPLE_DT
            ok, frame = cap.retrieve()
            if not ok or frame is None or frame.size == 0:
                continue
            s = _features(frame, orb, aruco)
            s["t"], s["shape"] = t, frame.shape[:2]
            if samples:
                pairs.append(_register(samples[-1], s, bf))
                samples[-1].update(gray=None, kp=None, des=None)
            samples.append(s)
    except cv2.error as e:
        warnings.append(f"decode stopped at {n_read / fps:.2f} s ({str(e).strip().splitlines()[-1][:80]})")
    finally:
        cap.release()
    return samples, pairs, fps, n_read, warnings, None if samples else f"no decodable frames: {video_path.name}"


def _verdict(samples: list[dict], pairs: list[tuple[float, float]], fps: float, n_read: int,
             warnings: list[str]) -> dict:
    h, w = samples[0]["shape"]
    duration = n_read / fps
    t = np.array([s["t"] for s in samples])
    shift = np.array([0.0] + [p[0] for p in pairs])
    residual = np.array([0.0] + [p[1] for p in pairs])
    blur = np.array([s["blur"] for s in samples])
    bright = np.array([s["brightness"] for s in samples])
    reasons = []

    static = float(np.mean(shift[1:] < STATIC_PX)) if pairs else None
    if static is None:
        reasons.append(f"too short to assess camera motion ({len(samples)} sample)")
    elif static < MIN_STATIC_FRACTION:
        reasons.append(f"camera moves (static {static * 100:.0f} %)")
    trim = _trim(t, residual, duration)
    if trim:
        warnings.append(f"duration {duration:.1f} s > {MAX_DURATION_S:.0f} s; trim to [{trim[0]}, {trim[1]}] s")
    b_range = float(bright.max() - bright.min())
    if b_range > MAX_BRIGHTNESS_RANGE:
        reasons.append(f"exposure drift (range {b_range:.0f})")
    blur_low = float(np.mean(blur < BLUR_VAR))
    if blur_low > MAX_BLUR_FRACTION:
        reasons.append(f"motion blur in {blur_low * 100:.0f} % of frames")
    if min(w, h) < MIN_SIDE:
        warnings.append(f"resolution below 720p ({w}x{h})")

    headline = f"static {(static or 0.0) * 100:.0f} %, {duration:.1f} s, {w}x{h}"
    # A handheld clip can still be used: the camera is then tracked frame by frame (moving-camera mode).
    soft = [r for r in reasons if r.startswith(("camera moves", "exposure drift"))]
    moving = bool(soft) and len(soft) == len(reasons)
    return {"accepted": not reasons, "moving_camera_ok": moving, "reasons": reasons, "warnings": warnings,
            "headline": headline,
            "duration_s": round(duration, 2), "fps": round(fps, 3), "width": int(w), "height": int(h),
            "n_samples": len(samples), "static_fraction": None if static is None else round(static, 3),
            "brightness_range": round(b_range, 1), "blur_low_fraction": round(blur_low, 3),
            "aruco_detect_fraction": round(float(np.mean([s["aruco"] > 0 for s in samples])), 3),
            "keyframes_t": _keyframes(t, residual), "trim": trim,
            "curve": {"t": [round(float(x), 2) for x in t],
                      "shift_px": [round(float(x), 2) if math.isfinite(x) else None for x in shift],
                      "residual": [round(float(x), 4) for x in residual],
                      "blur": [round(float(x), 1) for x in blur],
                      "brightness": [round(float(x), 1) for x in bright]}}


def probe(video_path: str | Path, out_dir: str | Path | None = None) -> dict:
    """QA verdict + motion curves for one video. Never raises on bad input; out_dir gets probe.json + motion.png."""
    samples, pairs, fps, n_read, warnings, error = _sample(Path(video_path))
    if error:
        res = _rejected(error, warnings)
    else:
        res = _verdict(samples, pairs, fps, n_read, warnings)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "probe.json").write_text(json.dumps(res, indent=1))
        c = res["curve"]
        _plot(out / "motion.png", np.array(c["t"], float), np.array(c["residual"], float),
              np.array([s is None or s >= STATIC_PX for s in c["shift_px"]]), res["keyframes_t"])
    return res


def _open(video_path: str | Path) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    return cap, _fps(cap) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def _frame_index(t_s: float, fps: float, n_frames: int) -> int:
    idx = max(0, round(t_s * fps))
    return min(idx, n_frames - 1) if n_frames > 0 else idx


def _seek_read(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def extract_frames(video_path: str | Path, times_s: list[float], out_dir: str | Path,
                   prefix: str = "frame") -> list[Path]:
    """Write the frame nearest each time as <prefix>_<frameidx:05d>.jpg after clearing existing <prefix>_*.jpg."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob(f"{prefix}_*.jpg"):
        p.unlink()
    cap, fps, n = _open(video_path)
    paths = []
    try:
        for idx in dict.fromkeys(_frame_index(t, fps, n) for t in times_s):
            frame = _seek_read(cap, idx)
            if frame is not None:
                p = out / f"{prefix}_{idx:05d}.jpg"
                cv2.imwrite(str(p), frame)
                paths.append(p)
    finally:
        cap.release()
    return paths


def read_frame(video_path: str | Path, t_s: float) -> np.ndarray:
    """BGR frame nearest t_s."""
    cap, fps, n = _open(video_path)
    try:
        frame = _seek_read(cap, _frame_index(t_s, fps, n))
    finally:
        cap.release()
    if frame is None:
        raise ValueError(f"cannot read frame at {t_s:.2f} s from {video_path}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="QA-gate a robot video before building its twin")
    ap.add_argument("video")
    ap.add_argument("--out")
    a = ap.parse_args()
    print(json.dumps(probe(a.video, a.out), indent=1))


if __name__ == "__main__":
    main()

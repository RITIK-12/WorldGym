"""Locked referee: the only reader of Penn Action ground truth (13 joints, visibility, boxes). Eval only.

Its numbers are logged next to each generation for the demo; they never reach the critic and never decide keep or
revert, so the loop cannot fit them. Held-out (test split) clips are scored once, aggregates only.

Metrics, all on visible GT joints:
- perception_pck   PCK@0.2 of the perceived human (SAM 3D Body 2D keypoints) against GT, threshold 0.2 x the longer
                   side of the GT box (the usual Penn Action protocol); `_lr` variants also accept a left/right swap.
- robot_limb_cos   mean cosine between each limb's image direction on the G1 (projected through the clip's camera)
                   and on the GT skeleton. Limb directions are proportion-free, so a robot with other limb lengths can
                   still score 1.
- robot_pck        PCK@0.2 of the projected G1 joints (lengths differ from the human's, so this is stricter).
- contact_err_fr   |frame of peak wrist speed| on the G1 vs on GT: for a swing, the bat meets the ball there.
- plant_err_fr     |frame the stride ankle stops| (after its peak speed, below 20 % of it) on the G1 vs GT.

Self-contained on purpose: reads .mat files and arrays, imports nothing from the harness except the keypoint maps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from humanoid.skeleton import MHR, PENN_LIMBS, PENN_LR_SWAP, mhr_to_penn

LABELS = Path(__file__).resolve().parents[1] / "data" / "Penn_Action" / "labels"
ALPHA = 0.2


def gt(clip: str) -> dict[str, np.ndarray]:
    m = sio.loadmat(LABELS / f"{clip}.mat")
    xy = np.stack([m["x"], m["y"]], -1).astype(float)          # (T, 13, 2)
    vis = m["visibility"].astype(bool)
    bbox = m["bbox"].astype(float)                            # (T, 4) x1 y1 x2 y2
    return {"xy": xy, "vis": vis, "bbox": bbox}


def _pck(pred: np.ndarray, g: dict) -> tuple[float, float]:
    T = min(len(pred), len(g["xy"]))
    pred, xy, vis, bbox = pred[:T], g["xy"][:T], g["vis"][:T], g["bbox"][:T]
    thr = ALPHA * np.maximum(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1])
    ok = np.linalg.norm(pred - xy, axis=-1) <= thr[:, None]
    ok_sw = np.linalg.norm(pred[:, PENN_LR_SWAP] - xy, axis=-1) <= thr[:, None]
    best = np.where(ok.sum(1, keepdims=True) >= ok_sw.sum(1, keepdims=True), ok, ok_sw)
    return float(ok[vis].mean()), float(best[vis].mean())


def _limb_cos(pred: np.ndarray, g: dict) -> float:
    T = min(len(pred), len(g["xy"]))
    cos = []
    for a, b in PENN_LIMBS:
        v = g["vis"][:T, a] & g["vis"][:T, b]
        u1 = pred[:T, b] - pred[:T, a]
        u2 = g["xy"][:T, b] - g["xy"][:T, a]
        c = (u1 * u2).sum(-1) / (np.linalg.norm(u1, axis=-1) * np.linalg.norm(u2, axis=-1) + 1e-9)
        cos.append(c[v])
    cos = np.concatenate(cos)
    return float(cos.mean()) if len(cos) else float("nan")


def _speed(xy: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=-1) / scale[1:]]


def events(xy: np.ndarray, scale: np.ndarray) -> dict[str, int]:
    """Swing events from a (T, 13, 2) Penn-order skeleton: contact = peak mean wrist speed; plant = stride ankle stop."""
    wrist = _speed((xy[:, 5] + xy[:, 6]) / 2, scale)
    contact = int(wrist.argmax())
    ankles = [_speed(xy[:, j], scale) for j in (11, 12)]
    stride = ankles[int(np.argmax([a[:contact + 1].max() for a in ankles]))]
    peak = int(stride[:contact + 1].argmax())
    after = np.flatnonzero(stride[peak:] < 0.2 * stride[peak])
    plant = peak + int(after[0]) if len(after) else peak
    return {"contact": contact, "plant": plant}


def score(clip: str, human_kp2d: np.ndarray | None = None, robot_penn2d: np.ndarray | None = None) -> dict:
    """human_kp2d: (T, 70, 2) MHR keypoints in pixels; robot_penn2d: (T, 13, 2) projected G1 joints in Penn order,
    sampled at the clip's own frames."""
    g = gt(clip)
    scale = np.maximum(g["bbox"][:, 2] - g["bbox"][:, 0], g["bbox"][:, 3] - g["bbox"][:, 1])
    ev_gt = events(g["xy"], scale)
    out = {"frames": int(len(g["xy"])), "gt_events": ev_gt}
    if human_kp2d is not None:
        p = mhr_to_penn(human_kp2d)
        out["perception_pck"], out["perception_pck_lr"] = _pck(p, g)
    if robot_penn2d is not None:
        T = min(len(robot_penn2d), len(g["xy"]))
        out["robot_pck"], out["robot_pck_lr"] = _pck(robot_penn2d, g)
        out["robot_limb_cos"] = _limb_cos(robot_penn2d, g)
        ev = events(robot_penn2d[:T], scale[:T])
        out["contact_err_fr"] = abs(ev["contact"] - ev_gt["contact"])
        out["plant_err_fr"] = abs(ev["plant"] - ev_gt["plant"])
    return out


def aggregate(rows: list[dict]) -> dict:
    keys = ["perception_pck", "perception_pck_lr", "robot_pck", "robot_pck_lr", "robot_limb_cos", "contact_err_fr",
            "plant_err_fr"]
    return {k: round(float(np.nanmean([r[k] for r in rows if k in r])), 4) for k in keys if any(k in r for r in rows)} | {
        "clips": len(rows)}


assert MHR["l_wrist"] == 62 and MHR["r_wrist"] == 41   # the MHR-70 layout this referee was written against

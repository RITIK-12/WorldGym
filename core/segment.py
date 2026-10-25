"""Object and robot masks from detector boxes.

Each box seeds GrabCut (box interior = probable foreground, outside = background), so masks work on neutral-coloured
props where colour keys fail. Boxes are Gemini-style `[ymin, xmin, ymax, xmax]` normalised to 0..1000.
Pixel coordinates follow OpenCV: integer values at pixel centres, v down.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_BOX_PX = 6      # below this GrabCut has too few samples: the box itself is the mask
CROP_MARGIN = 0.5   # GrabCut runs on the box grown by this fraction of its size (local background model)


def norm_box_to_px(box, w: int, h: int) -> tuple[int, int, int, int]:
    """`[ymin, xmin, ymax, xmax]` (0..1000) -> `(x0, y0, x1, y1)` pixels, x1/y1 exclusive, clamped, >= 1 px."""
    ymin, xmin, ymax, xmax = (float(v) for v in box)
    x0, x1 = sorted((xmin * w / 1000, xmax * w / 1000))
    y0, y1 = sorted((ymin * h / 1000, ymax * h / 1000))
    return _clamp_box((x0, y0, x1, y1), w, h)


def _clamp_box(box_xyxy, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    x0, x1 = int(np.clip(np.floor(min(x0, x1) + 1e-6), 0, w - 1)), int(np.clip(np.ceil(max(x0, x1) - 1e-6), 1, w))
    y0, y1 = int(np.clip(np.floor(min(y0, y1) + 1e-6), 0, h - 1)), int(np.clip(np.ceil(max(y0, y1) - 1e-6), 1, h))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _as_bgr8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img)


def _grabcut(img_bgr: np.ndarray, region: np.ndarray, iters: int, rect_mode: bool) -> np.ndarray:
    """GrabCut on a crop around `region` (probable foreground; everything outside it is background)."""
    h, w = region.shape
    vs, us = np.nonzero(region)
    x0, y0, x1, y1 = int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1
    if min(x1 - x0, y1 - y0) < MIN_BOX_PX:
        return region
    mx, my = max(8, int(CROP_MARGIN * (x1 - x0))), max(8, int(CROP_MARGIN * (y1 - y0)))
    cx0, cy0, cx1, cy1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    crop = _as_bgr8(img_bgr[cy0:cy1, cx0:cx1])
    gc = np.where(region[cy0:cy1, cx0:cx1], cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)  # GrabCut's k-means init uses OpenCV's global RNG: seed it so masks are reproducible
    try:
        cv2.grabCut(crop, gc, (x0 - cx0, y0 - cy0, x1 - x0, y1 - y0), bgd, fgd, iters,
                    cv2.GC_INIT_WITH_RECT if rect_mode else cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return region
    fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    if not fg.any():
        return region
    out = np.zeros((h, w), bool)
    out[cy0:cy1, cx0:cx1] = fg
    return out


def grabcut_mask(img_bgr: np.ndarray, box_xyxy, iters: int = 5) -> np.ndarray:
    """Boolean foreground mask (full image size) for the object inside `box_xyxy`; never raises on bad boxes
    (boxes under MIN_BOX_PX, or GrabCut failures, return the box itself)."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = _clamp_box(box_xyxy, w, h)
    rect = np.zeros((h, w), bool)
    rect[y0:y1, x0:x1] = True
    return _grabcut(img_bgr, rect, iters, rect_mode=True)


def masks_from_boxes(img_bgr: np.ndarray, boxes: dict[str, list[float]],
                     multi: dict[str, list[list[float]]] | None = None, iters: int = 5) -> dict[str, np.ndarray]:
    """One mask per id. All boxes of an id (`boxes[id]` plus `multi[id]`, e.g. pieces of a thin object) form one
    probable-foreground region segmented by a single GrabCut, so object pixels in a sibling box never seed the
    background model."""
    h, w = img_bgr.shape[:2]
    multi = multi or {}
    out = {}
    for oid in dict.fromkeys([*boxes, *multi]):
        bs = ([boxes[oid]] if oid in boxes else []) + list(multi.get(oid, []))
        if len(bs) == 1:
            out[oid] = grabcut_mask(img_bgr, norm_box_to_px(bs[0], w, h), iters)
            continue
        region = np.zeros((h, w), bool)
        for b in bs:
            x0, y0, x1, y1 = norm_box_to_px(b, w, h)
            region[y0:y1, x0:x1] = True
        out[oid] = _grabcut(img_bgr, region, iters, rect_mode=False) if bs else region
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    vs, us = np.nonzero(mask)
    return (float(us.mean()), float(vs.mean())) if us.size else None

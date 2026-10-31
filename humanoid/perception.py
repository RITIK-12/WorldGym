"""Perception tools (local on the Mac, no GT): SAM 3 finds the athlete, SAM 3D Body lifts them to 3D, frame by frame.

1. osam SAM 3 (text prompt, onnxruntime CPU) on the first frame -> the athlete's box (e.g. "baseball batter").
   It runs again only when the tracked box jumps, and then the detection closest to the prediction wins.
2. SAM 3D Body ONNX (gok11/sam-3d-body-onnx, parity with Meta's model: 0.4 px / 0.6 mm) per frame: box -> MHR-70
   keypoints in image pixels and in camera space (m, before cam_t), camera translation and focal length.
3. The next frame's box is this frame's body keypoint extent (the crop rule adds the 1.25 margin).

Results are cached in .worldgym/perception/<clip>.npz. Nothing here reads Penn Action labels.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from humanoid import penn

REPO = Path(__file__).resolve().parents[1]
ONNX = REPO / "assets" / "models" / "sam3d_body" / "sam-3d-body-vith-body.onnx"
CACHE = REPO / ".worldgym" / "perception"
BODY_KP = list(range(21)) + [41, 62]     # body, feet and both wrists: the box tracker's extent
CROP = 512

os.environ.setdefault("OSAM_ONNX_PROVIDERS", "CPUExecutionProvider")   # CoreML's compile cache does not fit the disk


def detect(image: np.ndarray, text: str) -> list[tuple[np.ndarray, float]]:
    """SAM 3 text-prompted detections: [(box x1 y1 x2 y2, score)], best first. SAM 3 is released afterwards: with
    SAM 3D Body also resident, a 16 GB Mac swaps hard."""
    import osam
    req = osam.types.GenerateRequest(model="sam3", image=image, prompt=osam.types.Prompt(texts=[text]))
    out = []
    for a in osam.apis.generate(request=req).annotations:
        b = a.bounding_box
        out.append((np.array([b.xmin, b.ymin, b.xmax, b.ymax], float), float(a.score or 0.0)))
    release_sam3()
    return sorted(out, key=lambda t: -t[1])


def release_sam3() -> None:
    import gc
    import osam.apis
    osam.apis.running_model = None
    gc.collect()


def detect_striker(clip: str, text: str = "baseball batter") -> tuple[np.ndarray, float]:
    """The striker's box on frame 1, cached in .worldgym/perception/<clip>.det.json."""
    p = CACHE / f"{clip}.det.json"
    if p.exists():
        d = json.loads(p.read_text())
        return np.array(d["box"]), d["score"]
    first = penn.frame(clip, 0)
    dets = detect(first, text) or detect(first, "person")
    if not dets:
        raise RuntimeError(f"{clip}: SAM 3 found no {text!r} and no person in frame 1")
    box, score = dets[0]
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"box": box.tolist(), "score": score, "text": text, "n_detections": len(dets)}))
    return box, score


def crop_params(box: np.ndarray, w: int, h: int) -> dict:
    """SAM 3D Body crop: centre of the box, size x1.25, widened to aspect 0.75, made square."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = (box[2] - box[0]) * 1.25, (box[3] - box[1]) * 1.25
    if bw < 0.75 * bh:
        bw = 0.75 * bh
    else:
        bh = bw / 0.75
    s = max(bw, bh)
    k = CROP / s
    f = float(np.hypot(w, h))
    return {"center": np.array([cx, cy], np.float32), "scale": np.array([s, s], np.float32),
            "affine": np.array([[k, 0, CROP / 2 - k * cx], [0, k, CROP / 2 - k * cy]], np.float32),
            "size": np.array([w, h], np.float32),
            "camera": np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)}


class Body3D:
    """SAM 3D Body on onnxruntime: one person per call."""

    def __init__(self, providers: tuple[str, ...] = ("CPUExecutionProvider",)):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(ONNX), so, providers=list(providers))
        self.inputs = {i.name: i.shape for i in self.sess.get_inputs()}
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def _feed(self, name: str, x: np.ndarray) -> np.ndarray:
        want = self.inputs[name]
        return x[None] if len(want) == x.ndim + 1 else x

    def __call__(self, image: np.ndarray, box: np.ndarray) -> dict[str, np.ndarray]:
        h, w = image.shape[:2]
        c = crop_params(box, w, h)
        crop = cv2.warpAffine(image, c["affine"], (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        feeds = {"image": crop.transpose(2, 0, 1).astype(np.float32) / 255.0, "bbox_center": c["center"],
                 "bbox_scale": c["scale"], "affine": c["affine"], "image_size": c["size"], "camera": c["camera"]}
        outs = self.sess.run(None, {k: self._feed(k, v) for k, v in feeds.items()})
        r = {k: np.asarray(v)[0] if np.asarray(v).ndim and np.asarray(v).shape[0] == 1 else np.asarray(v)
             for k, v in zip(self.outputs, outs)}
        r["principal"] = c["camera"][:2, 2].copy()
        return r


def _box_from_kp(kp2d: np.ndarray, w: int, h: int) -> np.ndarray:
    p = kp2d[BODY_KP]
    x1, y1 = p.min(0)
    x2, y2 = p.max(0)
    pad = 0.05 * max(x2 - x1, y2 - y1)
    return np.clip(np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad]), 0, [w - 1, h - 1, w - 1, h - 1])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def perceive(clip: str, text: str = "baseball batter", body: Body3D | None = None, force: bool = False,
             log=print) -> dict[str, np.ndarray]:
    out_path = CACHE / f"{clip}.npz"
    if out_path.exists() and not force:
        return dict(np.load(out_path))
    box, det_score = detect_striker(clip, text)
    body = body or Body3D()
    paths = penn.frame_paths(clip)
    first = penn.frame(clip, 0)
    h, w = first.shape[:2]
    keys = ["keypoints_2d", "keypoints_3d", "cam_t", "focal_length", "global_rot"]
    rec = {k: [] for k in keys}
    boxes, redetect = [], []
    prev = None   # last frame's keypoint box: a jump between keypoint boxes means the tracker lost the striker
    t0 = time.time()
    for i in range(len(paths)):
        img = first if i == 0 else penn.frame(clip, i)
        r = body(img, box)
        nxt = _box_from_kp(r["keypoints_2d"], w, h)
        jumped = prev is not None and _iou(nxt, prev) < 0.3
        if jumped:   # re-detect and take the detection closest to the last good keypoint box
            cands = detect(img, text) or detect(img, "person")
            if cands:
                box = max(cands, key=lambda c: _iou(c[0], prev))[0]
                r = body(img, box)
                nxt = _box_from_kp(r["keypoints_2d"], w, h)
        prev = nxt
        for k in keys:
            rec[k].append(np.asarray(r[k], np.float32).reshape(np.asarray(r[k]).shape))
        boxes.append(box.copy())
        redetect.append(jumped)
        box = nxt
        if i % 20 == 0:
            log(f"  {clip} frame {i + 1}/{len(paths)}  {(time.time() - t0) / (i + 1):.2f} s/frame")
    out = {k: np.stack(v) for k, v in rec.items()}
    out.update(boxes=np.stack(boxes), redetect=np.array(redetect), principal=np.array([w / 2, h / 2], np.float32),
               image_size=np.array([w, h]), det_score=np.array(det_score))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return out


def mirror(per: dict) -> dict:
    """The clip as if flipped horizontally: camera x negated, image u reflected, left and right swapped. A left-handed
    striker becomes a right-handed one (standard mirror augmentation; mirrored references are not scored against GT)."""
    from humanoid.skeleton import MHR_LR_SWAP
    out = dict(per)
    kp3 = per["keypoints_3d"][:, MHR_LR_SWAP].copy()
    kp3[..., 0] *= -1
    kp2 = per["keypoints_2d"][:, MHR_LR_SWAP].copy()
    kp2[..., 0] = float(per["image_size"][0]) - kp2[..., 0]
    cam = per["cam_t"].copy()
    cam[:, 0] *= -1
    out.update(keypoints_3d=kp3, keypoints_2d=kp2, cam_t=cam)
    return out

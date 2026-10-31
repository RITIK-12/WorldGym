"""Penn Action for the pipeline: frames and curation metadata only (class, coarse view, train/test split, size).

Ground truth (joints, visibility, bounding boxes) is read by `humanoid.referee` and nowhere else, so nothing the
harness or the critic sees can leak it. Picking clips by class label is dataset curation, not supervision.
The index of all 2,326 clips is cached in .worldgym/penn_index.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "Penn_Action"
INDEX = REPO / ".worldgym" / "penn_index.json"
FPS = 30.0   # Penn Action records no frame rate; its broadcast and web clips run at ~30 fps


def index() -> dict[str, dict]:
    if INDEX.exists():
        return json.loads(INDEX.read_text())
    out = {}
    for p in sorted((ROOT / "labels").glob("*.mat")):
        m = sio.loadmat(p, variable_names=["action", "pose", "train", "nframes", "dimensions"])
        h, w, _ = (int(v) for v in m["dimensions"][0])
        out[p.stem] = {"action": str(m["action"][0]), "view": str(m["pose"][0]), "nframes": int(m["nframes"][0][0]),
                       "split": "train" if int(m["train"][0][0]) == 1 else "test", "size": [w, h]}
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(out))
    return out


def clips(action: str = "baseball_swing", split: str | None = None) -> list[str]:
    return [c for c, v in index().items() if v["action"] == action and (split is None or v["split"] == split)]


def frame_paths(clip: str) -> list[Path]:
    return sorted((ROOT / "frames" / clip).glob("*.jpg"))


def frame(clip: str, i: int) -> np.ndarray:
    """RGB uint8 frame i (0-based) of a clip."""
    return cv2.cvtColor(cv2.imread(str(frame_paths(clip)[i])), cv2.COLOR_BGR2RGB)

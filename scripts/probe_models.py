"""Which models accept image input on this API key? One tiny synchronous call per model (+ one background check).

    uv run python scripts/probe_models.py [image.jpg]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.argon import Argon  # noqa: E402

CANDIDATES = ["gemini-3.8-flash", "gemini-robotics-er-2-preview", "gemini-3.5-flash-lite", "gemini-3.7-flash",
              "gemini-4-argon"]


class Box(BaseModel):
    label: str
    box_2d: list[float]


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "demo/synthetic_panda_sort.mp4"
    out = Path(".worldgym/probe")
    out.mkdir(parents=True, exist_ok=True)
    img = out / "probe.jpg"
    if src.endswith(".mp4"):
        from core.video_probe import read_frame
        frame = read_frame(src, 0.5)
    else:
        frame = cv2.imread(src)
    cv2.imwrite(str(img), cv2.resize(frame, (640, 360)))
    ar = Argon(out / "bundles")
    ar.use_cache = False
    rows = []
    for model, bg in [(m, False) for m in CANDIDATES] + [("gemini-3.8-flash", True)]:
        t0 = time.time()
        try:
            obj, resp = ar.ask_json(f"probe_{model}_{'bg' if bg else 'sync'}", "Precise spatial labeller.",
                                    "Give the box of the red cube as [ymin, xmin, ymax, xmax] normalised 0..1000.",
                                    Box, [img], model=model, thinking="low", max_output_tokens=1024, background=bg)
            rows.append({"model": model, "background": bg, "ok": True, "s": round(time.time() - t0, 1),
                         "box": obj.box_2d})
        except Exception as e:  # noqa: BLE001
            rows.append({"model": model, "background": bg, "ok": False, "s": round(time.time() - t0, 1),
                         "error": str(e)[:160]})
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()

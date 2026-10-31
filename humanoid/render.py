"""What the critic looks at: a contact sheet of key frames. Each row: the source frame with the perceived human
(cyan) and the projected G1 skeleton (orange), then the G1 rendered in MuJoCo from the side and from the front.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from humanoid import penn
from humanoid import skeleton as sk
from humanoid.retarget import IK_POINTS, Reference, project, ref_model

H = 240
LIMBS = sk.PENN_LIMBS + [(0, 1), (0, 2)]


def _skeleton(img: np.ndarray, pts: np.ndarray, colour) -> None:
    for a, b in LIMBS:
        cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), colour, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(int(v) for v in p), 3, colour, -1, cv2.LINE_AA)


def robot_penn2d(ref: Reference) -> np.ndarray:
    idx = [IK_POINTS.index(n) for n in sk.G1_PENN]
    return project(ref, ref.reached[:, idx])


def key_frames(ref: Reference, n: int = 4) -> list[int]:
    return sorted({int(round(x)) for x in np.linspace(0, len(ref.qpos) - 1, n)})


def contact_sheet(ref: Reference, per: dict, out: Path, frames: list[int] | None = None) -> Path:
    frames = frames or key_frames(ref)
    m = ref_model(ref)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, H)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 2.6, -10
    rob2d = robot_penn2d(ref)
    hum2d = sk.mhr_to_penn(per["keypoints_2d"])
    rows = []
    for t in frames:
        img = penn.frame(ref.clip, t).copy()
        _skeleton(img, hum2d[t], (0, 220, 255))
        _skeleton(img, rob2d[t], (255, 140, 0))
        img = cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))
        cv2.putText(img, f"frame {t + 1}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        d.qpos[:] = ref.qpos[t]
        mujoco.mj_forward(m, d)
        views = []
        for az in (90, 180):   # side (from +y), front (from +x looking back at the robot)
            cam.lookat[:] = d.qpos[:3] + np.array([0, 0, -0.2])
            cam.azimuth = az
            r.update_scene(d, cam)
            views.append(r.render())
        rows.append(np.hstack([img] + views))
    width = max(x.shape[1] for x in rows)
    sheet = np.vstack([np.pad(x, ((0, 0), (0, width - x.shape[1]), (0, 0))) for x in rows])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    r.close()
    return out

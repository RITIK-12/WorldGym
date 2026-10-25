"""Video QA gate: tripod clip accepted; handheld, exposure-drift and non-video inputs rejected; exact frame extraction.

Clips are rendered from the panda_sort scene (640x480, 30 fps, arm swinging on joint1) and written with libx264.
"""
import json
import math
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import pytest

from core.mjcf_compiler import compile_scene, load_task
from core.video_probe import extract_frames, probe, read_frame

REPO = Path(__file__).resolve().parents[1]
FPS, N = 30, 180


def _write(path: Path, frames) -> Path:
    w = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    for f in frames:
        w.append_data(f)
    w.close()
    return path


def _shake(frames, seed=0):
    """Hand-held camera: random-walk translation + rotation per frame, plus a viewpoint jump half way."""
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    walk, out = np.zeros(3), []
    for i, f in enumerate(frames):
        walk += rng.normal(0.0, [2.5, 2.5, 0.25])
        dx, dy, deg = walk + (np.array([40.0, -25.0, 4.0]) if i >= len(frames) // 2 else 0.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.05)
        M[:, 2] += (dx, dy)
        out.append(cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return out


@pytest.fixture(scope="module")
def frames():
    _, m = compile_scene(load_task(REPO / "tasks/panda_sort.json"))
    d = mujoco.MjData(m)
    q1 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint1")]
    q4 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint4")]
    r = mujoco.Renderer(m, 480, 640)
    out = []
    for i in range(N):
        d.qpos[q1] = 0.6 * math.sin(2 * math.pi * i / 90)
        d.qpos[q4] = -2.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="front")
        out.append(r.render().copy())
    r.close()
    return out


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    return tmp_path_factory.mktemp("clips")


@pytest.fixture(scope="module")
def tripod(frames, clips):
    return _write(clips / "tripod.mp4", frames)


def test_tripod_accepted(tripod, tmp_path):
    res = probe(tripod, tmp_path)
    assert res["accepted"], res["reasons"]
    assert res["static_fraction"] >= 0.8
    assert len(res["keyframes_t"]) >= 2
    assert np.all(np.diff(res["keyframes_t"]) >= 0.5)
    assert (res["width"], res["height"]) == (640, 480)
    assert any("720p" in w for w in res["warnings"]) and res["trim"] is None
    assert abs(res["duration_s"] - N / FPS) < 0.05
    assert json.loads((tmp_path / "probe.json").read_text()) == res
    assert cv2.imread(str(tmp_path / "motion.png")) is not None


def test_handheld_rejected(frames, clips):
    res = probe(_write(clips / "handheld.mp4", _shake(frames)))
    assert not res["accepted"]
    assert any("camera moves" in r for r in res["reasons"]), res["reasons"]


def test_exposure_drift_rejected(frames, clips):
    ramp = [np.clip(f.astype(np.int16) + round(60 * i / (N - 1)), 0, 255).astype(np.uint8)
            for i, f in enumerate(frames)]
    res = probe(_write(clips / "drift.mp4", ramp))
    assert not res["accepted"]
    assert any("exposure" in r for r in res["reasons"]), res["reasons"]
    assert not any("camera moves" in r for r in res["reasons"])


def test_non_video_rejected(tmp_path):
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("this is not a video\n" * 50)
    for path in (bogus, tmp_path / "missing.mp4"):
        res = probe(path)
        assert res["accepted"] is False and res["reasons"]
        json.dumps(res)


def test_extract_frames(tripod, tmp_path):
    (tmp_path / "frame_99999.jpg").write_bytes(b"stale")
    (tmp_path / "other_00001.jpg").write_bytes(b"keep")
    paths = extract_frames(tripod, [0.0, 1.0, 2.5, 2.5], tmp_path)
    assert [p.name for p in paths] == ["frame_00000.jpg", "frame_00030.jpg", "frame_00075.jpg"]
    assert sorted(p.name for p in tmp_path.glob("frame_*.jpg")) == [p.name for p in paths]
    assert (tmp_path / "other_00001.jpg").exists()

    cap = cv2.VideoCapture(str(tripod))
    seq = [cap.read()[1] for _ in range(76)]
    cap.release()
    assert np.array_equal(read_frame(tripod, 2.5), seq[75])
    jpg = cv2.imread(str(paths[2])).astype(float)
    assert np.abs(jpg - seq[75]).mean() < np.abs(jpg - seq[72]).mean()

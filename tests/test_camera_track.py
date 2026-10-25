"""Moving-camera tracking on a synthetic handheld orbit over a textured table: pose error vs ground truth."""
import math

import cv2
import imageio
import mujoco
import numpy as np
import pytest

from core.camera_solve import params_to_pose, set_camera
from core.camera_track import camera_at, track_camera
from core.mjcf_compiler import compile_scene, load_task
from core.schemas import CameraSpec


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    task = load_task("tasks/panda_sort.json")
    task = task.model_copy(update={"cameras": [CameraSpec(name="video", pos=[1.5, 0, 0.8], look_at=[0.4, 0, 0.1])]})
    xml, _ = compile_scene(task)
    xml = xml.replace('<geom name="table"', '<geom name="table" material="speckle"', 1)
    tex = tmp_path_factory.mktemp("tex") / "mat.png"
    rng = np.random.default_rng(0)
    blobs = cv2.resize(rng.integers(0, 255, (48, 48), dtype=np.uint8), (768, 768), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(tex), cv2.GaussianBlur(blobs, (5, 5), 0))
    xml = xml.replace("</asset>", f'<texture name="mat_tex" type="2d" file="{tex}"/>'
                      '<material name="speckle" texture="mat_tex" texrepeat="2 2"/></asset>', 1)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 544, 960)
    path = tmp_path_factory.mktemp("track") / "orbit.mp4"
    writer = imageio.get_writer(str(path), fps=10, codec="libx264", quality=9, macro_block_size=8)
    truth = []
    for k in range(20):
        a = k / 19
        p = dict(azimuth_deg=20 + 18 * a, elevation_deg=40 - 6 * a + 2 * math.sin(6 * a), distance_m=1.3 - 0.1 * a,
                 lookat_x=0.4, lookat_y=0.02 * math.sin(4 * a), lookat_z=0.05, roll_deg=2 * math.sin(5 * a),
                 fovy_deg=50)
        pos, xy, fovy = params_to_pose(p)
        set_camera(m, d, "video", pos, xy, fovy)
        r.update_scene(d, camera="video")
        writer.append_data(r.render())
        truth.append((k / 10, np.array(pos), np.array(xy)))
    writer.close()
    r.close()
    return path, truth


def test_track_matches_ground_truth(clip):
    path, truth = clip
    t0, p0, xy0 = truth[0]
    track = track_camera(path, 0.0, p0, xy0, 50.0, [t for t, _, _ in truth], table_rect=(0.35, 0.0, 1.0, 1.0))
    assert all(c["ok"] for c in track)
    pos_err = [np.linalg.norm(np.array(camera_at(track, t)["pos"]) - p) for t, p, _ in truth]
    rot_err = [math.degrees(math.acos(np.clip(np.dot(np.array(camera_at(track, t)["xyaxes"][:3]), xy[:3]), -1, 1)))
               for t, _, xy in truth]
    assert np.median(pos_err) < 0.02 and max(pos_err) < 0.05, pos_err
    assert np.median(rot_err) < 1.0 and max(rot_err) < 3.0, rot_err

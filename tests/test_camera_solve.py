"""Camera solve acceptance (synthetic): recover a known "video" camera from the robot silhouette alone, pinhole
projection agrees with the renderer, box-seeded GrabCut masks match the segmentation render of compact objects."""
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest

from core import camera_solve as cs
from core.mjcf_compiler import load_task
from core.rollout import build_sim
from core.segment import grabcut_mask, mask_centroid, mask_iou, masks_from_boxes, norm_box_to_px

REPO = Path(__file__).resolve().parents[1]
W, H = 640, 480
GT = dict(azimuth_deg=20.0, elevation_deg=35.0, distance_m=1.5, lookat_x=0.35, lookat_y=0.0, lookat_z=0.15,
          roll_deg=2.0, fovy_deg=50.0)
INIT = dict(azimuth_deg=40.0, elevation_deg=25.0, distance_m=1.8)    # verbal estimate; fovy left at the default 45
GT_POS, GT_XY, GT_FOVY = cs.params_to_pose(GT)


@pytest.fixture(scope="module")
def sim():
    task = cs.apply_solution(load_task(REPO / "tasks/panda_sort.json"), dict(pos=GT_POS, xyaxes=GT_XY, fovy=GT_FOVY),
                             W, H)
    s = build_sim(task)
    cid = mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_CAMERA, "video")
    s.compiled_cam = (s.m.cam_pos[cid].copy(), s.m.cam_quat[cid].copy())
    s.real_robot = cs.render_body_mask(s.m, s.d, "video", s.ix.robot_bodies, W, H)   # camera as compiled from MJCF
    return s


def _gt_camera(s):
    cs.set_camera(s.m, s.d, "video", GT_POS, GT_XY, GT_FOVY)


def test_apply_solution_and_quat(sim):
    task = sim.ix.task
    assert [c.name for c in task.cameras] == ["front", "side", "top", "ground", "video"]
    video = task.cameras[-1]
    assert (video.w, video.h, video.fovy) == (W, H, GT_FOVY) and np.allclose(video.xyaxes, GT_XY)
    pos, quat = sim.compiled_cam
    q = cs.xyaxes_to_quat(GT_XY)
    assert np.allclose(pos, GT_POS) and min(np.abs(q - quat).max(), np.abs(q + quat).max()) < 1e-6


def test_solve_camera_recovers_pose(sim):
    real = cv2.erode(sim.real_robot.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # 1 px boundary loss
    sol = cs.solve_camera(sim.m, sim.d, real, sim.ix.robot_bodies, init=INIT)
    pos_err = float(np.linalg.norm(np.array(sol["pos"]) - GT_POS))
    print(f"\ncamera solve: iou={sol['iou']:.4f} pos_err={pos_err * 100:.2f}cm "
          f"fovy_err={sol['fovy'] - GT_FOVY:+.2f}deg evals={sol['evals']} {sol['seconds']:.1f}s")
    assert sol["iou"] >= 0.85 and sol["accepted"]
    assert pos_err < 0.03
    solved = cs.apply_solution(sim.ix.task, sol, W, H).cameras[-1]
    assert solved.name == "video" and np.allclose(solved.pos, sol["pos"]) and solved.fovy == pytest.approx(sol["fovy"])


def test_project_points_matches_render(sim):
    _gt_camera(sim)
    m, d = sim.m, sim.d
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp_arm")
    rgba, size = m.site_rgba[sid].copy(), m.site_size[sid].copy()
    m.site_rgba[sid], m.site_size[sid, 0] = [1, 0, 1, 1], 0.012
    opt = mujoco.MjvOption()
    opt.sitegroup[5] = 1
    r = cs.seg_renderer(m, W, H)
    try:
        r.update_scene(d, camera="video", scene_option=opt)
        seg = r.render()
        robot = cs.render_body_mask(m, d, "video", sim.ix.robot_bodies, W, H, r)
        fingers = cs.render_body_mask(m, d, "video", {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                                                      for b in ("left_finger", "right_finger")}, W, H, r)
    finally:
        cs.close_renderer(r)
        m.site_rgba[sid], m.site_size[sid] = rgba, size
    marker = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_SITE)) & (seg[..., 0] == sid)
    assert marker.sum() > 20
    uv = cs.project_points(m, d, "video", d.site_xpos[sid][None], W, H)[0]
    assert np.linalg.norm(uv - mask_centroid(marker)) < 1.0
    vs, us = np.nonzero(fingers)
    hull = cv2.convexHull(np.stack([us, vs], 1).astype(np.float32))
    assert cv2.pointPolygonTest(hull, (float(uv[0]), float(uv[1])), False) >= 0     # TCP sits between the fingers
    hand = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")]
    uh = cs.project_points(m, d, "video", hand[None], W, H)[0]
    assert robot[int(round(uh[1])), int(round(uh[0]))]
    o, ray = cs.pixel_ray(m, d, "video", uv, W, H)
    p = d.site_xpos[sid]
    assert np.linalg.norm(o + ((p - o) @ ray) * ray - p) < 1e-9


def test_grabcut_masks_from_boxes(sim):
    _gt_camera(sim)
    m, d, ix = sim.m, sim.d, sim.ix
    r = mujoco.Renderer(m, H, W)
    try:
        r.update_scene(d, camera="video")
        bgr = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        truth = {oid: cs.render_body_mask(m, d, "video", {ix.obj_body[oid]}, W, H) for oid in ("cube", "bin")}
    finally:
        cs.close_renderer(r)
    boxes = {}
    for oid, gt in truth.items():
        vs, us = np.nonzero(gt)
        boxes[oid] = [vs.min() / H * 1000, us.min() / W * 1000, (vs.max() + 1) / H * 1000, (us.max() + 1) / W * 1000]
        assert norm_box_to_px(boxes[oid], W, H) == (us.min(), vs.min(), us.max() + 1, vs.max() + 1)
    masks = masks_from_boxes(bgr, boxes)
    for oid, gt in truth.items():
        assert mask_iou(masks[oid], gt) >= 0.7, oid

    y0, x0, y1, x1 = boxes["bin"]
    halves = [[y0, x0, y1, (x0 + x1) / 2], [y0, (x0 + x1) / 2, y1, x1]]
    union = masks_from_boxes(bgr, {}, multi={"bin": halves})["bin"]
    assert mask_iou(union, truth["bin"]) >= 0.7
    tiny = grabcut_mask(bgr, (100, 100, 103, 102))
    assert tiny.sum() == 6 and tiny[100:102, 100:103].all()
    assert grabcut_mask(bgr, (50, 50, 50, 50)).sum() == 1
    assert grabcut_mask(bgr, (-20, -20, W + 20, H + 20)).shape == (H, W)
    assert mask_centroid(np.zeros((4, 4), bool)) is None

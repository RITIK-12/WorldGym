"""Harness sanity: the referee scores a perfect skeleton perfectly, the ladder calls a standing G1 feasible, θ patches
validate, and the structured-output schema is in the shape the API accepts."""
import mujoco
import numpy as np
import pytest

from humanoid import referee
from humanoid.config import ConfigPatch, HarnessConfig, apply_patch
from humanoid.llm import _strict_schema
from humanoid.retarget import G1_SCENE, Reference, g1_model, g1_points, sole_points, stand_pose, IK_POINTS

needs_g1 = pytest.mark.skipif(not G1_SCENE.exists(), reason="Menagerie unitree_g1 not fetched")


def test_referee_perfect_and_events():
    T = 30
    t = np.arange(T)
    xy = np.tile(np.random.default_rng(0).uniform(60, 240, (13, 2)), (T, 1, 1))   # distinct joints, non-zero limbs
    xy[:, 5, 0] = xy[:, 6, 0] = 100 + 50 * np.tanh((t - 20) / 2.0)     # wrists sweep fastest at frame 20
    xy[:, 11, 0] = 100 + 30 * np.clip((t - 5) / 6.0, 0, 1)                # left ankle strides frames 5-11
    g = {"xy": xy, "vis": np.ones((T, 13), bool), "bbox": np.tile([50.0, 50.0, 250.0, 250.0], (T, 1))}
    assert referee._pck(xy, g) == (1.0, 1.0)
    assert referee._limb_cos(xy + 1e-3, g) > 0.99
    ev = referee.events(xy, np.full(T, 200.0))
    assert ev["contact"] == 20 and 9 <= ev["plant"] <= 12


def test_patch_validates():
    cfg = apply_patch(HarnessConfig(), ConfigPatch(hypothesis="slower", time_scale=2.0))
    assert cfg.time_scale == 2.0
    with pytest.raises(Exception):
        apply_patch(HarnessConfig(), ConfigPatch(hypothesis="bad", time_scale=99.0))


def test_strict_schema_shape():
    s = _strict_schema(ConfigPatch)
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    assert "minimum" not in str(s)


@needs_g1
def test_standing_g1_is_feasible():
    from humanoid import ladder
    m = g1_model()
    d = mujoco.MjData(m)
    q = stand_pose(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    q[2] -= sole_points(m, d)[..., 2].min()          # soles on the floor
    T = 20
    qpos = np.tile(q, (T, 1))
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    pts = np.tile(g1_points(m, d), (T, 1, 1))
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    human = np.zeros((T, 70, 3))
    ref = Reference(clip="synthetic", qpos=qpos, fps=30.0, targets=pts, reached=pts, human=human, human_raw=human,
                    at_limit=np.zeros((T, len(hinge)), bool), plane_rms_m=0.0, world_from_cam=np.eye(3),
                    cam_origin=np.zeros(3), scale=1.0, pin_offset=np.zeros((T, 3)), root_shift=np.zeros(T),
                    focal=np.full(T, 500.0), principal=np.zeros(2))
    cfg = HarnessConfig()
    met1, _ = ladder.l1(ref, cfg)
    met2, _ = ladder.l2(ref, cfg)
    assert met1["skate_cm_s"] < 0.5 and met1["speed_over_frac"] == 0 and met1["self_collision_frac"] == 0
    assert met2["ghost_force_mg"] < 0.02 and met2["torque_over_frac"] == 0
    assert len(IK_POINTS) == pts.shape[1]

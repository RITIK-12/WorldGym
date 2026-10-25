"""Loop 1 acceptance: displaced twin vs a synthetic frame -> alignment >= 90 %, xy error < 1 cm; patch validation;
composite objects compile to their declared extents."""
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from core.camera_solve import close_renderer, seg_renderer
from core.diagnostics import sim_object_masks
from core.fidelity import ScenePatch, ScenePatchError, _apply_edits, run_loop1
from core.locked_evaluator import body_aabb
from core.mjcf_compiler import compile_scene
from core.perturb import apply_overrides
from core.rollout import build_sim
from core.schemas import ObjectSpec, Part, TaskSpec
from core.video_probe import read_frame

REPO = Path(__file__).resolve().parents[1]
GT = json.loads((REPO / "demo/synthetic_panda_sort.gt.json").read_text())
TASK = TaskSpec.model_validate(GT["task"])


def test_loop1_recovers_displaced_twin(tmp_path):
    real = read_frame(REPO / "demo/synthetic_panda_sort.mp4", 0.5)
    h, w = real.shape[:2]
    s0 = build_sim(TASK)
    r = seg_renderer(s0.m, w, h)
    masks = sim_object_masks(s0.ix, s0.d, "video", w, h, r)
    close_renderer(r)
    wrong = apply_overrides(TASK, {"objects": {"cube": {"xy": [0.45, -0.2]}, "cyl": {"xy": [0.55, -0.14]},
                                               "barrier": {"yaw": 0.3, "xy": [0.5, 0.05]},
                                               "bin": {"dims_m": [0.13, 0.13, 0.07], "xy": [0.5, 0.18]}}})
    res = run_loop1(wrong, real, masks, tmp_path)
    assert res["history"][0]["alignment"] < 0.6 and res["alignment"] >= 0.9
    gt = {o.id: o for o in TASK.objects}
    for o in res["task"].objects:
        assert np.linalg.norm(np.subtract(o.xy, gt[o.id].xy)) < 0.01, o.id
    assert (tmp_path / "final" / "matrix.jpg").exists()


def test_scene_patch_validation():
    ok = _apply_edits(TASK, ScenePatch(edits=[{"object_id": "cube", "set": {"dims_m": [0.05, 0.05, 0.05]}}],
                                       remove=["barrier"]))
    assert ok.obj("cube").dims_m == [0.05, 0.05, 0.05] and "barrier" not in [o.id for o in ok.objects]
    with pytest.raises(ScenePatchError, match="cannot be edited"):
        _apply_edits(TASK, ScenePatch(edits=[{"object_id": "cube", "set": {"id": "x"}}]))
    with pytest.raises(ScenePatchError, match="unknown object"):
        _apply_edits(TASK, ScenePatch(edits=[{"object_id": "ghost", "set": {"yaw": 1.0}}]))


def test_composite_matches_dims():
    bar = ObjectSpec(id="grab_bar", shape="composite", dims_m=[0.06, 0.42, 0.10], xy=[0.45, 0.0], is_fixed=True,
                     parts=[Part(shape="box", dims_m=[0.06, 0.06, 0.05], pos=[0, -0.18, -0.025]),
                            Part(shape="box", dims_m=[0.06, 0.06, 0.05], pos=[0, 0.18, -0.025]),
                            Part(shape="capsule", dims_m=[0.02, 0.02, 0.38], pos=[0, 0, 0.04],
                                 euler=[math.pi / 2, 0, 0])])
    _, m = compile_scene(TaskSpec(name="c", robot="franka_panda", objects=[bar]))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = body_aabb(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "grab_bar"))
    assert np.allclose(hi - lo, [0.06, 0.42, 0.10], atol=1e-4) and abs(lo[2]) < 1e-3

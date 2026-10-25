"""Loop 2: with a cached critic response (replay backend), Draft N fails -> Draft N+1 succeeds."""
import json
from pathlib import Path

import pytest

from core.argon import Argon
from core.bench import dev_suite
from core.critic import PatchError, apply_patch, critique, param_diff
from core.mjcf_compiler import load_adapter, load_task
from core.perturb import apply_overrides
from core.rollout import run_episode
from core.schemas import PolicyConfig

REPO = Path(__file__).resolve().parents[1]
TASK = load_task(REPO / "tasks/panda_sort.json")
DRAFT1 = PolicyConfig.model_validate_json((REPO / "policies/panda_sort_draft1.json").read_text())

REL = lambda terms, off: {"terms": terms, "offset": off}  # noqa: E731
CACHED = {
    "from_iteration": 1,
    "critique": [{"timestamp_sec": 3.1, "video_frame_idx": 93,
                  "observation": "held cube bottom passes below the barrier top during transit_arc"}],
    "root_causes": ["absolute carry heights ignore the barrier height and the object hanging below the TCP"],
    "patch": [{"object_id": "*", "set": {
        "post_grasp_lift_z_m": REL(["max_scene_top_z", "held_below_tcp"], 0.05),
        "transit_apex_z_m": REL(["max_scene_top_z", "held_below_tcp"], 0.06),
        "pre_place_hover_z_m": REL(["target_rim_z", "held_below_tcp"], 0.04),
        "retract_z_m": REL(["max_scene_top_z"], 0.07),
        "reach_clearance_z_m": REL(["max_scene_top_z"], 0.07),
        "transit_via_mode": "none", "release_open_width_m": 0.06}}],
}


def test_patch_fixes_failing_draft(tmp_path):
    task = apply_overrides(TASK, dev_suite(TASK)[0].overrides)
    nominal = run_episode(task, DRAFT1).summary()
    assert not nominal["success"]
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "critic_iter1.json").write_text(json.dumps({"text": json.dumps(CACHED)}))
    argon = Argon(tmp_path / "bundles", backend="replay", replay_dir=replay)
    patch, draft2 = critique(task, DRAFT1, load_adapter("franka_panda"), nominal, [], argon, 1, [], tmp_path)
    assert draft2.name == "draft2" and patch.critique[0].get("video_frame_idx") == 93
    assert run_episode(task, draft2).success
    assert any(r["field"] == "transit_apex_z_m" for r in param_diff(DRAFT1, draft2))
    assert "transit_arc" in (tmp_path / "bundles" / "critic_iter1" / "prompt.md").read_text()


def test_apply_patch_rejects_unknown_and_locked_fields():
    with pytest.raises(PatchError, match="unknown or locked"):
        apply_patch(DRAFT1, {"patch": [{"object_id": "cube", "set": {"fly": True}}]})
    with pytest.raises(PatchError, match="unknown or locked"):
        apply_patch(DRAFT1, {"patch": [{"object_id": "cube", "set": {"target_id": "floor"}}]})
    with pytest.raises(PatchError):
        apply_patch(DRAFT1, {"patch": [{"object_id": "cube", "set": {"place_yaw_mode": "diagonal"}}]})
    with pytest.raises(PatchError, match="no primitive"):
        apply_patch(DRAFT1, {"patch": [{"object_id": "ghost", "set": {"speed_mps": 0.2}}]})
    p = apply_patch(DRAFT1, {"patch": [{"object_id": "cyl", "set": {"speed_mps": 0.2}}]})
    assert p.primitives[1].speed_mps == 0.2 and p.primitives[0].speed_mps == DRAFT1.primitives[0].speed_mps

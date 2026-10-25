"""Bimanual Trossen WidowX AI: compiles with both TCP sites, both arms pass the self-test, rest pose helper,
and the default (Panda) table and cameras are unchanged."""
import mujoco
import numpy as np

from core.mjcf_compiler import compile_scene, default_cameras, load_adapter
from core.robot_onboard import self_test, set_rest_pose
from core.rollout import build_sim
from core.schemas import TaskSpec


def test_biarm_self_test_and_rest_pose():
    a = load_adapter("trossen_wxai_biarm")
    r = self_test(a)
    assert r["ok"], {k: v["ok"] for k, v in r["checks"].items()}
    assert set(r["checks"]["ik_grid"]["per_arm"]) == {"left", "right"}
    sim = build_sim(TaskSpec(name="b", robot=a.robot_id, objects=[]), a)
    for name in ("tcp_left", "tcp_right"):
        assert mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    set_rest_pose(sim)
    q = sim.d.qpos[sim.ctrls["left"].qadr]
    assert np.allclose(q, a.arms[0].rest_qpos)


def test_default_table_and_cameras_unchanged():
    _, m = compile_scene(TaskSpec(name="p", robot="franka_panda", objects=[]))
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    assert np.allclose(m.geom_size[g], [0.55, 0.55, 0.02]) and np.allclose(m.geom_pos[g], [0.35, 0, -0.02])
    assert [c.pos for c in default_cameras()][:2] == [[1.45, 0.0, 0.75], [0.45, -1.15, 0.45]]


def test_biarm_hold_and_wipe_succeeds():
    """Left arm holds the bar, right arm (synced) wipes it with the sponge and puts the sponge back."""
    from pathlib import Path

    from core.mjcf_compiler import load_task
    from core.rollout import run_episode
    from core.schemas import PolicyConfig

    repo = Path(__file__).resolve().parents[1]
    task = load_task(repo / "tasks/biarm_wipe.json")
    pol = PolicyConfig.model_validate_json((repo / "policies/biarm_wipe_v1.json").read_text())
    ep = run_episode(task, pol)
    assert ep.success, ep.evaluation
    cov = next(p for p in ep.evaluation["predicates"] if p["kind"] == "wiped_coverage")
    assert cov["coverage"] >= 0.9 and ep.trace["metrics"]["unwanted_collision_count"] == 0
    tl = ep.trace["phase_timeline_per_arm"]
    t_hold = next(p["t0"] for p in tl["left"] if p["phase"] == "hold")
    t_reach = next(p["t0"] for p in tl["right"] if p["phase"] == "reach")
    assert t_reach >= t_hold   # sync_after: the right arm starts once the left arm holds

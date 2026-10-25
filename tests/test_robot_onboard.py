"""Robot onboarding: the stored Panda adapter passes all five self-test checks."""
from core.mjcf_compiler import load_adapter
from core.robot_onboard import mjcf_summary, self_test


def test_panda_self_test():
    r = self_test(load_adapter("franka_panda"))
    assert r["ok"], r
    assert set(r["checks"]) == {"compiles", "home_collision_free", "ik_grid", "gripper_hold_lift", "velocity_limits"}


def test_summary_lists_actuators():
    s = mjcf_summary("franka_emika_panda")
    assert "actuator actuator8 -> split" in s and "body hand" in s

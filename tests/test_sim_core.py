"""Sim core acceptance: hand-tuned pick-place succeeds, TCP lag p99 < 1 cm, deterministic run to run."""
from pathlib import Path

import numpy as np
import pytest

from core.mjcf_compiler import load_task
from core.rollout import run_episode
from core.schemas import PolicyConfig

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def episodes():
    task = load_task(REPO / "tasks/panda_sort.json")
    pol = PolicyConfig.model_validate_json((REPO / "policies/panda_sort_v3.json").read_text())
    return run_episode(task, pol), run_episode(task, pol)


def test_reference_policy_succeeds(episodes):
    ep = episodes[0]
    assert ep.success, ep.evaluation
    assert ep.trace["metrics"]["unwanted_collision_count"] == 0, ep.trace["events"]


def test_tcp_lag(episodes):
    assert episodes[0].trace["metrics"]["tcp_lag_p99_m"] < 0.01


def test_deterministic(episodes):
    a, b = episodes
    assert np.array_equal(a.arrays["qpos"], b.arrays["qpos"])

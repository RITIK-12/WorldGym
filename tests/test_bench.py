"""Benchmark protocol: settle-gated scenarios, held-out lock (sha + chmod 444), scored once per draft."""
import json
import stat
from pathlib import Path

import pytest

import core.bench as bench
from core.mjcf_compiler import load_task
from core.perturb import apply_overrides, default_family, layout_ok, rect_distance, sample_scenarios
from core.schemas import ObjectSpec, PolicyConfig

REPO = Path(__file__).resolve().parents[1]
TASK = load_task(REPO / "tasks/panda_sort.json")


def test_rect_distance():
    a = ObjectSpec(id="a", shape="box", dims_m=[0.1, 0.1, 0.1], xy=[0, 0])
    b = ObjectSpec(id="b", shape="box", dims_m=[0.1, 0.1, 0.1], xy=[0.15, 0])
    assert rect_distance(a, b) == pytest.approx(0.05, abs=1e-6)
    c = ObjectSpec(id="c", shape="box", dims_m=[0.1, 0.1, 0.1], xy=[0.12, 0], yaw=0.785)
    assert rect_distance(a, c) < 0.02


def test_scenarios_valid_and_deterministic():
    s1 = sample_scenarios(TASK, default_family(TASK), 3, 7, "d")
    s2 = sample_scenarios(TASK, default_family(TASK), 3, 7, "d")
    assert [s.overrides for s in s1] == [s.overrides for s in s2]
    for s in s1:
        assert layout_ok(apply_overrides(TASK, s.overrides))[0]


def test_held_out_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "BENCH", tmp_path)
    monkeypatch.setattr(bench, "N_HELD", 3)
    monkeypatch.setattr(bench, "N_HARDER", 1)
    info = bench.lock_held_out(TASK)
    path = Path(info["path"])
    assert info["created"] and not (path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    scen, digest = bench.verify_held_out(TASK)
    assert len(scen) == 3 and digest == info["sha256"] and {s.family for s in scen} == {"dev", "harder"}
    assert not bench.lock_held_out(TASK)["created"]  # never silently re-drawn

    pol = PolicyConfig.model_validate_json((REPO / "policies/panda_sort_v3.json").read_text())
    r1 = bench.score_held_out(TASK, pol, "v3")
    r2 = bench.score_held_out(TASK, pol, "v3-again")
    assert not r1["cached"] and r2["cached"] and r2["draft"] == "v3"
    ledger = json.loads((tmp_path / TASK.name / "held_out_ledger.json").read_text())
    assert len(ledger) == 1 and "clusters" not in ledger[0] and "scenario_id" not in json.dumps(ledger)

    path.chmod(0o644)
    path.write_text(path.read_text().replace('"held', '"HELD', 1))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        bench.verify_held_out(TASK)

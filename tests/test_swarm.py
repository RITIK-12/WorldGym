"""Loop 3 local worker: finds failing scenarios inside the dev ranges; inline sources exclude benchmarks/."""
from pathlib import Path

from core.mjcf_compiler import load_task
from core.perturb import apply_overrides, default_family, layout_ok
from core.schemas import PolicyConfig
from core.swarm import _inline_sources, adversarial_suite, biased_family, hypotheses_from_clusters, local_search

REPO = Path(__file__).resolve().parents[1]
TASK = load_task(REPO / "tasks/panda_sort.json")
DRAFT1 = PolicyConfig.model_validate_json((REPO / "policies/panda_sort_draft1.json").read_text())


def test_biased_family_inside_dev_ranges():
    fam = default_family(TASK)
    b = biased_family(TASK, fam, "barrier_clearance")
    lo, hi = fam.objects["barrier"].height_m
    blo, bhi = b.objects["barrier"].height_m
    assert lo <= blo < bhi <= hi and blo > lo


def test_local_search_breaks_draft1():
    rep = local_search(TASK, DRAFT1, default_family(TASK), "barrier_clearance", budget=9)
    assert rep.failing_scenarios and rep.rollouts <= 12
    assert any("BARRIER_COLLISION" in f.labels for f in rep.failing_scenarios)
    for s in adversarial_suite([rep]):
        assert layout_ok(apply_overrides(TASK, s.overrides))[0]


def test_hypothesis_order_and_sources():
    assert hypotheses_from_clusters({"RIM_STRIKE_WHILE_HELD": 2})[0] == "container_fit"
    targets = [s["target"] for s in _inline_sources({"POLICY.json": "{}"})]
    assert "worldgym/core/bench.py" in targets and "worldgym/POLICY.json" in targets
    assert not any("benchmarks/" in t or t.endswith(".mp4") for t in targets)

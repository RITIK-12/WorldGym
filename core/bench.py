"""Benchmark: dev suite (fixed seeds, full traces) + LOCKED held-out suite (aggregates only).

Held-out protocol:
- drawn from an os.urandom master seed BEFORE any hardening, including a harder family beyond the dev ranges;
- written to benchmarks/<task>/held_out.json + .sha256, chmod 444, sha-verified on every load;
- scored once per draft: a ledger keyed by the policy hash returns the recorded aggregate instead of re-running;
- only aggregates are persisted (no per-scenario traces or descriptions), and agent sandboxes never receive it.

CLI (used by swarm agents):  python -m core.bench POLICY.json --task tasks/x.json --suite scenarios.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from core.perturb import Family, Scenario, apply_overrides, default_family, sample_scenarios
from core.schemas import PolicyConfig, TaskSpec

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("WORLDGYM_BENCH", REPO / "benchmarks"))
DEV_SEED = 1234
N_DEV = 12
N_HELD = 20
N_HARDER = 6
WORKERS = min(8, os.cpu_count() or 4)

CLUSTER_OF_EVENT = {"ILLEGAL_OBSTACLE_COLLISION": "BARRIER_COLLISION", "CONTAINER_RIM_STRIKE": "RIM_STRIKE_WHILE_HELD",
                    "GRASP_MISSED": "GRASP_FAILURE", "OBJECT_LOST_FROM_GRASP": "GRASP_FAILURE",
                    "MOTION_BLOCKED": "MOTION_BLOCKED", "IK_TRACKING_ERROR": "IK_REACH_LIMIT",
                    "ROBOT_DISTURBED_OTHER_OBJECT": "KNOCKED_OTHER_OBJECT",
                    "OBJECT_OBJECT_COLLISION": "KNOCKED_OTHER_OBJECT"}


def policy_hash(policy: PolicyConfig) -> str:
    return hashlib.sha256(policy.model_dump_json().encode()).hexdigest()[:16]


def failure_labels(summary: dict) -> list[str]:
    """Rule-based failure clusters for one episode (Argon refines them)."""
    if summary["success"]:
        return []
    labels = {CLUSTER_OF_EVENT[e["type"]] for e in summary["trace"]["events"] if e["type"] in CLUSTER_OF_EVENT}
    for p in summary["evaluation"]["predicates"]:
        if p["kind"] == "inside_container" and not p["ok"]:
            labels.add("MISSED_CONTAINER" if not p.get("in_xy") or not p.get("bottom_ok") else "TIPPED_IN_CONTAINER")
        elif p["kind"] == "max_obstacle_displacement" and not p["ok"]:
            labels.add("BARRIER_COLLISION")
    if summary["trace"].get("timed_out"):
        labels.add("TIMEOUT")
    return sorted(labels) or ["UNLABELLED"]


def _run_one(args: tuple[str, str, str, str | None]) -> dict:
    task_json, policy_json, scen_json, video = args
    from core.rollout import run_episode

    task = TaskSpec.model_validate_json(task_json)
    sc = Scenario.model_validate_json(scen_json)
    ep = run_episode(apply_overrides(task, sc.overrides), PolicyConfig.model_validate_json(policy_json), video=video)
    s = ep.summary()
    return {"scenario_id": sc.scenario_id, "family": sc.family, "success": s["success"],
            "labels": failure_labels(s), "metrics": s["trace"]["metrics"], "events": s["trace"]["events"],
            "predicates": s["evaluation"]["predicates"], "resolved": s["resolved"], "wall_s": s["wall_s"]}


def run_suite(task: TaskSpec, policy: PolicyConfig, scenarios: list[Scenario], workers: int = WORKERS,
              video_dir: Path | None = None) -> list[dict]:
    jobs = [(task.model_dump_json(), policy.model_dump_json(), s.model_dump_json(),
             str(video_dir / f"{s.scenario_id}.mp4") if video_dir else None) for s in scenarios]
    if workers <= 1:
        return [_run_one(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_run_one, jobs))


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    ok = sum(r["success"] for r in results)
    clusters: dict[str, int] = {}
    for r in results:
        for lab in r["labels"]:
            clusters[lab] = clusters.get(lab, 0) + 1
    by_family: dict[str, list[int]] = {}
    for r in results:
        by_family.setdefault(r["family"], [0, 0])
        by_family[r["family"]][0] += r["success"]
        by_family[r["family"]][1] += 1
    return {"n": n, "successes": ok, "rate": round(ok / n, 3) if n else 0.0,
            "collisions": sum(r["metrics"]["unwanted_collision_count"] for r in results),
            "barrier_events": sum(sum(e["type"] == "ILLEGAL_OBSTACLE_COLLISION" for e in r["events"]) for r in results),
            "mean_episode_s": round(sum(r["metrics"]["episode_time_sec"] for r in results) / max(n, 1), 2),
            "clusters": dict(sorted(clusters.items(), key=lambda kv: -kv[1])),
            "by_family": {k: f"{v[0]}/{v[1]}" for k, v in by_family.items()}}


# ----------------------------------------------------------------------------------------------------------------------
# Dev suite
# ----------------------------------------------------------------------------------------------------------------------

def dev_suite(task: TaskSpec, fam: Family | None = None) -> list[Scenario]:
    return sample_scenarios(task, fam or default_family(task), N_DEV, DEV_SEED, "dev")


# ----------------------------------------------------------------------------------------------------------------------
# Locked held-out
# ----------------------------------------------------------------------------------------------------------------------

def _held_paths(task: TaskSpec) -> tuple[Path, Path, Path]:
    d = BENCH / task.name
    return d / "held_out.json", d / "held_out.sha256", d / "held_out_ledger.json"


def lock_held_out(task: TaskSpec, fam: Family | None = None, harder: Family | None = None,
                  force: bool = False) -> dict:
    path, sha_path, _ = _held_paths(task)
    if path.exists() and not force:
        return {"path": str(path), "sha256": verify_held_out(task)[1], "created": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    for p in (path, sha_path):
        if p.exists():
            p.chmod(stat.S_IWUSR | stat.S_IRUSR)
    master = int.from_bytes(os.urandom(4), "little")
    normal = sample_scenarios(task, fam or default_family(task), N_HELD - N_HARDER, master, "held")
    hard = sample_scenarios(task, harder or default_family(task, harder=True), N_HARDER, master + 1, "hard")
    payload = {"task": task.name, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
               "scenarios": [s.model_dump() for s in normal + hard]}
    raw = json.dumps(payload, indent=1).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    sha_path.write_text(digest + "\n")
    for p in (path, sha_path):
        p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 444
    return {"path": str(path), "sha256": digest, "created": True}


def verify_held_out(task: TaskSpec) -> tuple[list[Scenario], str]:
    path, sha_path, _ = _held_paths(task)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha_path.read_text().strip():
        raise RuntimeError(f"held-out suite hash mismatch: {digest} != recorded")
    if path.stat().st_mode & 0o222:
        raise RuntimeError("held-out suite is writable; it must stay chmod 444")
    return [Scenario.model_validate(s) for s in json.loads(raw)["scenarios"]], digest


def score_held_out(task: TaskSpec, policy: PolicyConfig, draft: str) -> dict:
    """Score a draft ONCE; repeated calls with the same policy return the recorded aggregate."""
    scenarios, digest = verify_held_out(task)
    _, _, ledger_path = _held_paths(task)
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    ph = policy_hash(policy)
    for row in ledger:
        if row["policy_hash"] == ph:
            return {**row, "cached": True}
    agg = aggregate(run_suite(task, policy, scenarios))
    agg.pop("clusters")  # aggregates only: no per-failure descriptions leave the evaluator
    row = {"draft": draft, "policy_hash": ph, "lock_sha256": digest[:12],
           "scored": datetime.now().astimezone().isoformat(timespec="seconds"), **agg}
    ledger.append(row)
    ledger_path.write_text(json.dumps(ledger, indent=1))
    return {**row, "cached": False}


def held_out_ledger(task: TaskSpec) -> list[dict]:
    p = _held_paths(task)[2]
    return json.loads(p.read_text()) if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a policy on a scenario suite (dev or agent-proposed)")
    ap.add_argument("policy")
    ap.add_argument("--task", required=True)
    ap.add_argument("--suite", help="JSON list of scenarios; default = the dev suite")
    ap.add_argument("--out", help="write per-scenario results here")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    task = TaskSpec.model_validate_json(Path(a.task).read_text())
    pol = PolicyConfig.model_validate_json(Path(a.policy).read_text())
    scen = ([Scenario.model_validate(s) for s in json.loads(Path(a.suite).read_text())] if a.suite
            else dev_suite(task))
    res = run_suite(task, pol, scen, a.workers)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"aggregate": aggregate(res),
                      "scenarios": [{"id": r["scenario_id"], "success": r["success"], "labels": r["labels"]}
                                    for r in res]}, indent=1))


if __name__ == "__main__":
    main()

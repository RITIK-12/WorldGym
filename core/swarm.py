"""Loop 3: adversarial curriculum swarm.

One worker per failure hypothesis searches the dev perturbation ranges for scenarios that break the current policy
and returns the least-extreme failing ones. Workers run either as Antigravity Managed Agents in Google-hosted
sandboxes (live) or locally (deterministic, same report contract). Argon then clusters the reports and hardens the
policy into a state-conditioned draft. Agent sandboxes receive the tracked source files only: never benchmarks/.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from core.argon import ARGON, Argon, check_budget, parse_json, record_spend
from core.bench import aggregate, run_suite
from core.critic import DSL_DOC, apply_patch, robot_facts
from core.perturb import Family, ObjRange, Scenario, apply_overrides, layout_ok, sample_overrides
from core.schemas import PolicyConfig, PolicyPatch, RobotAdapter, TaskSpec

REPO = Path(__file__).resolve().parents[1]
AGENT = os.environ.get("WORLDGYM_AGENT", "antigravity-preview-09-2026")
POLL_S = 10.0
BUDGET = 24
AGENT_MAX_TOKENS = int(os.environ.get("WORLDGYM_AGENT_MAX_TOKENS", 2_000_000))  # runaway guard per agent
ALLOWLIST = ["github.com", "*.github.com", "*.githubusercontent.com", "pypi.org", "files.pythonhosted.org",
             "astral.sh", "*.astral.sh"]

HYPOTHESES = {
    "barrier_clearance": "Taller or shifted obstacles versus the carry path (held object hanging below the TCP).",
    "container_fit": "Smaller or yawed containers and larger objects versus the place pose, opening and yaw.",
    "grasp_seating": "Object size, yaw, mass and friction versus grasp depth, force and finger opening.",
    "reach_limits": "Objects and containers near the workspace edges versus reach, yaw and joint limits.",
}
CLUSTER_TO_HYPOTHESIS = {"BARRIER_COLLISION": "barrier_clearance", "RIM_STRIKE_WHILE_HELD": "container_fit",
                         "MISSED_CONTAINER": "container_fit", "TIPPED_IN_CONTAINER": "container_fit",
                         "GRASP_FAILURE": "grasp_seating", "KNOCKED_OTHER_OBJECT": "grasp_seating",
                         "MOTION_BLOCKED": "reach_limits", "IK_REACH_LIMIT": "reach_limits",
                         "NOT_AT_REST": "container_fit", "MISSED_SUPPORT": "container_fit"}


class FailingScenario(BaseModel):
    scenario_id: str
    overrides: dict
    failed: bool = True
    labels: list[str] = []
    why: str = ""


class SwarmReport(BaseModel):
    hypothesis: str
    failing_scenarios: list[FailingScenario] = []
    passing_boundary: str = ""
    summary: str = ""
    rollouts: int = 0
    source: str = "local"


def hypotheses_from_clusters(clusters: dict[str, int]) -> list[str]:
    hs = [CLUSTER_TO_HYPOTHESIS[c] for c in clusters if c in CLUSTER_TO_HYPOTHESIS]
    out = list(dict.fromkeys(hs))
    for h in HYPOTHESES:  # always probe every hypothesis; observed ones first
        if h not in out:
            out.append(h)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Local worker (deterministic)
# ----------------------------------------------------------------------------------------------------------------------

def _top(r: list[float], frac: float = 0.4) -> list[float]:
    return [r[1] - (r[1] - r[0]) * frac, r[1]]


def _bottom(r: list[float], frac: float = 0.4) -> list[float]:
    return [r[0], r[0] + (r[1] - r[0]) * frac]


def biased_family(task: TaskSpec, fam: Family, hypothesis: str) -> Family:
    """A sub-range of the dev family that stresses one hypothesis (still inside the dev ranges)."""
    objs: dict[str, ObjRange] = {}
    for o in task.objects:
        r = fam.objects.get(o.id)
        if r is None:
            continue
        r = r.model_copy(deep=True)
        if hypothesis == "barrier_clearance" and o.is_obstacle and r.height_m:
            r.height_m = _top(r.height_m)
        elif hypothesis == "container_fit":
            if o.shape == "container":
                r.scale = _bottom(r.scale)
            elif not o.is_fixed and not o.is_obstacle:
                r.scale = _top(r.scale)
        elif hypothesis == "grasp_seating" and not o.is_fixed and not o.is_obstacle:
            r.scale = _top(r.scale)
            if r.friction:
                r.friction = _bottom(r.friction)
            r.mass_scale = _top(r.mass_scale)
        objs[o.id] = r
    return Family(name=f"adv_{hypothesis}", objects=objs)


def _edge_push(task: TaskSpec, ov: dict, rng: np.random.Generator) -> dict:
    """reach_limits: push one movable object toward the far / lateral edge of the reachable workspace."""
    movable = [o for o in task.objects if not o.is_fixed and not o.is_obstacle]
    if not movable:
        return ov
    o = movable[int(rng.integers(len(movable)))]
    f = ov["objects"].setdefault(o.id, {})
    x, y = f.get("xy", o.xy)
    f["xy"] = [round(x + rng.choice([-1, 1]) * rng.uniform(0.04, 0.08), 4), round(y + rng.uniform(-0.06, 0.06), 4)]
    return ov


def _scale_overrides(task: TaskSpec, ov: dict, alpha: float) -> dict:
    """Interpolate a scenario toward the nominal task (alpha = 0 nominal, 1 = as sampled)."""
    out = {"objects": {}}
    for oid, f in ov.get("objects", {}).items():
        o = task.obj(oid)
        g = {}
        for k, v in f.items():
            base = getattr(o, k)
            if isinstance(v, list):
                g[k] = [round(b + alpha * (x - b), 4) for b, x in zip(base, v)]
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                g[k] = round(base + alpha * (v - base), 4)
            else:
                g[k] = v
        out["objects"][oid] = g
    return out


def local_search(task: TaskSpec, policy: PolicyConfig, fam: Family, hypothesis: str, budget: int = BUDGET,
                 seed: int = 0) -> SwarmReport:
    rng = np.random.default_rng(abs(hash(hypothesis)) % 2 ** 31 + seed)
    bf = biased_family(task, fam, hypothesis)
    n_probe = budget * 2 // 3
    cands: list[Scenario] = []
    tries = 0
    while len(cands) < n_probe and tries < 300:
        tries += 1
        ov = sample_overrides(task, bf, np.random.default_rng(int(rng.integers(2 ** 31))))
        if hypothesis == "reach_limits":
            ov = _edge_push(task, ov, rng)
        if layout_ok(apply_overrides(task, ov))[0]:
            cands.append(Scenario(scenario_id=f"{hypothesis[:4]}{len(cands):02d}", family=bf.name, overrides=ov))
    res = run_suite(task, policy, cands)
    fails = [(s, r) for s, r in zip(cands, res) if not r["success"]]
    rollouts = len(cands)
    found: list[FailingScenario] = []
    for s, r in fails[: max(1, (budget - rollouts) // 2)]:
        best = (s, r)
        refine = []
        for alpha in (0.5, 0.75):
            ov = _scale_overrides(task, s.overrides, alpha)
            if layout_ok(apply_overrides(task, ov))[0]:
                refine.append(Scenario(scenario_id=f"{s.scenario_id}_a{int(alpha * 100)}", family=bf.name,
                                       overrides=ov))
        rr = run_suite(task, policy, refine) if refine else []
        rollouts += len(refine)
        for s2, r2 in zip(refine, rr):  # least extreme failing variant wins (alpha ascending)
            if not r2["success"]:
                best = (s2, r2)
                break
        bs, br = best
        why = "; ".join([f"{p['kind']}({p.get('object_id') or ''}) failed" for p in br["predicates"] if not p["ok"]]
                        + sorted({f"{e['type']}@{e['t']:.2f}s" for e in br["events"]
                                  if e["type"] not in ("OBJECT_RELEASED",)})[:4])
        found.append(FailingScenario(scenario_id=bs.scenario_id, overrides=bs.overrides, labels=br["labels"], why=why))
    for s, r in fails[len(found):]:
        found.append(FailingScenario(scenario_id=s.scenario_id, overrides=s.overrides, labels=r["labels"]))
    agg = aggregate(res)
    return SwarmReport(hypothesis=hypothesis, failing_scenarios=found, rollouts=rollouts,
                       passing_boundary=f"{agg['successes']}/{agg['n']} probes passed in {bf.name}",
                       summary=f"{len(fails)} of {len(cands)} adversarial probes failed; clusters {agg['clusters']}",
                       source="local")


# ----------------------------------------------------------------------------------------------------------------------
# Managed Agents (live)
# ----------------------------------------------------------------------------------------------------------------------

def _inline_sources(extra: dict[str, str]) -> list[dict]:
    files = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    keep = [f for f in files if not f.startswith(("benchmarks/", "demo/", "docs/")) and not f.endswith(".mp4")]
    src = [{"type": "inline", "target": f"worldgym/{f}", "encoding": "base64",
            "content": base64.b64encode((REPO / f).read_bytes()).decode()} for f in keep]
    for target, text in extra.items():
        src.append({"type": "inline", "target": f"worldgym/{target}", "encoding": "base64",
                    "content": base64.b64encode(text.encode()).decode()})
    return src


def agent_prompt(hypothesis: str, task_path: str, fam: Family) -> str:
    return f"""You are an adversarial test engineer in the WorldGym project (MuJoCo robot manipulation benchmark).
The repository is mounted at ./worldgym. Setup:
  cd worldgym && pip install uv && uv sync && scripts/setup_assets.sh
The policy under test is ./worldgym/POLICY.json; the task is ./worldgym/{task_path}.

Hypothesis to probe: {hypothesis} — {HYPOTHESES[hypothesis]}

Perturbation ranges you MUST stay inside (per object; xy_jitter is +- around the task pose; scale multiplies x/y
extents; height_m is the absolute z extent):
{fam.model_dump_json(indent=1)}

A scenario is {{"scenario_id": str, "family": "agent", "overrides": {{"objects": {{"<id>": {{"xy": [x, y], "yaw": r,
"dims_m": [x, y, z], "mass": m, "friction": f}}}}}}}} (any subset of fields). Write up to {BUDGET} scenarios to
scenarios.json (a JSON list), then run:
  uv run python -m core.bench POLICY.json --task {task_path} --suite scenarios.json --out results.json
Read results.json (success, labels, events per scenario), then refine around the failures: move each failing
scenario toward the nominal task until you find the least extreme variant that still fails. Total budget: {BUDGET}
rollouts. Write your final report to report.json AND print it as your final answer, as ONE JSON object:
{{"hypothesis": "{hypothesis}", "failing_scenarios": [{{"scenario_id", "overrides", "failed": true, "labels": [...],
"why": "facts from the events"}}], "passing_boundary": "where the policy stops failing", "summary": "...",
"rollouts": int, "source": "antigravity"}}"""


def launch_agents(task: TaskSpec, task_path: str, policy: PolicyConfig, fam: Family, hypotheses: list[str],
                  argon: Argon, log_dir: Path) -> list[dict]:
    client = argon.client
    extra = {"POLICY.json": policy.model_dump_json(indent=1), task_path: task.model_dump_json(indent=1)}
    env = {"type": "remote", "sources": _inline_sources(extra),
           "network": {"allowlist": [{"domain": d} for d in ALLOWLIST]}}
    launched = []
    for h in hypotheses:
        prompt = agent_prompt(h, task_path, fam)
        (log_dir / f"agent_{h}_prompt.md").write_text(prompt)
        check_budget("agent", f"swarm_{h}")
        it = client.interactions.create(agent=AGENT, agent_config={"type": "antigravity", "model": ARGON,
                                                                   "max_total_tokens": AGENT_MAX_TOKENS},
                                        input=prompt, environment=env, background=True)
        record_spend("agent", f"swarm_{h}", AGENT, None, True, it.id)
        launched.append({"hypothesis": h, "id": it.id, "status": it.status, "t0": time.time()})
    (log_dir / "agents.json").write_text(json.dumps(launched, indent=1))
    return launched


def _usage(it) -> dict:
    u = getattr(it, "usage", None)
    return u.model_dump() if u is not None and hasattr(u, "model_dump") else {}


def poll_agents(argon: Argon, launched: list[dict], log_dir: Path, timeout_s: float = 1800,
                on_update=None) -> list[SwarmReport]:
    client = argon.client
    pending = {a["id"]: a for a in launched}
    reports: dict[str, SwarmReport] = {}
    t0 = time.time()
    while pending and time.time() - t0 < timeout_s:
        for iid, a in list(pending.items()):
            it = client.interactions.get(iid)
            a["status"] = it.status
            if it.status in ("in_progress", "queued"):
                continue
            pending.pop(iid)
            record_spend("agent_usage", f"swarm_{a['hypothesis']}", AGENT, _usage(it), it.status == "completed", iid)
            text = it.output_text or ""
            try:
                rep = SwarmReport.model_validate(parse_json(text))
            except Exception:  # noqa: BLE001 - fall back to the report file in the sandbox
                try:
                    raw = client.environments.files.download(environment=it.environment_id,
                                                             path="worldgym/report.json")
                    rep = SwarmReport.model_validate(json.loads(raw))
                except Exception as e:  # noqa: BLE001
                    rep = SwarmReport(hypothesis=a["hypothesis"], summary=f"agent ended {it.status}: {e}"[:300],
                                      source="antigravity")
            rep.source = "antigravity"
            reports[a["hypothesis"]] = rep
            (log_dir / f"agent_{a['hypothesis']}_report.json").write_text(rep.model_dump_json(indent=1))
        if on_update:
            on_update(launched, reports)
        if pending:
            time.sleep(POLL_S)
    for a in pending.values():
        reports[a["hypothesis"]] = SwarmReport(hypothesis=a["hypothesis"], summary="timed out", source="antigravity")
    return [reports[a["hypothesis"]] for a in launched]


# ----------------------------------------------------------------------------------------------------------------------
# Hardening (Argon clusters reports -> state-conditioned policy)
# ----------------------------------------------------------------------------------------------------------------------

HARDEN_SYSTEM = """You harden a robot manipulation policy written in a closed parametric DSL. You receive adversarial
scenarios that break it (found by a swarm of test agents, one per failure hypothesis) and the dev-suite results.
Cluster the failures by mechanism, then return the smallest patch that makes the policy STATE-CONDITIONED so it
survives the whole perturbation family: express heights and clearances with {"terms": [...], "offset": x} features
instead of absolute numbers, keep openings narrow inside containers, align non-round objects with container walls.
Only DSL fields may change; never object_id/target_id/arm.

Output: ONE JSON object {"from_iteration": int, "critique": [{"timestamp_sec": float, "video_frame_idx": int,
"observation": str}], "root_causes": [str], "patch": [{"object_id": "<id or *>", "set": {...}}]}"""


def harden(task: TaskSpec, policy: PolicyConfig, adapter: RobotAdapter, reports: list[SwarmReport],
           dev_results: list[dict], argon: Argon, iteration: int, out_dir: Path) -> tuple[PolicyPatch, PolicyConfig]:
    rep_view = [{"hypothesis": r.hypothesis, "source": r.source, "summary": r.summary,
                 "passing_boundary": r.passing_boundary,
                 "failing": [f.model_dump() for f in r.failing_scenarios[:5]]} for r in reports]
    prompt = f"""{robot_facts(adapter)}

{DSL_DOC}

TASK:
{task.model_dump_json(indent=1)}

CURRENT POLICY (iteration {iteration}):
{policy.model_dump_json(indent=1)}

DEV SUITE: {sum(r['success'] for r in dev_results)}/{len(dev_results)}; failure clusters
{json.dumps(aggregate(dev_results)['clusters']) if dev_results else '{}'}

SWARM REPORTS:
{json.dumps(rep_view, indent=1)}

Return the hardening patch (from_iteration = {iteration})."""
    patch, _ = argon.ask_json(f"harden_iter{iteration}", HARDEN_SYSTEM, prompt, PolicyPatch, model=ARGON,
                              thinking="high")
    new = apply_patch(policy, patch, name=f"draft{iteration + 1}")
    (out_dir / f"harden_patch_iter{iteration}.json").write_text(patch.model_dump_json(indent=1))
    (out_dir / f"policy_draft{iteration + 1}.json").write_text(new.model_dump_json(indent=1))
    return patch, new


def adversarial_suite(reports: list[SwarmReport]) -> list[Scenario]:
    out = []
    for r in reports:
        for f in r.failing_scenarios:
            out.append(Scenario(scenario_id=f"{r.hypothesis[:4]}_{f.scenario_id}", family=f"adv_{r.hypothesis}",
                                overrides=f.overrides))
    return out

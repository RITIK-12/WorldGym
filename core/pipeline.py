"""Orchestrator (deterministic demo path): Python drives the stages in order and calls Argon only for judgements.

Twin half (video in):   qa -> video_architect -> robot_onboard -> camera_solve -> loop1_static -> motion_extract
Policy half (any task): replay -> benchmark -> loop2_critic -> loop3_swarm -> report

    uv run python -m core.pipeline --video demo/clip.mp4 --prompt "put both blocks in the bin"
    uv run python -m core.pipeline --task tasks/panda_sort.json --draft policies/panda_sort_draft1.json

Every stage writes into runs/<run_id>/ and updates manifest.json, which the Studio polls.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.argon import Argon, call_summary
from core.bench import aggregate, dev_suite, lock_held_out, run_suite, score_held_out
from core.critic import critique, event_keyframes, param_diff
from core.mjcf_compiler import load_adapter, load_task
from core.perturb import default_family
from core.rollout import run_episode
from core.runs import Refused, Run, new_run
from core.schemas import PolicyConfig, RobotAdapter, TaskSpec
from core.swarm import adversarial_suite, harden, hypotheses_from_clusters, launch_agents, local_search, poll_agents

TWIN_STAGES = ["qa", "video_architect", "robot_onboard", "camera_solve", "loop1_static", "motion_extract"]
POLICY_STAGES = ["replay", "benchmark", "loop2_critic", "loop3_swarm", "report"]


def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def _evaluate_draft(run: Run, task: TaskSpec, policy: PolicyConfig, draft: str, dev, extra=()) -> dict:
    """Nominal rollout with video, dev suite (+ adversarial suite), held-out scored once."""
    d = run.path(draft)
    d.mkdir(exist_ok=True)
    (d / "policy.json").write_text(policy.model_dump_json(indent=1))
    nominal = run_episode(task, policy, video=d / "nominal.mp4", out_dir=d / "nominal").summary()
    dev_res = run_suite(task, policy, dev)
    (d / "dev_results.json").write_text(json.dumps(dev_res, indent=1))
    out = {"draft": draft, "nominal_success": nominal["success"], "nominal": nominal,
           "dev": aggregate(dev_res), "dev_results": dev_res,
           "held_out": {k: v for k, v in score_held_out(task, policy, draft).items()}}
    if extra:
        adv = run_suite(task, policy, list(extra))
        out["adversarial"] = aggregate(adv)
    run.write_json(f"{draft}/summary.json", {k: v for k, v in out.items() if k not in ("nominal", "dev_results")})
    return out


def run_policy_loops(run: Run, task: TaskSpec, draft1: PolicyConfig, adapter: RobotAdapter, argon: Argon,
                     swarm: str = "local", replay_ref: dict | None = None) -> dict:
    ratchet: list[dict] = []

    with run.stage("replay") as s:
        d = run.path("draft1")
        d.mkdir(exist_ok=True)
        ep = run_episode(task, draft1, video=d / "nominal.mp4", out_dir=d / "nominal")
        nominal = ep.summary()
        s.data = {"success": ep.success, "metrics": nominal["trace"]["metrics"],
                  "dynamic": replay_ref or {}}
        s.headline = ("replay succeeds on the nominal twin" if ep.success else "replay fails on the nominal twin")
        if replay_ref and replay_ref.get("end_pose_err_mm") is not None:
            s.headline += f" · end error {replay_ref['end_pose_err_mm']:.0f} mm"
        s.artifacts += ["draft1/nominal.mp4", "draft1/nominal/result.json"]

    with run.stage("benchmark") as s:
        lock = lock_held_out(task, default_family(task), default_family(task, harder=True))
        dev = dev_suite(task)
        run.write_json("benchmark/dev_suite.json", [x.model_dump() for x in dev])
        run.write_json("benchmark/lock.json", {"sha256": lock["sha256"], "path": lock["path"], "n_dev": len(dev)})
        r1 = _evaluate_draft(run, task, draft1, "draft1", dev)
        ratchet.append({"draft": "draft1", "label": "Replay (Draft 1)", "dev": r1["dev"]["rate"],
                        "held_out": r1["held_out"]["rate"], "collisions": r1["held_out"]["collisions"],
                        "nominal": r1["nominal_success"]})
        s.headline = f"replay breaks: dev {r1['dev']['successes']}/{r1['dev']['n']} · held-out {_pct(r1['held_out']['rate'])}"
        s.data = {"lock_sha256": lock["sha256"], "dev": r1["dev"], "held_out": r1["held_out"]}
        s.artifacts += ["benchmark/dev_suite.json", "benchmark/lock.json", "draft1/summary.json"]

    policy, current = draft1, r1
    with run.stage("loop2_critic") as s:
        d = run.path("loop2")
        d.mkdir(exist_ok=True)
        video = run.path("draft1/nominal.mp4")
        kfs = event_keyframes(video, current["nominal"]["trace"]["events"], d / "keyframes")
        patch, draft2 = critique(task, policy, adapter, current["nominal"], current["dev_results"], argon, 1,
                                 [video, *kfs], d)
        r2 = _evaluate_draft(run, task, draft2, "draft2", dev)
        ratchet.append({"draft": "draft2", "label": "Argon critic (Draft 2)", "dev": r2["dev"]["rate"],
                        "held_out": r2["held_out"]["rate"], "collisions": r2["held_out"]["collisions"],
                        "nominal": r2["nominal_success"]})
        run.write_json("loop2/diff.json", param_diff(policy, draft2))
        s.data = {"critique": patch.critique, "root_causes": patch.root_causes, "diff": param_diff(policy, draft2),
                  "dev": r2["dev"], "held_out": r2["held_out"]}
        s.headline = f"dev {r2['dev']['successes']}/{r2['dev']['n']} · held-out {_pct(r2['held_out']['rate'])}"
        s.artifacts += [run.rel(p) for p in kfs] + ["loop2/patch_iter1.json", "loop2/diff.json", "draft2/summary.json"]
        policy, current = draft2, r2

    with run.stage("loop3_swarm") as s:
        d = run.path("loop3")
        d.mkdir(exist_ok=True)
        fam = default_family(task)
        hyps = hypotheses_from_clusters(current["dev"]["clusters"])
        t0 = time.time()
        if swarm == "antigravity":
            launched = launch_agents(task, f"tasks/{task.name}.json", policy, fam, hyps, argon, d)
            s.data["agents"] = launched
            run.save()
            reports = poll_agents(argon, launched, d)
        else:
            reports = []
            for h in hyps:
                reports.append(local_search(task, policy, fam, h))
                (d / f"agent_{h}_report.json").write_text(reports[-1].model_dump_json(indent=1))
                s.data["reports"] = [r.model_dump() for r in reports]
                run.save()
        adv = adversarial_suite(reports)
        run.write_json("loop3/adversarial_suite.json", [x.model_dump() for x in adv])
        patch, draft3 = harden(task, policy, adapter, reports, current["dev_results"], argon, 2, d)
        r3 = _evaluate_draft(run, task, draft3, "draft3", dev, extra=adv)
        ratchet.append({"draft": "draft3", "label": "Swarm-hardened (Draft 3)", "dev": r3["dev"]["rate"],
                        "held_out": r3["held_out"]["rate"], "collisions": r3["held_out"]["collisions"],
                        "nominal": r3["nominal_success"]})
        run.write_json("loop3/diff.json", param_diff(policy, draft3))
        s.data = {"reports": [r.model_dump() for r in reports], "swarm_mode": swarm, "swarm_s": round(time.time() - t0),
                  "critique": patch.critique, "root_causes": patch.root_causes, "diff": param_diff(policy, draft3),
                  "dev": r3["dev"], "held_out": r3["held_out"], "adversarial": r3.get("adversarial")}
        s.headline = (f"{sum(len(r.failing_scenarios) for r in reports)} adversarial scenarios · dev "
                      f"{r3['dev']['successes']}/{r3['dev']['n']} · held-out {_pct(r3['held_out']['rate'])}")
        s.artifacts += ["loop3/adversarial_suite.json", "loop3/diff.json", "draft3/summary.json"]
        policy, current = draft3, r3

    with run.stage("report") as s:
        report = {"task": task.name, "ratchet": ratchet, "lock_sha256": lock["sha256"],
                  "final_policy": policy.model_dump(), "model_calls": call_summary()}
        run.write_json("report.json", report)
        s.headline = " → ".join(_pct(r["held_out"]) for r in ratchet) + " held-out"
        s.data = report
        s.artifacts.append("report.json")
    return report


def run_from_task(task_path: str, draft_path: str, name: str | None = None, swarm: str = "local",
                  backend: str | None = None) -> Run:
    task = load_task(task_path)
    draft1 = PolicyConfig.model_validate_json(Path(draft_path).read_text())
    run = new_run("policy", name or task.name)
    run.plan(POLICY_STAGES)
    argon = Argon(run.dir / "argon", backend)
    run_policy_loops(run, task, draft1, load_adapter(task.robot), argon, swarm=swarm)
    return run


def run_from_video(video: str, prompt: str, name: str | None = None, swarm: str = "local",
                   backend: str | None = None, force: bool = False) -> Run:
    from core.twin import build_twin

    run = new_run("pipeline", name or Path(video).stem, video=video, prompt=prompt)
    run.plan(TWIN_STAGES + POLICY_STAGES)
    argon = Argon(run.dir / "argon", backend)
    try:
        twin = build_twin(run, video, prompt, argon, force=force)
    except Refused:
        return run
    run_policy_loops(run, twin["task"], twin["draft1"], twin["adapter"], argon, swarm=swarm,
                     replay_ref=twin.get("dynamic_ref"))
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description="WorldGym pipeline")
    ap.add_argument("--video")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--task")
    ap.add_argument("--draft")
    ap.add_argument("--name")
    ap.add_argument("--swarm", choices=["local", "antigravity"], default="local")
    ap.add_argument("--backend", choices=["gemini", "replay", "manual"])
    ap.add_argument("--force", action="store_true", help="continue past a QA refusal (testing only)")
    a = ap.parse_args()
    if a.video:
        run = run_from_video(a.video, a.prompt, a.name, a.swarm, a.backend, a.force)
    elif a.task and a.draft:
        run = run_from_task(a.task, a.draft, a.name, a.swarm, a.backend)
    else:
        ap.error("give --video (+ --prompt) or --task + --draft")
    print(run.dir)


if __name__ == "__main__":
    main()

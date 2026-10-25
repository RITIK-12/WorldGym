"""Run directory -> Studio story: the stage timeline and the eight narrated steps. Pure data (no viser).

Every reader tolerates partial runs: a missing stage, artifact or field yields a step with `done=False` and a
placeholder sentence instead of an exception.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.pipeline import POLICY_STAGES, TWIN_STAGES
from core.schemas import RunManifest, StageRecord, TaskSpec

STATUS_COLOUR = {"ok": "#2fb36d", "running": "#e0a020", "failed": "#e34948", "refused": "#e34948",
                 "pending": "#6b6b68"}
TITLES = ["Input", "Argon builds the twin", "Real ↔ Sim", "Replay reproduces the video", "The benchmark breaks it",
          "Critique and repair", "Held-out ratchet", "Your turn"]
NEXT = ["Next: Argon builds the twin →", "Next: check the twin against the real frame →",
        "Next: replay the motion in physics →", "Next: stress-test it on unseen scenes →",
        "Next: Argon critiques and repairs →", "Next: does the repair hold out? →",
        "Next: your turn, perturb it live →", "Back to the start ↺"]
DASH = "—"


@dataclass
class Chip:
    name: str
    status: str
    colour: str
    headline: str
    error: str | None
    seconds: float | None


@dataclass
class Step:
    n: int
    title: str
    done: bool
    sentence: str
    number: str = DASH
    number_label: str = ""
    media: str | None = None
    next_label: str = ""
    extra: dict = field(default_factory=dict)


def pct(x) -> str:
    return DASH if x is None else f"{100 * float(x):.0f} %"


def fmt_s(s: float | None) -> str:
    if s is None:
        return DASH
    s = int(round(s))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001 - missing or half-written artifact
        return None


def read_manifest(run_dir: Path) -> RunManifest | None:
    try:
        return RunManifest.model_validate_json((Path(run_dir) / "manifest.json").read_text())
    except Exception:  # noqa: BLE001
        return None


def get_stage(m: RunManifest, name: str) -> StageRecord | None:
    return next((s for s in m.stages if s.name == name), None)


def stage_seconds(s: StageRecord, now: datetime | None = None) -> float | None:
    if not s.t0:
        return None
    try:
        t0 = datetime.fromisoformat(s.t0)
        t1 = datetime.fromisoformat(s.t1) if s.t1 else (now or datetime.now().astimezone())
        return max(0.0, (t1 - t0).total_seconds())
    except ValueError:
        return None


def timeline(m: RunManifest) -> list[Chip]:
    order = {n: k for k, n in enumerate(TWIN_STAGES + POLICY_STAGES)}
    stages = sorted(m.stages, key=lambda s: order.get(s.name, len(order)))
    return [Chip(s.name, s.status, STATUS_COLOUR.get(s.status, "#6b6b68"), s.headline or "", s.error,
                 stage_seconds(s)) for s in stages]


def drafts(run_dir: Path) -> list[str]:
    """Drafts with a replayable nominal rollout, in ratchet order."""
    ds = [p.parent.parent.name for p in Path(run_dir).glob("draft*/nominal/telemetry.npz")
          if (p.parent / "scene.xml").exists()]
    return sorted(ds, key=lambda n: int(re.sub(r"\D", "", n) or 0))


def live_drafts(run_dir: Path) -> list[str]:
    ds = [p.parent.name for p in Path(run_dir).glob("draft*/policy.json") if (p.parent / "nominal/task.json").exists()]
    return sorted(ds, key=lambda n: int(re.sub(r"\D", "", n) or 0))


def static_task(run_dir: Path) -> tuple[TaskSpec, str] | None:
    """Best scene to show before any draft exists: the Loop 1 twin, else the onboarded robot alone."""
    d = Path(run_dir)
    p = d / "loop1/task.json"
    if p.exists():
        try:
            return TaskSpec.model_validate_json(p.read_text()), f"loop1@{p.stat().st_mtime_ns}"
        except Exception:  # noqa: BLE001 - being written
            pass
    m = read_manifest(d)
    ob = get_stage(m, "robot_onboard") if m else None
    src = ob.data.get("source") if ob is not None and ob.status == "ok" else None
    if src and str(src).endswith(".json"):
        return TaskSpec(name=m.run_id, robot=Path(src).stem, objects=[]), f"robot:{Path(src).stem}"
    return None


def _first(*paths: Path) -> str | None:
    return next((str(p) for p in paths if p is not None and p.exists()), None)


def _glob_first(d: Path, pattern: str) -> Path | None:
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def fmt_value(v) -> str:
    if isinstance(v, dict) and "terms" in v:
        off = float(v.get("offset", 0.0))
        return " + ".join(v["terms"]) + (f" {'+' if off >= 0 else '−'} {abs(off):.3g}" if off else "")
    if isinstance(v, float):
        return f"{v:.4g}"
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def _cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def diff_markdown(diff: list[dict]) -> str:
    if not diff:
        return ""
    rows = ["| object | field | old | new |", "|---|---|---|---|"]
    rows += [f"| {_cell(d.get('object_id') or d.get('arm', ''))} | {_cell(d.get('field', ''))} | "
             f"{_cell(fmt_value(d.get('old')))} | {_cell(fmt_value(d.get('new')))} |" for d in diff]
    return "\n".join(rows)


def ratchet(run_dir: Path, m: RunManifest) -> list[dict]:
    """report.json ratchet, or the same rows rebuilt from whichever policy stages have finished."""
    rep = read_json(run_dir / "report.json")
    if rep and rep.get("ratchet"):
        return rep["ratchet"]
    rows = []
    for draft, stage, label in (("draft1", "benchmark", "Replay (Draft 1)"),
                                ("draft2", "loop2_critic", "Argon critic (Draft 2)"),
                                ("draft3", "loop3_swarm", "Swarm-hardened (Draft 3)")):
        s = get_stage(m, stage)
        if s is None or s.status != "ok" or "held_out" not in s.data:
            continue
        rows.append({"draft": draft, "label": label, "dev": (s.data.get("dev") or {}).get("rate"),
                     "held_out": s.data["held_out"].get("rate"), "collisions": s.data["held_out"].get("collisions"),
                     "nominal": None})
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# The eight steps
# ----------------------------------------------------------------------------------------------------------------------

def _task_only(n: int, task_name: str) -> Step:
    what = {1: "no input video; the scene is a hand-authored task",
            2: "no twin was built; the task file defines the scene directly",
            3: "no real frame to compare against"}[n]
    return Step(n, TITLES[n - 1], False, f"Task-only run ({task_name}): {what}.", DASH, "task-only run")


def _step_input(d: Path, m: RunManifest) -> Step:
    qa = get_stage(m, "qa")
    video = Path(m.video).name if m.video else "(no video)"
    prompt = f" · “{m.prompt}”" if m.prompt else ""
    media = _first(_glob_first(d / "architect/keyframes", "kf_*.jpg"), d / "camera/rest_frame.jpg", d / "qa/motion.png")
    if qa is None or qa.status in ("pending", "running"):
        return Step(1, TITLES[0], False, f"{video}{prompt} — video QA is {qa.status if qa else 'not started'}.",
                    DASH, "QA verdict", media)
    accepted = bool(qa.data.get("accepted", qa.status == "ok"))
    verdict = ("accepted" if accepted and qa.status == "ok" else "moving camera" if qa.status == "ok"
               else "refused" if qa.status == "refused" else qa.status)
    reason = f" ({qa.error})" if qa.status in ("refused", "failed") and qa.error else ""
    return Step(1, TITLES[0], True, f"{video}{prompt} — QA: {qa.headline or verdict}{reason}.", verdict,
                "QA verdict on the clip", media, extra={"keyframes_t": qa.data.get("keyframes_t", [])})


def _step_twin(d: Path, m: RunManifest) -> Step:
    va, ob = get_stage(m, "video_architect"), get_stage(m, "robot_onboard")
    rows, total, n_ok = [], 0.0, 0
    for name in TWIN_STAGES:
        s = get_stage(m, name)
        if s is None:
            continue
        secs = stage_seconds(s) if s.t1 else None
        total += secs or 0.0
        n_ok += s.status == "ok"
        rows.append({"stage": name, "status": s.status, "seconds": secs, "headline": s.headline or s.error or ""})
    robot = ((va.data.get("robot") or {}).get("make_model_guess") if va else None) or "robot (not identified yet)"
    conf = (va.data.get("robot") or {}).get("confidence") if va else None
    verify = (ob.data.get("verify") if ob else None) or {}
    if verify.get("menagerie_dir") and verify["menagerie_dir"] != (va.data.get("robot") or {}).get("menagerie_dir"):
        robot = f"{verify['menagerie_dir']} (visual check of rendered candidates overruled the first guess, {robot})"
        conf = verify.get("confidence")
    n_obj = len(va.data.get("objects", [])) if va else 0
    n_ph = len(va.data.get("phases", [])) if va else 0
    done = va is not None and va.status == "ok"
    sentence = (f"Gemini 4 Argon identified a {robot}{f' ({conf:.0%} confident)' if conf else ''} with {n_obj} objects "
                f"and {n_ph} phases; {n_ok}/{len(TWIN_STAGES)} twin stages finished."
                if done else f"Argon is reading the clip: {n_ok}/{len(TWIN_STAGES)} twin stages finished.")
    if ob is not None and ob.headline:
        sentence += f" Robot onboarding: {ob.headline}."
    media = _first(d / "camera/robot_fit.jpg", _glob_first(d / "architect/keyframes", "kf_*.jpg"))
    table = ["| stage | status | time | headline |", "|---|---|---|---|"]
    table += [f"| {r['stage']} | {r['status']} | {fmt_s(r['seconds'])} | {_cell(r['headline'])} |" for r in rows]
    return Step(2, TITLES[1], done, sentence, fmt_s(total) if total else DASH, "wall clock, video → twin", media,
                extra={"stages": rows, "robot": robot, "table": "\n".join(table)})


def _step_real_sim(d: Path, m: RunManifest) -> Step:
    s = get_stage(m, "loop1_static")
    hist = (s.data.get("history") if s else None) or []
    by_iter = {h.get("iteration"): h for h in hist}
    iters = []
    for p in sorted((d / "loop1").glob("iter*/matrix.jpg"), key=lambda p: int(re.sub(r"\D", "", p.parent.name) or 0)):
        k = int(re.sub(r"\D", "", p.parent.name) or 0)
        h = by_iter.get(k, {})
        iters.append({"label": f"iteration {k}", "path": str(p),
                      "alignment": h.get("alignment_after_critic", h.get("alignment")),
                      "critique": [f"{c.get('object_id') or ''} ({c.get('view', '')}): {c.get('observation', '')}"
                                   for c in h.get("critique", [])]})
    final = d / "loop1/final/matrix.jpg"
    align = s.data.get("alignment") if s else None
    if final.exists():
        iters.append({"label": "final", "path": str(final), "alignment": align, "critique": []})
    done = s is not None and s.status == "ok" and align is not None
    if done:
        sentence = (f"The Twin Critic compared four views of the twin (ghost, top, ground, wire) with the real frame: "
                    f"{s.headline}.")
    elif s is not None and s.status in ("failed", "refused"):
        sentence = f"Loop 1 {s.status}: {s.error}."
    else:
        sentence = f"Loop 1 (static fidelity) is {s.status if s else 'not started'}."
    objs = (s.data.get("objects") if s else None) or {}
    return Step(3, TITLES[2], done, sentence, pct(align), "real ↔ sim alignment", iters[-1]["path"] if iters else None,
                extra={"iterations": iters, "objects": objs})


def _step_replay(d: Path, m: RunManifest) -> Step:
    rp, mx, rc = get_stage(m, "replay"), get_stage(m, "motion_extract"), get_stage(m, "replay_critic")
    dyn = next((s.data["dynamic"] for s in (rc, mx, rp) if s is not None and s.data.get("dynamic")), {})
    err = dyn.get("end_pose_err_mm")
    success = rp.data.get("success") if rp else dyn.get("replay_success")
    has = "draft1" in drafts(d)
    if err is not None:
        number, label = f"{err:.0f} mm", "end-pose error vs the last video frame"
    elif success is not None:
        number, label = ("✓" if success else "✗"), "nominal replay " + ("succeeds" if success else "fails")
    else:
        number, label = DASH, "nominal replay"
    who = "the motion extracted from the video" if m.kind == "pipeline" else "the hand-written Draft 1 policy"
    if rp is not None and rp.status == "ok":
        sentence = f"Draft 1 replays {who} in MuJoCo: {rp.headline}."
        if rc is not None and rc.status == "ok" and rc.headline:
            sentence += f" Argon's replay critic, watching real and sim side by side: {rc.headline}."
        sentence += " Watch it in the 3D view."
    elif rp is not None and rp.status in ("failed", "refused"):
        sentence = f"Replay {rp.status}: {rp.error}."
    else:
        sentence = f"Replay is {rp.status if rp else 'not started'}."
    segs = len((mx.data.get("segments") if mx else None) or [])
    return Step(4, TITLES[3], bool(rp and rp.status == "ok" and has), sentence, number, label,
                extra={"draft": "draft1", "segments": segs, "dynamic": dyn})


def _step_benchmark(d: Path, m: RunManifest) -> Step:
    b = get_stage(m, "benchmark")
    if b is None or b.status != "ok":
        msg = f"Benchmark {b.status}: {b.error}." if b and b.error else f"Benchmark is {b.status if b else 'not started'}."
        return Step(5, TITLES[4], False, msg, DASH, "held-out success, Draft 1")
    dev, held, lock = b.data.get("dev") or {}, b.data.get("held_out") or {}, b.data.get("lock_sha256") or ""
    sentence = (f"Draft 1 passes {dev.get('successes', '?')}/{dev.get('n', '?')} perturbed dev scenes and only "
                f"{pct(held.get('rate'))} of the {held.get('n', '?')} held-out scenes, locked before any repair "
                f"(sha256 {lock[:12]}).")
    return Step(5, TITLES[4], True, sentence, pct(held.get("rate")), "held-out success, Draft 1",
                extra={"clusters": dev.get("clusters") or {}, "lock": lock, "by_family": held.get("by_family") or {},
                       "dev": dev, "held_out": held})


def _step_critique(d: Path, m: RunManifest) -> Step:
    s = get_stage(m, "loop2_critic")
    if s is None or s.status != "ok":
        msg = (f"Loop 2 {s.status}: {s.error}." if s and s.error
               else f"Loop 2 (critic) is {s.status if s else 'not started'}.")
        return Step(6, TITLES[5], False, msg, DASH, "parameters Argon rewrote")
    crit = sorted(s.data.get("critique") or [], key=lambda c: c.get("timestamp_sec") or 0.0)
    diff = s.data.get("diff") or []
    causes = s.data.get("root_causes") or []
    kfs = sorted((d / "loop2/keyframes").glob("*.jpg"))
    media = None
    if kfs:
        idx = crit[0].get("video_frame_idx") if crit else None
        num = lambda p: int(re.sub(r"\D", "", p.stem) or 0)  # noqa: E731
        media = str(min(kfs, key=lambda p: abs(num(p) - idx)) if idx is not None else kfs[0])
    sentence = (f"Argon watched the Draft 1 video and named the root cause: {causes[0]}." if causes
                else "Argon watched the Draft 1 video and patched the policy.")
    held = (s.data.get("held_out") or {}).get("rate")
    if held is not None:
        sentence += f" Draft 2 scores {pct(held)} held-out."
    return Step(6, TITLES[5], True, sentence, str(len(diff)), "policy parameters Argon rewrote", media,
                extra={"critique": crit, "root_causes": causes, "diff": diff, "diff_md": diff_markdown(diff)})


def _step_ratchet(d: Path, m: RunManifest) -> Step:
    rows = ratchet(d, m)
    rep = read_json(d / "report.json") or {}
    lock = rep.get("lock_sha256") or ((get_stage(m, "benchmark") or StageRecord(name="")).data.get("lock_sha256") or "")
    sw = get_stage(m, "loop3_swarm")
    reports = (sw.data.get("reports") if sw else None) or []
    swarm = [{"hypothesis": r.get("hypothesis", "?"), "failing": len(r.get("failing_scenarios") or []),
              "summary": r.get("summary", ""), "source": r.get("source", "")} for r in reports]
    if not rows:
        return Step(7, TITLES[6], False, "No draft has been scored on the held-out suite yet.", DASH,
                    "held-out, final draft", extra={"ratchet": [], "lock": lock, "swarm": swarm})
    held = " → ".join(pct(r.get("held_out")) for r in rows)
    coll = " → ".join(str(r.get("collisions", DASH)) for r in rows)
    sentence = (f"Held-out success {held} across the drafts on one suite locked before any hardening "
                f"(sha256 {lock[:12]}); held-out collisions {coll}.")
    if swarm:
        sentence += (f" The swarm ({(sw.data.get('swarm_mode') or 'local')}) found "
                     f"{sum(x['failing'] for x in swarm)} adversarial scenarios across {len(swarm)} hypotheses.")
    return Step(7, TITLES[6], bool(rep.get("ratchet")), sentence, pct(rows[-1].get("held_out")),
                f"held-out success, {rows[-1].get('label') or rows[-1].get('draft')}",
                extra={"ratchet": rows, "lock": lock, "swarm": swarm})


def _step_your_turn(d: Path, m: RunManifest) -> Step:
    ds = live_drafts(d)
    task = d / ds[0] / "nominal/task.json" if ds else None
    sentence = ("Move the obstacle and the objects, pick a draft and run the physics live; the locked evaluator "
                "judges the result." if ds else "No draft policy is available yet to run live.")
    return Step(8, TITLES[7], bool(ds), sentence, DASH, "run a live episode",
                extra={"drafts": ds, "task": str(task) if task else None})


def build_story(run_dir: str | Path) -> list[Step]:
    d = Path(run_dir)
    m = read_manifest(d)
    if m is None:
        return [Step(k + 1, t, False, "manifest.json is missing or unreadable.", next_label=NEXT[k])
                for k, t in enumerate(TITLES)]
    rep = read_json(d / "report.json") or {}
    task_name = rep.get("task") or (read_json(d / "draft1/nominal/task.json") or {}).get("name") or m.run_id
    twin = m.kind == "pipeline" or get_stage(m, "qa") is not None
    steps = ([_step_input(d, m), _step_twin(d, m), _step_real_sim(d, m)] if twin
             else [_task_only(k, task_name) for k in (1, 2, 3)])
    steps += [_step_replay(d, m), _step_benchmark(d, m), _step_critique(d, m), _step_ratchet(d, m), _step_your_turn(d, m)]
    for k, s in enumerate(steps):
        s.next_label = NEXT[k]
    return steps

"""The self-improving real-to-sim harness: Penn Action clip -> perception -> θ -> G1 reference -> ladder -> critic.

Inner loop (per clip): retarget with θ, score the GT-free ladder, ask the critic for a patch, keep it only if the
ladder score rises. Outer loop (across clips): consolidate the kept patches into new default θ and a critic playbook,
adopted only if they raise the mean ladder score on validation clips. Later clips start from better defaults.
The locked referee scores every generation against Penn Action GT for the demo; those numbers are written to
eval/ files and never shown to the critic or used to keep or revert.

    uv run python -m humanoid.harness perceive --split train --n 12
    uv run python -m humanoid.harness improve --clips 0174,0177 --iters 6          # critic = Claude via the API
    uv run python -m humanoid.harness step --run runs/<id> --clip 0174 [--patch '{"hypothesis": ..., ...}']
    uv run python -m humanoid.harness summary --run runs/<id>                     # reveals the GT eval, at the end
    uv run python -m humanoid.harness consolidate --run runs/<id> --val 0186,0188
    uv run python -m humanoid.harness heldout --n 58        # test split, once, aggregates only

`step` is the agent-as-critic mode: the coding agent reads the printed ladder, facts, history and contact sheet and
passes its patch back with --patch. It prints no referee numbers, so the critic stays blind to GT.
Perception targets the striker only: SAM 3 "baseball batter", then SAM 3D Body on that one person.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from core.runs import Run, new_run
from humanoid import ladder, penn, perception, referee, render
from humanoid.config import ConfigPatch, HarnessConfig, apply_patch, diff
from humanoid.critic import ClaudeCritic
from humanoid.retarget import retarget

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "humanoid" / "learned.json"     # adopted defaults + playbook (the harness's own memory)
ACTION = "baseball_swing"


def learned() -> tuple[HarnessConfig, str]:
    if STATE.exists():
        s = json.loads(STATE.read_text())
        return HarnessConfig.model_validate(s["defaults"]), s.get("playbook", "")
    return HarnessConfig(), ""


def dev_clips(n: int, split: str = "train") -> list[str]:
    """Shortest side-view clips first: side views show the swing plane, short clips are cheap to perceive."""
    idx = penn.index()
    cs = penn.clips(ACTION, split)
    return sorted(cs, key=lambda c: (idx[c]["view"] not in ("right", "left"), idx[c]["nframes"], c))[:n]


def save_reference(path: Path, ref) -> None:
    """Everything the RL step and the rollout referee need: the motion and the clip camera it was seen through."""
    np.savez_compressed(path, qpos=ref.qpos, fps=ref.fps, clip=ref.clip, world_from_cam=ref.world_from_cam,
                        cam_origin=ref.cam_origin, scale=ref.scale, pin_offset=ref.pin_offset, focal=ref.focal,
                        principal=ref.principal, top_hand=ref.info.get("top_hand", "right"),
                        bat=ref.info.get("bat", "none"))


def generation(clip: str, per: dict, cfg: HarnessConfig) -> tuple[object, dict, dict]:
    ref = retarget(clip, per, cfg, penn.FPS)
    ev = ladder.evaluate(ref, cfg)
    gt = referee.score(clip, per["keypoints_2d"], render.robot_penn2d(ref))   # eval only
    return ref, ev, gt


def improve_clip(run: Run, clip: str, cfg0: HarnessConfig, critic, iters: int, log=print) -> dict:
    per = perception.perceive(clip, log=log)
    d = f"clips/{clip}"
    cfg = cfg0
    ref, ev, gt = generation(clip, per, cfg)
    sheet = render.contact_sheet(ref, per, run.path(d, "gen0.png"))
    run.write_json(f"{d}/gen0.json", {"cfg": cfg, "ladder": ev})
    run.write_json(f"{d}/eval/gen0.json", gt)
    log(f"{clip} gen0 ladder {ev['score']:.3f}  [eval: limb cos {gt.get('robot_limb_cos', float('nan')):.3f}, "
        f"perception PCK {gt.get('perception_pck', float('nan')):.3f}]")
    curve = [{"iter": 0, "score": ev["score"], "kept": True, "eval": gt}]
    history = []
    for it in range(1, iters + 1):
        if ev["score"] >= 3.0:
            break
        t0 = time.time()
        try:
            patch = critic.propose(clip, it, cfg, ladder.table(ev), ev["facts"], history, sheet, run.path(d, "critic"))
            cand = apply_patch(cfg, patch)
        except Exception as e:  # noqa: BLE001 - a bad or missing proposal ends this clip's loop, recorded
            log(f"{clip} iter {it}: critic failed: {type(e).__name__}: {e}")
            history.append({"iter": it, "error": f"{type(e).__name__}: {e}"})
            break
        ref2, ev2, gt2 = generation(clip, per, cand)
        kept = ev2["score"] > ev["score"]
        h = {"iter": it, "hypothesis": patch.hypothesis, "diff": diff(cfg, cand), "score_before": ev["score"],
             "score_after": ev2["score"], "kept": kept, "seconds": round(time.time() - t0, 1)}
        history.append(h)
        run.write_json(f"{d}/gen{it}.json", {"cfg": cand, "ladder": ev2, "patch": patch, "kept": kept})
        run.write_json(f"{d}/eval/gen{it}.json", gt2)
        log(f"{clip} iter {it}: {h['diff']} -> {ev2['score']:.3f} ({'kept' if kept else 'reverted'})  "
            f"[eval: limb cos {gt2.get('robot_limb_cos', float('nan')):.3f}]  {patch.hypothesis[:90]}")
        curve.append({"iter": it, "score": ev2["score"], "kept": kept, "eval": gt2})
        if kept:
            cfg, ref, ev, gt = cand, ref2, ev2, gt2
            sheet = render.contact_sheet(ref, per, run.path(d, f"gen{it}.png"))
    save_reference(run.path(d, "reference.npz"), ref)
    out = {"clip": clip, "cfg": cfg, "score": ev["score"], "passed": {k: r["passed"] for k, r in ev["rungs"].items()},
           "history": history, "curve": curve, "eval": gt}
    run.write_json(f"{d}/summary.json", out)
    return out


def cmd_perceive(a) -> None:
    clips = a.clips.split(",") if a.clips else dev_clips(a.n, a.split)
    for c in clips:   # pass 1: SAM 3 finds the striker on every clip, then it is released
        box, score = perception.detect_striker(c)
        print(f"{c}: striker box {np.round(box).astype(int).tolist()} score {score:.2f}", flush=True)
    body = perception.Body3D()   # pass 2: SAM 3D Body alone in memory
    for i, c in enumerate(clips):
        t0 = time.time()
        per = perception.perceive(c, body=body)
        print(f"[{i + 1}/{len(clips)}] {c}: {len(per['cam_t'])} frames, {int(per['redetect'].sum())} re-detects, "
              f"{time.time() - t0:.0f} s", flush=True)


def cmd_step(a) -> None:
    """One agent-as-critic step on one clip: gen0 on the first call, then apply --patch, keep it if the ladder rises."""
    from core.runs import load_run
    run = load_run(a.run) if a.run else new_run("humanoid", "agent_critic")
    clip, d = a.clip, f"clips/{a.clip}"
    st_path = run.path(d, "state.json")
    per = perception.perceive(clip)
    if not st_path.exists():
        cfg, playbook = learned()
        ref, ev, gt = generation(clip, per, cfg)
        render.contact_sheet(ref, per, run.path(d, "gen0.png"))
        run.write_json(f"{d}/gen0.json", {"cfg": cfg, "ladder": ev})
        run.write_json(f"{d}/eval/gen0.json", gt)
        save_reference(run.path(d, "reference.npz"), ref)
        st = {"cfg": cfg.model_dump(), "ladder": ev, "iter": 0, "sheet": f"{d}/gen0.png", "history": [],
              "curve": [{"iter": 0, "score": ev["score"], "kept": True}], "playbook": playbook}
    else:
        st = json.loads(st_path.read_text())
        if a.patch:
            it = st["iter"] + 1
            patch = ConfigPatch.model_validate_json(a.patch)
            cfg = HarnessConfig.model_validate(st["cfg"])
            cand = apply_patch(cfg, patch)
            ref, ev, gt = generation(clip, per, cand)
            kept = ev["score"] > st["ladder"]["score"]
            h = {"iter": it, "hypothesis": patch.hypothesis, "diff": diff(cfg, cand),
                 "score_before": st["ladder"]["score"], "score_after": ev["score"], "kept": kept}
            run.write_json(f"{d}/gen{it}.json", {"cfg": cand, "ladder": ev, "patch": patch, "kept": kept})
            run.write_json(f"{d}/eval/gen{it}.json", gt)
            st["history"].append(h)
            st["curve"].append({"iter": it, "score": ev["score"], "kept": kept})
            st["iter"] = it
            if kept:
                render.contact_sheet(ref, per, run.path(d, f"gen{it}.png"))
                save_reference(run.path(d, "reference.npz"), ref)
                st.update(cfg=cand.model_dump(), ladder=ev, sheet=f"{d}/gen{it}.png")
            print(f"iter {it}: {h['diff']} -> {ev['score']:.3f} ({'KEPT' if kept else 'reverted'})")
    run.write_json(f"{d}/state.json", st)
    ev = st["ladder"]
    print(f"run {run.dir}\nclip {clip} ({penn.index()[clip]['view']} view, {len(per['cam_t'])} frames), "
          f"iteration {st['iter']}\n\ncurrent θ: {json.dumps(st['cfg'])}\n\n{ladder.table(ev)}\n\nfacts:")
    print("\n".join(f"- {f}" for f in ev["facts"]))
    print("\ntried:\n" + ("\n".join(f"- iter {h['iter']}: {json.dumps(h['diff'])} {h['score_before']:.3f}->"
                                    f"{h['score_after']:.3f} {'kept' if h['kept'] else 'reverted'}: {h['hypothesis']}"
                                    for h in st["history"]) or "(nothing yet)"))
    print(f"\ncontact sheet: {run.dir / st['sheet']}")


def cmd_summary(a) -> None:
    """End of a session: ladder curve per clip, then (only now) the referee's GT numbers, first vs final generation."""
    run_dir = Path(a.run)
    rows = []
    for st_path in sorted(run_dir.glob("clips/*/state.json")):
        st = json.loads(st_path.read_text())
        clip = st_path.parent.name
        kept_it = max([h["iter"] for h in st["history"] if h["kept"]], default=0)
        first = json.loads((st_path.parent / "eval" / "gen0.json").read_text())
        final = json.loads((st_path.parent / "eval" / f"gen{kept_it}.json").read_text())
        rows.append({"clip": clip, "curve": [round(c["score"], 3) for c in st["curve"]],
                     "ladder_start": st["curve"][0]["score"], "ladder_end": st["ladder"]["score"],
                     "passed": {k: r["passed"] for k, r in st["ladder"]["rungs"].items()}, "eval_first": first,
                     "eval_final": final})
    out = {"clips": len(rows), "ladder_start": float(np.mean([r["ladder_start"] for r in rows])),
           "ladder_end": float(np.mean([r["ladder_end"] for r in rows])),
           "eval_first": referee.aggregate([r["eval_first"] for r in rows]),
           "eval_final": referee.aggregate([r["eval_final"] for r in rows]), "per_clip": rows}
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per_clip"}, indent=1))
    for r in rows:
        print(r["clip"], r["curve"], r["passed"])


def cmd_improve(a) -> None:
    cfg0, playbook = learned()
    critic = ClaudeCritic(playbook)
    clips = a.clips.split(",") if a.clips else dev_clips(a.n)
    run = new_run("humanoid", "improve_claude")
    run.write_json("start.json", {"cfg": cfg0, "playbook": playbook, "critic": "claude", "clips": clips})
    rows = []
    for c in clips:
        with run.stage(f"clip_{c}") as s:
            r = improve_clip(run, c, cfg0, critic, a.iters)
            s.headline = f"ladder {r['curve'][0]['score']:.2f} -> {r['score']:.2f} in {len(r['history'])} iters"
            s.data = {"score": r["score"], "passed": r["passed"]}
            rows.append(r)
    summary = {"critic": "claude", "clips": len(rows),
               "ladder_start": float(np.mean([r["curve"][0]["score"] for r in rows])),
               "ladder_end": float(np.mean([r["score"] for r in rows])),
               "iters_used": float(np.mean([len(r["history"]) for r in rows])),
               "eval_start": referee.aggregate([r["curve"][0]["eval"] for r in rows]),
               "eval_end": referee.aggregate([r["eval"] for r in rows])}
    run.write_json("summary.json", summary)
    print(json.dumps(summary, indent=1))
    print("run:", run.dir)


class Consolidation(ConfigPatch):
    playbook: str


def cmd_consolidate(a) -> None:
    """Outer loop: kept patches across clips -> new defaults + playbook, adopted only if validation improves."""
    from humanoid import llm
    run_dir = Path(a.run)
    rows = [json.loads(p.read_text()) for p in sorted(run_dir.glob("clips/*/summary.json"))]
    cfg0, playbook0 = learned()
    notebook = []
    for r in rows:
        for h in r["history"]:
            if "diff" in h:
                notebook.append(f"clip {r['clip']}: {json.dumps(h['diff'])} {h['score_before']:.3f}->{h['score_after']:.3f} "
                                f"{'kept' if h['kept'] else 'reverted'}; {h['hypothesis']}")
    prompt = (f"Current defaults:\n{cfg0.model_dump_json(indent=1)}\n\nCurrent playbook:\n{playbook0 or '(none)'}\n\n"
              "Lab notebook from the inner loop (each line: patch, ladder score before->after, kept or reverted, "
              "hypothesis):\n" + "\n".join(notebook) + "\n\nPropose new defaults (as a patch on the current defaults) "
              "that would have helped most clips from the start, and a short playbook (at most 8 rules of the form "
              "'when <fact pattern>, change <field> <direction> first') for the critic. Put the rules in `playbook`.")
    system = ("You consolidate what a self-improving real-to-sim harness learned across clips into better starting "
              "defaults and rules for its critic. Prefer changes that were kept on several clips; ignore one-offs.")
    if a.show:
        print(prompt)
        return
    if a.patch:   # agent mode: the coding agent read the same notebook and wrote the consolidation itself
        c = Consolidation.model_validate_json(a.patch)
        (run_dir / "consolidate").mkdir(parents=True, exist_ok=True)
        (run_dir / "consolidate" / "notebook.md").write_text(prompt)
    else:
        c = llm.ask("consolidate", system, prompt, Consolidation, bundle_dir=run_dir / "consolidate", effort="high")
    cand = apply_patch(cfg0, ConfigPatch(**c.model_dump(exclude={"playbook"})))
    val = a.val.split(",") if a.val else dev_clips(a.n_val + len(rows))[len(rows):]
    old = [generation(v, perception.perceive(v), cfg0)[1]["score"] for v in val]
    new = [generation(v, perception.perceive(v), cand)[1]["score"] for v in val]
    adopt = float(np.mean(new)) > float(np.mean(old))
    result = {"diff": diff(cfg0, cand), "playbook": c.playbook, "hypothesis": c.hypothesis, "val_clips": val,
              "val_old": float(np.mean(old)), "val_new": float(np.mean(new)), "adopted": adopt}
    (run_dir / "consolidate" / "result.json").write_text(json.dumps(result, indent=1, default=str))
    if adopt:
        STATE.write_text(json.dumps({"defaults": cand.model_dump(), "playbook": c.playbook,
                                     "from_run": run_dir.name, "val": [result["val_old"], result["val_new"]]}, indent=1))
    print(json.dumps(result, indent=1, default=str))


def cmd_heldout(a) -> None:
    """Test split, scored once with the adopted defaults (no critic iterations), aggregates only."""
    cfg, _ = learned()
    clips = dev_clips(a.n, "test")
    rows, ladders = [], []
    for c in clips:
        _, ev, gt = generation(c, perception.perceive(c), cfg)
        rows.append(gt)
        ladders.append(ev["score"])
    out = {"split": "test", "clips": len(clips), "ladder_mean": float(np.mean(ladders)),
           "ladder_pass_L2": float(np.mean([s >= 3.0 for s in ladders])), "eval": referee.aggregate(rows)}
    run = new_run("humanoid", "heldout")
    run.write_json("heldout.json", out)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("perceive")
    p.add_argument("--clips", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=12)
    p = sub.add_parser("improve")
    p.add_argument("--clips", default="")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--iters", type=int, default=6)
    p = sub.add_parser("step")
    p.add_argument("--run", default="")
    p.add_argument("--clip", required=True)
    p.add_argument("--patch", default="")
    p = sub.add_parser("summary")
    p.add_argument("--run", required=True)
    p = sub.add_parser("consolidate")
    p.add_argument("--run", required=True)
    p.add_argument("--val", default="")
    p.add_argument("--n-val", type=int, default=4)
    p.add_argument("--patch", default="", help="agent mode: Consolidation JSON (patch fields + playbook + hypothesis)")
    p.add_argument("--show", action="store_true", help="print the lab notebook and exit")
    p = sub.add_parser("heldout")
    p.add_argument("--n", type=int, default=58)
    a = ap.parse_args()
    {"perceive": cmd_perceive, "improve": cmd_improve, "step": cmd_step, "summary": cmd_summary,
     "consolidate": cmd_consolidate, "heldout": cmd_heldout}[a.cmd](a)


if __name__ == "__main__":
    main()

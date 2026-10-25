# WorldGym — Progress Log

Built at the Gemini 4 Argon SF Hackathon, 25 Oct 2026 (build window 10:30–17:00 PT).
Everything in this repository was written today.

**What it is.** One video of a robot doing a tabletop task goes in. Out come a MuJoCo twin of the scene with a
fidelity score, a replay of the demonstrated motion, a perturbation benchmark with a locked held-out suite, and a
policy hardened by Gemini 4 Argon critique and Antigravity Managed Agents, with a number at every step.

## Quick start

```bash
uv sync
scripts/setup_assets.sh          # sparse-clones MuJoCo Menagerie (Panda + Robotiq)
cp .env.example .env             # add GEMINI_API_KEY
```

## Stage status

| # | Stage | Status | Headline |
|---|---|---|---|
| 0 | Repo, env, deps, Menagerie | done | mujoco 3.17, google-genai 2.33 |
| 1 | Sim core: compiler, controller, policy DSL, telemetry, rollout | done | reference pick-place 2/2 in bin, TCP lag p99 1.9 mm, 15.7 s sim in 2.7 s wall |
| 2 | Locked evaluator | done, locked | 10 referee tests; sha256 `d6e94519…` |

## How to run

```bash
uv run python -m core.rollout tasks/panda_sort.json policies/panda_sort_v3.json --video runs/v3.mp4
uv run pytest -q
```

## Log (25 Oct 2026, PT)

- **11:45** Repo scaffold: uv project (Python 3.12), dependencies, `.env` kept out of git, Menagerie setup script.
- **12:00** Sim core. Menagerie Panda patched via ElementTree (gravcomp, TCP site at the fingertip pads, tendon
  force gripper, named collision geoms). DLS IK on a scratch MjData with velocity feed-forward and per-joint limits,
  collision reflex, primitive DSL with state-conditioned parameters, fact-only contact telemetry, split-view video.
  Reference task `panda_sort` (cube + cylinder into a bin past a barrier) succeeds with zero unwanted contacts and is
  bit-identical across runs and with/without video. Evaluator tested (placed / outside / tilted / yawed cylinder /
  on-top-of / obstacle displacement / rest) and locked.

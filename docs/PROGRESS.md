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
| 3 | Run registry + model adapter (gemini / replay / manual) | done | manifest per run, request bundles |
| 4 | Video QA gate | done | synthetic tripod clip: static 97 %, accepted; handheld / exposure-drift clips refused |
| 5 | Robot onboarding + self-test | done | Panda 5/5 checks (IK grid max err 0.09 mm, 0.1 kg box lifted 0.218 m) |
| 6 | Video Architect (Argon) | code done | awaiting GEMINI_API_KEY for the first live call |
| 7 | Benchmark + locked held-out | done | dev 12 scenarios in ~9 s (8 workers); Draft 1 5/12, v3 10/12 |
| 8 | Loop 2 critic | done (replay-tested) | cached critique turns failing Draft 1 into a passing Draft 2 |
| 9 | Loop 3 swarm | done (local), live path written | 4 hypotheses x 24 rollouts each find failures in v3 |
| 10 | Pipeline orchestrator (policy half) | done (replay-tested) | replay → benchmark → Loop 2 → Loop 3 → report, all stages in the run manifest |
| 11 | Live Gemini API + spend guard | done | Argon text + video/image calls verified; ~$0.006 spent on smoke tests |
| 12 | Video Architect on a real clip | done | 36 s handheld dual-arm wiping clip: 1 call, 142 s, ~$0.11, valid JSON first try |

## Runaway-loop guard (not a spend limit)

Argon runs at `thinking_level="high"` with the full 65,536-token output window. Live calls go through
`core/argon.py`, which serves identical requests from a local cache and only trips on runaway loops:
`WORLDGYM_MAX_CALLS` (400 live calls per process), `WORLDGYM_MAX_REPEAT` (same request name 4x),
`WORLDGYM_MAX_AGENTS` (16 Managed Agents per process), `WORLDGYM_AGENT_MAX_TOKENS` (2M per agent). Token usage
is logged to `.worldgym/api_ledger.jsonl` (gitignored). `WORLDGYM_LLM=replay` runs from cached responses.

## How to run

```bash
uv run python -m core.rollout tasks/panda_sort.json policies/panda_sort_v3.json --video runs/v3.mp4
uv run pytest -q
```

## Log (25 Oct 2026, PT)

- **11:48** Repo scaffold: uv project (Python 3.12), dependencies, `.env` kept out of git, Menagerie setup script.
- **12:00** Sim core. Menagerie Panda patched via ElementTree (gravcomp, TCP site at the fingertip pads, tendon
  force gripper, named collision geoms). DLS IK on a scratch MjData with velocity feed-forward and per-joint limits,
  collision reflex, primitive DSL with state-conditioned parameters, fact-only contact telemetry, split-view video.
  Reference task `panda_sort` (cube + cylinder into a bin past a barrier) succeeds with zero unwanted contacts and is
  bit-identical across runs and with/without video. Evaluator tested (placed / outside / tilted / yawed cylinder /
  on-top-of / obstacle displacement / rest) and locked.
- **12:02** Run registry (`runs/<id>/manifest.json`, stages pending/running/ok/failed/refused) and one model adapter
  with three backends: live Interactions API in background mode, replay of cached responses, and manual bundles.
- **12:06** Video QA gate (ORB homography camera-motion check that ignores a large moving robot, exposure, blur,
  motion-energy keyframes), Video Architect prompt + schema, robot onboarding with a 5-check self-test, and a
  synthetic 1280x720 robot clip with full ground truth (`demo/synthetic_panda_sort.*`) as the oracle for the camera
  solve, Loop 1 and motion extraction.
- **12:09** Benchmark. Perturbation family (object poses/sizes/mass/friction, obstacle heights, container sizes)
  with yawed-rectangle rejection sampling and a 2 mm settle gate. Held-out suite: 14 dev-range + 6 harder-family
  scenarios from an `os.urandom` seed, chmod 444 + sha256, scored once per draft via a ledger, aggregates only.
  Draft 1 (absolute heights) passes nominal but goes 5/12 on dev (6 barrier collisions); state-conditioned v3: 10/12.
- **12:15** Loop 2 critic: Argon gets the rollout video, event keyframes, fact-only trace, resolved parameters and
  dev failures; `apply_patch` rejects unknown/locked fields and bad enums. Loop 3 swarm: one worker per failure
  hypothesis (barrier clearance, container fit, grasp seating, reach limits) searches inside the dev ranges and
  bisects toward the least extreme failing scenario. Live workers are Antigravity Managed Agents whose remote
  sandbox gets the tracked source files inline (the repo is private and `benchmarks/` must never reach an agent).
- **12:24** Pipeline orchestrator for the policy half, verified offline with the replay backend. Live API wired
  in: key loaded from `.env` (repo or parent folder), per-call `max_output_tokens` caps, `thinking_level` medium by
  default (high only for the Video Architect, critic and hardening, low for dense labels), and the spend guard above.
- **12:30** First live Video Architect run on a real 36 s handheld clip of a dual-arm robot wiping a steel bar with a
  sponge. The QA gate refuses it for the twin pipeline (camera static 13 %, exposure range 46, over 30 s), as
  designed. Run directly, Argon identified a Trossen WidowX AI pair (`trossen_wxai`, confidence 0.9), both objects,
  ten phases (left arm holds the bar, right arm wipes 11.8-23.8 s, places and releases the sponge) and 15 keyframes
  whose boxes, fingertip points and open/held states check out on an overlay. Fixed the architect to send the full
  clip so video time and keyframe timestamps share one time base.

## Lessons (API quirks found today)

- `background=True` requests are served by an `-agent` variant that rejects image/video parts for every model
  ("Image input modality is not enabled for models/<model>-agent"). Media calls run synchronously with a 20-minute
  client timeout; text-only calls stay in background mode. `scripts/probe_models.py` checks this per model.
- Synchronously, gemini-4-argon, gemini-3.8-flash, gemini-3.7-flash, gemini-3.5-flash-lite and
  gemini-robotics-er-2-preview all accept images and agree on a test box within a few units of 1000; Robotics-ER 2
  answered fastest (1.6 s) and is the dense spatial labeller.
- Setting `response_mime_type` without a full `response_format` schema is a 400 in synchronous mode. JSON is
  enforced by the prompt contract + pydantic validation + one re-ask instead.
- Video costs ~1.2k input tokens for a 16 s 640 px clip, an image ~1.1k: a full Video Architect call is cents.

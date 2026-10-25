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
| 2 | Locked evaluator | v2 locked | 11 referee tests incl. wipe coverage; sha256 `d94d76e9…` (v1 `d6e94519…`) |
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
| 13 | Camera solve from the robot silhouette | done | synthetic: IoU 0.962, camera position error 0.52 cm |
| 14 | 4-view diagnostic matrix + Loop 1 | done (analytic), Twin Critic wired | displaced twin 47.5 % → 98.3 % alignment in 1 iteration, xy error ≤ 1.1 mm |
| 15 | Motion extraction + Replay Critic | done | synthetic: carry waypoints 3.0 mm median from ground truth; Draft 1 replays successfully |
| 16 | Live Managed Agents | done | smoke test 34 s; live swarm agent: 24 sandbox rollouts in 10 min, 6 failing scenarios with a mechanism |
| 17 | Studio (Viser) | done | 8-step story mode, live run timeline, 3D replay, live "your turn" runs; Panda meshes welded 397k → 70k vertices |
| 18 | Bimanual Trossen WidowX AI | done | onboarding agent wrote the adapter in its sandbox; both arms pass the self-test |
| 19 | Dual-arm hold + wipe primitives | done | synthetic bar wipe: coverage 100 %, sponge returned, 0 unwanted contacts |
| 20 | Robot identity check by rendered candidates | done | user's clip: Argon corrected "Trossen WidowX AI" → `i2rt_yam` (0.98) |
| 21 | Bimanual i2rt YAM | done | generated with MjSpec.attach; both arms pass the self-test |
| 22 | Moving camera (handheld) | done | synthetic orbit ≤ 0.5 mm; real clip: metric table grid tracks the mat over 3-33 s |
| 23 | Multi-arm motion extraction | done | synthetic labels: hold at -0.158 m (truth -0.16), 4 strokes (truth 4), replay succeeds |
| 24 | NVIDIA SimReady assets as a Scene Author tool | done | search → fetch → USD convert → mesh object; wipe with a SimReady sponge succeeds |

## Agent architecture (video → sim)

Problem statement 2: build a system that renders what it made, watches the result, critiques it against the goal and
fixes it in a loop. Python orchestrates deterministically; every agent is a role prompt + a typed output contract +
a numeric verification signal it is judged by (never another model's opinion).

| Agent | Model | Watches | Produces | Verified by |
|---|---|---|---|---|
| Video Architect | gemini-4-argon, high | full clip + keyframes in one context | robot, objects, phases, keyframes, rest frame | downstream fits |
| Spatial labellers (parallel) | gemini-robotics-er-2-preview | frame batches | per-frame boxes, fingertip points, open/held | track consistency |
| Robot onboarding | Argon / Antigravity agent | Menagerie MJCF summary | RobotAdapter | 5-check self-test |
| Camera solver | numeric (render-and-compare) | robot silhouette at rest | camera pose + fovy, metric scale | silhouette IoU |
| Scene Author | Argon, high | rest frame + masks + back-projected positions | TaskSpec (primitives / composites) | static fidelity |
| **Twin Critic — Loop 1** | Argon, high, multi-turn | **4-view matrix**: ghost overlay, ortho top 5 cm grid, ground view, collision wireframe | ScenePatch | per-object IoU / offset / alignment |
| Replay Critic | Argon, high | real clip vs synced sim replay | waypoint / timing patch | end-pose error, event timing |
| Skill Critic — Loop 2 | Argon, high | failing rollout video + event frames + fact trace | PolicyPatch | dev suite, held-out once |
| Adversarial swarm — Loop 3 | Antigravity Managed Agents (parallel) | perturbation ranges + bench CLI | least-extreme failing scenarios | rollouts in sandbox |
| Hardener | Argon, high | swarm reports | state-conditioned policy | dev + adversarial + held-out |
| Referee | locked evaluator (no model) | final sim state | success predicates | sha256 lock |

The Twin Critic keeps server-side state across iterations (`previous_interaction_id`), so each critique sees what it
already tried; independent critiques (per object / per view / per hypothesis) run in parallel.

## Runaway-loop guard (not a spend limit)

Argon runs at `thinking_level="high"` with the full 65,536-token output window. Live calls go through
`core/argon.py`, which serves identical requests from a local cache and only trips on runaway loops:
`WORLDGYM_MAX_CALLS` (400 live calls per process), `WORLDGYM_MAX_REPEAT` (same request name 4x),
`WORLDGYM_MAX_AGENTS` (16 Managed Agents per process), `WORLDGYM_AGENT_MAX_TOKENS` (2M per agent). Token usage
is logged to `.worldgym/api_ledger.jsonl` (gitignored). `WORLDGYM_LLM=replay` runs from cached responses.

## How to run

```bash
uv run python -m core.pipeline --video demo/synthetic_panda_sort.mp4 --prompt "put the cube and cylinder in the bin"
uv run python -m app.studio --port 8080          # open http://localhost:8080
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
- **12:47** Twin side of the PS2 loop. Composite objects (parts rescaled to `dims_m`) so real items such as a grab
  bar on two blocks can be described. 4-view diagnostic matrix: ghost overlay, ortho top with 5 cm grid and the real
  footprints (silhouette back-projected at the table and at the object top, intersected), ground view with height
  lines, collision wireframe. Loop 1: metrics per spec, analytic footprint patch (xy, yaw, extents) then the Twin
  Critic (Argon, multi-turn) returning a validated ScenePatch. Camera solve from the robot silhouette landed
  (grid + Levenberg-Marquardt + Powell; IoU 0.962, 0.52 cm).
- **13:08** First live end-to-end run on the synthetic clip. Video Architect: Franka Emika Panda, 4 objects,
  13 phases (156 s). Camera solve from Argon's robot box + GrabCut: silhouette IoU 0.98, fovy 48.9° vs 48.0°, position
  error 3.0 cm. Scene Author: every object, role and predicate right (barrier = obstacle, bin = fixed container,
  inside_container x2). Twin Critic iteration 1, reading the views: "the barrier is too thick because its footprint
  included the perspective projection of its height" and corrected it to 2.4 cm (truth 2.0 cm) at 8 mm from the true
  position. Iteration 2 hit a server 400 (malformed JSON); the adapter now retries with the server's message and
  Loop 1 keeps the measured twin if a critique fails. Replay Critic added: real | sim side by side, aligned at the
  first grasp; on the synthetic clip it shows the extracted speed lagging the demonstration.
- **13:08** Live Antigravity swarm agent (barrier clearance, v3 policy): installed the repo in its sandbox, ran 24
  rollouts through `core.bench`, and reported a mechanism the state-conditioned heights miss: grasping the cylinder
  with a fixed 90° yaw turns the ~10 cm hand housing toward the barrier, so with the cylinder within ~4 cm of a
  >= 12 cm barrier the hand strikes its top during descend_grasp (~250 N). 10 min, 1.34M input tokens.
- **13:14** Studio: run-centric Viser app polling `runs/*/manifest.json` every second (stage chips, errors and
  refusals included), 3D mirror of the twin with welded meshes, telemetry replay scrubber, and an 8-step story mode
  (input, twin build with timings, real↔sim matrices per iteration, replay, benchmark, critique with jump-to-time,
  held-out ratchet, live "your turn" runs with obstacle/object sliders). Draft 3 with the barrier raised to 17 cm and
  the cube moved succeeded live with 0 unwanted contacts.
- **13:38** Toward the user's real clip (two Trossen WidowX AI arms wiping a steel bar). The Antigravity onboarding
  agent wrote a single-arm WidowX AI adapter in its sandbox against the self-test (920 s); re-verified locally it only
  passed with the base raised 0.2 m, so the self-test was made fair instead: grid in each arm's own base frame,
  robot-relative grid heights, and opt-in servo damping for Menagerie's undamped servos (kv = 0.1 kp). With the base on
  the table both arms of the bimanual model now pass every check. New primitives: `hold` (grasp and keep holding until
  the other arms finish), `wipe` (press the held wiper onto a target's top and stroke along its long axis, then put it
  back) and `sync_after`. Telemetry attributes every robot contact to the arm that made it (arm-arm collisions too) and
  records wiper-on-target contact points. Evaluator v2 adds `wiped_coverage` (1 cm bins along the target's long axis),
  with tests, and was re-locked (sha256 `d94d76e9…`). The IK null-space pull to home is now applied only to redundant
  (> 6 DOF) arms: on 6-DOF arms it made a holding arm drift off its grasp. An arm-clearance sweep showed wiping the
  bar left of x = 0.05 m makes the two elbows meet, so the synthetic task wipes [0.08, 0.17] m: 100 % coverage, 0
  unwanted contacts.
- **14:05** Real-clip path. Argon's first guess for the user's robot (Trossen WidowX AI, 0.9) was wrong: a new check
  renders 13 Menagerie candidates on one contact sheet and Argon picks by appearance ("QDD actuators with green LEDs,
  white forearm, D405 bracket, triangular truss fingers") → `i2rt_yam`, 0.98. YAM adapter with an IK-found top-down
  home; its soft elbow servo overshoots commanded joint speed by ~35 %, so commanded speed is capped at 70 % of the
  (estimated) 3.14 rad/s limit for this arm, and the self-test judges sustained speed (99.5th percentile). Bimanual
  YAM built reproducibly with `MjSpec.attach` (bases 0.6 m apart, both facing the workspace); both arms pass. The QA
  gate now treats a handheld clip as moving-camera mode instead of refusing it: the camera is solved once from the
  robot silhouette (Robotics-ER box per arm → GrabCut; IoU 0.73 on the user's clip with the bases out of frame) and
  tracked on the table plane (SIFT + RANSAC homographies, sequential with re-anchoring); a 5 cm metric grid
  re-projected through the tracked camera lies on the real mat from 3 s to 33 s. Multi-arm Draft 1: fixed held
  objects become `hold`, push/wipe phases become `wipe` (span and stroke count from the fingertip track along the
  target), other arms sync after the holding arm. NVIDIA SimReady: the public USD Search API (metric bounding boxes
  included) + `nvidia/simready-assets` packages (CC-BY 4.0, attribution recorded), converted with pxr (instanced
  geometry, convex collision meshes, authored mass, grasp points); the Scene Author can call it as a function tool
  (Interactions `requires_action` → `function_result` loop) and sees thumbnails. A SimReady sponge in the bimanual
  wipe: 100 % coverage, 0 unwanted contacts. Incomplete interactions (output budget) now retry with a doubled budget.

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
- A long hardening prompt at `thinking_level="high"` can end `incomplete` (thinking + answer share the output
  window); text-only calls can be retried in background mode with a larger `max_output_tokens`.
- Function calling: the interaction stops with `requires_action` and `function_call` steps; answering with
  `function_result` steps and `previous_interaction_id` continues the same chain. Results may include images.

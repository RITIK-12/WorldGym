"""Loop 2 skill critic: failing rollout video + event keyframes + fact-only trace -> Argon -> PolicyPatch.

Only PolicyConfig fields may change. apply_patch merges, validates with pydantic and rejects unknown fields or enum
values, so a bad patch fails loudly instead of silently doing nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.argon import ARGON, Argon
from core.schemas import FEATURES, PolicyConfig, PolicyPatch, Primitive, RobotAdapter, TaskSpec

LOCKED_FIELDS = {"object_id", "target_id", "arm"}

DSL_DOC = """POLICY DSL (closed; no code). One primitive per object, executed in order:
plan -> reach -> descend_grasp -> close (0.35 s) -> lift_clear -> transit_arc -> lower_place -> release (0.35 s) -> retract

Fields of a primitive (numbers in metres / radians / newtons / m/s):
- grasp_xy_offset_m [dx, dy] in the object frame; grasp_z_depth_m: TCP depth below the object top (default 0.018)
- grasp_yaw_mode: "fixed_world" | "object_relative"; grasp_yaw_rad: closing-axis yaw (world, or relative to object)
- pre_grasp_clearance_m (0.08): hover height above the grasp pose before descending
- pre_grasp_open_width_m (0.08): finger opening while reaching; release_open_width_m: opening at release
- reach_clearance_z_m (null): if set, the reach rises to this z, crosses, then descends (avoids sweeping obstacles)
- grip_force_n (35): squeeze force per finger
- post_grasp_lift_z_m: TCP z after lifting; transit_apex_z_m: TCP z at the top of the transit
- transit_via_mode: "none" (rise, cross at apex, descend) | "midpoint" (arc through the midpoint at apex) |
  "absolute" (arc through transit_via_point_xy_m at apex)
- place_xy_offset_m [dx, dy] in the target frame (target = container/support)
- place_yaw_mode: "keep" (keep TCP yaw) | "fixed_world" | "container_relative": place_yaw_rad is the OBJECT yaw
  (world, or relative to the container)
- pre_place_hover_z_m: TCP z above the target before lowering
- release_drop_height_m: object bottom height above the target floor at release
- retract_z_m: TCP z after releasing; speed_mps (0.35): descend/lower x0.45, lift x0.7

Primitive kinds: "pick_place" (above), "hold" (grasp object_id and keep holding it until every other arm is done;
hold_lift_m lifts it), "wipe" (grasp object_id = the wiper, press it onto target_id's top and stroke along the
target's long axis between wipe_start_m and wipe_end_m (metres from the target centre), wipe_strokes passes at
wipe_speed_mps, wiper bottom wipe_press_m below the surface top (wipe_surface_z_m overrides the top), offset
wipe_lateral_m across the axis; place_back returns the wiper to where it was picked). sync_after = an arm name: start
only once that arm is holding (or idle). Two arms colliding is reported as ARM_ARM_COLLISION.

Any numeric field may instead be a state-conditioned value {"terms": [feature, ...], "offset": x} = sum of the
features measured when the action starts + offset. Features: """ + ", ".join(FEATURES) + """.
- obj_top_z / obj_height: the primitive's object; held_below_tcp: how far the held object hangs below the TCP
  (measured at grasp); max_obstacle_top_z: tallest obstacle top; max_scene_top_z: tallest top of every other object
  (obstacles, containers, other objects); target_rim_z / target_floor_z: the target's rim / floor-top z; table_z = 0.
Release z = target_floor_z + release_drop_height_m + held_below_tcp.
Carry heights must account for the object hanging below the TCP."""

SYSTEM = """You are the skill critic of WorldGym. You watch a simulated robot rollout (video + event keyframes) and read a
fact-only trace of contacts and predicate outcomes, then propose the smallest set of policy parameter changes
that fixes the failures AND generalises across scene perturbations (object poses and sizes, obstacle heights,
container sizes, friction). Prefer state-conditioned values over absolute numbers when a failure depends on scene
geometry. Cite evidence with timestamps (seconds) and frame indices from the trace/video. Only fields of the
policy DSL may change; never change object_id, target_id or arm.

Output: ONE JSON object
{"from_iteration": int,
 "critique": [{"timestamp_sec": float, "video_frame_idx": int, "observation": str}],
 "root_causes": [str],
 "patch": [{"object_id": "<id or *>", "set": {"<field>": <number | {"terms": [...], "offset": x} | enum>}}]}"""


class PatchError(ValueError):
    pass


def robot_facts(adapter: RobotAdapter) -> str:
    a = adapter.arms[0]
    g = a.gripper
    return (f"Robot {adapter.robot_id}: TCP = midpoint of the fingertip pads ({a.tcp.body} + {a.tcp.pos}); fingertips "
            f"reach ~0.008 m below the TCP; fingers close along the TCP {a.closing_axis} axis; max opening "
            f"{g.max_width_m} m; the finger envelope is about opening + 0.04 m wide; grasps are top-down.")


def apply_patch(policy: PolicyConfig, patch: PolicyPatch | dict, name: str | None = None) -> PolicyConfig:
    entries = patch.patch if isinstance(patch, PolicyPatch) else patch["patch"]
    data = policy.model_dump()
    allowed = set(Primitive.model_fields) - LOCKED_FIELDS
    errors = []
    for e in entries:
        sel = e.get("object_id", "*")
        arm = e.get("arm")
        fields = e.get("set", {})
        bad = sorted(set(fields) - allowed)
        if bad:
            errors.append(f"unknown or locked fields {bad} for {sel}")
            continue
        hits = [p for p in data["primitives"] if (sel in ("*", None) or p["object_id"] == sel)
                and (arm is None or p["arm"] == arm)]
        if not hits:
            errors.append(f"no primitive matches object_id={sel!r} arm={arm!r}")
        for p in hits:
            p.update(fields)
    if errors:
        raise PatchError("; ".join(errors))
    data["name"] = name or f"{policy.name}+patch"
    try:
        return PolicyConfig.model_validate(data)
    except Exception as ex:  # noqa: BLE001 - surface enum / type errors as patch errors
        raise PatchError(str(ex)) from ex


def _trace_for_prompt(summary: dict) -> dict:
    tr = summary["trace"]
    return {"success": summary["success"], "predicates": summary["evaluation"]["predicates"],
            "metrics": tr["metrics"], "events": tr["events"][:60], "phase_timeline": tr["phase_timeline"],
            "timed_out": tr.get("timed_out", False)}


def critique(task: TaskSpec, policy: PolicyConfig, adapter: RobotAdapter, nominal: dict, dev_results: list[dict],
             argon: Argon, iteration: int, attachments: list[Path], out_dir: Path) -> tuple[PolicyPatch, PolicyConfig]:
    """nominal = rollout summary with video; dev_results = per-scenario dev outcomes (never held-out)."""
    failing = [r for r in dev_results if not r["success"]]
    dev_view = [{"scenario_id": r["scenario_id"], "labels": r["labels"],
                 "failed_predicates": [p for p in r["predicates"] if not p["ok"]],
                 "events": [e for e in r["events"] if e["type"] not in ("OBJECT_RELEASED",)][:12],
                 "resolved": r["resolved"]} for r in failing[:6]]
    prompt = f"""{robot_facts(adapter)}

{DSL_DOC}

TASK (TaskSpec):
{task.model_dump_json(indent=1)}

CURRENT POLICY (iteration {iteration}):
{policy.model_dump_json(indent=1)}

RESOLVED NUMERIC VALUES at execution time (nominal scene):
{json.dumps(nominal["resolved"], indent=1)}

NOMINAL ROLLOUT TRACE (facts only; the attached video is this rollout):
{json.dumps(_trace_for_prompt(nominal), indent=1)}

DEV SUITE: {sum(r["success"] for r in dev_results)}/{len(dev_results)} succeeded. Failing scenarios (overrides are
applied to the TaskSpec objects):
{json.dumps(dev_view, indent=1)}

Attachments: the nominal rollout video (left = perspective, right = side clearance camera; red border = obstacle
contact) and keyframes named by frame index at the critical events.

Write the critique and a patch (from_iteration = {iteration})."""
    patch, _ = argon.ask_json(f"critic_iter{iteration}", SYSTEM, prompt, PolicyPatch, attachments, model=ARGON,
                              thinking="high")
    try:
        new = apply_patch(policy, patch, name=f"draft{iteration + 1}")
    except PatchError as e:
        fix_prompt = prompt + f"\n\nYour previous patch was rejected: {e}\nReturn a corrected patch."
        patch, _ = argon.ask_json(f"critic_iter{iteration}_retry", SYSTEM, fix_prompt, PolicyPatch, attachments,
                                  model=ARGON, thinking="high")
        new = apply_patch(policy, patch, name=f"draft{iteration + 1}")
    (out_dir / f"patch_iter{iteration}.json").write_text(patch.model_dump_json(indent=1))
    (out_dir / f"policy_draft{iteration + 1}.json").write_text(new.model_dump_json(indent=1))
    return patch, new


def param_diff(old: PolicyConfig, new: PolicyConfig) -> list[dict]:
    rows = []
    for a, b in zip(old.primitives, new.primitives):
        da, db = a.model_dump(), b.model_dump()
        for k in da:
            if da[k] != db[k]:
                rows.append({"object_id": a.object_id, "field": k, "old": da[k], "new": db[k]})
    return rows


SEVERITY = {"ILLEGAL_OBSTACLE_COLLISION": 9, "OBJECT_ON_FLOOR": 8, "GRASP_MISSED": 8, "OBJECT_LOST_FROM_GRASP": 8,
            "MOTION_BLOCKED": 7, "CONTAINER_RIM_STRIKE": 7, "ROBOT_DISTURBED_OTHER_OBJECT": 6,
            "HAND_BODY_OBJECT_CONTACT": 5, "OBJECT_OBJECT_COLLISION": 5, "ROBOT_CONTAINER_CONTACT": 5,
            "CONTAINER_WALL_CONTACT_WHILE_HELD": 4, "GRASP_SLIP": 4, "ROBOT_TABLE_COLLISION": 4,
            "IK_TRACKING_ERROR": 3, "OBJECT_RELEASED": 1}


def event_keyframes(video: str | Path, events: list[dict], out_dir: str | Path, max_n: int = 8) -> list[Path]:
    """Frames of the most severe events (Gemini samples video sparsely, so these go alongside the mp4)."""
    from core.video_probe import extract_frames

    ranked = sorted(events, key=lambda e: (-SEVERITY.get(e["type"], 0), -(e.get("peak_force_n") or 0), e["t"]))
    times = []
    for e in ranked:
        t = e.get("t_peak", e["t"])
        if all(abs(t - u) > 0.15 for u in times):
            times.append(t)
        if len(times) == max_n:
            break
    return extract_frames(video, sorted(times), out_dir, prefix="event")

"""Video Architect: one Argon call (plus SimReady search tool turns) on the clip, keyframes and rendered robot models.

It returns the robot (picked against rendered candidates), every object with what it is, its size, position, long
axis, behaviour and SimReady asset, one motion entry per arm (kind, contact, start and end along the target,
passes, direction) and the timed phases. Nothing in the prompt is specific to a task: the video decides. Gemini
Robotics-ER 2 subagents then label frames densely, in parallel batches, for tracking and motion extraction.

Coordinates follow the Gemini convention: boxes [ymin, xmin, ymax, xmax] and points [y, x], normalised 0..1000.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from pydantic import BaseModel

from core.argon import ARGON, ROBOTICS_ER, STATE as STATE_DIR, Argon, run_parallel
from core.mjcf_compiler import MENAGERIE
from core.schemas import Keyframe, VideoUnderstanding

SHAPES = ("box, cylinder, sphere, capsule, tblock (T-shaped block), container (open bin/box/tray), "
          "rail (a bar or handle: a horizontal rod on two end posts)")

SYSTEM = """You are the Video Architect of WorldGym. WorldGym turns one video of a robot doing a manipulation task
into a MuJoCo simulation that repeats the same action. Any robot, any objects, any task: the video decides what is
there and what happens, never the prompt and never an assumption about the task type. You watch the whole clip
before you describe anything, you report only what is physically visible, and you measure sizes and positions
against the robot itself. When unsure, give your best estimate and lower the confidence; never refuse."""

CONVENTIONS = """Conventions:
- Boxes: [ymin, xmin, ymax, xmax] normalised to 0..1000 of the keyframe image. Points: [y, x] normalised 0..1000.
- Times are seconds from the start of the video.
- Robot frame, in metres: origin at the robot base on the table (for two arms, the midpoint between the two arm
  bases), +x = forward from the base(s) across the workspace, +y = the robot's left, +z up. "Left end" and
  "right end" always mean the robot's left and right.
- camera_guess = {"azimuth_deg": direction from the robot base to the camera, measured around +z from +x
  (0 = camera in front of the robot looking back at it, 90 = camera on the robot's left, -90 = on its right),
  "elevation_deg": angle above the table plane, "distance_m": distance from the robot base}."""


class KeyframeBatch(BaseModel):
    keyframes: list[Keyframe]


def menagerie_models() -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(MENAGERIE), "ls-tree", "-d", "--name-only", "HEAD"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return [d for d in out if not d.startswith(".") and d not in ("assets", "python", "test")]
    except Exception:  # noqa: BLE001 - listing is a hint only
        return []


def downscale(video: str | Path, out: str | Path, width: int = 640, max_s: float = 30.0,
              trim: list[float] | None = None) -> Path:
    """Small H.264 copy for the model (keeps inline uploads small)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    if trim:
        cmd += ["-ss", f"{trim[0]:.2f}", "-t", f"{trim[1] - trim[0]:.2f}"]
    cmd += ["-i", str(video), "-t", f"{max_s:.1f}", "-vf", f"scale={width}:-2", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True, timeout=120)
    return out


def _kf_list(paths: list[Path], times: list[float]) -> str:
    return "\n".join(f"- {p.name}  t={t:.2f} s" for p, t in zip(paths, times))


def understand(video: str | Path, probe: dict, prompt: str, argon: Argon, out_dir: str | Path,
               n_extra: int = 6, review: bool = False) -> VideoUnderstanding:
    from core.video_probe import extract_frames

    out_dir = Path(out_dir)
    dur = float(probe.get("duration_s") or 0)
    times = sorted(set(round(t, 2) for t in probe.get("keyframes_t", [])))
    if dur > 0:
        times = sorted(set(times) | {round(dur * (k + 0.5) / n_extra, 2) for k in range(n_extra)})
    times = _dedupe(times, 0.4)[:16]
    frames = extract_frames(video, times, out_dir / "keyframes", prefix="kf")
    small = downscale(video, out_dir / "clip_small.mp4", max_s=max(dur + 1.0, 30.0))  # same time base as keyframes
    from core.robot_onboard import ARM_CANDIDATES, render_candidates

    sheet = STATE_DIR / "robot_candidates.jpg"     # deterministic renders, shared by every run
    if not sheet.exists():
        sheet.parent.mkdir(parents=True, exist_ok=True)
        render_candidates(ARM_CANDIDATES, sheet)
    text = f"""User prompt: "{prompt}"

Attachments: the video (clip_small.mp4, {dur:.1f} s, original {probe.get('width')}x{probe.get('height')}).{_trim_note(probe)}
then keyframes as JPEGs:
{_kf_list(frames, times)}
then robot_candidates.jpg: rendered MuJoCo robot models, each labelled with its directory name, two views each.

Object types: {SHAPES}.

{CONVENTIONS}

Work in this order, then return ONE JSON object.

1. Watch the whole clip. Find the phases and the moments of contact: which arm touches what, with which part of
   its gripper, and what the other arm is doing at that time.

2. robot: make_model_guess, arms (1 or 2), gripper type, confidence 0..1, and menagerie_dir: compare the real arm
   with robot_candidates.jpg by the whole arm (link shapes and colours, proportions, joint layout, gripper), not
   by a wrist close-up. menagerie_dir must be one of: {', '.join(ARM_CANDIDATES)}.

3. objects: every object the robot touches, plus anything that holds, supports or blocks them (containers,
   fixtures, stands, obstacles). Not the table or mat itself. For each:
   - id: short snake_case, colour + noun (e.g. red_mug, blue_cube, wooden_board). description: what it is.
   - type from the object types; colour: one plain colour word; behaviour: rigid, articulated or deformable.
   - size_guess_m: full extents [x, y, z] in the robot frame (cylinder: [diameter, diameter, height]). A size is
     a claim: measure it against the gripper and the arm links in the frames, and check it from more than one
     keyframe because a single view hides depth.
   - offset_m: [x, y] of its centre at the start; end_offset_m: where it is at the end, if it moved.
   - long_axis: "x" (long side points forward/back), "y" (long side runs left-right) or "none".
   - fixed: true if it never moves (clamped, mounted, too heavy to move).
   - placement: one sentence on where it is relative to the arms and the other objects.
   - role: what it does in the task, in a few words.
   - asset: call search_simready_assets with what the object is (its description, not its colour) for every
     object that is a real product; you may search for several objects at once. Each result has a metric
     bounding box and a thumbnail: pick the candidate whose kind and proportions match the object and set its
     name, or null when nothing fits (custom fixtures, plain blocks, rods, parts of the rig).

4. motions: one entry per arm for the whole clip, the structure the simulation must reproduce:
   - kind: what the arm does, e.g. hold, pick_place, wipe, push, pour, insert, open, close, press, handover,
     idle (use another single word if none fits).
   - object: what is in its gripper; target: what it acts on or puts it onto; t0, t1.
   - contact: which part of the gripper touches which surface of the object.
   - for motions along a target's surface: start_frac and end_frac = where the contact starts and ends along the
     target's long side (-1 = its right end, +1 = its left end); strokes = number of passes; pattern = "one_way"
     when every pass goes the same direction, "back_and_forth" when it alternates. Watch the direction of every
     pass; do not assume.
   - place_back: whether the object ends where it was picked up; after: what happens to it at the end.
   - summary: one sentence.

5. phases: contiguous phases covering the clip with t0, t1, action (a short verb such as reach, grasp, lift,
   transit, place, release, retract, push, wipe, pour, insert, open, hold, idle), arm ("arm" for one arm,
   "left"/"right" for two), object, target.

6. keyframes: exactly one entry per keyframe with its exact t: robot_box and object_boxes for every visible
   object id (leave gripper empty; a labelling swarm does that frame by frame).

7. rest_frame_t: a keyframe time where the robot is at rest and fully visible. success_intent: one sentence
   comparing the first state and the last state. camera_guess."""
    try:   # the architect picks SimReady assets itself while it watches the clip
        from core.argon import ask_with_tools
        from core.assets import TOOL, run_tool

        vu, resp = ask_with_tools(argon, "video_architect", SYSTEM, text, VideoUnderstanding,
                                  {TOOL["name"]: (TOOL, run_tool)}, [small, *frames, sheet], thinking="medium",
                                  max_rounds=4)
    except Exception:  # noqa: BLE001 - the twin needs the understanding more than the asset choice
        vu, resp = argon.ask_json("video_architect", SYSTEM, text, VideoUnderstanding, [small, *frames, sheet],
                                  model=ARGON, thinking="medium")
    snap = {round(t, 2) for t in times}
    for kf in vu.keyframes:
        kf.t = min(snap, key=lambda s: abs(s - kf.t))
    if review:
        vu = review_labels(video, vu, argon, out_dir / "review", resp.get("interaction_id"))
    return vu


REVIEW_SYSTEM = SYSTEM + """
You are now reviewing your own labels. Each attached image is a keyframe with your labels drawn on it: object boxes
(colour per id, labelled), the robot box (white), and the gripper point (circle: red = closed, green = open, with the
held object). Fix every box that is not tight on its object, every gripper point not at the fingertip midpoint, and
every wrong open/held state. Return the full corrected VideoUnderstanding (same ids, same keyframe times)."""


def draw_labels(frame, kf: Keyframe) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    rng = np.random.default_rng(7)
    cols = {}

    def box(b, col, label):
        y0, x0, y1, x1 = b
        p0, p1 = (int(x0 * w / 1000), int(y0 * h / 1000)), (int(x1 * w / 1000), int(y1 * h / 1000))
        cv2.rectangle(img, p0, p1, col, 2)
        cv2.putText(img, label, (p0[0], max(14, p0[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    if kf.robot_box:
        box(kf.robot_box, (255, 255, 255), "robot")
    for oid, b in kf.object_boxes.items():
        cols.setdefault(oid, tuple(int(c) for c in rng.integers(60, 255, 3)))
        box(b, cols[oid], oid)
    for g in kf.gripper:
        if g.uv:
            u, v = int(g.uv[1] * w / 1000), int(g.uv[0] * h / 1000)
            col = (0, 255, 0) if g.open else (0, 0, 255)
            cv2.circle(img, (u, v), 10, col, 3)
            cv2.putText(img, f"{g.arm} {'open' if g.open else 'closed'}{' ' + g.held if g.held else ''}", (u + 14, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    cv2.putText(img, f"t={kf.t:.2f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def review_labels(video: str | Path, vu: VideoUnderstanding, argon: Argon, out_dir: Path,
                  previous_id: str | None = None) -> VideoUnderstanding:
    """PS2 loop on perception: render the labels onto the keyframes, let Argon check and correct them."""
    from core.video_probe import read_frame

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kf in vu.keyframes:
        p = out_dir / f"labels_{kf.t:06.2f}.jpg"
        cv2.imwrite(str(p), draw_labels(read_frame(video, kf.t), kf))
        paths.append(p)
    text = f"""Your labels, drawn on {len(paths)} keyframes (file names carry the time). Current JSON:
{vu.model_dump_json()}

{CONVENTIONS}

Return the corrected VideoUnderstanding."""
    try:
        fixed, _ = argon.ask_json("label_review", REVIEW_SYSTEM, text, VideoUnderstanding, paths, model=ARGON,
                                  thinking="high", previous_id=previous_id)
    except Exception:  # noqa: BLE001 - keep the first-pass labels if the review fails
        return vu
    if {o.id for o in fixed.objects} != {o.id for o in vu.objects} or len(fixed.keyframes) != len(vu.keyframes):
        return vu
    for a, b in zip(fixed.keyframes, vu.keyframes):
        a.t = b.t
    return fixed


def dense_labels(video: str | Path, times: list[float], vu: VideoUnderstanding, argon: Argon,
                 out_dir: str | Path, batch: int = 8, model: str = ROBOTICS_ER) -> list[Keyframe]:
    """Parallel spatial-labelling subagents: gripper point/open/held + object boxes on extra frames."""
    from core.video_probe import extract_frames

    out_dir = Path(out_dir)
    ids = [o.id for o in vu.objects]
    arms = sorted({p.arm for p in vu.phases} or {"arm"})

    def job(b: int):
        ts = times[b:b + batch]
        frames = extract_frames(video, ts, out_dir / f"dense_{b // batch:02d}", prefix="f")
        text = f"""Robot: {vu.robot.make_model_guess} ({vu.robot.arms} arm(s): {', '.join(arms)}). Objects: {', '.join(ids)}.
Task: {vu.success_intent}
Frames attached:
{_kf_list(frames, ts)}

{CONVENTIONS}

For EVERY frame return a keyframe entry with its exact t: robot_box, object_boxes for each visible object id
(use only these ids), and gripper = [{{arm, uv, open, held}}] per visible arm, where uv is the fingertip midpoint,
open is whether the fingers are open, and held is the id of the object between the fingers (else null)."""
        out, _ = argon.ask_json(f"dense_labels_{b // batch:02d}", SYSTEM, text, KeyframeBatch, frames, model=model,
                                thinking="high", max_output_tokens=32768)
        for kf in out.keyframes:
            kf.t = min(ts, key=lambda s: abs(s - kf.t))
        return out.keyframes

    results = run_parallel([lambda b=b: job(b) for b in range(0, len(times), batch)])
    labels: list[Keyframe] = []
    for r in results:
        if isinstance(r, Exception):
            raise r
        labels += r
    return sorted(labels, key=lambda k: k.t)


def _trim_note(probe: dict) -> str:
    t = probe.get("trim")
    return f" Most of the task motion happens within [{t[0]:.1f}, {t[1]:.1f}] s." if t else ""


def _dedupe(times: list[float], gap: float) -> list[float]:
    out: list[float] = []
    for t in sorted(times):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out

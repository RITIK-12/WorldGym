"""Video Architect: one Argon multimodal call on the clip + keyframes -> VideoUnderstanding.

Argon identifies the robot (closest Menagerie model), the objects (shape vocabulary + full-extent size guesses),
task phases, per-keyframe robot/object boxes and the gripper point/open/held state, a rest frame for the camera
solve and a verbal camera estimate. A review pass draws those labels onto the keyframes and Argon corrects its own
mistakes (the render-watch-fix loop applied to perception). Gemini Robotics-ER 2 subagents (spatial pointing specialists) then label extra
frames densely, in parallel batches, for tracking and motion extraction.

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

SHAPES = "box, cylinder, sphere, capsule, tblock (T-shaped block), container (open bin/box/tray)"

SYSTEM = """You are the Video Architect of WorldGym, a system that turns one video of a robot doing a tabletop
manipulation task into a physics twin in MuJoCo. You watch the video and keyframes carefully and report what is
physically there, as structured data. Be precise about geometry: boxes must tightly enclose the visible pixels of
each item in that exact keyframe. Never invent objects that are not visible. If unsure, give your best estimate and
lower the confidence; never refuse."""

CONVENTIONS = """Conventions:
- Boxes: [ymin, xmin, ymax, xmax] normalised to 0..1000 of the keyframe image. Points: [y, x] normalised 0..1000.
- The gripper point `uv` is the midpoint between the fingertips (the grasp centre), even when closed.
- Times are seconds from the start of the video.
- Robot frame: origin at the robot base on the table, +x = the robot's forward direction (from its base toward the
  workspace), +y = the robot's left, +z up.
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
    models = menagerie_models()
    from core.robot_onboard import ARM_CANDIDATES, render_candidates

    sheet = STATE_DIR / "robot_candidates.jpg"     # deterministic renders, shared by every run
    if not sheet.exists():
        sheet.parent.mkdir(parents=True, exist_ok=True)
        render_candidates(ARM_CANDIDATES, sheet)
    text = f"""TASK PROMPT FROM THE USER: "{prompt}"

The video is attached (clip_small.mp4, {dur:.1f} s, original {probe.get('width')}x{probe.get('height')}).{_trim_note(probe)}
Keyframes are attached as JPEGs with these timestamps:
{_kf_list(frames, times)}
robot_candidates.jpg (the last attachment) shows rendered robot models for the robot identification.

Robot models available (MuJoCo Menagerie directories): {', '.join(models)}
Object shape vocabulary: {SHAPES}.

{CONVENTIONS}

Produce:
- robot: make_model_guess, arms (1 or 2), gripper type, confidence 0..1, and menagerie_dir chosen by comparing
  the real arm with robot_candidates.jpg (rendered MuJoCo models, each labelled with its directory, two views).
  Judge the whole arm: link shapes and colours (white/grey shells vs all-black links), proportions, joint layout,
  gripper design; wrist close-ups alone are ambiguous. menagerie_dir must be one of: {', '.join(ARM_CANDIDATES)}.
- objects: every object the robot manipulates plus containers, fixtures and obstacles on the table (not the table
  itself). Miss nothing the robot touches. id = short snake_case with colour (e.g. red_cube, white_sponge,
  silver_bar). type from the vocabulary. colour = a plain colour word. size_guess_m = FULL extents [x, y, z] in the
  robot frame, in metres (cylinder: [diameter, diameter, height]); use the robot's known size as the scale reference.
  Layout, in the robot frame (for two arms the origin is the midpoint between the two arm bases):
  offset_m = [x, y] of the object's centre at the start of the video; long_axis = "x" if its long side points
  forward/back, "y" if it runs left-right (e.g. parallel to the line between two arm bases), "none" if it has no
  long side; fixed = true when it is clamped or never moves; placement = one sentence of where it is relative to
  the arms and the other objects; role = what it does in the task (e.g. "wiping tool held by the right arm",
  "surface being wiped, held by the left arm").
  asset: call search_simready_assets for each real product (sponge, cup, bottle, can, tool, box...) and set the name
  of the candidate whose shape and size match; null when nothing matches (bars, custom fixtures, plain blocks).
- phases: contiguous phases covering the task, with t0, t1, action (one of reach, grasp, lift, transit, place,
  release, retract, push, idle), arm ("arm" for single-arm robots, "left"/"right" for dual-arm), object, target.
- keyframes: exactly one entry per attached keyframe, using its exact t: robot_box, object_boxes for every visible
  object id, gripper = [{{arm, uv, open, held}}] (held = object id or null).
- rest_frame_t: one of the keyframe times where the robot is at a rest/home pose and fully visible.
- success_intent: one sentence of what success means for this task.
- camera_guess."""
    try:   # the architect picks SimReady assets itself while it watches the clip
        from core.argon import ask_with_tools
        from core.assets import TOOL, run_tool

        vu, resp = ask_with_tools(argon, "video_architect", SYSTEM, text, VideoUnderstanding,
                                  {TOOL["name"]: (TOOL, run_tool)}, [small, *frames, sheet], max_rounds=4)
    except Exception:  # noqa: BLE001 - the twin needs the understanding more than the asset choice
        vu, resp = argon.ask_json("video_architect", SYSTEM, text, VideoUnderstanding, [small, *frames, sheet],
                                  model=ARGON, thinking="high")
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

"""Video Architect: one Argon multimodal call on the clip + keyframes -> VideoUnderstanding.

Argon identifies the robot (closest Menagerie model), the objects (shape vocabulary + full-extent size guesses),
task phases, per-keyframe robot/object boxes and the gripper point/open/held state, a rest frame for the camera
solve and a verbal camera estimate. Argon at low thinking then labels extra frames densely for motion extraction
(Flash does not accept image input on this account, so it is kept for text-only checks).

Coordinates follow the Gemini convention: boxes [ymin, xmin, ymax, xmax] and points [y, x], normalised 0..1000.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
from pydantic import BaseModel

from core.argon import ARGON, Argon
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
               n_extra: int = 6) -> VideoUnderstanding:
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
    text = f"""TASK PROMPT FROM THE USER: "{prompt}"

The video is attached (clip_small.mp4, {dur:.1f} s, original {probe.get('width')}x{probe.get('height')}).{_trim_note(probe)}
Keyframes are attached as JPEGs with these timestamps:
{_kf_list(frames, times)}

Robot models available (MuJoCo Menagerie directories): {', '.join(models)}
Object shape vocabulary: {SHAPES}.

{CONVENTIONS}

Produce:
- robot: make_model_guess, arms (1 or 2), gripper type, menagerie_dir (closest match from the list), confidence 0..1.
- objects: every object the robot manipulates plus containers and obstacles on the table (not the table itself).
  id = short snake_case with colour (e.g. red_cube, blue_cylinder, grey_bin, yellow_barrier). type from the
  vocabulary. colour = a plain colour word. size_guess_m = FULL extents [x, y, z] in metres
  (cylinder: [diameter, diameter, height]); use the robot's known size as the scale reference.
- phases: contiguous phases covering the task, with t0, t1, action (one of reach, grasp, lift, transit, place,
  release, retract, push, idle), arm ("arm" for single-arm robots, "left"/"right" for dual-arm), object, target.
- keyframes: exactly one entry per attached keyframe, using its exact t: robot_box, object_boxes for every visible
  object id, gripper = [{{arm, uv, open, held}}] (held = object id or null).
- rest_frame_t: one of the keyframe times where the robot is at a rest/home pose and fully visible.
- success_intent: one sentence of what success means for this task.
- camera_guess."""
    vu, _ = argon.ask_json("video_architect", SYSTEM, text, VideoUnderstanding, [small, *frames], model=ARGON,
                           thinking="high")
    snap = {round(t, 2) for t in times}
    for kf in vu.keyframes:
        kf.t = min(snap, key=lambda s: abs(s - kf.t))
    return vu


def dense_labels(video: str | Path, times: list[float], vu: VideoUnderstanding, argon: Argon,
                 out_dir: str | Path, batch: int = 8) -> list[Keyframe]:
    """Argon (low thinking) labels gripper point/open/held + object boxes on extra frames for motion extraction."""
    from core.video_probe import extract_frames

    out_dir = Path(out_dir)
    ids = [o.id for o in vu.objects]
    labels: list[Keyframe] = []
    for b in range(0, len(times), batch):
        ts = times[b:b + batch]
        frames = extract_frames(video, ts, out_dir / f"dense_{b // batch:02d}", prefix="f")
        text = f"""Robot: {vu.robot.make_model_guess} ({vu.robot.arms} arm(s)). Objects: {', '.join(ids)}.
Task: {vu.success_intent}
Frames attached:
{_kf_list(frames, ts)}

{CONVENTIONS}

For EVERY frame return a keyframe entry with its exact t: robot_box, object_boxes for each visible object id
(use only these ids), and gripper = [{{arm, uv, open, held}}] where uv is the fingertip midpoint, open is whether
the fingers are open, and held is the id of the object currently held between the fingers (else null)."""
        out, _ = argon.ask_json(f"dense_labels_{b // batch:02d}", SYSTEM, text, KeyframeBatch, frames, model=ARGON,
                                thinking="low", max_output_tokens=16384)
        snap = list(ts)
        for kf in out.keyframes:
            kf.t = min(snap, key=lambda s: abs(s - kf.t))
        labels += out.keyframes
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

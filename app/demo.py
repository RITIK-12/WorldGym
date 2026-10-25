"""WorldGym demo: upload a robot clip, press Convert, watch the MuJoCo world appear and run it.

    .venv/bin/python -m app.demo [--port 8080] [--video clip.mp4] [--run runs/<id>] [--cached]

Sidebar: upload, a looping preview of the clip, Convert to sim, a four-step checklist with a progress bar and one
live status line. Convert starts `core.pipeline --twin-only` with Argon's response cache off (WORLDGYM_CACHE=0)
unless --cached. Results arrive as generations (runs/<id>/gens), each a world plus the demonstrated motion:
Gen 1 from Argon's understanding of the clip, Gen 2 with measured positions and the labelling swarm's
motion, Gen 3 after Argon critiques the original against four views of the Gen 2 run. "Run simulation" plays
the selected generation's policy live in MuJoCo physics, in real time.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import viser

import core.runs as runs
from app.scene import SceneMirror
from core.first_world import robot_frame
from core.mjcf_compiler import FLOOR_Z, load_adapter
from core.rollout import build_sim, run_episode
from core.schemas import PolicyConfig, TaskSpec

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / ".worldgym"
PROMPT = "Reconstruct the demonstrated task as a physics simulation."
BLUE, INK, INK2, OK, BAD = "#3987e5", "#e8e8ea", "#9a9aa2", "#3fb950", "#f85149"
SPIN = ('<style>@keyframes wgspin{to{transform:rotate(360deg)}}</style><span style="display:inline-block;width:11px;'
        f'height:11px;border:2px solid {BLUE};border-top-color:transparent;border-radius:50%;'
        'animation:wgspin .9s linear infinite"></span>')

# (checklist label, milestones); a milestone is a pipeline stage or a published generation
STEPS = [("Video understood", ("qa", "video_architect", "robot_onboard")),
         ("Assets matched", ("first_world", "gen1")),
         ("World constructed", ("camera_solve", "motion_extract", "gen2")),
         ("Refining simulation", ("final_critic", "gen3"))]
ORDER = [m for _, ms in STEPS for m in ms]
GEN_STAGE = {1: "first_world", 2: "motion_extract", 3: "final_critic"}   # each generation is published in its stage
DOING = {"qa": "Checking the clip",
         "video_architect": "Argon is watching the video: objects, layout, actions, SimReady assets",
         "robot_onboard": "Identifying the robot by comparing rendered candidates",
         "first_world": "Building Gen 1 from Argon's understanding",
         "gen1": "Building Gen 1 from Argon's understanding",
         "camera_solve": "Calibrating the camera from the robot's silhouette",
         "motion_extract": "Measuring the objects and extracting the motion from the labelling swarm",
         "gen2": "Measuring the objects and extracting the motion from the labelling swarm",
         "final_critic": "Argon is watching the original and four views of the simulation",
         "gen3": "Argon is watching the original and four views of the simulation"}


def preview_html(video: Path) -> str:
    """A small H.264 copy of the clip, embedded so every viewer sees the input."""
    out = WORK / "previews" / f"{video.stem}_{int(video.stat().st_mtime)}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", "scale=480:-2", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "30", "-an", "-movflags", "+faststart", str(out)], check=True)
    b64 = base64.b64encode(out.read_bytes()).decode()
    wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip().strip(",").split(",")
    ar = f"{wh[0]} / {wh[1]}" if len(wh) == 2 else "16 / 9"
    # the whole frame at its own aspect ratio: never cropped or zoomed
    return (f'<video autoplay muted loop playsinline controls style="width:100%;height:auto;aspect-ratio:{ar};'
            f'object-fit:contain;border-radius:10px;display:block;background:#000" '
            f'src="data:video/mp4;base64,{b64}"></video>'
            f'<div style="font-size:12px;color:{INK2};padding:4px 2px 0">{html.escape(video.name)}</div>')


class Demo:
    def __init__(self, server: viser.ViserServer, cached: bool):
        self.server, self.gui, self.cached = server, server.gui, cached
        self.mirror = SceneMirror(server)
        self.lock = threading.RLock()
        self.video: Path | None = None
        self.proc: subprocess.Popen | None = None
        self.launched = 0.0
        self.name = ""
        self.run_dir: Path | None = None
        self.shown = ""            # what the 3D view shows (robot / gen key)
        self.running = False       # a live physics run is streaming
        self.follow = True         # jump to each new generation
        self.result = ""
        self._build()

    # ------------------------------------------------------------------------------------------------------------------
    def _build(self) -> None:
        gui = self.gui
        gui.configure_theme(dark_mode=True, control_width="large", show_logo=False, show_share_button=False,
                            brand_color=(57, 135, 229))
        gui.main_panel.dock_left()
        gui.main_panel.set_width(420)
        gui.set_panel_label("Create your world")
        gui.add_html(f'<div style="display:flex;justify-content:space-between;align-items:baseline;padding:2px 0 8px">'
                     f'<span style="font-size:20px;font-weight:700;color:{INK}">WorldGym</span>'
                     f'<span style="font-size:12px;color:{INK2}">Video → Physics World</span></div>')
        self.upload = gui.add_upload_button("Upload video", icon=viser.Icon.UPLOAD, mime_type="video/*")
        self.preview = gui.add_html("", visible=False)
        self.go = gui.add_button("Convert to sim", icon=viser.Icon.WAND, color="blue", disabled=True)
        self.steps = gui.add_html("")
        self.bar = gui.add_progress_bar(0.0, animated=False)
        self.pct = gui.add_html("")
        self.live = gui.add_html("")
        self.gen_dd = gui.add_dropdown("Generation", ("—",), visible=False)
        self.run_btn = gui.add_button("Run simulation", icon=viser.Icon.PLAYER_PLAY, color="green", visible=False)
        self.note = gui.add_html("")
        self.server.scene.add_grid("/floor", width=3.0, height=3.0, position=(0.0, 0.2, FLOOR_Z), cell_size=0.1,
                                   section_size=0.5, cell_color=(55, 55, 58), section_color=(85, 85, 90))
        self.upload.on_upload(lambda e: self.on_upload(e.target.value))
        self.go.on_click(lambda _: self.convert())
        self.gen_dd.on_update(lambda _: self._pick_gen())
        self.run_btn.on_click(lambda _: self.run_sim())
        self.refresh()

    # ------------------------------------------------------------------------------------------------------------------
    def on_upload(self, f) -> None:
        d = WORK / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        p = d / Path(f.name).name
        p.write_bytes(f.content)
        self.attach(p)

    def attach(self, p: Path) -> None:
        with self.lock:
            self.video = p
            try:
                self.preview.content = preview_html(p)
                self.preview.visible = True
            except Exception as e:  # noqa: BLE001 - no preview is not fatal
                self.note.content = f'<div style="color:{INK2};font-size:12px">preview unavailable: {e}</div>'
            self.go.disabled = self._alive()

    def convert(self) -> None:
        with self.lock:
            if self.video is None or self._alive():
                return
            self.name = "".join(c if c.isalnum() else "_" for c in self.video.stem)[:40] or "clip"
            env = {**os.environ, "WORLDGYM_CACHE": "1" if self.cached else "0"}
            log = open(WORK / f"demo_{self.name}.log", "w")
            self.launched = time.time()
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "core.pipeline", "--video", str(self.video), "--prompt", PROMPT,
                 "--name", self.name, "--twin-only"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)   # a UI restart must not kill a live run
            self.run_dir, self.shown, self.follow, self.result = None, "", True, ""
            self.mirror.clear()
            self.gen_dd.visible = self.run_btn.visible = False
            self.go.disabled = True
            self.refresh()

    def open_run(self, d: Path) -> None:
        """Attach to an existing (finished or still running) run without launching one."""
        with self.lock:
            self.run_dir = d
            m = self._manifest() or {}
            if m.get("video") and Path(m["video"]).exists():
                self.attach(Path(m["video"]).resolve())

    # ------------------------------------------------------------------------------------------------------------------
    def _alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        m = self._manifest() if self.run_dir is not None else None
        return any(s["status"] == "running" for s in (m or {}).get("stages", []))

    def _find_run(self) -> None:
        if self.run_dir is not None or not self.name:
            return
        hits = [d for d in runs.RUNS.glob(f"*_{self.name}") if (d / "manifest.json").exists()
                and d.stat().st_mtime >= self.launched - 2]
        if hits:
            self.run_dir = max(hits, key=lambda d: d.stat().st_mtime)

    def _manifest(self) -> dict | None:
        try:
            return json.loads((self.run_dir / "manifest.json").read_text())
        except (OSError, ValueError, TypeError):
            return None

    def _gens(self) -> list[dict]:
        try:
            return json.loads((self.run_dir / "gens" / "gens.json").read_text())
        except (OSError, ValueError, TypeError):
            return []

    def status(self) -> tuple[dict, dict, list[dict]]:
        """milestone -> status, stage name -> record, generations."""
        m = self._manifest() if self.run_dir is not None else None
        stages = {s["name"]: s for s in (m or {}).get("stages", [])}
        gens = self._gens() if self.run_dir is not None else []
        st = {k: v["status"] for k, v in stages.items()}
        have = {g["n"] for g in gens}
        for n, stage in GEN_STAGE.items():
            st[f"gen{n}"] = "ok" if n in have else ("running" if st.get(stage) == "running" else "pending")
        return st, stages, gens

    # ------------------------------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        st, stages, gens = self.status()
        self._progress(st)
        self._live_line(st, stages, gens)
        self._gens_ui(gens)
        self._scene(stages, gens)
        if self.go.disabled and self.video is not None and not self._alive():
            self.go.disabled = False

    def _progress(self, st: dict) -> None:
        rows = []
        for label, ms in STEPS:
            ss = [st.get(m, "pending") for m in ms]
            if any(s in ("failed", "refused") for s in ss):
                icon, col = "✕", BAD
            elif all(s == "ok" for s in ss):
                icon, col = "✓", OK
            elif any(s in ("running", "ok") for s in ss):
                icon, col = SPIN, BLUE
            else:
                icon, col = "○", INK2
            rows.append(f'<div style="display:flex;gap:10px;align-items:center;padding:3px 0;font-size:14px;color:{INK}">'
                        f'<span style="color:{col};width:16px;text-align:center;font-weight:700">{icon}</span>{label}</div>')
        _set(self.steps, '<div style="padding:8px 0 2px">' + "".join(rows) + "</div>")
        done = sum(st.get(m) == "ok" for m in ORDER) + 0.5 * sum(st.get(m) == "running" for m in ORDER)
        v = round(100.0 * done / len(ORDER), 1) if self.run_dir is not None else 0.0
        if abs(self.bar.value - v) > 0.05:
            self.bar.value = v
        anim = self._alive()
        if self.bar.animated != anim:
            self.bar.animated = anim
        _set(self.pct, f'<div style="display:flex;justify-content:space-between;font-size:12px;color:{INK2}">'
                       f'<span>Progress</span><span>{v:.0f}%</span></div>')

    def _live_line(self, st: dict, stages: dict, gens: list[dict]) -> None:
        """One status line, replaced as the run moves on (no log)."""
        bad = next((s for s in stages.values() if s["status"] in ("failed", "refused")), None)
        if bad is not None:
            msg, col = f"Stopped: {(bad.get('error') or bad['status'])[:160]}", BAD
        elif self._alive():
            k = next((k for k in ORDER if st.get(k) == "running"), None)
            msg, col = (DOING.get(k, "Working") if k else "Starting the pipeline") + "…", INK
            s = stages.get(k) if k else None
            if s and s.get("t0"):
                try:
                    msg += f" {time.time() - datetime.fromisoformat(s['t0']).timestamp():.0f} s"
                except ValueError:
                    pass
        elif gens and st.get("final_critic") == "ok":
            msg, col = "Simulation ready. Pick a generation and run it.", OK
        elif gens:
            msg, col = "Run ended early; the newest generation is shown.", INK2
        elif self.video is not None:
            msg, col = "Ready. Press Convert to sim.", INK2
        else:
            msg, col = "Upload a robot video to begin.", INK2
        _set(self.live, f'<div style="font-size:13px;color:{col};padding:6px 0 2px;min-height:20px">{html.escape(msg)}</div>')

    def _gens_ui(self, gens: list[dict]) -> None:
        if not gens:
            return
        opts = tuple(f"Gen {g['n']} · {g['title']}" for g in gens)
        if tuple(self.gen_dd.options) != opts:
            self.gen_dd.options = opts
            self.gen_dd.visible = True
            if self.follow and not self.running:
                self.gen_dd.value = opts[-1]
        g = self._selected(gens)
        self.run_btn.visible = True
        runnable = bool(g and g.get("replay")) and not self.running
        if self.run_btn.disabled == runnable:
            self.run_btn.disabled = not runnable
        sub = (self.result or (html.escape(g["headline"]) if g else "")) if not self.running else "Running in MuJoCo…"
        if g and not g.get("replay"):
            sub = "This generation has no motion yet; the next one will."
        _set(self.note, f'<div style="font-size:12px;color:{INK2};padding-top:4px">{sub}</div>')

    def _selected(self, gens: list[dict]) -> dict | None:
        return next((g for g in gens if self.gen_dd.value.startswith(f"Gen {g['n']} ")), gens[-1] if gens else None)

    def _pick_gen(self) -> None:
        opts = tuple(self.gen_dd.options)
        self.follow = bool(opts) and self.gen_dd.value == opts[-1]
        self.result = ""

    # ------------------------------------------------------------------------------------------------------------------
    def _scene(self, stages: dict, gens: list[dict]) -> None:
        if self.running or self.run_dir is None:
            return
        g = self._selected(gens)
        if g is not None:
            d = self.run_dir / "gens" / f"gen{g['n']}"
            key = f"{d}@{g['t']}"
            if key != self.shown and not self.shown.startswith(f"ran:{d}@{g['t']}"):
                try:
                    sim = build_sim(TaskSpec.model_validate_json((d / "task.json").read_text()))
                except Exception:  # noqa: BLE001 - still being written
                    return
                self._show(key, sim)
            return
        ob = stages.get("robot_onboard")
        if ob and ob["status"] == "ok" and ob.get("data", {}).get("adapter"):
            key = f"robot:{ob['data']['adapter']}"
            if key != self.shown:
                self._show(key, build_sim(TaskSpec(name="robot", robot=ob["data"]["adapter"], objects=[])))

    def _show(self, key: str, sim) -> None:
        with self.lock:
            first = not self.shown or (self.shown.startswith("robot:") and not key.startswith("robot:"))
            self.shown = key
            self.mirror.load(sim.m)
            self.mirror.update(sim.d.xpos, sim.d.xquat)
            if first:
                frame_camera(self.server, sim.ix.task.robot, sim.d.xpos)

    def run_sim(self) -> None:
        g = self._selected(self._gens()) if self.run_dir is not None else None
        if g is None or not g.get("replay") or self.running:
            return
        d = self.run_dir / "gens" / f"gen{g['n']}"
        task = TaskSpec.model_validate_json((d / "task.json").read_text())
        policy = PolicyConfig.model_validate_json((d / "policy.json").read_text())
        self.running, self.result = True, ""
        self.run_btn.disabled = True
        threading.Thread(target=self._run_worker, args=(task, policy, g, d), daemon=True).start()

    def _run_worker(self, task: TaskSpec, policy: PolicyConfig, g: dict, d: Path) -> None:
        clock: list[float] = []

        def on_frame(sim) -> None:
            if not clock:
                self.mirror.load(sim.m)
                clock.append(time.perf_counter() - sim.d.time)
            self.mirror.update(sim.d.xpos, sim.d.xquat)
            lag = clock[0] + sim.d.time - time.perf_counter()
            if lag > 0:
                time.sleep(lag)

        t0 = time.perf_counter()
        try:
            run_episode(task, policy, on_frame=on_frame)
            self.result = f"Gen {g['n']} ran the demonstrated task in MuJoCo ({time.perf_counter() - t0:.0f} s)."
        except Exception as e:  # noqa: BLE001 - report, keep the server alive
            self.result = f"run failed: {type(e).__name__}: {str(e)[:120]}"
        finally:
            self.shown = f"ran:{d}@{g['t']}"   # keep the final state on screen until another generation is picked
            self.running = False


def _set(handle, content: str) -> None:
    if handle.content != content:
        handle.content = content


def frame_camera(server: viser.ViserServer, robot: str | None, xpos: np.ndarray) -> None:
    """Start every viewer where the demonstration was filmed from: behind and above the robot base(s), looking
    forward across the table, so the sim reads like the clip."""
    try:
        origin, f, _ = robot_frame(load_adapter(robot))
    except Exception:  # noqa: BLE001 - unknown robot: centre on the scene
        p = xpos[1:]
        origin, f = (p.min(0)[:2] + p.max(0)[:2]) / 2 - np.array([0.3, 0.0]), np.array([1.0, 0.0])
    look = np.array([*(origin + 0.3 * f), 0.05])
    pos = np.array([*(origin - 0.8 * f), 0.85])
    server.initial_camera.position, server.initial_camera.look_at = tuple(pos), tuple(look)
    for c in server.get_clients().values():
        c.camera.position, c.camera.look_at = tuple(pos), tuple(look)


def main() -> None:
    ap = argparse.ArgumentParser(description="WorldGym demo")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--video", type=Path, help="pre-attach a clip (Convert still has to be pressed)")
    ap.add_argument("--run", type=Path, help="show an existing run instead of starting one")
    ap.add_argument("--cached", action="store_true", help="let Argon reuse cached responses (rehearsals only)")
    a = ap.parse_args()
    server = viser.ViserServer(host=a.host, port=a.port, label="WorldGym")
    server.scene.set_up_direction("+z")
    server.initial_camera.position, server.initial_camera.look_at = (0.0, -0.8, 0.85), (0.0, 0.3, 0.05)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = server.initial_camera.position
        client.camera.look_at = server.initial_camera.look_at

    demo = Demo(server, cached=a.cached)
    if a.video:
        demo.attach(a.video.expanduser().resolve())
    if a.run:
        demo.open_run(a.run.expanduser().resolve())
    print(f"[demo] http://localhost:{a.port}", flush=True)
    try:
        while True:
            try:
                demo._find_run()
                demo.refresh()
            except Exception as e:  # noqa: BLE001 - a bad poll must not kill the server
                print(f"[demo] poll: {type(e).__name__}: {e}", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

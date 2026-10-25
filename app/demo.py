"""WorldGym demo: a chat-style sidebar (upload a clip, prompt, convert, progress) and the 3D twin as the page.

    .venv/bin/python -m app.demo [--port 8090] [--video clip.mp4] [--run runs/<id>] [--cached]

Convert starts `core.pipeline --twin-only` with Argon's response cache off (WORLDGYM_CACHE=0) unless --cached.
Results arrive as generations (runs/<id>/gens): Gen 1 is the first world, Gen 2 the critiqued twin replaying the
demonstrated motion, Gen 3 the motion after the Replay Critic. The newest generation is always on screen.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import viser

import core.runs as runs
from app.scene import FPS, Replay, SceneMirror, load_replay, static_replay
from core.mjcf_compiler import FLOOR_Z
from core.rollout import build_sim
from core.schemas import TaskSpec

REPO = Path(__file__).resolve().parents[1]
UPLOADS = REPO / ".worldgym" / "uploads"
PROMPT = "Reconstruct this robotic manipulation task as an interactive physics simulation."
BLUE, INK, INK2, CARD, OK, BAD = "#3987e5", "#e8e8ea", "#9a9aa2", "#25262b", "#3fb950", "#f85149"
SPIN = ('<style>@keyframes wgspin{to{transform:rotate(360deg)}}</style><span style="display:inline-block;width:11px;'
        f'height:11px;border:2px solid {BLUE};border-top-color:transparent;border-radius:50%;'
        'animation:wgspin .9s linear infinite"></span>')

# (checklist label, milestones); a milestone is a pipeline stage or a published generation
STEPS = [("Video understood", ("qa", "video_architect", "robot_onboard")),
         ("Assets matched", ("camera_solve", "gen1")),
         ("World constructed", ("loop1_static",)),
         ("Refining simulation", ("motion_extract", "gen2", "replay_critic", "gen3"))]
ORDER = [m for _, ms in STEPS for m in ms]
SAY = {"qa": "Clip checked", "video_architect": "Argon watched the video", "robot_onboard": "Robot identified",
       "camera_solve": "Camera calibrated from the robot's silhouette", "loop1_static": "Twin Critic (4-view)",
       "motion_extract": "Motion extracted by the labelling swarm", "replay_critic": "Replay Critic (real vs sim)",
       "gen1": "Scene Author is placing assets (SimReady)", "gen2": "Replaying the demonstrated motion",
       "gen3": "Replaying the refined motion"}


def _duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n / fps


def _bubble(text: str, user: bool = False, colour: str = CARD) -> str:
    side = "flex-end" if user else "flex-start"
    bg = BLUE if user else colour
    return (f'<div style="display:flex;justify-content:{side}"><div style="max-width:88%;background:{bg};color:{INK};'
            f'padding:8px 11px;border-radius:12px;font-size:13px;line-height:1.4">{text}</div></div>')


class Demo:
    def __init__(self, server: viser.ViserServer, cached: bool):
        self.server, self.gui, self.cached = server, server.gui, cached
        self.mirror = SceneMirror(server)
        self.lock = threading.RLock()
        self.video: Path | None = None
        self.prompt_sent = ""
        self.proc: subprocess.Popen | None = None
        self.launched = 0.0
        self.name = ""
        self.run_dir: Path | None = None
        self.shown = ""            # key of what the 3D view shows
        self.replay: Replay | None = None
        self.t = 0.0
        self.playing = True
        self.follow = True         # jump to each new generation
        self.chat_sig = ""
        self.caption = ""
        self._build()
        threading.Thread(target=self._player, daemon=True).start()

    # ------------------------------------------------------------------------------------------------------------------
    def _build(self) -> None:
        gui = self.gui
        gui.configure_theme(dark_mode=True, control_width="large", show_logo=False, show_share_button=False,
                            brand_color=(57, 135, 229))
        gui.main_panel.dock_left()
        gui.main_panel.set_width(400)
        gui.set_panel_label("Create your world")
        gui.add_html(f'<div style="display:flex;justify-content:space-between;align-items:baseline;padding:2px 0 6px">'
                     f'<span style="font-size:20px;font-weight:700;color:{INK}">WorldGym</span>'
                     f'<span style="font-size:12px;color:{INK2}">Video → Physics World</span></div>')
        self.chat = gui.add_html("")
        self.upload = gui.add_upload_button("Upload video", icon=viser.Icon.UPLOAD, mime_type="video/*")
        self.prompt = gui.add_text("Prompt", PROMPT, multiline=True)
        self.go = gui.add_button("Convert to sim", icon=viser.Icon.PLAYER_PLAY, color="blue")
        self.steps = gui.add_html("")
        self.bar = gui.add_progress_bar(0.0, animated=False)
        self.pct = gui.add_html("")
        self.gen_dd = gui.add_dropdown("Showing", ("—",), visible=False)
        self.play = gui.add_button("Pause replay", icon=viser.Icon.PLAYER_PAUSE, visible=False)
        self.now = gui.add_html("")
        self.server.scene.add_grid("/floor", width=3.0, height=3.0, position=(0.0, 0.2, FLOOR_Z), cell_size=0.1,
                                   section_size=0.5, cell_color=(55, 55, 58), section_color=(85, 85, 90))
        self.upload.on_upload(lambda e: self.on_upload(e.target.value))
        self.go.on_click(lambda _: self.convert())
        self.gen_dd.on_update(lambda _: self._pick_gen())
        self.play.on_click(lambda _: self._toggle_play())
        self.render_chat()

    # ------------------------------------------------------------------------------------------------------------------
    def on_upload(self, f) -> None:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        p = UPLOADS / Path(f.name).name
        p.write_bytes(f.content)
        self.attach(p)

    def attach(self, p: Path) -> None:
        with self.lock:
            self.video = p
            self.render_chat()

    def convert(self) -> None:
        with self.lock:
            if self.video is None:
                self.now.content = f'<div style="color:{BAD};font-size:13px">Upload a video first.</div>'
                return
            if self.proc is not None and self.proc.poll() is None:
                return
            self.prompt_sent = self.prompt.value.strip() or PROMPT
            self.name = "".join(c if c.isalnum() else "_" for c in self.video.stem)[:40] or "clip"
            env = {**os.environ, "WORLDGYM_CACHE": "1" if self.cached else "0"}
            log = open(REPO / ".worldgym" / f"demo_{self.name}.log", "w")
            self.launched = time.time()
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "core.pipeline", "--video", str(self.video), "--prompt", self.prompt_sent,
                 "--name", self.name, "--twin-only"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)   # a UI restart must not kill a live run
            self.run_dir, self.shown, self.replay, self.follow = None, "", None, True
            self.mirror.clear()
            self.go.disabled = True
            self.render_chat()

    def open_run(self, d: Path) -> None:
        """Attach to an existing run (no pipeline launch)."""
        with self.lock:
            self.run_dir, self.launched = d, 0.0
            m = self._manifest() or {}
            self.prompt_sent = m.get("prompt") or PROMPT
            self.video = Path(m["video"]) if m.get("video") else self.video

    # ------------------------------------------------------------------------------------------------------------------
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
        # each generation is published inside its stage: gen1 by the Scene Author at the start of loop1_static,
        # gen2 at the end of motion_extract, gen3 at the end of replay_critic
        for n, stage in ((1, "loop1_static"), (2, "motion_extract"), (3, "replay_critic")):
            st[f"gen{n}"] = "ok" if n in have else ("running" if st.get(stage) == "running" else "pending")
        if st.get("loop1_static") == "running" and 1 not in have:
            st["loop1_static"] = "pending"   # the Scene Author is still building the first world
        return st, stages, gens

    # ------------------------------------------------------------------------------------------------------------------
    def render_chat(self) -> None:
        st, stages, gens = self.status()
        parts = []
        if self.video is not None:
            dur = _duration(self.video) if self.video.exists() else 0.0
            parts.append(_bubble(f"🎬 <b>{html.escape(self.video.name)}</b><br><span style='color:#dfe9fa;"
                                 f"font-size:12px'>{dur:.0f} s · video input</span>", user=True))
        if self.prompt_sent:
            parts.append(_bubble(html.escape(self.prompt_sent), user=True))
        gen_by = {f"gen{g['n']}": g for g in gens}
        failed = False
        for k in ORDER:
            s = stages.get(k)
            if k in gen_by:
                g = gen_by[k]
                parts.append(_bubble(f"<b>Gen {g['n']} · {html.escape(g['title'])}</b><br>"
                                     f"<span style='color:{INK2}'>{html.escape(g['headline'])}</span> "
                                     f"<span style='color:{OK}'>→ on screen</span>"))
            elif s is not None and s["status"] in ("ok", "failed", "refused"):
                bad = s["status"] != "ok"
                failed |= bad
                txt = s.get("error") if bad else s.get("headline")
                parts.append(_bubble(f"<b>{SAY.get(k, k)}</b><br><span style='color:{BAD if bad else INK2}'>"
                                     f"{html.escape((txt or '')[:220])}</span>"))
        running = next((k for k in ORDER if st.get(k) == "running"), None)
        if running and not failed and self._alive():
            s = stages.get(running)
            label = SAY[running]
            secs = ""
            if s and s.get("t0"):
                try:
                    from datetime import datetime
                    secs = f" · {time.time() - datetime.fromisoformat(s['t0']).timestamp():.0f} s"
                except ValueError:
                    pass
            parts.append(_bubble(f"<span style='color:{INK2}'>⋯ {label}{secs}</span>"))
        if st.get("replay_critic") == "ok" and 3 in {g["n"] for g in gens}:
            parts.append(_bubble(f"<span style='color:{OK}'>✓ Simulation ready.</span> Orbit, pan and zoom the world; "
                                 "it replays the demonstrated task in MuJoCo physics."))
        elif self.run_dir is not None and not self._alive() and not failed and self.launched:
            parts.append(_bubble(f"<span style='color:{BAD}'>The pipeline stopped; see .worldgym/demo_{self.name}.log"
                                 f"</span>"))
        if not parts:
            parts.append(_bubble("Upload a robot video and tell me what to build. I'll turn it into a MuJoCo "
                                 "world you can replay."))
        body = ('<div style="display:flex;flex-direction:column;gap:7px;max-height:44vh;overflow-y:auto;'
                'padding:2px 0 8px">' + "".join(parts) + "</div>")
        if body != self.chat_sig:
            self.chat_sig = body
            self.chat.content = body
        self._progress(st)

    def _alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        m = self._manifest() if self.run_dir is not None else None   # attached run (--run): judge by its stages
        return any(s["status"] == "running" for s in (m or {}).get("stages", []))

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
        html_rows = '<div style="padding:6px 0 2px">' + "".join(rows) + "</div>"
        if self.steps.content != html_rows:
            self.steps.content = html_rows
        done = sum(st.get(m) == "ok" for m in ORDER) + 0.5 * sum(st.get(m) == "running" for m in ORDER)
        v = round(100.0 * done / len(ORDER), 1) if self.run_dir is not None else 0.0
        if abs(self.bar.value - v) > 0.05:
            self.bar.value = v
        self.bar.animated = self._alive()
        txt = (f'<div style="display:flex;justify-content:space-between;font-size:12px;color:{INK2}">'
               f'<span>{"Working…" if self._alive() else ("Done" if v >= 99.9 else "Progress")}</span><span>{v:.0f}%</span></div>')
        if self.pct.content != txt:
            self.pct.content = txt
        if self.go.disabled and not self._alive():
            self.go.disabled = False

    # ------------------------------------------------------------------------------------------------------------------
    def refresh_scene(self) -> None:
        if self.run_dir is None:
            return
        st, stages, gens = self.status()
        if gens:
            opts = tuple(f"Gen {g['n']} · {g['title']}" for g in gens)
            if tuple(self.gen_dd.options) != opts:
                self.gen_dd.options = opts
                self.gen_dd.visible = True
                if self.follow:
                    self.gen_dd.value = opts[-1]
            pick = next((g for g in gens if self.gen_dd.value.startswith(f"Gen {g['n']} ")), gens[-1])
            self._show_gen(pick)
            return
        ob = stages.get("robot_onboard")
        if ob and ob["status"] == "ok" and ob.get("data", {}).get("adapter"):
            key = f"robot:{ob['data']['adapter']}"
            if key != self.shown:
                sim = build_sim(TaskSpec(name="robot", robot=ob["data"]["adapter"], objects=[]))
                self._load(key, static_replay(sim.m, sim.d), "the verified robot, before the world is built")

    def _show_gen(self, g: dict) -> None:
        d = self.run_dir / "gens" / f"gen{g['n']}"
        key = f"{d}@{g['t']}"
        if key == self.shown:
            return
        try:
            if g.get("replay") and (d / "nominal" / "telemetry.npz").exists():
                rep = load_replay(d / "nominal")
            else:
                sim = build_sim(TaskSpec.model_validate_json((d / "task.json").read_text()))
                rep = static_replay(sim.m, sim.d)
        except Exception as e:  # noqa: BLE001 - files may still be being written
            self.now.content = f'<div style="color:{INK2};font-size:12px">loading Gen {g["n"]}… ({type(e).__name__})</div>'
            return
        self._load(key, rep, f"Gen {g['n']} · {g['title']}")

    def _load(self, key: str, rep: Replay, caption: str) -> None:
        with self.lock:
            first = not self.shown or (self.shown.startswith("robot:") and not key.startswith("robot:"))
            self.shown, self.replay, self.t = key, rep, 0.0
            self.mirror.load(rep.m)
            self.mirror.update(rep.xpos[0], rep.xquat[0])
            if first:
                frame_camera(self.server, rep)
            self.play.visible = len(rep.t) > 1
            self.caption = caption
            self._caption(0)

    def _caption(self, k: int) -> None:
        rep = self.replay
        if rep is None:
            return
        tail = f" · t = {rep.t[k]:.1f} s · {html.escape(rep.phase[k] or '')}" if len(rep.t) > 1 else ""
        self.now.content = (f'<div style="font-size:12px;color:{INK2};padding-top:4px">Showing <b style="color:{INK}">'
                            f'{html.escape(self.caption)}</b>{tail}</div>')

    def _pick_gen(self) -> None:
        opts = tuple(self.gen_dd.options)
        self.follow = bool(opts) and self.gen_dd.value == opts[-1]

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play.label = "Pause replay" if self.playing else "Play replay"
        self.play.icon = viser.Icon.PLAYER_PAUSE if self.playing else viser.Icon.PLAYER_PLAY

    def _player(self) -> None:
        last_cap = 0.0
        while True:
            time.sleep(1.0 / FPS)
            rep = self.replay
            if rep is None or len(rep.t) < 2 or not self.playing:
                continue
            with self.lock:
                self.t = self.t + 1.0 / FPS
                if self.t > rep.t[-1] + 1.0:   # hold the last frame for a second, then loop
                    self.t = 0.0
                k = rep.index(min(self.t, float(rep.t[-1])))
                self.mirror.update(rep.xpos[k], rep.xquat[k])
                if time.time() - last_cap > 0.25:
                    last_cap = time.time()
                    self._caption(k)


def frame_camera(server: viser.ViserServer, rep: Replay) -> None:
    p = rep.xpos[0][1:]
    lo, hi = p.min(0), p.max(0)
    look = (lo + hi) / 2
    look[2] = max(lo[2], FLOOR_Z) + 0.1
    r = max(float(np.max(hi - lo)) / 2, 0.5)
    pos = look + np.array([0.62, -0.62, 0.48]) * 3.4 * r
    server.initial_camera.position, server.initial_camera.look_at = tuple(pos), tuple(look)
    for c in server.get_clients().values():
        c.camera.position, c.camera.look_at = tuple(pos), tuple(look)


def main() -> None:
    ap = argparse.ArgumentParser(description="WorldGym demo")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--video", type=Path, help="pre-attach a clip (the user still presses Convert)")
    ap.add_argument("--run", type=Path, help="show an existing run instead of starting one")
    ap.add_argument("--cached", action="store_true", help="let Argon reuse cached responses (rehearsals only)")
    a = ap.parse_args()
    server = viser.ViserServer(host=a.host, port=a.port, label="WorldGym")
    server.scene.set_up_direction("+z")
    server.initial_camera.position, server.initial_camera.look_at = (1.4, -1.2, 1.0), (0.0, 0.25, 0.1)

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
                demo.render_chat()
                demo.refresh_scene()
            except Exception as e:  # noqa: BLE001 - a bad poll must not kill the server
                print(f"[demo] poll: {type(e).__name__}: {e}", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

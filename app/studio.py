"""WorldGym Studio: run-centric viser app. The live MuJoCo twin in 3D is the main surface; the story narrates one run.

    .venv/bin/python -m app.studio [--port 8080] [--runs DIR]

Left panel: run picker, stage timeline, the eight story steps and the replay transport.
Right panel: the current step (one media, one sentence, one big number, details) and the full run timeline.
runs/*/manifest.json is polled every second, so new runs and stage transitions appear without a restart.
"""
from __future__ import annotations

import argparse
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import plotly.graph_objects as go
import viser

import core.runs as runs
from app.scene import FPS, Replay, SceneMirror, contact_segments, load_replay, static_replay
from app.story import DASH, TITLES, Step, build_story, drafts, pct, read_manifest, static_task, timeline
from app.widgets import big_number, clusters_figure, load_image, ratchet_figure, runs_html, strip_html, timeline_html
from core.mjcf_compiler import FLOOR_Z, load_task
from core.perturb import apply_overrides, layout_ok
from core.rollout import build_sim, run_episode
from core.schemas import PolicyConfig, RunManifest, TaskSpec

CAM_POS, CAM_LOOK = (1.35, -0.95, 0.85), (0.42, 0.0, 0.08)
STATIC = "twin (static)"
NONE = "—"


class Studio:
    def __init__(self, server: viser.ViserServer):
        self.server, self.gui = server, server.gui
        self.mirror = SceneMirror(server)
        self.lock = threading.RLock()
        self.started = datetime.now().astimezone()
        self.run_ids: list[str] = []
        self.runs_sig = ""
        self.run_dir: Path | None = None
        self.manifest_text = ""
        self.steps: list[Step] = []
        self.step_i = 0
        self.shown = ""
        self.replay: Replay | None = None
        self.replay_key = ""
        self.playing = False
        self.live = False
        self.live_result: dict | None = None
        self.live_task: TaskSpec | None = None
        self.live_sliders: dict[tuple[str, str], viser.GuiSliderHandle] = {}
        self.live_btn = self.live_md = self.live_draft = None
        self.contacts = None
        self.framed = ""
        self.quiet = False
        self._build_gui()
        threading.Thread(target=self._player, daemon=True).start()

    # ------------------------------------------------------------------------------------------------------------------
    def _build_gui(self) -> None:
        gui = self.gui
        gui.configure_theme(dark_mode=True, control_width="large", show_logo=False,
                            show_share_button=False, brand_color=(57, 135, 229))
        gui.main_panel.dock_right()
        nav = gui.add_panel()
        with nav.add_tab("WorldGym", viser.Icon.ROBOT):
            self.run_dd = gui.add_dropdown("Run", ("(no runs yet)",))
            self.strip = gui.add_html("")
            self.step_btns = []
            for k, title in enumerate(TITLES):
                b = gui.add_button(f"○ {k + 1} · {title}", color="dark")
                b.on_click(lambda _, k=k: self.show_step(k, act=True))
                self.step_btns.append(b)
            with gui.add_folder("Replay"):
                self.draft_dd = gui.add_dropdown("Draft", (NONE,))
                self.t_slider = gui.add_slider("t (s)", min=0.0, max=1.0, step=1 / FPS, initial_value=0.0)
                self.t_slider.precision = 2
                self.play_btn = gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
                self.replay_md = gui.add_markdown("")
        nav.dock_left()
        nav.set_width(330)
        tabs = gui.add_tab_group()
        self.tab_story = tabs.add_tab("Story", viser.Icon.PRESENTATION)
        with self.tab_story:
            self.title_md = gui.add_markdown("", order=1)
            self.image = gui.add_image(np.zeros((8, 8, 3), np.uint8), format="jpeg", jpeg_quality=85, order=2,
                                       visible=False)
            self.plot = gui.add_plotly(go.Figure(), aspect=1.5, order=3, visible=False)
            self.sentence = gui.add_markdown("", order=4)
            self.number = gui.add_html("", order=5)
            self.extras = gui.add_folder("Details", order=6)
            self.next_btn = gui.add_button("Next →", order=7, color="blue")
        with tabs.add_tab("Runs", viser.Icon.LIST_DETAILS):
            self.timeline_html = gui.add_html("")
            self.runs_html = gui.add_html("")
        self.server.scene.add_grid("/floor", width=3.0, height=3.0, position=(0.35, 0.0, FLOOR_Z), cell_size=0.1,
                                   section_size=0.5, cell_color=(55, 55, 58), section_color=(85, 85, 90))

        self.run_dd.on_update(lambda _: None if self.quiet else self.select_run(self.run_dd.value))
        self.draft_dd.on_update(lambda _: None if self.quiet else self.load_draft(self.draft_dd.value))
        self.t_slider.on_update(lambda _: self.show_frame(self.t_slider.value))
        self.play_btn.on_click(lambda _: self.set_playing(not self.playing))
        self.next_btn.on_click(lambda _: self.show_step((self.step_i + 1) % len(TITLES), act=True))

    @staticmethod
    def _set(handle, attr: str, value) -> None:
        if getattr(handle, attr) != value:
            setattr(handle, attr, value)

    # ------------------------------------------------------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------------------------------------------------------
    def poll(self) -> None:
        ms = runs.list_runs()
        ids = [m.run_id for m in ms]
        sig = repr([(m.run_id, [(s.name, s.status) for s in m.stages]) for m in ms])
        if sig != self.runs_sig:
            self.runs_sig = sig
            self.runs_html.content = runs_html(ms)
        if ids != self.run_ids:
            fresh = [m.run_id for m in ms if m.run_id not in self.run_ids and _created(m) > self.started]
            self.run_ids = ids
            with self._quiet():
                self.run_dd.options = ids or ["(no runs yet)"]
            cur = self.run_dir.name if self.run_dir is not None else None
            target = fresh[0] if fresh else (cur if cur in ids else (ids[0] if ids else None))
            if target is not None:
                with self._quiet():
                    self.run_dd.value = target
                if target != cur:
                    self.select_run(target)
        self.refresh()

    @contextmanager
    def _quiet(self) -> Iterator[None]:
        """Programmatic dropdown changes must not re-enter the selection callbacks."""
        self.quiet = True
        try:
            yield
        finally:
            self.quiet = False

    def select_run(self, run_id: str) -> None:
        d = runs.RUNS / run_id
        if not (d / "manifest.json").exists():
            return
        with self.lock:
            self.set_playing(False)
            self.run_dir, self.manifest_text, self.shown, self.replay_key = d, "", "", ""
            self.replay, self.live_result = None, None
            self.refresh()

    def refresh(self) -> None:
        if self.run_dir is None:
            return
        with self.lock:
            try:
                text = (self.run_dir / "manifest.json").read_text()
            except OSError:
                return
            media = sorted(str(p.relative_to(self.run_dir)) for p in self.run_dir.glob("loop*/*/*.jpg"))
            text += repr(media) + repr(drafts(self.run_dir))
            m = read_manifest(self.run_dir)
            if m is None:
                return
            chips = timeline(m)
            self._set(self.strip, "content", strip_html(chips))
            self._set(self.timeline_html, "content", timeline_html(m, chips))
            self._sync_drafts()
            if text == self.manifest_text:
                return
            self.manifest_text = text
            self.steps = build_story(self.run_dir)
            for b, s in zip(self.step_btns, self.steps):
                self._set(b, "label", f"{'✓' if s.done else '○'} {s.n} · {s.title}")
            if repr(self.steps[self.step_i]) != self.shown and not (self.step_i == 7 and self.live):
                self.show_step(self.step_i, act=False)

    def _replay_key(self, name: str) -> str:
        st = static_task(self.run_dir) if name == STATIC else None
        return f"{self.run_dir}/{name}" + (f"/{st[1]}" if st else "")

    def _sync_drafts(self) -> None:
        opts = drafts(self.run_dir)
        if not opts and static_task(self.run_dir) is not None:
            opts = [STATIC]
        opts = tuple(opts or [NONE])
        if opts != tuple(self.draft_dd.options):
            with self._quiet():
                cur = self.draft_dd.value
                self.draft_dd.options = opts
                self.draft_dd.value = cur if cur in opts else opts[0]
        if self.replay_key != self._replay_key(self.draft_dd.value):
            self.load_draft(self.draft_dd.value)

    # ------------------------------------------------------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------------------------------------------------------
    def load_draft(self, name: str) -> None:
        with self.lock:
            if self.run_dir is None or name == NONE:
                self.replay, self.replay_key = None, ""
                self.mirror.clear()
                self.replay_md.content = "no replayable draft in this run yet"
                return
            key = self._replay_key(name)
            if key == self.replay_key:
                return
            self.replay_key = key
            try:
                if name == STATIC:
                    st = static_task(self.run_dir)
                    if st is None:
                        raise FileNotFoundError("no twin task yet")
                    sim = build_sim(st[0])
                    rep = static_replay(sim.m, sim.d)
                else:
                    rep = load_replay(self.run_dir / name / "nominal")
            except Exception as e:  # noqa: BLE001 - partial artifacts: show why, keep the previous scene
                self.replay_md.content = f"replay unavailable: {type(e).__name__}: {e}"
                return
            self.replay = rep
            if self.framed != str(self.run_dir):
                self.framed = str(self.run_dir)
                self.frame_camera(rep)
            self.t_slider.value = 0.0
            self.t_slider.max = max(float(rep.t[-1]), 1.0 / FPS)
            self.show_frame(0.0)

    def frame_camera(self, rep: Replay) -> None:
        """Point every client at this run's scene (a Panda cell and a bimanual table sit in different places)."""
        p = rep.xpos[0][1:]
        lo, hi = p.min(0), p.max(0)
        look = (lo + hi) / 2
        look[2] = max(lo[2], FLOOR_Z) + 0.1
        r = max(float(np.max(hi - lo)) / 2, 0.5)
        pos = look + np.array([0.62, -0.62, 0.48]) * 3.4 * r
        self.server.initial_camera.position, self.server.initial_camera.look_at = tuple(pos), tuple(look)
        for c in self.server.get_clients().values():
            c.camera.position, c.camera.look_at = tuple(pos), tuple(look)

    def show_frame(self, t: float) -> None:
        rep = self.replay
        if rep is None or self.live:
            return
        if self.mirror.model is not rep.m:
            self.mirror.load(rep.m)
        k = rep.index(t)
        self.mirror.update(rep.xpos[k], rep.xquat[k])
        self.replay_md.content = f"`{self.draft_dd.value}` · **t = {rep.t[k]:.2f} s** · {rep.phase[k] or DASH}"

    def set_playing(self, on: bool) -> None:
        rep = self.replay
        if on and rep is not None and self.t_slider.value >= rep.t[-1] - 1e-6:
            self.t_slider.value = 0.0
        self.playing = bool(on and rep is not None and len(rep.t) > 1)
        self._set(self.play_btn, "label", "Pause" if self.playing else "Play")
        self.play_btn.icon = viser.Icon.PLAYER_PAUSE if self.playing else viser.Icon.PLAYER_PLAY

    def _player(self) -> None:
        nxt = time.perf_counter()
        while True:
            nxt += 1.0 / FPS
            time.sleep(max(0.0, nxt - time.perf_counter()))
            nxt = max(nxt, time.perf_counter() - 0.5)
            rep = self.replay
            if not self.playing or self.live or rep is None:
                continue
            try:
                t = self.t_slider.value + 1.0 / FPS
                if t >= rep.t[-1]:
                    t = float(rep.t[-1])
                    self.set_playing(False)
                self.t_slider.value = t
            except Exception as e:  # noqa: BLE001 - keep the player thread alive
                print(f"[studio] player: {type(e).__name__}: {e}", flush=True)

    def play_draft(self, name: str, t: float = 0.0, play: bool = True) -> None:
        opts = tuple(self.draft_dd.options)
        if name not in opts:
            name = opts[-1] if opts and opts[-1] != NONE else None
        if name is None:
            return
        if self.draft_dd.value != name:
            self.draft_dd.value = name
        self.load_draft(name)
        if self.replay is not None:
            self.t_slider.value = min(float(t), float(self.replay.t[-1]))
        self.set_playing(play)

    # ------------------------------------------------------------------------------------------------------------------
    # Story
    # ------------------------------------------------------------------------------------------------------------------
    def show_step(self, k: int, act: bool = False) -> None:
        with self.lock:
            if not self.steps:
                return
            self.step_i = k
            s = self.steps[k]
            self.shown = repr(s)
            for i, b in enumerate(self.step_btns):
                self._set(b, "color", "blue" if i == k else "dark")
            self.title_md.content = f"## {s.n} · {s.title}"
            img = load_image(s.media) if s.media else None
            fig = None
            if s.n == 5 and s.extra.get("clusters"):
                fig = clusters_figure(s.extra["clusters"])
            elif s.n == 7 and s.extra.get("ratchet"):
                fig = ratchet_figure(s.extra["ratchet"])
            if img is not None:
                self.image.image = img
            self.image.visible = img is not None
            if fig is not None:
                self.plot.aspect = 2.2 if s.n == 5 else 1.4
                self.plot.figure = fig
            self.plot.visible = fig is not None
            self.sentence.content = s.sentence
            self.number.content = self._number_html(s)
            self.next_btn.label = s.next_label
            self._rebuild_extras(s)
            if act:
                self._act(s)

    def _number_html(self, s: Step) -> str:
        r = self.live_result
        if s.n != 8 or r is None:
            return big_number(s.number, s.number_label)
        if r.get("running"):
            return big_number("…", f"{r['draft']} running live in the twin")
        if r.get("error"):
            return big_number("✗", f"{r['draft']}: {r['error'][:160]}")
        why = f"failed: {', '.join(r['failed'])}" if r["failed"] else ("timed out" if r.get("timed_out") else "")
        label = f"{r['draft']} {'succeeds' if r['success'] else why or 'fails'} · {r['collisions']} unwanted collisions"
        return big_number("✓" if r["success"] else "✗", label)

    def _act(self, s: Step) -> None:
        if s.n in (1, 2, 3):
            self.play_draft(self.draft_dd.options[0], 0.0, play=False)
        elif s.n == 4:
            self.play_draft("draft1", 0.0, play=True)
        elif s.n == 6:
            crit = s.extra.get("critique") or []
            self.play_draft("draft1", float(crit[0].get("timestamp_sec") or 0.0) if crit else 0.0, play=False)
        elif s.n == 7:
            last = (s.extra.get("ratchet") or [{}])[-1].get("draft", "draft3")
            self.play_draft(last, 0.0, play=True)
        elif s.n == 8 and not self.live:
            ds = s.extra.get("drafts") or []
            if ds:
                self.play_draft(ds[-1], 0.0, play=False)

    def _rebuild_extras(self, s: Step) -> None:
        self.extras.remove()
        self.live_sliders, self.live_btn, self.live_md, self.live_draft = {}, None, None, None
        with self.tab_story:
            self.extras = self.gui.add_folder("Details", order=6, expand_by_default=True)
        with self.extras:
            builder = getattr(self, f"_extras_{s.n}", None)
            shown = bool(builder(s)) if builder else False
        self.extras.visible = shown

    def _extras_2(self, s: Step) -> bool:
        if s.extra.get("table"):
            self.gui.add_markdown(s.extra["table"])
        return bool(s.extra.get("table"))

    def _extras_3(self, s: Step) -> bool:
        iters = s.extra.get("iterations") or []
        if not iters:
            return False
        cap = self.gui.add_markdown("")

        def show(i: int) -> None:
            it = iters[int(i)]
            img = load_image(it["path"])
            if img is not None:
                self.image.image = img
                self.image.visible = True
            lines = [f"**{it['label']}** · alignment {pct(it['alignment'])}"] + [f"- {c}" for c in it["critique"][:6]]
            cap.content = "\n".join(lines)

        if len(iters) > 1:
            sl = self.gui.add_slider("Iteration", min=0, max=len(iters) - 1, step=1, initial_value=len(iters) - 1,
                                     order=0)
            sl.on_update(lambda _: show(sl.value))
        show(len(iters) - 1)
        return True

    def _extras_4(self, s: Step) -> bool:
        dyn = s.extra.get("dynamic") or {}
        per = dyn.get("per_object_mm") or {}
        lines = []
        if s.extra.get("segments"):
            lines.append(f"{s.extra['segments']} grasp → release segment(s) extracted from the video.")
        before = (dyn.get("before_critic") or {}).get("end_pose_err_mm")
        if before is not None:
            lines.append(f"End-pose error before Argon's replay critic: {before:.0f} mm.")
        if per:
            lines += ["| object | end-pose error |", "|---|---|"] + [f"| {k} | {v:.0f} mm |" for k, v in per.items()]
        if lines:
            self.gui.add_markdown("\n".join(lines))
        return bool(lines)

    def _extras_5(self, s: Step) -> bool:
        if not s.extra.get("lock"):
            return False
        dev, held = s.extra.get("dev") or {}, s.extra.get("held_out") or {}
        fam = " · ".join(f"{k} {v}" for k, v in (s.extra.get("by_family") or {}).items())
        self.gui.add_markdown(
            f"Locked held-out sha256 `{s.extra['lock'][:32]}…`\n\n"
            f"| suite | success | collisions |\n|---|---|---|\n"
            f"| dev | {dev.get('successes', '?')}/{dev.get('n', '?')} | {dev.get('collisions', DASH)} |\n"
            f"| held-out | {held.get('successes', '?')}/{held.get('n', '?')} | {held.get('collisions', DASH)} |\n\n"
            + (f"Held-out by family: {fam}" if fam else ""))
        return True

    def _extras_6(self, s: Step) -> bool:
        crit = (s.extra.get("critique") or [])[:12]
        if crit:
            self.gui.add_markdown("**Argon's timestamped critique** (jump seeks the Draft 1 replay)")
        for c in crit:
            ts = float(c.get("timestamp_sec") or 0.0)
            self.gui.add_markdown(f"**{ts:.1f} s** · {c.get('observation', '')}")
            self.gui.add_button(f"Jump to {ts:.1f} s", icon=viser.Icon.PLAYER_TRACK_NEXT).on_click(
                lambda _, ts=ts: self.play_draft("draft1", ts, play=False))
        if s.extra.get("root_causes"):
            self.gui.add_markdown("**Root causes**\n" + "\n".join(f"- {r}" for r in s.extra["root_causes"]))
        if s.extra.get("diff_md"):
            self.gui.add_markdown("**Policy diff, Draft 1 → Draft 2**\n\n" + s.extra["diff_md"])
        return bool(crit or s.extra.get("diff_md") or s.extra.get("root_causes"))

    def _extras_7(self, s: Step) -> bool:
        rows = s.extra.get("ratchet") or []
        if not rows:
            return False
        md = ["| draft | dev | held-out | held-out collisions |", "|---|---|---|---|"]
        md += [f"| {r.get('label') or r.get('draft')} | {pct(r.get('dev'))} | {pct(r.get('held_out'))} | "
               f"{r.get('collisions', DASH)} |" for r in rows]
        if s.extra.get("lock"):
            md += ["", f"Locked held-out sha256 `{s.extra['lock'][:32]}…` (scored once per draft)"]
        if s.extra.get("swarm"):
            md += ["", "**Swarm hypotheses**", "", "| hypothesis | failing scenarios | source |", "|---|---|---|"]
            md += [f"| {w['hypothesis']} | {w['failing']} | {w['source'] or DASH} |" for w in s.extra["swarm"]]
        self.gui.add_markdown("\n".join(md))
        return True

    def _extras_8(self, s: Step) -> bool:
        ds = s.extra.get("drafts") or []
        if not ds or not s.extra.get("task"):
            return False
        try:
            task = load_task(s.extra["task"])
        except Exception as e:  # noqa: BLE001
            self.gui.add_markdown(f"cannot load the base task: {e}")
            return True
        self.live_task = task
        self.live_draft = self.gui.add_dropdown("Policy", tuple(ds), initial_value=ds[-1])
        for o in task.objects:
            if o.is_obstacle:
                h = o.dims_m[2]
                self.live_sliders[(o.id, "h")] = self.gui.add_slider(
                    f"{o.id} height (m)", min=round(max(0.01, h - 0.1), 3), max=round(h + 0.1, 3), step=0.005,
                    initial_value=round(h, 3))
            elif not o.is_fixed:
                for ax in "xy":
                    v = o.xy["xy".index(ax)]
                    sl = self.gui.add_slider(f"{o.id} {ax} (m)", min=round(v - 0.1, 3), max=round(v + 0.1, 3),
                                             step=0.005, initial_value=round(v, 3))
                    sl.on_update(lambda _, oid=o.id: self._preview(oid))
                    self.live_sliders[(o.id, ax)] = sl
        self.live_btn = self.gui.add_button("Run live", icon=viser.Icon.PLAYER_PLAY, color="green", disabled=self.live)
        self.live_btn.on_click(lambda _: self.run_live())
        self.live_md = self.gui.add_markdown("")
        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Your turn: a live episode streamed into the 3D view
    # ------------------------------------------------------------------------------------------------------------------
    def _preview(self, oid: str) -> None:
        if self.live:
            return
        x, y = self.live_sliders.get((oid, "x")), self.live_sliders.get((oid, "y"))
        if x is not None and y is not None:
            self.set_playing(False)
            self.mirror.nudge(oid, (x.value, y.value))

    def overrides(self) -> dict:
        ov: dict = {"objects": {}}
        for (oid, k), sl in self.live_sliders.items():
            o, f = self.live_task.obj(oid), ov["objects"].setdefault(oid, {})
            if k == "h":
                f["dims_m"] = [*o.dims_m[:2], round(float(sl.value), 4)]
            else:
                f.setdefault("xy", list(o.xy))["xy".index(k)] = round(float(sl.value), 4)
        return ov

    def run_live(self) -> None:
        if self.live or self.run_dir is None or self.live_task is None or self.live_draft is None:
            return
        draft = self.live_draft.value
        try:
            policy = PolicyConfig.model_validate_json((self.run_dir / draft / "policy.json").read_text())
            task = apply_overrides(self.live_task, self.overrides())
        except Exception as e:  # noqa: BLE001
            self.live_result = {"draft": draft, "error": f"{type(e).__name__}: {e}"}
            self.number.content = self._number_html(self.steps[7])
            return
        ok, why = layout_ok(task)
        self.live, self.live_result = True, {"draft": draft, "running": True}
        self.set_playing(False)
        self.live_btn.disabled = True
        self.live_md.content = "" if ok else f"⚠ layout check: {why} (running anyway)"
        self.number.content = self._number_html(self.steps[7])
        threading.Thread(target=self._live_worker, args=(task, policy, draft), daemon=True).start()

    def _live_worker(self, task: TaskSpec, policy: PolicyConfig, draft: str) -> None:
        clock: list[float] = []

        def on_frame(sim) -> None:
            if not clock:
                self.mirror.load(sim.m)
                clock.append(time.perf_counter() - sim.d.time)
            self.mirror.update(sim.d.xpos, sim.d.xquat)
            self._draw_contacts(contact_segments(sim.m, sim.d))
            lag = clock[0] + sim.d.time - time.perf_counter()
            if lag > 0:
                time.sleep(lag)

        try:
            ep = run_episode(task, policy, on_frame=on_frame)
            s = ep.summary()
            res = {"draft": draft, "success": ep.success, "timed_out": s["trace"].get("timed_out"),
                   "collisions": s["trace"]["metrics"].get("unwanted_collision_count", 0),
                   "failed": [p["kind"] + (f"({p['object_id']})" if p.get("object_id") else "")
                              for p in s["evaluation"]["predicates"] if not p["ok"]]}
        except Exception as e:  # noqa: BLE001 - report in the UI, keep the server alive
            res = {"draft": draft, "error": f"{type(e).__name__}: {e}"}
        finally:
            self.live = False
        self._draw_contacts(np.zeros((0, 2, 3)))
        with self.lock:
            self.live_result = res
            if self.live_btn is not None:
                self.live_btn.disabled = False
            if self.step_i == 7 and self.steps:
                self.number.content = self._number_html(self.steps[7])

    def _draw_contacts(self, seg: np.ndarray) -> None:
        if len(seg) == 0:
            if self.contacts is not None:
                self.contacts.remove()
                self.contacts = None
            return
        self.contacts = self.server.scene.add_line_segments("/contacts", seg.astype(np.float32), colors=(255, 120, 40),
                                                            thickness=4.0, thickness_units="screen")


def _created(m: RunManifest) -> datetime:
    try:
        return datetime.fromisoformat(m.created)
    except ValueError:
        return datetime.min.replace(tzinfo=datetime.now().astimezone().tzinfo)


def main() -> None:
    ap = argparse.ArgumentParser(description="WorldGym Studio")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--runs", type=Path, help="run registry (default: $WORLDGYM_RUNS or runs/)")
    a = ap.parse_args()
    if a.runs:
        runs.RUNS = a.runs.expanduser().resolve()
    server = viser.ViserServer(host=a.host, port=a.port, label="WorldGym Studio")
    server.initial_camera.position, server.initial_camera.look_at = CAM_POS, CAM_LOOK

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = server.initial_camera.position
        client.camera.look_at = server.initial_camera.look_at

    server.scene.set_up_direction("+z")
    studio = Studio(server)
    print(f"[studio] watching {runs.RUNS}", flush=True)
    try:
        while True:
            try:
                studio.poll()
            except Exception as e:  # noqa: BLE001 - a bad poll must not kill the server
                print(f"[studio] poll: {type(e).__name__}: {e}", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

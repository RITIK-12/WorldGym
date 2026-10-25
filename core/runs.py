"""Run registry: every CLI/agent stage writes runs/<run_id>/manifest.json, the single source of truth for the Studio.

    run = new_run("pipeline", "panda_sort", video="demo.mp4")
    with run.stage("qa") as s:
        ...
        s.headline = "static 96 %"
        s.artifacts.append("qa/probe.json")

A stage that raises is recorded as `failed` with the exception text; `Refused` is recorded as `refused` (QA verdicts).
Manifests are written atomically so a polling reader never sees a half-written file.
"""
from __future__ import annotations

import json
import os
import re
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from core.schemas import RunManifest, StageRecord

REPO = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("WORLDGYM_RUNS", REPO / "runs"))


class Refused(Exception):
    """A stage declined its input (e.g. QA gate). Recorded as `refused`, not as a crash."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")[:40] or "run"


class Run:
    def __init__(self, run_dir: Path, manifest: RunManifest):
        self.dir = run_dir
        self.manifest = manifest

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, p: str | Path) -> str:
        return str(Path(p).resolve().relative_to(self.dir.resolve()))

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / ".manifest.json.tmp"
        tmp.write_text(self.manifest.model_dump_json(indent=1))
        os.replace(tmp, self.dir / "manifest.json")

    def get_stage(self, name: str) -> StageRecord | None:
        return next((s for s in self.manifest.stages if s.name == name), None)

    def plan(self, names: list[str]) -> None:
        """Pre-register stages as `pending` so the UI can show the whole pipeline up front."""
        for n in names:
            if self.get_stage(n) is None:
                self.manifest.stages.append(StageRecord(name=n))
        self.save()

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        rec = self.get_stage(name)
        if rec is None:
            rec = StageRecord(name=name)
            self.manifest.stages.append(rec)
        rec.status, rec.t0, rec.t1, rec.error = "running", _now(), None, None
        self.save()
        try:
            yield rec
        except Refused as e:
            rec.status, rec.t1, rec.error = "refused", _now(), str(e)
            self.save()
            raise
        except BaseException as e:  # noqa: BLE001 - record then re-raise
            rec.status, rec.t1 = "failed", _now()
            rec.error = f"{type(e).__name__}: {e}"
            (self.path("errors", f"{_slug(name)}.txt")).write_text(traceback.format_exc())
            self.save()
            raise
        else:
            rec.status, rec.t1 = "ok", _now()
            self.save()

    def write_json(self, rel_path: str, obj) -> str:
        p = self.path(rel_path)
        p.write_text(json.dumps(obj, indent=1, default=_default))
        return rel_path


def _default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def new_run(kind: str = "pipeline", name: str = "", video: str | None = None, prompt: str | None = None) -> Run:
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_id = f"{stamp}_{_slug(name or kind)}"
    run_dir = RUNS / run_id
    k = 1
    while run_dir.exists():
        k += 1
        run_dir = RUNS / f"{run_id}-{k}"
    m = RunManifest(run_id=run_dir.name, kind=kind, video=video, prompt=prompt, created=_now())
    run = Run(run_dir, m)
    run.save()
    return run


def load_run(run_id: str | Path) -> Run:
    d = Path(run_id) if Path(run_id).is_dir() else RUNS / str(run_id)
    return Run(d, RunManifest.model_validate_json((d / "manifest.json").read_text()))


def list_runs() -> list[RunManifest]:
    out = []
    if not RUNS.exists():
        return out
    for p in RUNS.glob("*/manifest.json"):
        try:
            out.append(RunManifest.model_validate_json(p.read_text()))
        except Exception:  # noqa: BLE001 - a run being written; skip this poll
            continue
    return sorted(out, key=lambda m: m.created, reverse=True)

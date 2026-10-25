"""Run registry + model adapter (offline backends only)."""
import json

import pytest
from pydantic import BaseModel

import core.runs as runs
from core.argon import Argon, ArgonError, ManualPending, parse_json, strip_fences


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    return tmp_path / "runs"


def test_stage_lifecycle(runs_dir):
    run = runs.new_run("pipeline", "demo clip", video="demo.mp4")
    run.plan(["qa", "camera_solve"])
    with run.stage("qa") as s:
        s.headline = "static 96 %"
        s.artifacts.append(run.write_json("qa/probe.json", {"ok": True}))
    with pytest.raises(RuntimeError):
        with run.stage("camera_solve"):
            raise RuntimeError("robot IoU 0.41 < 0.75")
    m = runs.load_run(run.run_id).manifest
    st = {s.name: s for s in m.stages}
    assert st["qa"].status == "ok" and st["qa"].headline == "static 96 %" and st["qa"].artifacts == ["qa/probe.json"]
    assert st["camera_solve"].status == "failed" and "0.41" in st["camera_solve"].error
    assert (run.dir / "errors" / "camera_solve.txt").exists()
    assert [r.run_id for r in runs.list_runs()] == [run.run_id]


def test_refused(runs_dir):
    run = runs.new_run("pipeline", "x")
    with pytest.raises(runs.Refused):
        with run.stage("qa"):
            raise runs.Refused("camera moves (static 5 %)")
    assert runs.load_run(run.run_id).manifest.stages[0].status == "refused"


def test_strip_fences_and_parse():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert parse_json('Sure! {"a": {"b": [1, 2]}} hope that helps') == {"a": {"b": [1, 2]}}


class Thing(BaseModel):
    a: int
    b: str


def test_replay_backend(tmp_path):
    rd = tmp_path / "replay"
    rd.mkdir()
    (rd / "thing.json").write_text(json.dumps({"text": '```json\n{"a": 3, "b": "x"}\n```', "model": "m"}))
    img = tmp_path / "k.jpg"
    img.write_bytes(b"\xff\xd8fake")
    ar = Argon(tmp_path / "bundles", backend="replay", replay_dir=rd)
    obj, resp = ar.ask_json("thing", "sys", "prompt", Thing, attachments=[img])
    assert obj == Thing(a=3, b="x")
    b = tmp_path / "bundles" / "thing"
    assert (b / "system.md").read_text() == "sys" and (b / "attachments" / "k.jpg").exists()
    assert json.loads((b / "response.json").read_text())["backend"] == "replay"
    with pytest.raises(ArgonError):
        ar.ask_text("missing", "s", "p")


def test_replay_fix_path(tmp_path):
    rd = tmp_path / "replay"
    rd.mkdir()
    (rd / "t.json").write_text(json.dumps({"text": '{"a": "not-int"}'}))
    (rd / "t__fix.json").write_text(json.dumps({"text": '{"a": 1, "b": "ok"}'}))
    obj, _ = Argon(tmp_path / "b", backend="replay", replay_dir=rd).ask_json("t", "s", "p", Thing)
    assert obj.a == 1


def test_manual_backend(tmp_path):
    ar = Argon(tmp_path / "b", backend="manual")
    with pytest.raises(ManualPending):
        ar.ask_text("q", "s", "p")
    (tmp_path / "b" / "q" / "response.json").write_text(json.dumps({"text": "answer"}))
    assert ar.ask_text("q", "s", "p")[0] == "answer"


def test_stale_attachments_removed(tmp_path):
    ar = Argon(tmp_path / "b", backend="manual")
    old = tmp_path / "old.jpg"
    old.write_bytes(b"1")
    with pytest.raises(ManualPending):
        ar.ask_text("q", "s", "p", [old])
    new = tmp_path / "new.jpg"
    new.write_bytes(b"2")
    with pytest.raises(ManualPending):
        ar.ask_text("q", "s", "p", [new])
    assert sorted(p.name for p in (tmp_path / "b" / "q" / "attachments").iterdir()) == ["new.jpg"]


def test_budget_cap_blocks_before_network(tmp_path, monkeypatch):
    import core.argon as a
    monkeypatch.setattr(a, "STATE", tmp_path / "state")
    monkeypatch.setenv("WORLDGYM_MAX_CALLS", "0")
    ar = Argon(tmp_path / "b", backend="gemini")
    monkeypatch.setattr(Argon, "client", property(lambda self: (_ for _ in ()).throw(AssertionError("network"))))
    with pytest.raises(a.BudgetExceeded):
        ar.ask_text("q", "s", "p")


def test_cache_hit_costs_nothing(tmp_path, monkeypatch):
    import core.argon as a
    monkeypatch.setattr(a, "STATE", tmp_path / "state")
    monkeypatch.setenv("WORLDGYM_MAX_CALLS", "0")
    key = Argon._cache_key(a.ARGON, "s", "p", [], "high", False, None, a.DEFAULT_MAX_OUTPUT)
    (tmp_path / "state" / "cache").mkdir(parents=True)
    (tmp_path / "state" / "cache" / f"{key}.json").write_text(json.dumps({"text": "cached answer"}))
    text, resp = Argon(tmp_path / "b", backend="gemini").ask_text("q", "s", "p")
    assert text == "cached answer" and resp["cached"]


def test_repeat_guard_and_ledger(tmp_path, monkeypatch):
    import core.argon as a
    monkeypatch.setattr(a, "STATE", tmp_path / "state")
    monkeypatch.setattr(a, "CALLS", [a.CallLog("loop", a.ARGON, "gemini", 1.0, True) for _ in range(4)])
    a.check_budget("model", "other")
    with pytest.raises(a.LoopGuardTripped, match="repeated"):
        a.check_budget("model", "loop")
    a.record_spend("model", "x", a.ARGON, {"total_input_tokens": 1000, "total_output_tokens": 500}, True)
    assert a.spend_summary()["usd_est_today"] == pytest.approx(0.007)

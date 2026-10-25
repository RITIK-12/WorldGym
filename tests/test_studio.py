"""Studio: welded scene mirroring, manifest -> story extraction, and a headless server smoke test (no browser)."""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import cv2
import numpy as np
import pytest

import app.studio  # noqa: F401 - the entry module must import cleanly
import core.runs as runs
from app.scene import load_replay, scene_geoms
from app.story import STATUS_COLOUR, build_story, drafts, read_manifest, timeline
from core.mjcf_compiler import compile_scene, load_task
from core.schemas import RunManifest, StageRecord

REPO = Path(__file__).resolve().parents[1]
LOCK = "ab" * 32


@pytest.fixture(scope="module")
def scene():
    task = load_task(REPO / "tasks/panda_sort.json")
    xml, m = compile_scene(task)
    return task, xml, m


def _stage(name, status="ok", **kw) -> StageRecord:
    t = {"t0": "2026-10-25T12:00:00-07:00", "t1": "2026-10-25T12:00:30-07:00"} if status != "pending" else {}
    if status == "running":
        t.pop("t1")
    return StageRecord(name=name, status=status, **t, **kw)


def make_run(root: Path, scene) -> Path:
    task, xml, m = scene
    d = root / "2026-10-25T12-00-00_demo"
    for rel in ("loop1/iter1", "loop1/final", "camera", "architect/keyframes", "draft1/nominal"):
        (d / rel).mkdir(parents=True, exist_ok=True)
    img = np.full((60, 80, 3), 128, np.uint8)
    for rel in ("loop1/iter1/matrix.jpg", "loop1/final/matrix.jpg", "camera/robot_fit.jpg",
                "architect/keyframes/kf_00000.jpg"):
        cv2.imwrite(str(d / rel), img)
    nd = d / "draft1/nominal"
    (nd / "scene.xml").write_text(xml)
    (nd / "task.json").write_text(task.model_dump_json())
    (d / "draft1/policy.json").write_text((REPO / "policies/panda_sort_draft1.json").read_text())
    n = 1000
    np.savez_compressed(nd / "telemetry.npz", t=np.arange(1, n + 1, dtype=np.float32) * 0.002,
                        qpos=np.tile(m.qpos0.astype(np.float32), (n, 1)), phase=np.array(["reach"] * n),
                        phase_obj=np.array(["cube"] * n))
    hist = [{"iteration": 0, "stage": "initial", "alignment": 0.62},
            {"iteration": 1, "stage": "critic", "alignment": 0.9, "alignment_after_critic": 0.94,
             "critique": [{"view": "top", "object_id": "bin", "observation": "bin sits 2 cm too far left"}]}]
    stages = [
        _stage("qa", data={"accepted": True, "keyframes_t": [0.0]}, headline="static 97 %, 16.7 s"),
        _stage("video_architect", data={"robot": {"make_model_guess": "Franka Emika Panda", "confidence": 0.99},
                                        "objects": [{}] * 4, "phases": [{}] * 3}),
        _stage("robot_onboard", data={"source": "robots/franka_panda.json"}, headline="franka_panda · self-test 5/5"),
        _stage("camera_solve", data={"iou": 0.98}),
        _stage("loop1_static", data={"alignment": 0.94, "history": hist},
               headline="alignment 62 % → 94 % in 1 iteration(s)"),
        _stage("motion_extract", data={"segments": [{}, {}], "dynamic": {"replay_success": True, "end_pose_err_mm": 12.4}}),
        _stage("replay", data={"success": True}, headline="replay succeeds on the nominal twin"),
        _stage("benchmark", data={"lock_sha256": LOCK, "dev": {"n": 12, "successes": 5, "rate": 0.417,
                                                               "clusters": {"BARRIER_COLLISION": 6}},
                                  "held_out": {"n": 20, "successes": 6, "rate": 0.3, "collisions": 115}}),
        _stage("loop2_critic", data={"critique": [{"timestamp_sec": 3.1, "video_frame_idx": 93,
                                                   "observation": "clips the barrier"}],
                                     "root_causes": ["carry height ignores the barrier"],
                                     "diff": [{"object_id": "cube", "field": "transit_apex_z_m", "old": 0.17,
                                               "new": {"terms": ["max_scene_top_z"], "offset": 0.06}},
                                              {"object_id": "cyl", "field": "transit_via_mode", "old": "midpoint",
                                               "new": "none"}],
                                     "dev": {"n": 12, "successes": 6, "rate": 0.5},
                                     "held_out": {"n": 20, "successes": 14, "rate": 0.7, "collisions": 68}}),
        _stage("loop3_swarm", "running"),
        _stage("report", "pending"),
    ]
    man = RunManifest(run_id=d.name, kind="pipeline", video="demo/clip.mp4", prompt="sort the blocks",
                      created="2026-10-25T12:00:00-07:00", stages=stages)
    (d / "manifest.json").write_text(man.model_dump_json(indent=1))
    return d


def test_scene_geoms_weld_panda_meshes(scene):
    _, _, m = scene
    specs = scene_geoms(m)
    meshes = [s for s in specs if s.kind == "mesh"]
    raw, welded = sum(s.raw_vertices for s in meshes), sum(len(s.vertices) for s in meshes)
    assert raw > 300_000 and welded < raw / 4, (raw, welded)
    assert all(s.faces.max() < len(s.vertices) for s in meshes)
    assert {"box", "cylinder"} <= {s.kind for s in specs}                   # table + task objects
    assert all(int(m.geom_group[s.gid]) in (0, 2) for s in specs)           # no collision-only geoms


def test_story_steps_from_partial_run(tmp_path, scene):
    d = make_run(tmp_path, scene)
    steps = build_story(d)
    assert [s.n for s in steps] == list(range(1, 9))
    num = {s.n: s.number for s in steps}
    assert num == {1: "accepted", 2: "3 min 00 s", 3: "94 %", 4: "12 mm", 5: "30 %", 6: "2", 7: "70 %", 8: "—"}
    assert [s.done for s in steps] == [True, True, True, True, True, True, False, True]
    assert [it["label"] for it in steps[2].extra["iterations"]] == ["iteration 1", "final"]
    assert steps[2].extra["iterations"][0]["alignment"] == 0.94
    assert [r["draft"] for r in steps[6].extra["ratchet"]] == ["draft1", "draft2"]
    assert steps[5].extra["critique"][0]["timestamp_sec"] == 3.1 and "max_scene_top_z + 0.06" in steps[5].extra["diff_md"]
    assert steps[7].extra["drafts"] == ["draft1"] and drafts(d) == ["draft1"]
    chips = {c.name: c for c in timeline(read_manifest(d))}
    assert chips["loop3_swarm"].colour == STATUS_COLOUR["running"] and chips["report"].colour == STATUS_COLOUR["pending"]
    rep = load_replay(d / "draft1/nominal")
    assert len(rep.t) == 60 and rep.xpos.shape[1:] == (scene[2].nbody, 3) and rep.phase[0] == "reach cube"


@pytest.mark.skipif(not os.environ.get("WORLDGYM_EXAMPLE_RUN"), reason="set WORLDGYM_EXAMPLE_RUN to a finished run")
def test_story_example_run():
    steps = build_story(os.environ["WORLDGYM_EXAMPLE_RUN"])
    assert [s.done for s in steps[:3]] == [False, False, False]              # policy-kind: task-only placeholders
    assert (steps[4].number, steps[5].number, steps[6].number) == ("30 %", "10", "65 %")
    assert [round(r["held_out"], 2) for r in steps[6].extra["ratchet"]] == [0.3, 0.7, 0.65]


def test_server_starts_and_stops(tmp_path, scene, monkeypatch):
    import viser

    from app.studio import Studio

    d = make_run(tmp_path / "runs", scene)
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
    try:
        studio = Studio(server)
        studio.poll()
        assert studio.run_dir == d and studio.replay is not None
        assert studio.mirror.stats["welded_vertices"] < studio.mirror.stats["raw_vertices"] / 4
        for k in range(8):
            studio.show_step(k, act=True)
        assert json.loads(json.dumps(studio.overrides()))["objects"]["barrier"]["dims_m"][2] == pytest.approx(0.09)
    finally:
        server.stop()

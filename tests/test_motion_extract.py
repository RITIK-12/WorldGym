"""Motion extraction acceptance on the synthetic clip: labels synthesised from ground truth (projected TCP, gripper
width, phase) -> carry waypoint error < 2 cm and the Draft 1 replay succeeds in the nominal twin."""
import json
from pathlib import Path

import numpy as np

from core.camera_solve import project_points
from core.motion_extract import extract
from core.rollout import build_sim, run_episode
from core.schemas import GripperLabel, Keyframe, TaskSpec

REPO = Path(__file__).resolve().parents[1]
GT = json.loads((REPO / "demo/synthetic_panda_sort.gt.json").read_text())
CARRY = {"close", "lift_clear", "transit_arc", "lower_place"}


def _labels(sim, w, h):
    out = []
    for f in GT["frames"][::8]:
        ph = next((p for p in GT["phase_timeline"] if p["t0"] <= f["t"] <= p["t1"]), None)
        u, v = project_points(sim.m, sim.d, "video", [f["tcp"]], w, h)[0]
        closed = f["width"] < 0.07
        held = ph["object"] if ph and ph["phase"] in CARRY and closed else None
        out.append(Keyframe(t=f["t"], gripper=[GripperLabel(uv=[v / h * 1000, u / w * 1000], open=not closed,
                                                            held=held)]))
    return out


def test_extract_matches_ground_truth_and_replays():
    task = TaskSpec.model_validate(GT["task"])
    sim = build_sim(task)
    pol, log = extract(task, _labels(sim, 1280, 720), sim, 1280, 720)
    assert [p.object_id for p in pol.primitives] == ["cube", "cyl"]
    assert all(p.target_id == "bin" for p in pol.primitives)
    gt_t = np.array([f["t"] for f in GT["frames"]])
    gt_tcp = np.array([f["tcp"] for f in GT["frames"]])
    errs = [np.linalg.norm(np.array(c["xyz"]) - gt_tcp[np.argmin(np.abs(gt_t - c["t"]))])
            for s in log["segments"] for c in s["carry"]]
    assert np.median(errs) < 0.02 and len(errs) > 10
    assert run_episode(task, pol).success

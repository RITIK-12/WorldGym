"""Re-run only the four-view critic (Gen 3) of an existing run, starting from its Gen 2.

    .venv/bin/python -m scripts.redo_final_critic runs/<id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from core.argon import Argon
from core.camera_track import camera_at
from core.first_world import arm_bases, demonstrated_motion, robot_frame
from core.mjcf_compiler import load_adapter
from core.replay_critic import run_final_critic
from core.runs import load_run
from core.schemas import Keyframe, PolicyConfig, TaskSpec, VideoUnderstanding
from core.twin import end_pose_error, publish_gen, real_end_positions


def main(run_dir: str) -> None:
    run = load_run(Path(run_dir))
    d = run.dir
    m = run.manifest
    task = TaskSpec.model_validate_json((d / "gens/gen2/task.json").read_text())
    draft = PolicyConfig.model_validate_json((d / "gens/gen2/policy.json").read_text())
    adapter = load_adapter(task.robot)
    labels = [Keyframe.model_validate(k) for k in json.loads((d / "motion/labels.json").read_text())]
    track_p = d / "camera/track.json"
    track = json.loads(track_p.read_text()) if track_p.exists() else None
    cam_at = (lambda t: camera_at(track, t)) if track else None
    log = json.loads((d / "motion/motion_extract.json").read_text())
    segs = log.get("segments") or []
    vu = json.loads((d / "architect/vu.json").read_text())
    grasp_t = segs[0]["grasp_t"] if segs else min((p["t0"] for p in vu["phases"] if p["action"] == "grasp"), default=0.0)
    real_end = real_end_positions(m.video, task, labels, cam_at)
    origin, fwd, left = robot_frame(adapter)
    frame_txt = (f"World frame: the robot's forward is world {fwd.round(3).tolist()}, its left is world "
                 f"{left.round(3).tolist()}, origin {origin.round(3).tolist()}; arm bases "
                 f"{ {k: v.round(3).tolist() for k, v in arm_bases(adapter).items()} }.\n" + demonstrated_motion(VideoUnderstanding.model_validate(vu)))
    argon = Argon(d / "argon")
    with run.stage("final_critic") as s:
        rc = run_final_critic(m.video, d / "architect/clip_small.mp4", task, draft, adapter, argon, grasp_t,
                              lambda t, e: end_pose_error(t, e, real_end)[0], d / "final_critic", cam_at=cam_at,
                              robot_frame=frame_txt)
        e0, e1 = rc["start"]["end_pose_err_mm"], rc["end_pose_err_mm"]
        kept = any(h.get("kept") for h in rc["history"])
        errored = bool(rc["history"]) and all(h.get("error") for h in rc["history"])
        s.headline = ("four-view critique " + ("applied" if kept else "did not complete (API error): Gen 2 kept"
                                               if errored else "proposed, not better: Gen 2 kept")
                      + (f" · end error {e0:.0f} → {e1:.0f} mm" if e0 is not None and e1 is not None else ""))
        s.data = {"history": rc["history"]}
        publish_gen(run, 3, "Refined by the four-view critic", s.headline, rc["task"], adapter,
                    rc["draft"].model_copy(update={"name": "draft1"}))
    print(s.headline)


if __name__ == "__main__":
    main(sys.argv[1])

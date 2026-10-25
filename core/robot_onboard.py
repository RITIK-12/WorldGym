"""Robot onboarding: Menagerie model -> RobotAdapter JSON -> self-test.

Known robots load robots/<id>.json. For an unfamiliar robot, Argon reads a structural summary of the Menagerie MJCF
(bodies, joints, actuators, sites, tendons) and writes the adapter; the self-test result is fed back until it passes.

Self-test (must pass before the robot is used):
1. the model compiles; 2. the home pose is collision-free; 3. IK reaches a 5x5 grid over the workspace
(x 0.3-0.6, y -0.25-0.25) at z 0.02 / 0.15 / 0.30 with error < 5 mm; 4. the gripper holds a 0.1 kg box during a
0.2 m lift; 5. joint velocities stay within the datasheet limits.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from core.argon import ARGON, Argon
from core.controller import ArmController, top_down_R
from core.mjcf_compiler import MENAGERIE, ROBOTS, compile_scene
from core.schemas import ObjectSpec, PolicyConfig, Predicate, Primitive, RobotAdapter, TaskSpec

IK_TOL_M = 0.005


def known_adapters() -> dict[str, RobotAdapter]:
    out = {}
    for p in ROBOTS.glob("*.json"):
        a = RobotAdapter.model_validate_json(p.read_text())
        out[a.menagerie_dir] = a
    return out


def ensure_menagerie(dir_name: str) -> Path:
    d = MENAGERIE / dir_name
    if not d.exists():
        subprocess.run(["git", "-C", str(MENAGERIE), "sparse-checkout", "add", dir_name], check=True, timeout=300)
    return d


def mjcf_summary(dir_name: str) -> str:
    """Compact structural description of a Menagerie robot for the onboarding prompt."""
    d = ensure_menagerie(dir_name)
    xmls = sorted(p.name for p in d.glob("*.xml") if "scene" not in p.name and "mjx" not in p.name)
    lines = [f"directory: {dir_name}", f"xml files: {xmls}"]
    for name in xmls[:3]:
        try:
            m = mujoco.MjModel.from_xml_path(str(d / name))
        except Exception as e:  # noqa: BLE001
            lines.append(f"{name}: does not compile standalone ({e})")
            continue
        nm = lambda t, i: mujoco.mj_id2name(m, t, i)  # noqa: E731
        lines.append(f"\n== {name}: nq={m.nq} nu={m.nu}")
        for b in range(1, m.nbody):
            lines.append(f"body {nm(mujoco.mjtObj.mjOBJ_BODY, b)} parent={nm(mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b])} "
                         f"pos={np.round(m.body_pos[b], 4).tolist()}")
        for j in range(m.njnt):
            lines.append(f"joint {nm(mujoco.mjtObj.mjOBJ_JOINT, j)} type={int(m.jnt_type[j])} "
                         f"range={np.round(m.jnt_range[j], 4).tolist()} body={nm(mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j])}")
        for a in range(m.nu):
            tgt = (nm(mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]) if m.actuator_trntype[a] == 0
                   else nm(mujoco.mjtObj.mjOBJ_TENDON, m.actuator_trnid[a, 0]))
            lines.append(f"actuator {nm(mujoco.mjtObj.mjOBJ_ACTUATOR, a)} -> {tgt} gain={np.round(m.actuator_gainprm[a, :1], 4).tolist()} "
                         f"bias={np.round(m.actuator_biasprm[a, :3], 4).tolist()} ctrlrange={np.round(m.actuator_ctrlrange[a], 4).tolist()}")
        for s in range(m.nsite):
            lines.append(f"site {nm(mujoco.mjtObj.mjOBJ_SITE, s)} body={nm(mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[s])} "
                         f"pos={np.round(m.site_pos[s], 4).tolist()}")
        for t in range(m.ntendon):
            lines.append(f"tendon {nm(mujoco.mjtObj.mjOBJ_TENDON, t)}")
        if m.nkey:
            lines.append(f"keyframe qpos={np.round(m.key_qpos[0], 4).tolist()}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------------------------------

def _empty_task(adapter: RobotAdapter, objects=(), preds=()) -> TaskSpec:
    return TaskSpec(name=f"selftest_{adapter.robot_id}", robot=adapter.robot_id, objects=list(objects),
                    immutable_success_predicates=list(preds))


def arm_base(m: mujoco.MjModel, d: mujoco.MjData, arm) -> tuple[np.ndarray, float]:
    """Position and yaw of an arm's own base link (the parent of its first joint)."""
    first = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, arm.joints[0])
    b = m.body_parentid[m.jnt_bodyid[first]]
    R = d.xmat[b].reshape(3, 3)
    return d.xpos[b].copy(), float(np.arctan2(R[1, 0], R[0, 0]))


def reach_workspace(m: mujoco.MjModel, ctrl: ArmController, base: np.ndarray, n: int = 6000, seed: int = 0) -> dict:
    """Workspace grid scaled to the arm, in its base frame: sample joint configurations, keep those with the TCP
    pointing down at a table-top height, and use the 90th percentile of their horizontal reach (a Panda gives roughly
    x 0.3-0.6 m) and the median height reached pointing down inside that band (capped at 0.30 m)."""
    rng = np.random.default_rng(seed)
    reach = []
    saved = ctrl.q_cmd.copy()
    for _ in range(n):
        ctrl.q_cmd = rng.uniform(ctrl.lo, ctrl.hi)
        p, R = ctrl.cmd_pose()
        if R[2, 2] < -0.9 and 0.02 <= p[2] <= 0.30:
            reach.append((float(np.linalg.norm(p[:2] - base[:2])), float(p[2])))
    ctrl.q_cmd = saved
    r = float(np.percentile([h for h, _ in reach], 90)) if len(reach) >= 20 else 0.6
    near = [z for h, z in reach if 0.4 * r <= h <= 0.75 * r]
    ztop = float(np.clip(np.percentile(near, 50), 0.10, 0.30)) if len(near) >= 10 else 0.30
    return {"x": [round(max(0.15, 0.4 * r), 3), round(0.75 * r, 3)], "y": round(0.3 * r, 3),
            "z": [0.02, round((0.02 + ztop) / 2, 3), round(ztop, 3)], "reach_m": round(r, 3),
            "n_downward_samples": len(reach)}


def frame_diagnostics(m: mujoco.MjModel, d: mujoco.MjData, ctrl: ArmController, arm) -> dict:
    """Where the TCP frame points and where the fingers are, at the home pose (feedback for fixing an adapter)."""
    saved = d.qpos.copy()
    d.qpos[ctrl.g_qadr] = arm.gripper.max_width_m / 2   # fingers open so their separation shows the closing axis
    mujoco.mj_forward(m, d)
    p, R = ctrl.tcp_pose()
    fingers = [d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy() for b in arm.finger_bodies]
    width = ctrl.gripper_width()
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    out = {"tcp_world": np.round(p, 4).tolist(),
           "tcp_axes_world": {k: np.round(R[:, i], 3).tolist() for i, k in enumerate("xyz")},
           "gripper_width_open": round(width, 4)}
    if len(fingers) >= 2:
        mid = np.mean(fingers, 0)
        sep = fingers[0] - fingers[1]
        out["finger_bodies_mid_in_tcp_frame"] = np.round(R.T @ (mid - p), 4).tolist()
        out["finger_separation_dir_in_tcp_frame"] = np.round(R.T @ sep / (np.linalg.norm(sep) + 1e-9), 3).tolist()
    return out


def _arm_checks(adapter: RobotAdapter, m: mujoco.MjModel, d: mujoco.MjData, arm) -> dict[str, dict]:
    from core.rollout import run_episode  # local import: rollout imports this module's neighbours

    checks: dict[str, dict] = {}
    ctrl = ArmController(m, d, arm)
    robot_b = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in arm.robot_bodies}
    pen = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1),
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2), round(float(d.contact[i].dist), 4))
           for i in range(d.ncon) if d.contact[i].dist < -0.001
           and (m.geom_bodyid[d.contact[i].geom1] in robot_b or m.geom_bodyid[d.contact[i].geom2] in robot_b)]
    checks["home_collision_free"] = {"ok": not pen, "penetrations": pen[:6]}
    checks["frames"] = {"ok": True, **frame_diagnostics(m, d, ctrl, arm)}

    base, yaw = arm_base(m, d, arm)
    c, s = np.cos(yaw), np.sin(yaw)
    to_world = lambda x, y: base[:2] + np.array([c * x - s * y, s * x + c * y])  # noqa: E731
    ws = reach_workspace(m, ctrl, base)
    errs, fails = [], []
    for z in ws["z"]:
        for x in np.linspace(*ws["x"], 5):
            for y in np.linspace(-ws["y"], ws["y"], 5):
                ctrl.q_cmd = np.array(arm.home_qpos, float)
                _, e = ctrl.solve_ik(np.array([*to_world(x, y), z]), top_down_R(yaw, arm.closing_axis))
                errs.append(e)
                if e >= IK_TOL_M:
                    fails.append([round(float(x), 3), round(float(y), 3), z, round(e * 1000, 1)])
    errs = np.array(errs)
    checks["ik_grid"] = {"ok": bool(errs.max() < IK_TOL_M), "max_err_mm": round(float(errs.max()) * 1000, 2),
                         "n_fail": int((errs >= IK_TOL_M).sum()), "n": len(errs), "workspace": ws,
                         "worst": sorted(fails, key=lambda f: -f[3])[:5]}

    bx, by = (round(float(v), 3) for v in to_world(float(np.mean(ws["x"])), 0.0))
    box = ObjectSpec(id="probe_box", shape="box", dims_m=[0.04, 0.04, 0.04], xy=[bx, by], yaw=yaw, mass=0.1,
                     friction=0.9)
    pol = PolicyConfig(name="selftest", primitives=[Primitive(arm=arm.name, object_id="probe_box", target_id=None,
                                                              post_grasp_lift_z_m=0.24, transit_apex_z_m=0.24,
                                                              pre_place_hover_z_m=0.24, place_xy_offset_m=[bx, by],
                                                              release_drop_height_m=0.2)])
    try:
        box_task = _empty_task(adapter, [box])
        _, mb = compile_scene(box_task, adapter)
        ep = run_episode(box_task, pol, adapter, max_time=20)
        z = ep.arrays["qpos"][:, mb.jnt_qposadr[mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_JOINT, "probe_box_free")] + 2]
        rise = float(z.max() - z[0])
        checks["gripper_hold_lift"] = {"ok": rise > 0.18, "box_rise_m": round(rise, 3)}
        qadr = [mb.jnt_qposadr[mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm.joints]
        qv = np.diff(ep.arrays["qpos"][:, qadr], axis=0) / mb.opt.timestep
        ratio = float((np.abs(qv) / np.array(arm.vel_limits)).max())
        checks["velocity_limits"] = {"ok": ratio <= 1.1, "max_ratio": round(ratio, 3)}
    except Exception as e:  # noqa: BLE001
        checks["gripper_hold_lift"] = {"ok": False, "error": str(e)[:500]}
    return checks


def self_test(adapter: RobotAdapter) -> dict:
    """Every arm: home collision-free, frame diagnostics, IK grid in its base frame, gripper hold + lift, velocity
    limits. Single-arm results are flat; with several arms each check aggregates `per_arm` details."""
    checks: dict[str, dict] = {}
    try:
        _, m = compile_scene(_empty_task(adapter), adapter)
        checks["compiles"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        checks["compiles"] = {"ok": False, "error": str(e)[:500]}
        return {"ok": False, "checks": checks}
    d = mujoco.MjData(m)
    try:
        ctrls = [ArmController(m, d, arm) for arm in adapter.arms]
    except Exception as e:  # noqa: BLE001
        checks["names"] = {"ok": False, "error": f"adapter names do not match the model: {e}"}
        return {"ok": False, "checks": checks}
    for arm, ctrl in zip(adapter.arms, ctrls):
        ctrl.set_joints(arm.home_qpos)
    mujoco.mj_forward(m, d)
    per_arm = {arm.name: _arm_checks(adapter, m, d, arm) for arm in adapter.arms}
    if len(per_arm) == 1:
        checks.update(next(iter(per_arm.values())))
    else:
        for name in next(iter(per_arm.values())):
            checks[name] = {"ok": all(pa.get(name, {"ok": False})["ok"] for pa in per_arm.values()),
                            "per_arm": per_arm and {a: pa.get(name) for a, pa in per_arm.items()}}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def set_rest_pose(sim) -> None:
    """Put every arm at its rest pose (camera calibration frames show the robot idle)."""
    for arm in sim.ix.adapter.arms:
        sim.ctrls[arm.name].set_joints(arm.rest_qpos or arm.home_qpos)
    mujoco.mj_forward(sim.m, sim.d)


# ----------------------------------------------------------------------------------------------------------------------
# Onboarding
# ----------------------------------------------------------------------------------------------------------------------

ONBOARD_SYSTEM = """You onboard robot arms into WorldGym, a MuJoCo pipeline. Given a structural summary of a MuJoCo
Menagerie robot, write a RobotAdapter JSON. Rules:
- `tcp` is the point midway between the fingertip pads, on the last hand body, with its quat chosen so the TCP z axis
  points OUT of the fingers (from the wrist toward the object) and the fingers close along TCP `closing_axis`.
- gripper.kind is "tendon_force" when the gripper actuator drives a tendon, else "joint_force"; `joints` are the finger
  joints whose positions sum to the opening width; max_width_m is the full opening.
- home_qpos is a ready pose with the gripper pointing straight DOWN (TCP z axis ~ [0, 0, -1] in the world), the TCP
  0.2-0.4 m above the table and in front of the base, all joints inside their ranges, away from singularities.
- vel_limits are datasheet joint speed limits in rad/s; robot_bodies lists every arm and hand body; finger_bodies
  lists the bodies that touch the object.
The self-test reports `frames` (TCP position and axes in the world at home, finger midpoint and separation direction
in the TCP frame): use it to check the TCP orientation and the home pose."""


def onboard(menagerie_dir: str, argon: Argon | None = None, max_iters: int = 3, log_dir: Path | None = None) -> dict:
    """Return {adapter, self_test, source}. Uses a stored adapter when one exists for this Menagerie model."""
    known = known_adapters()
    if menagerie_dir in known:
        a = known[menagerie_dir]
        return {"adapter": a, "self_test": self_test(a), "source": f"robots/{a.robot_id}.json"}
    if argon is None:
        raise ValueError(f"no stored adapter for {menagerie_dir} and no model available to write one")
    summary = mjcf_summary(menagerie_dir)
    example = (ROBOTS / "franka_panda.json").read_text()
    feedback = ""
    for it in range(max_iters):
        prompt = (f"Robot summary:\n{summary}\n\nExample adapter (Franka Panda):\n{example}\n\n"
                  f"Write the RobotAdapter for `{menagerie_dir}`.{feedback}")
        adapter, _ = argon.ask_json(f"onboard_{menagerie_dir}_{it}", ONBOARD_SYSTEM, prompt, RobotAdapter, model=ARGON)
        result = self_test(adapter)
        if log_dir:
            (Path(log_dir) / f"onboard_{it}.json").write_text(json.dumps({"adapter": adapter.model_dump(),
                                                                          "self_test": result}, indent=1))
        if result["ok"]:
            (ROBOTS / f"{adapter.robot_id}.json").write_text(adapter.model_dump_json(indent=1))
            return {"adapter": adapter, "self_test": result, "source": f"argon (iteration {it + 1})"}
        feedback = f"\n\nYour previous adapter failed the self-test:\n{json.dumps(result['checks'])}\nFix it."
    return {"adapter": adapter, "self_test": result, "source": "argon (failed self-test)"}


if __name__ == "__main__":
    import sys

    print(json.dumps(self_test(RobotAdapter.model_validate_json(Path(sys.argv[1]).read_text())), indent=1))


AGENT_ONBOARD_PROMPT = """You are onboarding the robot `{menagerie_dir}` into WorldGym (a MuJoCo pipeline mounted at ./worldgym).
Setup: cd worldgym && pip install uv && uv sync && scripts/setup_assets.sh {menagerie_dir}
Goal: write robots/{robot_id}.json (a RobotAdapter, see core/schemas.py and the example robots/franka_panda.json) so
that `uv run python -m core.robot_onboard robots/{robot_id}.json` reports "ok": true for every check (compiles,
home_collision_free, ik_grid, gripper_hold_lift, velocity_limits). Read the Menagerie MJCF under
assets/menagerie/{menagerie_dir}/ and core/robot_onboard.py, core/controller.py, core/mjcf_compiler.py to see how the
adapter is used. Write small scripts to inspect the model (body frames, joint ranges, finger axes) and to search for a
home pose with the gripper pointing down. Iterate until the self-test passes or you run out of ideas.
Do NOT modify anything under core/ or tests/: only robots/{robot_id}.json and your own scratch scripts.
Finish with ONE JSON object as your final answer: {{"adapter": <the RobotAdapter JSON>, "self_test": <the final
self-test JSON>, "notes": "what you found about this robot"}}."""


def onboard_with_agent(menagerie_dir: str, robot_id: str, argon: Argon, log_dir: Path,
                       timeout_s: float = 2400) -> dict:
    """Antigravity Managed Agent writes the adapter in a sandbox with the self-test as its verification loop; the
    returned adapter is re-verified locally before it is stored."""
    import time

    from core.argon import check_budget, parse_json, record_spend
    from core.swarm import AGENT, AGENT_MAX_TOKENS, ALLOWLIST, _inline_sources

    client = argon.client
    env = {"type": "remote", "sources": _inline_sources({}),
           "network": {"allowlist": [{"domain": d} for d in ALLOWLIST]}}
    prompt = AGENT_ONBOARD_PROMPT.format(menagerie_dir=menagerie_dir, robot_id=robot_id)
    (log_dir / "agent_prompt.md").write_text(prompt)
    check_budget("agent", f"onboard_{robot_id}")
    it = client.interactions.create(agent=AGENT, agent_config={"type": "antigravity", "model": ARGON,
                                                               "max_total_tokens": AGENT_MAX_TOKENS},
                                    input=prompt, environment=env, background=True)
    record_spend("agent", f"onboard_{robot_id}", AGENT, None, True, it.id)
    t0 = time.time()
    while it.status in ("in_progress", "queued") and time.time() - t0 < timeout_s:
        time.sleep(15)
        it = client.interactions.get(it.id)
    u = it.usage.model_dump() if getattr(it, "usage", None) is not None else {}
    record_spend("agent_usage", f"onboard_{robot_id}", AGENT, u, it.status == "completed", it.id)
    text = it.output_text or ""
    (log_dir / "agent_output.md").write_text(text)
    try:
        out = parse_json(text)
        adapter = RobotAdapter.model_validate(out["adapter"])
    except Exception as e:  # noqa: BLE001
        return {"adapter": None, "self_test": {"ok": False}, "source": f"agent ({it.status}): {e}"[:300],
                "seconds": round(time.time() - t0)}
    local = self_test(adapter)   # trust, but verify on our machine
    if local["ok"]:
        (ROBOTS / f"{adapter.robot_id}.json").write_text(adapter.model_dump_json(indent=1))
    return {"adapter": adapter, "self_test": local, "agent_self_test": out.get("self_test"),
            "notes": out.get("notes", ""), "source": "antigravity agent", "seconds": round(time.time() - t0)}

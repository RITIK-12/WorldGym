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


def self_test(adapter: RobotAdapter) -> dict:
    checks: dict[str, dict] = {}
    try:
        _, m = compile_scene(_empty_task(adapter), adapter)
        checks["compiles"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        checks["compiles"] = {"ok": False, "error": str(e)[:500]}
        return {"ok": False, "checks": checks}

    d = mujoco.MjData(m)
    arm = adapter.arms[0]
    try:
        ctrl = ArmController(m, d, arm)
    except Exception as e:  # noqa: BLE001
        checks["names"] = {"ok": False, "error": f"adapter names do not match the model: {e}"}
        return {"ok": False, "checks": checks}
    ctrl.set_joints(arm.home_qpos)
    mujoco.mj_forward(m, d)
    robot_b = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in arm.robot_bodies}
    pen = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1),
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2), round(float(d.contact[i].dist), 4))
           for i in range(d.ncon) if d.contact[i].dist < -0.001
           and (m.geom_bodyid[d.contact[i].geom1] in robot_b or m.geom_bodyid[d.contact[i].geom2] in robot_b)]
    checks["home_collision_free"] = {"ok": not pen, "penetrations": pen[:6]}

    errs = []
    for z in (0.02, 0.15, 0.30):
        for x in np.linspace(0.3, 0.6, 5):
            for y in np.linspace(-0.25, 0.25, 5):
                ctrl.q_cmd = np.array(arm.home_qpos, float)
                _, e = ctrl.solve_ik(np.array([x, y, z]), top_down_R(0.0, arm.closing_axis))
                errs.append(e)
    errs = np.array(errs)
    checks["ik_grid"] = {"ok": bool(errs.max() < IK_TOL_M), "max_err_mm": round(float(errs.max()) * 1000, 2),
                         "n_fail": int((errs >= IK_TOL_M).sum()), "n": len(errs)}

    from core.rollout import run_episode  # local import: rollout imports this module's neighbours

    box = ObjectSpec(id="probe_box", shape="box", dims_m=[0.04, 0.04, 0.04], xy=[0.45, 0.0], mass=0.1, friction=0.9)
    pol = PolicyConfig(name="selftest", primitives=[Primitive(arm=arm.name, object_id="probe_box", target_id=None,
                                                              post_grasp_lift_z_m=0.24, transit_apex_z_m=0.24,
                                                              pre_place_hover_z_m=0.24, place_xy_offset_m=[0.45, 0.0],
                                                              release_drop_height_m=0.2)])
    try:
        box_task = _empty_task(adapter, [box])
        _, mb = compile_scene(box_task, adapter)
        ep = run_episode(box_task, pol, adapter, max_time=20)
        a = ep.arrays
        lifted = a["phase"] == "lift_clear"
        z = a["qpos"][:, mb.jnt_qposadr[mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_JOINT, "probe_box_free")] + 2]
        rise = float(z[lifted].max() - z[0]) if lifted.any() else 0.0
        checks["gripper_hold_lift"] = {"ok": rise > 0.18, "box_rise_m": round(rise, 3)}
        qadr = [mb.jnt_qposadr[mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm.joints]
        qv = np.diff(a["qpos"][:, qadr], axis=0) / mb.opt.timestep
        ratio = float((np.abs(qv) / np.array(arm.vel_limits)).max())
        checks["velocity_limits"] = {"ok": ratio <= 1.1, "max_ratio": round(ratio, 3)}
    except Exception as e:  # noqa: BLE001
        checks["gripper_hold_lift"] = {"ok": False, "error": str(e)[:500]}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


# ----------------------------------------------------------------------------------------------------------------------
# Onboarding
# ----------------------------------------------------------------------------------------------------------------------

ONBOARD_SYSTEM = """You onboard robot arms into WorldGym, a MuJoCo pipeline. Given a structural summary of a MuJoCo
Menagerie robot, write a RobotAdapter JSON. Rules: `tcp` is the point midway between the fingertip pads (add an
offset on the last hand body if no such site exists); `closing_axis` is the TCP-frame axis along which the fingers
close ("x" or "y") with the TCP z axis pointing out of the fingers; gripper.kind is "tendon_force" when the gripper
actuator drives a tendon, else "joint_force"; home_qpos is a natural ready pose with the hand ~0.4 m above the
table pointing down; vel_limits are datasheet joint speed limits in rad/s; robot_bodies lists every body of the arm
and hand; finger_bodies lists the bodies that touch the object."""


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

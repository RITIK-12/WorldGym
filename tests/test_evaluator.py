"""Referee tests: placed / outside / tilted / yawed-cylinder cases. Kinematic only (mj_forward, no dynamics)."""
import math

import mujoco
import numpy as np
import pytest

from core.locked_evaluator import evaluate, geom_aabb
from core.mjcf_compiler import compile_scene
from core.schemas import ObjectSpec, Predicate, TaskSpec

REST = [(0.0, 0.0), (1.0, 0.0)]


def _task(objs, preds):
    return TaskSpec(name="t", robot="franka_panda", objects=objs, immutable_success_predicates=preds)


def _bin(inner=0.10):
    w = 0.008
    return ObjectSpec(id="bin", shape="container", dims_m=[inner + 2 * w, inner + 2 * w, 0.06], xy=[0.5, 0.2],
                      is_fixed=True, wall_m=w)


def _place(m, d, name, pos, quat=(1, 0, 0, 0)):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
    a = m.jnt_qposadr[j]
    d.qpos[a:a + 3] = pos
    d.qpos[a + 3:a + 7] = quat


def _eval(task, placements):
    _, m = compile_scene(task)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    init = {o.id: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o.id)].copy() for o in task.objects}
    for name, pos, quat in placements:
        _place(m, d, name, pos, quat)
    mujoco.mj_forward(m, d)
    return evaluate(m, d, task, init, REST), m, d


def _quat_z(yaw):
    return (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))


def _quat_x(a):
    return (math.cos(a / 2), math.sin(a / 2), 0, 0)


CUBE = ObjectSpec(id="cube", shape="box", dims_m=[0.04, 0.04, 0.04], xy=[0.4, -0.1])
PRED = [Predicate(kind="inside_container", object_id="cube", target_id="bin")]


def test_placed_inside():
    r, *_ = _eval(_task([CUBE, _bin()], PRED), [("cube", (0.5, 0.2, 0.0282), (1, 0, 0, 0))])
    # bin centre z = 0.0301, floor top = 0.0081 -> cube centre = 0.0281
    assert r["success"], r


def test_outside():
    r, *_ = _eval(_task([CUBE, _bin()], PRED), [("cube", (0.5, 0.32, 0.0201), (1, 0, 0, 0))])
    assert not r["success"]
    assert not r["predicates"][0]["in_xy"]


def test_straddling_wall_fails():
    r, *_ = _eval(_task([CUBE, _bin()], PRED), [("cube", (0.5, 0.26, 0.0601), (1, 0, 0, 0))])
    assert not r["success"]


def test_tilted_cylinder_fails():
    cyl = ObjectSpec(id="cyl", shape="cylinder", dims_m=[0.04, 0.04, 0.07], xy=[0.4, -0.1])
    pred = [Predicate(kind="inside_container", object_id="cyl", target_id="bin", max_tilt_deg=25)]
    r, *_ = _eval(_task([cyl, _bin(0.10)], pred), [("cyl", (0.5, 0.2, 0.008 + 0.02 + 0.0001), _quat_x(math.pi / 2))])
    p = r["predicates"][0]
    assert p["in_xy"] and p["tilt_deg"] > 80 and not r["success"], p


def test_cube_on_side_is_upright():
    r, *_ = _eval(_task([CUBE, _bin()], PRED), [("cube", (0.5, 0.2, 0.0281), _quat_x(math.pi / 2))])
    assert r["success"], r


def test_yawed_cylinder_exact_aabb():
    # inner 0.064: a d=0.06 cylinder fits at any yaw; a box approximation (r, r) yawed 45deg would be 0.085 wide
    cyl = ObjectSpec(id="cyl", shape="cylinder", dims_m=[0.06, 0.06, 0.05], xy=[0.4, -0.1])
    pred = [Predicate(kind="inside_container", object_id="cyl", target_id="bin")]
    r, m, d = _eval(_task([cyl, _bin(0.064)], pred), [("cyl", (0.5, 0.2, 0.008 + 0.025 + 0.0001), _quat_z(math.pi / 4))])
    assert r["success"], r
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cyl")
    lo, hi = geom_aabb(m, d, g)
    assert hi[0] - lo[0] == pytest.approx(0.06, abs=1e-6)


def test_obstacle_displacement_and_rest():
    bar = ObjectSpec(id="bar", shape="box", dims_m=[0.2, 0.02, 0.08], xy=[0.45, 0.0], is_obstacle=True)
    preds = [Predicate(kind="max_obstacle_displacement", object_id="bar", limit=0.01), Predicate(kind="terminal_rest")]
    task = _task([bar], preds)
    r, *_ = _eval(task, [("bar", (0.45, 0.005, 0.0401), (1, 0, 0, 0))])
    assert r["success"], r
    r, *_ = _eval(task, [("bar", (0.45, 0.05, 0.0401), (1, 0, 0, 0))])
    assert not r["predicates"][0]["ok"]
    _, m = compile_scene(task)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    moving = evaluate(m, d, task, {"bar": d.xpos[m.nbody - 1].copy()}, [(0.0, 0.0), (0.9, 0.05), (1.0, 0.0)])
    assert not moving["predicates"][1]["ok"]


def test_on_top_of():
    base = ObjectSpec(id="base", shape="box", dims_m=[0.06, 0.06, 0.04], xy=[0.45, 0.1])
    top = ObjectSpec(id="top", shape="box", dims_m=[0.04, 0.04, 0.04], xy=[0.4, -0.1])
    pred = [Predicate(kind="on_top_of", object_id="top", target_id="base")]
    task = _task([base, top], pred)
    r, *_ = _eval(task, [("top", (0.455, 0.1, 0.0601), (1, 0, 0, 0))])
    assert r["success"], r
    r, *_ = _eval(task, [("top", (0.52, 0.1, 0.0201), (1, 0, 0, 0))])
    assert not r["success"]


def test_unknown_object_raises():
    with pytest.raises(KeyError):
        _eval(_task([CUBE, _bin()], [Predicate(kind="inside_container", object_id="nope", target_id="bin")]), [])


def test_aabb_box_rotation():
    task = _task([CUBE], [])
    _, m = compile_scene(task)
    d = mujoco.MjData(m)
    _place(m, d, "cube", (0.4, 0, 0.1), _quat_z(math.pi / 4))
    mujoco.mj_forward(m, d)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    lo, hi = geom_aabb(m, d, g)
    assert hi[0] - lo[0] == pytest.approx(0.04 * math.sqrt(2), abs=1e-6)
    assert np.allclose((lo + hi) / 2, [0.4, 0, 0.1])


def test_wiped_coverage():
    bar = ObjectSpec(id="bar", shape="box", dims_m=[0.40, 0.03, 0.03], xy=[0.45, 0.0], is_fixed=True)
    sponge = ObjectSpec(id="sponge", shape="box", dims_m=[0.08, 0.05, 0.03], xy=[0.45, -0.2])
    pred = [Predicate(kind="wiped_coverage", object_id="sponge", target_id="bar", limit=0.6)]
    task = _task([bar, sponge], pred)
    _, m = compile_scene(task)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    full = [[0.45 + s, 0.0, 0.03] for s in np.linspace(-0.16, 0.16, 200)]
    half = [[0.45 + s, 0.0, 0.03] for s in np.linspace(-0.16, 0.0, 100)]
    ok = evaluate(m, d, task, {}, REST, wipe_contacts={"sponge|bar": full})
    part = evaluate(m, d, task, {}, REST, wipe_contacts={"sponge|bar": half})
    none = evaluate(m, d, task, {}, REST)
    assert ok["success"] and ok["predicates"][0]["coverage"] > 0.95
    assert not part["success"] and 0.4 < part["predicates"][0]["coverage"] < 0.6
    assert not none["success"] and none["predicates"][0]["coverage"] == 0.0

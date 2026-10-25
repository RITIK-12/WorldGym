"""TaskSpec + RobotAdapter -> MJCF.

The Menagerie robot XML is patched with ElementTree (not MjSpec) so the compiler does not depend on MuJoCo's
editing API. Steps: absolute meshdir, gravcomp, TCP site, named collision geoms, force-controlled gripper, no keyframe,
then table + objects + cameras + lights are appended.
"""
from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from core.schemas import CameraSpec, ObjectSpec, RobotAdapter, TaskSpec

REPO = Path(__file__).resolve().parents[1]
MENAGERIE = REPO / "assets" / "menagerie"
ROBOTS = REPO / "robots"

TIMESTEP = 0.002
PAD_CONTACT = dict(friction="1.5 0.02 0.0002", condim="4", solimp="0.95 0.99 0.001", solref="0.004 1")
OBJ_SOLVER = dict(condim="4", solimp="0.95 0.99 0.001", solref="0.004 1")
TABLE_SIZE = (0.55, 0.55, 0.02)
TABLE_POS = (0.35, 0.0, -0.02)
FLOOR_Z = -0.76

DEFAULT_CAMERAS = [
    CameraSpec(name="front", pos=[1.45, 0.0, 0.75], look_at=[0.45, 0.0, 0.05], fovy=45),
    CameraSpec(name="side", pos=[0.45, -1.15, 0.45], look_at=[0.45, 0.0, 0.08], fovy=45),
    CameraSpec(name="top", pos=[0.45, 0.0, 1.5], xyaxes=[0, 1, 0, -1, 0, 0], fovy=0.9, projection="orthographic"),
]


def load_adapter(robot_id: str) -> RobotAdapter:
    return RobotAdapter.model_validate_json((ROBOTS / f"{robot_id}.json").read_text())


def load_task(path: str | Path) -> TaskSpec:
    return TaskSpec.model_validate_json(Path(path).read_text())


# ----------------------------------------------------------------------------------------------------------------------
# Geometry helpers (shared with the evaluator, features and benchmark)
# ----------------------------------------------------------------------------------------------------------------------

def object_height(o: ObjectSpec) -> float:
    return o.dims_m[0] if o.shape == "sphere" else o.dims_m[2]


def object_bottom_z(task: TaskSpec, o: ObjectSpec) -> float:
    if o.z is not None:
        return o.z
    if o.on_top_of:
        s = task.obj(o.on_top_of)
        return object_bottom_z(task, s) + object_height(s)
    return 0.0


def look_at_xyaxes(pos, look, up=(0.0, 0.0, 1.0)) -> list[float]:
    pos, look, up = (np.asarray(v, float) for v in (pos, look, up))
    f = look - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:  # looking straight down/up: image right = +y
        x = np.array([0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return [float(v) for v in (*x, *y)]


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def object_geoms(o: ObjectSpec) -> list[tuple[str, str, list[float], list[float]]]:
    """(name, type, size, pos-in-body) for every geom of an object; body origin = bounding-box centre."""
    d = o.dims_m
    if o.shape == "box":
        return [(o.id, "box", [d[0] / 2, d[1] / 2, d[2] / 2], [0, 0, 0])]
    if o.shape == "cylinder":
        return [(o.id, "cylinder", [d[0] / 2, d[2] / 2], [0, 0, 0])]
    if o.shape == "sphere":
        return [(o.id, "sphere", [d[0] / 2], [0, 0, 0])]
    if o.shape == "capsule":
        r = d[0] / 2
        return [(o.id, "capsule", [r, max(d[2] / 2 - r, 1e-4)], [0, 0, 0])]
    if o.shape == "tblock":
        fd = o.flange_d_m or 0.4 * d[1]
        sw = o.stem_w_m or 0.4 * d[1]
        stem_len = d[1] - fd
        return [(f"{o.id}_flange", "box", [d[0] / 2, fd / 2, d[2] / 2], [0, d[1] / 2 - fd / 2, 0]),
                (f"{o.id}_stem", "box", [sw / 2, stem_len / 2, d[2] / 2], [0, -fd / 2, 0])]
    if o.shape == "container":
        X, Y, H, w = d[0], d[1], d[2], o.wall_m
        return [(f"{o.id}_floor", "box", [X / 2, Y / 2, w / 2], [0, 0, -H / 2 + w / 2]),
                (f"{o.id}_wall_px", "box", [w / 2, Y / 2, H / 2], [X / 2 - w / 2, 0, 0]),
                (f"{o.id}_wall_nx", "box", [w / 2, Y / 2, H / 2], [-X / 2 + w / 2, 0, 0]),
                (f"{o.id}_wall_py", "box", [X / 2 - w, w / 2, H / 2], [0, Y / 2 - w / 2, 0]),
                (f"{o.id}_wall_ny", "box", [X / 2 - w, w / 2, H / 2], [0, -Y / 2 + w / 2, 0])]
    raise ValueError(o.shape)


def _geom_volume(gtype: str, size: list[float]) -> float:
    if gtype == "box":
        return 8 * size[0] * size[1] * size[2]
    if gtype == "cylinder":
        return math.pi * size[0] ** 2 * 2 * size[1]
    if gtype == "sphere":
        return 4 / 3 * math.pi * size[0] ** 3
    if gtype == "capsule":
        return math.pi * size[0] ** 2 * 2 * size[1] + 4 / 3 * math.pi * size[0] ** 3
    raise ValueError(gtype)


# ----------------------------------------------------------------------------------------------------------------------
# Robot patching
# ----------------------------------------------------------------------------------------------------------------------

def _find_body(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(f"body {name!r} not in robot XML")


def _patch_robot(adapter: RobotAdapter, base_pose: list[float]) -> ET.Element:
    src = MENAGERIE / adapter.menagerie_dir / adapter.mjcf
    root = ET.parse(src).getroot()

    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    for key in ("meshdir", "texturedir", "assetdir"):
        comp.set(key, str((src.parent / comp.get(key, "")).resolve()))
    comp.set("angle", comp.get("angle", "radian"))

    for tag in ("keyframe",):
        for el in root.findall(tag):
            root.remove(el)

    wb = root.find("worldbody")
    for light in wb.findall("light"):
        wb.remove(light)
    robot_top = list(wb.findall("body"))
    base = ET.Element("body", name="robot_base", pos=_fmt(base_pose[:3]), euler=f"0 0 {base_pose[3]:.6g}")
    for b in robot_top:
        wb.remove(b)
        base.append(b)
    wb.insert(0, base)
    for b in base.iter("body"):
        b.set("gravcomp", "1")

    for arm in adapter.arms:
        tcp = _find_body(root, arm.tcp.body)
        tcp.append(ET.Element("site", name=f"tcp_{arm.name}", pos=_fmt(arm.tcp.pos), quat=_fmt(arm.tcp.quat),
                              size="0.006", rgba="1 0.2 0.2 0.0", group="5"))
        g = arm.gripper
        for act in root.iter():
            if act.get("name") == g.actuator and act.tag in ("general", "position", "motor"):
                act.tag = "general"
                for k in ("kp", "kv", "dampratio"):
                    act.attrib.pop(k, None)
                act.set("gainprm", "1 0 0")
                act.set("biasprm", "0 0 0")
                act.set("biastype", "none")
                act.set("ctrlrange", f"{-g.force_range} {g.force_range}")
                act.set("forcerange", f"{-g.force_range} {g.force_range}")
        for j in root.iter("joint"):
            if j.get("name") in g.joints:
                j.set("solreflimit", "0.004 1")
                j.set("solimplimit", "0.95 0.99 0.001")
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        fb = arm.finger_bodies
        for i in range(len(fb)):
            for k in range(i + 1, len(fb)):
                contact.append(ET.Element("exclude", body1=fb[i], body2=fb[k]))
    return root


def _name_robot_geoms(root: ET.Element, adapter: RobotAdapter) -> None:
    """Name unnamed collision geoms `<body>_col<k>` / `<body>_pad<k>` so telemetry can cite them."""
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    fingers = {b for a in adapter.arms for b in a.finger_bodies}
    for arm in adapter.arms:
        for bname in arm.robot_bodies:
            bel = _find_body(root, bname)
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
            adr = m.body_geomadr[bid]
            n_col = n_pad = 0
            for k, gel in enumerate(bel.findall("geom")):
                gid = adr + k
                collides = m.geom_contype[gid] != 0 or m.geom_conaffinity[gid] != 0
                if not collides:
                    continue
                if bname in fingers:
                    for key, val in PAD_CONTACT.items():
                        gel.set(key, val)
                if gel.get("name"):
                    continue
                if bname in fingers and m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    gel.set("name", f"{bname}_pad{n_pad}")
                    n_pad += 1
                else:
                    gel.set("name", f"{bname}_col{n_col}")
                    n_col += 1


# ----------------------------------------------------------------------------------------------------------------------
# Scene assembly
# ----------------------------------------------------------------------------------------------------------------------

def _scene_globals(root: ET.Element) -> None:
    for el in root.findall("option"):
        root.remove(el)
    root.insert(1, ET.Element("option", timestep=str(TIMESTEP), integrator="implicitfast", cone="elliptic",
                              impratio="1"))
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    for tag in ("global", "headlight", "map", "quality", "scale"):
        for el in vis.findall(tag):
            vis.remove(el)
    vis.append(ET.Element("global", offwidth="1920", offheight="1080", azimuth="150", elevation="-25"))
    vis.append(ET.Element("headlight", diffuse="0.35 0.35 0.35", ambient="0.28 0.28 0.28", specular="0 0 0"))
    vis.append(ET.Element("map", force="0.004"))
    vis.append(ET.Element("quality", shadowsize="2048"))
    vis.append(ET.Element("scale", forcewidth="0.012", contactwidth="0.02", contactheight="0.004"))
    root.append(ET.Element("statistic", center="0.4 0 0.15", extent="0.9"))


def _append_world(root: ET.Element, task: TaskSpec) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    asset.append(ET.Element("texture", name="wg_floor_tex", type="2d", builtin="checker", width="512", height="512",
                            rgb1="0.32 0.33 0.35", rgb2="0.27 0.28 0.30"))
    asset.append(ET.Element("material", name="wg_floor_mat", texture="wg_floor_tex", texrepeat="6 6"))
    wb = root.find("worldbody")
    wb.append(ET.Element("light", name="key", pos="0.9 -0.7 1.8", dir="-0.35 0.3 -1", diffuse="0.45 0.45 0.45",
                         specular="0.1 0.1 0.1", castshadow="true"))
    wb.append(ET.Element("light", name="fill", pos="-0.3 0.9 1.4", dir="0.35 -0.5 -1", diffuse="0.18 0.18 0.18",
                         specular="0 0 0", castshadow="false"))
    wb.append(ET.Element("geom", name="floor", type="plane", size="3 3 0.05", pos=f"0 0 {FLOOR_Z}",
                         material="wg_floor_mat"))
    wb.append(ET.Element("geom", name="table", type="box", size=_fmt(TABLE_SIZE), pos=_fmt(TABLE_POS),
                         rgba="0.58 0.52 0.44 1", friction="0.8 0.005 0.0001"))

    for o in task.objects:
        geoms = object_geoms(o)
        h = object_height(o)
        cz = object_bottom_z(task, o) + h / 2 + 1e-4
        body = ET.SubElement(wb, "body", name=o.id, pos=_fmt([o.xy[0], o.xy[1], cz]), euler=f"0 0 {o.yaw:.6g}")
        if not o.is_fixed:
            body.append(ET.Element("freejoint", name=f"{o.id}_free"))
        vol = sum(_geom_volume(t, s) for _, t, s, _ in geoms)
        density = o.mass / vol
        for name, gtype, size, pos in geoms:
            body.append(ET.Element("geom", name=name, type=gtype, size=_fmt(size), pos=_fmt(pos),
                                   rgba=_fmt(o.colour), density=f"{density:.6g}",
                                   friction=f"{o.friction:.4g} 0.005 0.0001", **OBJ_SOLVER))

    for c in (task.cameras or DEFAULT_CAMERAS):
        xy = c.xyaxes or look_at_xyaxes(c.pos, c.look_at or [0.45, 0, 0])
        attrs = dict(name=c.name, pos=_fmt(c.pos), xyaxes=_fmt(xy), fovy=f"{c.fovy:.6g}")
        if c.projection == "orthographic":
            attrs["projection"] = "orthographic"
        wb.append(ET.Element("camera", **attrs))


def compile_scene(task: TaskSpec, adapter: RobotAdapter | None = None) -> tuple[str, mujoco.MjModel]:
    adapter = adapter or load_adapter(task.robot)
    base_pose = (task.base_poses or adapter.base_poses)[0]
    root = _patch_robot(adapter, base_pose)
    root.set("model", f"worldgym_{task.name}")
    _scene_globals(root)
    _append_world(root, task)
    _name_robot_geoms(root, adapter)
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    return xml, mujoco.MjModel.from_xml_string(xml)


def save_scene(run_dir: str | Path, task: TaskSpec, xml: str) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scene.xml").write_text(xml)
    (run_dir / "task.json").write_text(task.model_dump_json(indent=1))


# ----------------------------------------------------------------------------------------------------------------------
# Index of ids used by the controller, telemetry and evaluator
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class SceneIndex:
    m: mujoco.MjModel
    task: TaskSpec
    adapter: RobotAdapter
    obj_body: dict[str, int] = field(default_factory=dict)
    obj_geoms: dict[str, list[int]] = field(default_factory=dict)
    obj_qadr: dict[str, int] = field(default_factory=dict)      # freejoint qpos address (free objects only)
    obj_dadr: dict[str, int] = field(default_factory=dict)
    geom_owner: dict[int, str] = field(default_factory=dict)    # geom id -> object id
    robot_geoms: set[int] = field(default_factory=set)
    finger_geoms: set[int] = field(default_factory=set)
    robot_bodies: set[int] = field(default_factory=set)
    table: int = -1
    floor: int = -1

    @classmethod
    def build(cls, m: mujoco.MjModel, task: TaskSpec, adapter: RobotAdapter) -> "SceneIndex":
        ix = cls(m=m, task=task, adapter=adapter)
        name2id = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
        for o in task.objects:
            b = name2id(mujoco.mjtObj.mjOBJ_BODY, o.id)
            ix.obj_body[o.id] = b
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            ix.obj_geoms[o.id] = gs
            for g in gs:
                ix.geom_owner[g] = o.id
            if not o.is_fixed:
                j = name2id(mujoco.mjtObj.mjOBJ_JOINT, f"{o.id}_free")
                ix.obj_qadr[o.id] = int(m.jnt_qposadr[j])
                ix.obj_dadr[o.id] = int(m.jnt_dofadr[j])
        for arm in adapter.arms:
            for bn in arm.robot_bodies:
                b = name2id(mujoco.mjtObj.mjOBJ_BODY, bn)
                ix.robot_bodies.add(b)
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
                        ix.robot_geoms.add(g)
                        if bn in arm.finger_bodies:
                            ix.finger_geoms.add(g)
        ix.table = name2id(mujoco.mjtObj.mjOBJ_GEOM, "table")
        ix.floor = name2id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        return ix

    def geom_name(self, g: int) -> str:
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"


if __name__ == "__main__":
    import sys

    t = load_task(sys.argv[1])
    xml, model = compile_scene(t)
    print(json.dumps({"nbody": model.nbody, "ngeom": model.ngeom, "nu": model.nu, "nq": model.nq}))
    print(copy.copy(xml[:200]))

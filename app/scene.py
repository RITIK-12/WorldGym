"""MuJoCo -> viser mirror: one viser frame per body, geoms sent once, body poses streamed per frame.

Robot visual geoms (group 1 or 2) and world/object geoms (group 0) are drawn; collision-only geoms and planes are not.
Mesh vertices are welded before sending (the Panda meshes are ~400k split vertices raw, ~70k welded), and every
geom node is keyed by its content, so switching drafts or perturbing one object re-sends only what changed.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import viser

FPS = 30
ROOT = "/twin"
DRAWN_GROUPS = (0, 1, 2)       # objects 0; robot visuals are group 1 (Trossen) or 2 (Panda, YAM); 3+ = collision
G = mujoco.mjtGeom
_WELD_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_MODEL_CACHE: dict[str, mujoco.MjModel] = {}


@dataclass
class GeomSpec:
    gid: int
    body: int
    kind: str                       # box | sphere | ellipsoid | cylinder | capsule | mesh
    size: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    rgba: np.ndarray
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    raw_vertices: int = 0
    key: str = ""


def weld(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, inv = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    return u.astype(np.float32), inv.reshape(-1)[f].astype(np.uint32)


def capsule_mesh(r: float, half: float, seg: int = 20, rings: int = 6) -> tuple[np.ndarray, np.ndarray]:
    a = np.concatenate([np.linspace(np.pi / 2, 0, rings + 1), np.linspace(0, -np.pi / 2, rings + 1)])
    z = np.concatenate([half + r * np.sin(a[:rings + 1]), -half + r * np.sin(a[rings + 1:])])
    rad, th = r * np.cos(a), np.linspace(0, 2 * np.pi, seg, endpoint=False)
    v = np.stack([np.outer(rad, np.cos(th)), np.outer(rad, np.sin(th)), np.repeat(z[:, None], seg, 1)], -1).reshape(-1, 3)
    k, j = np.meshgrid(np.arange(len(a) - 1), np.arange(seg), indexing="ij")
    p0, p1 = k * seg + j, k * seg + (j + 1) % seg
    q0, q1 = p0 + seg, p1 + seg
    f = np.concatenate([np.stack([p0, q0, p1], -1).reshape(-1, 3), np.stack([p1, q0, q1], -1).reshape(-1, 3)])
    return v.astype(np.float32), f.astype(np.uint32)


def _rgba(m: mujoco.MjModel, g: int) -> np.ndarray:
    mat = m.geom_matid[g]
    rgba = m.mat_rgba[mat] if mat >= 0 and np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1]) else m.geom_rgba[g]
    return np.asarray(rgba, float).copy()


def scene_geoms(m: mujoco.MjModel) -> list[GeomSpec]:
    """Drawable geoms with welded mesh data (pure: no viser)."""
    kinds = {G.mjGEOM_BOX: "box", G.mjGEOM_SPHERE: "sphere", G.mjGEOM_ELLIPSOID: "ellipsoid",
             G.mjGEOM_CYLINDER: "cylinder", G.mjGEOM_CAPSULE: "capsule", G.mjGEOM_MESH: "mesh"}
    out = []
    for g in range(m.ngeom):
        kind = kinds.get(int(m.geom_type[g]))
        if kind is None or int(m.geom_group[g]) not in DRAWN_GROUPS or m.geom_rgba[g][3] == 0:
            continue
        s = GeomSpec(g, int(m.geom_bodyid[g]), kind, m.geom_size[g].copy(), m.geom_pos[g].copy(), m.geom_quat[g].copy(),
                     _rgba(m, g))
        ident = ""
        if kind == "mesh":
            mid = int(m.geom_dataid[g])
            va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh{mid}"
            ck = (name, int(vn), int(fn), float(m.mesh_vert[va:va + vn].sum()))
            if ck not in _WELD_CACHE:
                _WELD_CACHE[ck] = weld(m.mesh_vert[va:va + vn], m.mesh_face[fa:fa + fn])
            s.vertices, s.faces = _WELD_CACHE[ck]
            s.raw_vertices, ident = int(vn), repr(ck)
        elif kind == "capsule":
            s.vertices, s.faces = capsule_mesh(float(s.size[0]), float(s.size[1]))
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, s.body) or f"body{s.body}"
        sig = repr((body, kind, np.round(s.size, 5).tolist(), np.round(s.pos, 5).tolist(), np.round(s.quat, 5).tolist(),
                    np.round(s.rgba, 3).tolist(), ident))
        s.key = hashlib.sha1(sig.encode()).hexdigest()[:12]
        out.append(s)
    return out


class SceneMirror:
    """Keeps a viser subtree in sync with a MuJoCo model; `update` streams body poses in one atomic batch."""

    def __init__(self, server: viser.ViserServer, root: str = ROOT):
        self.server, self.root = server, root
        self.lock = threading.RLock()
        self.model: mujoco.MjModel | None = None
        self.frames: dict[str, viser.FrameHandle] = {}
        self.nodes: dict[str, object] = {}
        self.body_frames: list[viser.FrameHandle | None] = []
        self.body_index: dict[str, int] = {}
        self._last: np.ndarray | None = None
        self.stats = {"geoms": 0, "raw_vertices": 0, "welded_vertices": 0, "faces": 0, "bytes": 0}

    def _add_geom(self, name: str, s: GeomSpec):
        sc, rgb = self.server.scene, tuple(float(c) for c in s.rgba[:3])
        kw = dict(position=s.pos, wxyz=s.quat, opacity=None if s.rgba[3] >= 0.999 else float(s.rgba[3]))
        if s.vertices is not None:
            side = "double" if s.kind == "capsule" else "front"
            return sc.add_mesh_simple(name, s.vertices, s.faces, color=rgb, side=side, **kw)
        if s.kind == "box":
            return sc.add_box(name, color=rgb, dimensions=tuple(2 * s.size[:3]), **kw)
        if s.kind == "cylinder":
            return sc.add_cylinder(name, radius=float(s.size[0]), height=float(2 * s.size[1]), color=rgb, **kw)
        if s.kind == "sphere":
            return sc.add_icosphere(name, radius=float(s.size[0]), color=rgb, **kw)
        return sc.add_icosphere(name, radius=1.0, color=rgb, scale=tuple(float(x) for x in s.size[:3]), **kw)

    def load(self, m: mujoco.MjModel) -> None:
        with self.lock:
            if m is self.model:
                return
            specs = scene_geoms(m)
            bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}" for b in range(m.nbody)]
            want = {f"{self.root}/{bname[s.body]}/{s.key}": s for s in specs}
            bodies = {bname[s.body] for s in specs}
            d = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d)
            with self.server.atomic():
                for name in set(self.nodes) - set(want):
                    self.nodes.pop(name).remove()
                for b in set(self.frames) - bodies:
                    self.frames.pop(b).remove()
                for b, nm in enumerate(bname):
                    if nm in bodies and nm not in self.frames:
                        self.frames[nm] = self.server.scene.add_frame(f"{self.root}/{nm}", show_axes=False,
                                                                      position=d.xpos[b], wxyz=d.xquat[b])
                for name, s in want.items():
                    if name not in self.nodes:
                        self.nodes[name] = self._add_geom(name, s)
            self.model = m
            self.body_frames = [self.frames.get(nm) if b > 0 else None for b, nm in enumerate(bname)]
            self.body_index = {nm: b for b, nm in enumerate(bname)}
            self._last = None
            meshes = [s for s in specs if s.vertices is not None]
            self.stats = {"geoms": len(specs), "raw_vertices": sum(s.raw_vertices for s in meshes),
                          "welded_vertices": sum(len(s.vertices) for s in meshes),
                          "faces": sum(len(s.faces) for s in meshes),
                          "bytes": sum(s.vertices.nbytes + s.faces.nbytes for s in meshes)}
            self.update(d.xpos, d.xquat)

    def clear(self) -> None:
        """Nothing to show for this run yet: drop the previous run's scene instead of leaving it on screen."""
        with self.lock, self.server.atomic():
            for h in [*self.nodes.values(), *self.frames.values()]:
                h.remove()
            self.nodes, self.frames, self.body_frames, self.body_index = {}, {}, [], {}
            self.model, self._last = None, None

    def update(self, xpos: np.ndarray, xquat: np.ndarray) -> None:
        with self.lock:
            pose = np.concatenate([xpos, xquat], 1)
            moved = np.ones(len(pose), bool) if self._last is None or len(self._last) != len(pose) else \
                np.abs(pose - self._last).max(1) > 1e-6
            with self.server.atomic():
                for b, fr in enumerate(self.body_frames):
                    if fr is not None and moved[b]:
                        fr.position, fr.wxyz = xpos[b], xquat[b]
            self._last = pose.copy()

    def nudge(self, body: str, xy: tuple[float, float]) -> None:
        """Preview a planar move of one body (e.g. a slider before a live run)."""
        with self.lock:
            b = self.body_index.get(body)
            fr = self.body_frames[b] if b is not None and b < len(self.body_frames) else None
            if fr is not None:
                fr.position = (float(xy[0]), float(xy[1]), float(fr.position[2]))
                self._last = None


# ----------------------------------------------------------------------------------------------------------------------
# Replay: telemetry.npz (500 Hz qpos) -> per-frame body poses at 30 fps
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Replay:
    m: mujoco.MjModel
    t: np.ndarray                   # (F,)
    xpos: np.ndarray                # (F, nbody, 3)
    xquat: np.ndarray               # (F, nbody, 4)
    phase: list[str]

    def index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t - 1e-6), 0, len(self.t) - 1))


def load_model(xml_path: str | Path) -> mujoco.MjModel:
    """Compiled models are shared between drafts with identical scene.xml (same object -> no scene re-send)."""
    text = Path(xml_path).read_text()
    key = hashlib.sha1(text.encode()).hexdigest()
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(text)
    return _MODEL_CACHE[key]


def poses(m: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = mujoco.MjData(m)
    xpos, xquat = np.zeros((len(qpos), m.nbody, 3)), np.zeros((len(qpos), m.nbody, 4))
    for k, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        xpos[k], xquat[k] = d.xpos, d.xquat
    return xpos, xquat


def load_replay(nominal_dir: str | Path, fps: int = FPS) -> Replay:
    nd = Path(nominal_dir)
    m = load_model(nd / "scene.xml")
    z = np.load(nd / "telemetry.npz")
    t, q = z["t"].astype(float), z["qpos"].astype(float)
    if q.shape[1] != m.nq:
        raise ValueError(f"telemetry nq {q.shape[1]} != model nq {m.nq}")
    ts = np.arange(t[0], t[-1] + 1e-9, 1.0 / fps)
    idx = np.clip(np.searchsorted(t, ts), 0, len(t) - 1)
    ph = z["phase"][idx] if "phase" in z.files else np.full(len(idx), "")
    obj = z["phase_obj"][idx] if "phase_obj" in z.files else np.full(len(idx), "")
    xpos, xquat = poses(m, q[idx])
    return Replay(m, ts, xpos, xquat, [f"{p} {o}".strip() for p, o in zip(ph, obj)])


def static_replay(m: mujoco.MjModel, d: mujoco.MjData) -> Replay:
    return Replay(m, np.zeros(1), d.xpos[None].copy(), d.xquat[None].copy(), ["twin at rest"])


def contact_segments(m: mujoco.MjModel, d: mujoco.MjData, min_n: float = 0.5, m_per_n: float = 0.004,
                     max_len: float = 0.15) -> np.ndarray:
    """(N, 2, 3) normal-force segments from the solved contacts (needs a stepped MjData)."""
    f6, out = np.zeros(6), []
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        if f6[0] < min_n:
            continue
        c = d.contact[i]
        out.append([c.pos, c.pos + c.frame[:3] * min(f6[0] * m_per_n, max_len)])
    return np.asarray(out, float).reshape(-1, 2, 3)

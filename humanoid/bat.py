"""A regulation wood baseball bat in the G1's hands (no bat exists in the SimReady catalog, so it is built here).

The bat is a surface of revolution along its own +z (knob at z = 0, end cap at z = L): knob, handle, taper, barrel,
rounded end. The visual mesh carries the mass (uniform wood, inertia from the mesh); four capsules along the axis are
the collision geometry (smooth ball-bat contact). Presets follow the rules: MLB wood is at most 2.61 in thick; youth
"USSSA small barrel" bats are 2 1/4 in.

Default is the youth bat, because the G1 is child-sized: its wrist pitch/yaw motors are rated 5 N·m, and a 34 in /
32 oz MLB bat needs ~0.91 kg x 9.81 x 0.55 m = 4.9 N·m at the wrist just to hold it level; the 28 in / 18 oz youth
bat needs ~2.1 N·m.

Grip: the bat is rigid in the top hand (the hand nearer the barrel), its axis along the hand's +z (the thumb side
when the arm hangs) through the palm site; the bottom hand holds the handle near the knob. Kinematic retargeting
enforces the bottom hand's grip as an IK task; physics enforces it with a site-to-site connect constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "props" / "baseball_bat"
PALM = np.array([0.08, 0.0, 0.0])   # G1 palm site in the wrist_yaw_link frame (Menagerie g1)
IN, OZ = 0.0254, 0.0283495


@dataclass(frozen=True)
class BatSpec:
    name: str
    length: float
    barrel_d: float
    handle_d: float
    knob_d: float
    mass: float

    @property
    def top_grip(self) -> float:      # top hand centre, metres from the knob end
        return 0.035 * self.length + 0.085

    @property
    def bottom_grip(self) -> float:   # bottom hand centre (just above the knob)
        return 0.035 * self.length


PRESETS = {
    "youth": BatSpec("youth", 28 * IN, 2.25 * IN, 0.875 * IN, 1.40 * IN, 18 * OZ),
    "mlb": BatSpec("mlb", 34 * IN, 2.58 * IN, 0.94 * IN, 1.55 * IN, 32 * OZ),
}


def radius(b: BatSpec, s: np.ndarray) -> np.ndarray:
    """Radius at distance s from the knob end."""
    L, rk, rh, rb = b.length, b.knob_d / 2, b.handle_d / 2, b.barrel_d / 2
    x = s / L
    r = np.empty_like(s)
    knob = x < 0.012
    r[knob] = rk * np.sqrt(np.clip(1 - ((0.012 - x[knob]) / 0.012) ** 2, 0.04, 1))
    neck = (x >= 0.012) & (x < 0.035)
    t = (x[neck] - 0.012) / 0.023
    r[neck] = rk + (rh - rk) * (3 * t ** 2 - 2 * t ** 3)
    handle = (x >= 0.035) & (x < 0.40)
    r[handle] = rh * (1 + 0.12 * (x[handle] - 0.035) / 0.365)
    taper = (x >= 0.40) & (x < 0.68)
    t = (x[taper] - 0.40) / 0.28
    r[taper] = rh * 1.12 + (rb - rh * 1.12) * (3 * t ** 2 - 2 * t ** 3)
    barrel = (x >= 0.68) & (x < 0.975)
    r[barrel] = rb
    end = x >= 0.975
    r[end] = rb * np.sqrt(np.clip(1 - ((x[end] - 0.975) / 0.025) ** 2, 0.0, 1))
    return r


def write_mesh(b: BatSpec, n_axial: int = 120, n_theta: int = 28) -> Path:
    path = OUT / f"bat_{b.name}.obj"
    if path.exists():
        return path
    s = np.concatenate([np.linspace(0, 0.04 * b.length, 20), np.linspace(0.04 * b.length, 0.97 * b.length, n_axial)[1:],
                        np.linspace(0.97 * b.length, b.length, 16)[1:]])
    r = radius(b, s)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    V = [[0.0, 0.0, 0.0]]
    for si, ri in zip(s, r):
        V += [[ri * np.cos(t), ri * np.sin(t), si] for t in th]
    V.append([0.0, 0.0, b.length])
    F = []
    for k in range(n_theta):                                  # knob cap
        F.append([0, 1 + (k + 1) % n_theta, 1 + k])
    for i in range(len(s) - 1):
        a, c = 1 + i * n_theta, 1 + (i + 1) * n_theta
        for k in range(n_theta):
            k1 = (k + 1) % n_theta
            F += [[a + k, a + k1, c + k1], [a + k, c + k1, c + k]]
    last, top = 1 + (len(s) - 1) * n_theta, len(V) - 1
    for k in range(n_theta):                                  # end cap
        F.append([top, last + k, last + (k + 1) % n_theta])
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# {b.name} wood bat: {b.length / IN:.0f} in, barrel {b.barrel_d / IN:.2f} in, {b.mass / OZ:.0f} oz\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    return path


def capsules(b: BatSpec) -> list[tuple[float, float, float]]:
    """(z0, z1, radius) collision capsules along the axis (endpoints inset by the radius)."""
    L = b.length
    segs = [(0.03, 0.40), (0.40, 0.54), (0.54, 0.68), (0.68, 0.985)]
    out = []
    for a, c in segs:
        rr = float(radius(b, np.linspace(a * L, c * L, 9)).mean())
        out.append((a * L + rr * 0.5, c * L - rr, rr))
    return out


def attach(spec: mujoco.MjSpec, top_hand: str, preset: str = "youth") -> None:
    """Add the bat to a G1 MjSpec: rigid in `top_hand` ("right"/"left"), grip sites on both ends of the handle."""
    b = PRESETS[preset]
    mesh = spec.add_mesh(name="bat_mesh", file=str(write_mesh(b)))
    hand = spec.body(f"{top_hand}_wrist_yaw_link")
    bat = hand.add_body(name="bat", pos=(PALM - np.array([0, 0, b.top_grip])).tolist())
    bat.add_geom(name="bat_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, mass=b.mass,
                 rgba=[0.78, 0.6, 0.38, 1], contype=0, conaffinity=0, group=2)
    for k, (z0, z1, rr) in enumerate(capsules(b)):
        bat.add_geom(name=f"bat_collision{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[rr, 0, 0],
                     fromto=[0, 0, z0, 0, 0, z1], mass=0, contype=0, conaffinity=0, group=3, rgba=[0.8, 0.5, 0.2, 0.3])
    bat.add_site(name="bat_grip_top", pos=[0, 0, b.top_grip])
    bat.add_site(name="bat_grip_bottom", pos=[0, 0, b.bottom_grip])
    bat.add_site(name="bat_sweet_spot", pos=[0, 0, b.length - 0.17])   # ~6.5 in from the end


def bottom_hand(top_hand: str) -> str:
    return "left" if top_hand == "right" else "right"


def add_grip_constraint(spec: mujoco.MjSpec, top_hand: str) -> None:
    """Physics: the bottom hand's palm stays on the handle (site-to-site connect)."""
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE,
                           name1=f"{bottom_hand(top_hand)}_palm", name2="bat_grip_bottom")
    eq.solref = [0.01, 1.0]

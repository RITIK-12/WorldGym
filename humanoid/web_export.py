"""Export the batting cage for the browser: physics MJCF, decimated body meshes, and the swing.

    uv run python -m humanoid.web_export --ref runs/<id>/clips/0278/reference.npz

Writes web/assets/:
- cage.xml       physics only: the G1 (scene_mjx collision primitives and actuators), the bat (explicit inertia, collision
                 capsules), the ball (free body, 145 g, 7.32 cm) with a ball-floor contact pair, and the field floor.
                 No mesh files, so the WASM build loads it from a string.
- visuals.json   one triangle mesh per body in the body frame (vertex clustering at 4 mm brings the G1's 400k
                 triangles to about 110k), colour per body; float32 positions then uint32 indices, base64 in `blob`
                 (artifact hosting serves no .bin). three.js poses them from xpos/xquat.
- swing.json     the reference resampled to 50 Hz with the padding the RL env uses, plus the contact frame (the
                 sweet spot moving fastest toward the pitcher, who stands off the lead shoulder), the sweet spot's
                 position and velocity there, and the bat's collision capsules.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import mujoco
import numpy as np

from humanoid import bat as batlib
from humanoid.retarget import G1_SCENE
from humanoid.track_env import CTRL_HZ, resample

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "web" / "assets"
BALL_R, BALL_M = 0.0366, 0.145


def build_spec(top_hand: str, preset: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(G1_SCENE))
    batlib.attach(spec, top_hand, preset)
    batlib.add_grip_constraint(spec, top_hand)
    ball = spec.worldbody.add_body(name="ball", pos=[12.0, 0.0, 1.0])
    ball.add_freejoint(name="ball_free")
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], mass=BALL_M,
                  contype=0, conaffinity=0, rgba=[0.97, 0.96, 0.92, 1])
    pair = spec.add_pair(geomname1="ball_geom", geomname2="floor")
    pair.condim, pair.solref, pair.friction = 3, [0.004, 0.25], [0.6, 0.6, 0.005, 0.0001, 0.0001]
    return spec


def physics_xml(spec: mujoco.MjSpec, compiled: mujoco.MjModel) -> str:
    """Strip visual meshes (inertias are explicit), give the bat its compiled inertia, drop textures."""
    b = compiled.body("bat").id
    batb = spec.body("bat")
    batb.explicitinertial = True
    batb.mass = float(compiled.body_mass[b])
    batb.ipos = compiled.body_ipos[b].tolist()
    batb.iquat = compiled.body_iquat[b].tolist()
    batb.inertia = compiled.body_inertia[b].tolist()
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    for mat in list(spec.materials):
        spec.delete(mat)
    for tex in list(spec.textures):
        spec.delete(tex)
    for g in spec.geoms:
        g.material = ""
    return spec.to_xml()


def _quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def decimate(V: np.ndarray, F: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.floor(V / cell).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    out = np.zeros((n, 3))
    np.add.at(out, inv, V)
    out /= np.bincount(inv, minlength=n)[:, None]
    F2 = inv[F]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return out, F2[keep]


def visuals(m: mujoco.MjModel, cell: float = 0.004) -> tuple[dict, bytes]:
    bodies, pos_chunks, idx_chunks = [], [], []
    vofs = iofs = 0
    for b in range(1, m.nbody):
        Vs, Fs, rgba, base = [], [], None, 0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] > 2:
                continue
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            V = m.mesh_vert[v0:v0 + nv] @ _quat_mat(m.geom_quat[g]).T + m.geom_pos[g]
            F = m.mesh_face[f0:f0 + nf]
            if m.body(b).name != "bat":
                V, F = decimate(V, F, cell)
            Vs.append(V)
            Fs.append(F + base)
            base += len(V)
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        if not Vs:
            continue
        V, F = np.vstack(Vs).astype(np.float32), np.vstack(Fs).astype(np.uint32)
        bodies.append({"name": m.body(b).name, "rgba": [round(float(x), 3) for x in rgba], "vofs": vofs,
                       "vcount": int(len(V)), "iofs": iofs, "icount": int(F.size)})
        pos_chunks.append(V.tobytes())
        idx_chunks.append(F.tobytes())
        vofs += len(V)
        iofs += F.size
    pos = b"".join(pos_chunks)
    meta = {"bodies": bodies, "positions_bytes": len(pos), "vertices": vofs, "triangles": iofs // 3}
    return meta, pos + b"".join(idx_chunks)


def swing(ref: dict, top_hand: str, preset: str) -> dict:
    from humanoid.retarget import g1_model
    m = g1_model(G1_SCENE, preset, top_hand)          # robot + bat only: the ball's free joint comes after these
    q50, _ = resample(ref["qpos"], float(ref["fps"]), m)
    d = mujoco.MjData(m)
    sweet = m.site("bat_sweet_spot").id
    P = np.zeros((len(q50), 3))
    for t, q in enumerate(q50):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        P[t] = d.site_xpos[sweet]
    V = np.gradient(P, 1.0 / CTRL_HZ, axis=0)
    # the pitcher stands off the lead shoulder: the right one (-y) for a left-handed striker (left hand on top) ...
    side = np.array([0.0, -1.0 if top_hand == "left" else 1.0, 0.0])
    k = int((V @ side).argmax())                       # contact: the barrel moving fastest toward the pitcher
    # ... and the stride foot steps toward the pitcher, which fixes the exact pitch line (batters stand open or closed)
    lead = m.body(f"{'right' if top_hand == 'left' else 'left'}_ankle_roll_link").id
    feet = []
    for t in (0, k):
        d.qpos[:] = q50[t]
        mujoco.mj_kinematics(m, d)
        feet.append(d.xpos[lead].copy())
    stride = feet[1] - feet[0]
    stride[2] = 0.0
    to_pitcher = stride / np.linalg.norm(stride) if np.linalg.norm(stride) > 0.05 and stride @ side > 0 else side
    b = batlib.PRESETS[preset]
    return {"clip": str(ref["clip"]), "hz": CTRL_HZ, "nq": int(m.nq), "frames": len(q50),
            "qpos": np.round(q50, 5).ravel().tolist(), "contact_frame": k,
            "sweet_spot": np.round(P[k], 4).tolist(), "sweet_spot_vel": np.round(V[k], 3).tolist(),
            "bat_speed_mps": round(float(np.linalg.norm(V[k])), 2),
            "bat": {"preset": preset, "length_m": b.length, "mass_kg": b.mass, "capsules": batlib.capsules(b)},
            "time_scale": round(30.0 / float(ref["fps"]), 3), "to_pitcher": np.round(to_pitcher, 4).tolist(),
            "stride_m": round(float(np.linalg.norm(stride)), 3),
            "top_hand": top_hand}


def export(ref_path: Path, top_hand: str, preset: str = "youth", out: Path = OUT) -> dict:
    ref = dict(np.load(ref_path))
    spec = build_spec(top_hand, preset)
    full = spec.compile()
    meta, blob = visuals(full)
    sw = swing(ref, top_hand, preset)
    xml = physics_xml(spec, full)
    check = mujoco.MjModel.from_xml_string(xml)
    assert check.nq == full.nq and check.nu == full.nu, "physics model lost joints or actuators"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cage.xml").write_text(xml)
    (out / "visuals.json").write_text(json.dumps(meta | {"blob": base64.b64encode(blob).decode()}))
    (out / "visuals.bin").unlink(missing_ok=True)
    (out / "swing.json").write_text(json.dumps(sw))
    return {"vertices": meta["vertices"], "triangles": meta["triangles"], "bin_mb": round(len(blob) / 1e6, 2),
            "xml_kb": round(len(xml) / 1e3, 1), "swing_frames": sw["frames"], "contact_frame": sw["contact_frame"],
            "bat_speed_mps": sw["bat_speed_mps"], "sweet_spot": sw["sweet_spot"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--top-hand", default="")
    ap.add_argument("--bat", default="youth")
    a = ap.parse_args()
    ref_path = Path(a.ref)
    top = a.top_hand or str(np.load(ref_path).get("top_hand", "right"))
    print(json.dumps(export(ref_path, top, a.bat), indent=1))


if __name__ == "__main__":
    main()


def export_policy(policy_path: Path, ref_path: Path, out: Path = OUT) -> dict:
    """The tracking policy for the browser: observation normaliser, actor MLP weights (ELU), the actuator map, and one
    reference obs -> action pair so the page can check it computes what PyTorch computes."""
    import torch
    from humanoid.track_env import ACTION_SCALE, TrackEnv
    from humanoid.train import ActorCritic
    ref = np.load(ref_path)
    env = TrackEnv(ref["qpos"], float(ref["fps"]), bat=str(ref["bat"]), top_hand=str(ref["top_hand"]))
    ac = ActorCritic(env.obs_dim, env.nu)
    ac.load_state_dict(torch.load(policy_path, map_location="cpu"))
    ac.eval()
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
    layers = [m for m in ac.actor if isinstance(m, torch.nn.Linear)]
    obs = env.reset()
    with torch.no_grad():
        act = ac.actor(ac.norm(torch.as_tensor(obs)[None]))[0].numpy()
    pol = {"obs_dim": env.obs_dim, "act_dim": env.nu, "action_scale": ACTION_SCALE, "ctrl_hz": 50,
           "mean": b64(ac.norm.mean.numpy()), "var": b64(ac.norm.var.numpy()),
           "layers": [{"in": l.in_features, "out": l.out_features, "W": b64(l.weight.detach().numpy()),
                       "b": b64(l.bias.detach().numpy())} for l in layers],
           "jadr": env.jadr.tolist(), "vadr": env.vadr.tolist(), "lo": env.lo.tolist(), "hi": env.hi.tolist(),
           "pelvis": "pelvis", "frames": env.T, "test_obs": obs.tolist(), "test_action": act.tolist(),
           "checkpoint": str(policy_path), "trained_minutes": None}
    log = policy_path.parent / "log.jsonl"
    if log.exists():
        last = json.loads(log.read_text().splitlines()[-1])
        pol["trained_minutes"] = last["min"]
        pol["train_stats"] = {k: last[k] for k in ("iter", "reward", "fall_rate", "key_err_cm", "assist")}
    (out / "policy.json").write_text(json.dumps(pol))
    return {"bytes": (out / "policy.json").stat().st_size, "frames": env.T, "trained": pol.get("train_stats")}

"""NVIDIA SimReady assets as object geometry: search (text or image), fetch, convert USD -> MuJoCo meshes.

Search goes through the public SimReady Central USD Search API (SigLIP2 embeddings over the catalog); packages come
from the Hugging Face dataset `nvidia/simready-assets` (CC-BY 4.0, attribution kept in asset.json). Conversion reads
the USD with pxr: render meshes become one visual OBJ, collision meshes (UsdPhysics.CollisionAPI) become convex OBJs
(convex hull of the visual mesh when none are authored), everything re-centred on the bounding-box centre (our body
origin convention). The Scene Author calls `search_simready_assets` as a function tool and sees thumbnails.
"""
from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "assets" / "simready"
SEARCH_URL = "https://nvidia-simready-central.hf.space/search"
DATASET = "nvidia/simready-assets"
URL_RE = re.compile(r"huggingface\.co/datasets/(?P<ds>[^/]+/[^/]+)/blob/[0-9a-f]+/(?P<name>[^/]+)/(?P<rel>.+\.usd[a-z]?)$")


def _parse(url: str) -> dict | None:
    m = URL_RE.search(url)
    return m.groupdict() if m else None


def search_assets(text: str | None = None, image: Path | np.ndarray | None = None, limit: int = 8) -> list[dict]:
    """Candidates from nvidia/simready-assets: name, usd path, score, metric bbox dims, thumbnail url, license."""
    import httpx

    body: dict = {"limit": limit * 2, "return_metadata": True, "return_root_prims": True}
    if text:
        body["description"] = text
    if image is not None:
        import cv2

        img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
        ok, buf = cv2.imencode(".jpg", img)
        body["image_similarity_search"] = [base64.b64encode(buf.tobytes()).decode()]
    try:
        r = httpx.post(SEARCH_URL, json=body, timeout=90)
        r.raise_for_status()
        items = r.json()
    except Exception:  # noqa: BLE001 - search is a hint; the scene falls back to primitives
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        p = _parse(it.get("url", ""))
        if not p or p["ds"] != DATASET:
            continue
        rp = (it.get("root_prims") or [{}])[0]
        lo, hi = rp.get("bbox_min"), rp.get("bbox_max")
        dims = [round(float(b - a), 4) for a, b in zip(lo, hi)] if lo and hi else None
        usd_file = p["rel"].split("/")[-1]
        out.append({"name": p["name"], "usd": p["rel"], "score": round(float(it.get("score", 0)), 3), "dims_m": dims,
                    "thumbnail_url": f"https://huggingface.co/datasets/{DATASET}/resolve/main/{p['name']}/"
                                     f"simready_usd/.thumbs/256x256/{usd_file}.png",
                    "license": "CC-BY-4.0", "url": it["url"]})
        if len(out) >= limit:
            break
    return out


def fetch_asset(name: str) -> Path:
    """Download only what conversion needs (root USD, payloads, albedo, thumbnail, license) over HTTP."""
    import fnmatch

    import httpx

    keep = [f"{name}/simready_usd/*.usd", f"{name}/simready_usd/payloads/*", f"{name}/simready_usd/payloads/*/*",
            f"{name}/simready_usd/Textures/*_alb.png", f"{name}/.metadata/LICENSE.txt",
            f"{name}/simready_usd/.thumbs/256x256/*.png"]
    base = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/{name}?recursive=true"
    files = [f["path"] for f in httpx.get(base, timeout=60).json() if f.get("type") == "file"]
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for path in files:
            if not any(fnmatch.fnmatch(path, k) for k in keep):
                continue
            dst = CACHE / "pkg" / path
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = client.get(f"https://huggingface.co/datasets/{DATASET}/resolve/main/{path}")
            r.raise_for_status()
            dst.write_bytes(r.content)
    return CACHE / "pkg" / name


def _write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    lines = [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v] + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    path.write_text("\n".join(lines) + "\n")


def convert_asset(pkg: Path) -> dict:
    """USD package -> assets/simready/<name>/mujoco/{visual.obj, collision_*.obj, asset.json} (cached)."""
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics

    name = pkg.name
    out = CACHE / name / "mujoco"
    info_path = out / "asset.json"
    if info_path.exists():
        return json.loads(info_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    root = next(p for p in sorted((pkg / "simready_usd").glob("*.usd")))
    stage = Usd.Stage.Open(str(root))
    for prim in stage.Traverse():   # prefer the MuJoCo physics variant when the package authors one
        vsets = prim.GetVariantSets()
        if "MuJoCo" in vsets.GetNames():
            vs = vsets.GetVariantSet("MuJoCo")
            if "Enabled" in vs.GetVariantNames():
                vs.SetVariantSelection("Enabled")
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.XformCache()
    visual, collision, grasps, mass = [], [], [], None
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):   # geometry is instanced
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = np.array(mesh.GetPointsAttr().Get() or [], float)
            counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
            idx = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
            if len(pts) == 0 or not counts:
                continue
            M = np.array(cache.GetLocalToWorldTransform(prim), float)   # row-vector convention
            wp = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3] * mpu
            tris, k = [], 0
            for c in counts:
                for j in range(1, c - 1):
                    tris.append([idx[k], idx[k + j], idx[k + j + 1]])
                k += c
            item = (wp, np.array(tris, int))
            purpose = UsdGeom.Imageable(prim).ComputePurpose()
            if prim.HasAPI(UsdPhysics.CollisionAPI) or purpose == UsdGeom.Tokens.guide:
                collision.append(item)
            else:
                visual.append(item)
        if prim.GetName().lower().startswith("grasp_point"):
            M = np.array(cache.GetLocalToWorldTransform(prim), float)
            grasps.append({"name": prim.GetName(), "pos": (M[3, :3] * mpu).tolist()})
        if prim.HasAPI(UsdPhysics.MassAPI):
            mv = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if mv:
                mass = float(mv)
    if not visual:
        visual = collision
    v_all = np.concatenate([v for v, _ in visual])
    lo, hi = v_all.min(0), v_all.max(0)
    centre = (lo + hi) / 2
    vv, ff, off = [], [], 0
    for v, f in visual:
        vv.append(v - centre)
        ff.append(f + off)
        off += len(v)
    _write_obj(out / "visual.obj", np.concatenate(vv), np.concatenate(ff))
    hulls = collision or [(v_all, np.concatenate(ff))]
    col_files, volume = [], 0.0
    for k, (v, f) in enumerate(hulls[:16]):
        hull = trimesh.Trimesh(vertices=v - centre, faces=f, process=False).convex_hull
        if hull.volume <= 1e-9:
            continue
        _write_obj(out / f"collision_{k}.obj", hull.vertices, hull.faces)
        col_files.append(f"collision_{k}.obj")
        volume += float(hull.volume)
    lic = pkg / ".metadata" / "LICENSE.txt"
    info = {"name": name, "url": str(root), "license": "CC-BY-4.0" if lic.exists() else "unknown",
            "attribution": f"{name} from NVIDIA SimReady (nvidia/simready-assets, CC-BY 4.0)",
            "native_dims_m": [round(float(x), 5) for x in hi - lo], "bbox_center": centre.tolist(), "mass_kg": mass,
            "collision": col_files, "volume_m3": volume,
            "grasp_points": [{**g, "pos": (np.array(g["pos"]) - centre).round(5).tolist()} for g in grasps]}
    info_path.write_text(json.dumps(info, indent=1))
    return info


def ensure_asset(name: str) -> dict:
    info_path = CACHE / name / "mujoco" / "asset.json"
    if info_path.exists():
        return json.loads(info_path.read_text())
    return convert_asset(fetch_asset(name))


def asset_mjcf(obj_id: str, info: dict, dims_m, rgba, mass: float, contact: dict) -> tuple[list, list, float]:
    """(mesh asset elements, body geoms, density) scaled per axis so the asset's bbox matches dims_m."""
    d = CACHE / info["name"] / "mujoco"
    scale = np.asarray(dims_m[:3], float) / np.maximum(np.asarray(info["native_dims_m"], float), 1e-6)
    sc = " ".join(f"{x:.6g}" for x in scale)
    density = mass / max(info.get("volume_m3", 0.0) * float(np.prod(scale)), 1e-9)
    assets = [ET.Element("mesh", name=f"{obj_id}_vis", file=str(d / "visual.obj"), scale=sc)]
    rgba_s = " ".join(f"{float(x):.4g}" for x in rgba)
    geoms = [ET.Element("geom", name=f"{obj_id}_vis", type="mesh", mesh=f"{obj_id}_vis", contype="0",
                        conaffinity="0", group="0", rgba=rgba_s, density="0")]
    for k, f in enumerate(info["collision"]):
        assets.append(ET.Element("mesh", name=f"{obj_id}_col{k}", file=str(d / f), scale=sc))
        geoms.append(ET.Element("geom", name=obj_id if k == 0 else f"{obj_id}_col{k}", type="mesh",
                                mesh=f"{obj_id}_col{k}", group="3", rgba="0 0 0 0", density=f"{density:.6g}",
                                **contact))
    return assets, geoms, density


TOOL = {"type": "function", "name": "search_simready_assets",
        "description": "Search NVIDIA SimReady (3.5k physically-ready 3D assets, CC-BY 4.0) for a real object. Returns "
                       "candidates with asset name, metric bounding-box size and a thumbnail.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what the object is, e.g. 'yellow kitchen sponge'"},
            "max_results": {"type": "integer", "description": "1-6"}}, "required": ["query"]}}


def run_tool(args: dict) -> list[dict]:
    """Function result: compact JSON of the candidates + their thumbnails as images."""
    import httpx

    cands = search_assets(args.get("query", ""), limit=int(min(max(args.get("max_results", 4), 1), 6)))
    parts: list[dict] = [{"type": "text", "text": json.dumps([{k: c[k] for k in ("name", "dims_m", "score", "license")}
                                                             for c in cands])}]
    for c in cands:
        try:
            r = httpx.get(c["thumbnail_url"], timeout=30, follow_redirects=True)
            if r.status_code == 200:
                parts += [{"type": "text", "text": f"thumbnail of {c['name']}:"},
                          {"type": "image", "data": base64.b64encode(r.content).decode(), "mime_type": "image/png"}]
        except Exception:  # noqa: BLE001
            continue
    return parts

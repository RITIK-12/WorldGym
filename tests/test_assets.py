"""SimReady assets: search -> fetch -> convert -> compile a mesh object (network; skipped offline)."""
import mujoco
import numpy as np
import pytest

from core.assets import ensure_asset, search_assets
from core.locked_evaluator import body_aabb
from core.mjcf_compiler import compile_scene
from core.schemas import AssetRef, ObjectSpec, TaskSpec


@pytest.fixture(scope="module")
def sponge():
    c = search_assets("kitchen sponge", limit=3)
    if not c:
        pytest.skip("SimReady search unavailable")
    return c[0]


def test_search_and_convert(sponge):
    assert "sponge" in sponge["name"] and sponge["license"] == "CC-BY-4.0"
    info = ensure_asset(sponge["name"])
    assert info["collision"] and 0.02 < max(info["native_dims_m"]) < 0.2 and "CC-BY" in info["attribution"]


def test_mesh_object_compiles_at_requested_size(sponge):
    o = ObjectSpec(id="sp", shape="mesh", dims_m=[0.12, 0.08, 0.04], xy=[0.45, 0.0], mass=0.03,
                   asset=AssetRef(name=sponge["name"], url=sponge["url"]))
    _, m = compile_scene(TaskSpec(name="a", robot="franka_panda", objects=[o]))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = body_aabb(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sp"))
    assert np.allclose(hi - lo, [0.12, 0.08, 0.04], rtol=0.08)

"""θ: every knob between cached perception and the G1 reference motion, as one closed schema.

The critic may only patch these fields. `apply_patch` validates the merged config, so an unknown field or an
out-of-range value fails loudly instead of silently doing nothing (same contract as core.critic.apply_patch).
Perception (detector prompt, box tracking) is not here: it is expensive and cached once per clip.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HarnessConfig(Strict):
    # world frame from camera-space keypoints
    smooth_sigma: float = Field(1.0, ge=0.0, le=6.0, description="temporal Gaussian on world keypoints, frames")
    up_mode: Literal["feet_plane", "camera"] = Field("feet_plane", description="ground normal: fitted to the feet, or camera -y")
    plane_band_m: float = Field(0.03, ge=0.005, le=0.15, description="foot points within this of the lowest count as ground")
    root_motion: Literal["camera", "pinned"] = Field("camera", description="root translation from cam_t, or pelvis pinned horizontally")
    # retargeting onto the G1
    retarget_mode: Literal["bone_directions", "uniform"] = Field("bone_directions", description="human bone directions with G1 bone lengths, or one uniform scale")
    w_feet: float = Field(3.0, ge=0.0, le=20.0)
    w_legs: float = Field(2.0, ge=0.0, le=20.0)
    w_arms: float = Field(1.0, ge=0.0, le=20.0)
    w_head: float = Field(0.3, ge=0.0, le=20.0)
    w_orient: float = Field(1.0, ge=0.0, le=20.0, description="pelvis and torso orientation tasks")
    ik_iters: int = Field(40, ge=5, le=200)
    ik_damping: float = Field(0.05, ge=1e-4, le=1.0)
    w_smooth: float = Field(0.1, ge=0.0, le=5.0, description="pull toward the previous frame's joints")
    ground_snap: Literal["smoothed", "none"] = Field("smoothed", description="shift the root so the lower foot sole sits on the floor")
    mirror: bool = Field(False, description="flip the clip horizontally (left-handed striker -> right-handed)")
    # the bat: built by humanoid.bat, rigid in the top hand, axis from the striker's knuckle lines
    bat: Literal["youth", "mlb", "none"] = Field("youth", description="bat preset in the G1's hands")
    w_bat: float = Field(2.0, ge=0.0, le=20.0, description="IK task: bat axis along the striker's knuckle-line axis")
    w_grip: float = Field(6.0, ge=0.0, le=50.0, description="IK task: bottom hand's palm on the handle")
    grip_release_m: float = Field(0.14, ge=0.06, le=0.4, description="hands farther apart than this let go with the bottom hand")
    # timing and contact
    time_scale: float = Field(1.0, ge=0.5, le=4.0, description="robot seconds per demo second (>1 slows the motion)")
    contact_height_m: float = Field(0.025, ge=0.005, le=0.08, description="a foot sole this close to the floor is in contact")


class ConfigPatch(Strict):
    """Partial θ plus the hypothesis behind it (logged as the lab notebook)."""
    hypothesis: str
    smooth_sigma: float | None = None
    up_mode: Literal["feet_plane", "camera"] | None = None
    plane_band_m: float | None = None
    root_motion: Literal["camera", "pinned"] | None = None
    retarget_mode: Literal["bone_directions", "uniform"] | None = None
    w_feet: float | None = None
    w_legs: float | None = None
    w_arms: float | None = None
    w_head: float | None = None
    w_orient: float | None = None
    ik_iters: int | None = None
    ik_damping: float | None = None
    w_smooth: float | None = None
    ground_snap: Literal["smoothed", "none"] | None = None
    mirror: bool | None = None
    bat: Literal["youth", "mlb", "none"] | None = None
    w_bat: float | None = None
    w_grip: float | None = None
    grip_release_m: float | None = None
    time_scale: float | None = None
    contact_height_m: float | None = None


def apply_patch(cfg: HarnessConfig, patch: ConfigPatch) -> HarnessConfig:
    changes = patch.model_dump(exclude={"hypothesis"}, exclude_none=True)
    return HarnessConfig.model_validate({**cfg.model_dump(), **changes})


def diff(a: HarnessConfig, b: HarnessConfig) -> dict:
    da, db = a.model_dump(), b.model_dump()
    return {k: [da[k], db[k]] for k in da if da[k] != db[k]}

"""Data contracts shared by every stage (pydantic v2).

Conventions: SI units, table top at z = 0, +z up. Single arm: world origin = robot base, +x away from the robot.
`dims_m` are FULL extents (MuJoCo half-sizes are derived in the compiler).
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------------------------------------------------
# Robot adapter
# ----------------------------------------------------------------------------------------------------------------------

class TCP(Strict):
    body: str
    pos: list[float] = [0.0, 0.0, 0.0]
    quat: list[float] = [1.0, 0.0, 0.0, 0.0]


class Gripper(Strict):
    kind: Literal["tendon_force", "joint_force"] = "tendon_force"
    actuator: str
    joints: list[str]
    max_width_m: float
    force_range: float
    finger_force_per_ctrl: float = 0.5


class ArmSpec(Strict):
    name: str = "arm"
    joints: list[str]
    actuators: list[str]
    tcp: TCP
    closing_axis: Literal["x", "y"] = "y"
    gripper: Gripper
    home_qpos: list[float]
    vel_limits: list[float]
    robot_bodies: list[str]
    finger_bodies: list[str]


class RobotAdapter(Strict):
    robot_id: str
    menagerie_dir: str
    mjcf: str
    arms: list[ArmSpec]
    base_poses: list[list[float]] = [[0.0, 0.0, 0.0, 0.0]]  # (x, y, z, yaw) per arm


# ----------------------------------------------------------------------------------------------------------------------
# Task / scene
# ----------------------------------------------------------------------------------------------------------------------

Shape = Literal["box", "cylinder", "sphere", "capsule", "tblock", "container", "composite"]


class Part(Strict):
    """One primitive of a composite object, in the object frame (origin = bounding-box centre).
    Parts are rescaled per axis so their joint bounding box matches the object's dims_m."""
    shape: Literal["box", "cylinder", "sphere", "capsule"]
    dims_m: list[float]                    # full extents, same convention as ObjectSpec
    pos: list[float] = [0.0, 0.0, 0.0]
    euler: list[float] = [0.0, 0.0, 0.0]   # xyz radians; cylinders/capsules run along the part's z axis


class ObjectSpec(Strict):
    id: str
    shape: Shape
    dims_m: list[float]                    # box/container/tblock: [x, y, z]; cylinder/capsule: [d, d, h]; sphere: [d]*3
    colour: list[float] = [0.8, 0.2, 0.2, 1.0]
    xy: list[float]
    yaw: float = 0.0
    on_top_of: str | None = None
    z: float | None = None                 # explicit bottom height override
    mass: float = 0.1
    friction: float = 0.8
    is_fixed: bool = False
    is_obstacle: bool = False
    wall_m: float = 0.008                  # container wall / floor thickness
    stem_w_m: float | None = None          # tblock stem width (default = 0.4 * y extent)
    flange_d_m: float | None = None        # tblock flange depth along y (default = 0.4 * y extent)
    parts: list[Part] | None = None        # composite only
    material: str | None = None


class CameraSpec(Strict):
    name: str
    pos: list[float]
    xyaxes: list[float] | None = None
    look_at: list[float] | None = None     # used to derive xyaxes when xyaxes is None
    fovy: float = 45.0
    w: int = 640
    h: int = 480
    projection: Literal["perspective", "orthographic"] = "perspective"


class Phase(Strict):
    t0: float
    t1: float
    action: str
    arm: str = "arm"
    object: str | None = None
    target: str | None = None


PredicateKind = Literal["inside_container", "on_top_of", "max_obstacle_displacement", "terminal_rest",
                        "max_unwanted_collisions", "pushed_to_region"]


class Predicate(Strict):
    kind: PredicateKind
    object_id: str | None = None
    target_id: str | None = None
    limit: float | None = None             # metres / count depending on kind
    max_tilt_deg: float = 25.0
    region: list[float] | None = None      # pushed_to_region: [xmin, ymin, xmax, ymax]


class TaskSpec(Strict):
    name: str
    robot: str                             # adapter id (robots/<id>.json)
    base_poses: list[list[float]] | None = None
    objects: list[ObjectSpec]
    phases: list[Phase] = []
    cameras: list[CameraSpec] = []
    immutable_success_predicates: list[Predicate] = []
    settle_s: float = 1.0                  # post-policy settle before evaluating

    def obj(self, oid: str) -> ObjectSpec:
        for o in self.objects:
            if o.id == oid:
                return o
        raise KeyError(oid)


# ----------------------------------------------------------------------------------------------------------------------
# Policy DSL
# ----------------------------------------------------------------------------------------------------------------------

FEATURES = ("obj_top_z", "obj_height", "held_below_tcp", "max_obstacle_top_z", "max_scene_top_z",
            "target_rim_z", "target_floor_z", "table_z")


class Rel(Strict):
    """State-conditioned value: sum of measured scene features + offset, resolved when the primitive starts."""
    terms: list[Literal["obj_top_z", "obj_height", "held_below_tcp", "max_obstacle_top_z", "max_scene_top_z",
                        "target_rim_z", "target_floor_z", "table_z"]]
    offset: float = 0.0


Num = Union[float, Rel]


class Primitive(Strict):
    arm: str = "arm"
    object_id: str
    target_id: str | None = None
    grasp_xy_offset_m: list[float] = [0.0, 0.0]
    grasp_z_depth_m: Num = 0.018
    grasp_yaw_mode: Literal["fixed_world", "object_relative"] = "object_relative"
    grasp_yaw_rad: float = 0.0
    pre_grasp_clearance_m: Num = 0.08
    pre_grasp_open_width_m: float = 0.08
    reach_clearance_z_m: Num | None = None
    grip_force_n: float = 35.0
    post_grasp_lift_z_m: Num = 0.15
    transit_apex_z_m: Num = 0.20
    transit_via_mode: Literal["none", "absolute", "midpoint"] = "none"
    transit_via_point_xy_m: list[float] | None = None
    place_xy_offset_m: list[float] = [0.0, 0.0]
    place_yaw_mode: Literal["keep", "fixed_world", "container_relative"] = "keep"
    place_yaw_rad: float = 0.0
    pre_place_hover_z_m: Num = 0.15
    release_drop_height_m: Num = 0.01
    release_open_width_m: float = 0.06
    retract_z_m: Num = 0.20
    speed_mps: float = 0.35


class PolicyConfig(Strict):
    name: str = "draft"
    primitives: list[Primitive]


class PolicyPatch(Strict):
    from_iteration: int = 0
    critique: list[dict] = []
    root_causes: list[str] = []
    patch: list[dict]                      # [{object_id|arm, set{field: value}}]


# ----------------------------------------------------------------------------------------------------------------------
# Video understanding
# ----------------------------------------------------------------------------------------------------------------------

class RobotGuess(Strict):
    make_model_guess: str
    arms: int = 1
    gripper: str = "parallel_jaw"
    menagerie_dir: str | None = None       # closest MuJoCo Menagerie directory
    confidence: float | None = None


class VUObject(Strict):
    id: str
    type: Shape | str
    colour: str
    size_guess_m: list[float]


class GripperLabel(Strict):
    arm: str = "arm"
    uv: list[float] | None = None          # fingertip point [y, x] normalised 0..1000 (Gemini convention)
    open: bool = True
    held: str | None = None


class Keyframe(Strict):
    t: float
    robot_box: list[float] | None = None   # [ymin, xmin, ymax, xmax] normalised 0..1000 (Gemini convention)
    object_boxes: dict[str, list[float]] = {}
    gripper: list[GripperLabel] = []


class VideoUnderstanding(Strict):
    robot: RobotGuess
    phases: list[Phase] = []
    objects: list[VUObject] = []
    keyframes: list[Keyframe] = []
    rest_frame_t: float = 0.0
    success_intent: str = ""
    camera_guess: dict | None = None       # {azimuth_deg, elevation_deg, distance_m}


# ----------------------------------------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------------------------------------

class StaticObjectFidelity(Strict):
    iou: float
    size_ratio: float
    offset_px: float
    xy_err_mm: float | None = None


class FidelityReport(Strict):
    static: dict[str, StaticObjectFidelity] = {}
    alignment: float | None = None
    camera: dict = {}
    dynamic: dict = {}


StageStatus = Literal["pending", "running", "ok", "failed", "refused"]


class StageRecord(Strict):
    name: str
    status: StageStatus = "pending"
    t0: str | None = None
    t1: str | None = None
    headline: str | None = None
    artifacts: list[str] = []
    error: str | None = None
    data: dict = {}


class RunManifest(Strict):
    run_id: str
    kind: str = "pipeline"
    video: str | None = None
    prompt: str | None = None
    created: str
    stages: list[StageRecord] = Field(default_factory=list)

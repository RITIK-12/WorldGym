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
    rest_qpos: list[float] | None = None   # pose the robot sits in when idle (camera calibration); default home_qpos
    vel_limits: list[float]
    servo_damping: float | None = None     # kv = servo_damping * kp for undamped position servos (opt-in)
    cmd_speed_fraction: float = 1.0        # commanded joint speed cap as a fraction of vel_limits (soft servos)
    robot_bodies: list[str]
    finger_bodies: list[str]


class TableSpec(Strict):
    """Table slab under the workspace; its top is always z = 0."""
    center_xy: list[float] = [0.35, 0.0]
    size_xy: list[float] = [1.1, 1.1]      # full extents
    thickness: float = 0.04


class RobotAdapter(Strict):
    robot_id: str
    menagerie_dir: str
    mjcf: str
    arms: list[ArmSpec]
    base_poses: list[list[float]] = [[0.0, 0.0, 0.0, 0.0]]  # (x, y, z, yaw) per arm
    table: TableSpec | None = None         # default table for this robot's workspace
    build: dict | None = None              # generate `mjcf` from a single-arm model: {"from", "root", "arms": {name: [x, y, yaw]}}


# ----------------------------------------------------------------------------------------------------------------------
# Task / scene
# ----------------------------------------------------------------------------------------------------------------------

Shape = Literal["box", "cylinder", "sphere", "capsule", "tblock", "container", "composite", "mesh"]


class AssetRef(Strict):
    """A library asset used as an object's geometry (shape "mesh"); rescaled so its bounding box matches dims_m."""
    source: str = "simready"
    name: str                              # e.g. "cleaning_sponge_b01"
    url: str                               # root USD of the asset package
    license: str = "CC-BY-4.0"
    attribution: str = ""
    local_dir: str | None = None           # converted MuJoCo meshes (visual.obj, collision_*.obj, asset.json)
    native_dims_m: list[float] | None = None
    mass_kg: float | None = None


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
    asset: AssetRef | None = None          # mesh only
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
                        "max_unwanted_collisions", "pushed_to_region", "wiped_coverage"]


class Predicate(Strict):
    kind: PredicateKind
    object_id: str | None = None
    target_id: str | None = None
    limit: float | None = None             # metres / count depending on kind
    max_tilt_deg: float = 25.0
    region: list[float] | None = None      # pushed_to_region: [xmin, ymin, xmax, ymax]
    # wiped_coverage: object_id = the wiper, target_id = the wiped surface, limit = required fraction of its long axis


class TaskSpec(Strict):
    name: str
    robot: str                             # adapter id (robots/<id>.json)
    base_poses: list[list[float]] | None = None
    objects: list[ObjectSpec]
    phases: list[Phase] = []
    cameras: list[CameraSpec] = []
    table: TableSpec | None = None         # default: the adapter's table, else TableSpec()
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
    """kind = "pick_place": pick object_id, place it on/in target_id (or at place_xy_offset_m when no target).
    kind = "hold": reach and grasp object_id and keep holding it until every other arm is done, then release.
    kind = "wipe": pick object_id (the wiper), press it on target_id's top and stroke along the target's long axis
    between wipe_start_m and wipe_end_m (metres from the target centre), then put it back where it was picked."""
    kind: Literal["pick_place", "hold", "wipe"] = "pick_place"
    arm: str = "arm"
    object_id: str
    target_id: str | None = None
    sync_after: str | None = None          # wait until this arm is holding (or idle) before starting
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
    wipe_start_m: float = -0.15
    wipe_end_m: float = 0.15
    wipe_strokes: int = 3                  # passes over the surface (start -> end is one)
    wipe_pattern: Literal["back_and_forth", "one_way"] = "back_and_forth"   # one_way: lift and return between passes
    wipe_press_m: Num = 0.006              # wiper bottom commanded this far below the surface top (contact pressure)
    wipe_lateral_m: float = 0.0            # offset perpendicular to the wipe axis
    wipe_speed_mps: float = 0.12
    wipe_surface_z_m: Num | None = None    # surface top; default = target top (target_rim_z)
    place_back: bool = True
    hold_lift_m: float = 0.0               # hold: lift the held object by this much (0 = hold in place)


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
    # layout in the robot frame (+x forward from the base / the midpoint between the arm bases, +y the robot's left)
    offset_m: list[float] | None = None    # [x, y] of the object's centre
    long_axis: str = "none"                # "x" (front-back) | "y" (left-right) | "none"
    fixed: bool = False                    # clamped / fixture: does not move during the video
    placement: str = ""                    # where it is relative to the arms and the other objects
    role: str = ""                         # its role in the demonstrated actions
    asset: str | None = None               # NVIDIA SimReady asset name chosen with the search tool


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


class ArmMotion(Strict):
    """What one arm does over the whole demonstration, as structure (the phases give the timing detail)."""
    arm: str
    kind: str                              # hold | wipe | pick_place | push | idle
    object: str | None = None              # what its gripper holds
    target: str | None = None              # what it acts on / places onto
    t0: float = 0.0
    t1: float = 0.0
    # along the target's long side, in the robot frame: -1 = the robot's right end, +1 = its left end
    start_frac: float | None = None
    end_frac: float | None = None
    strokes: int = 1                       # contact passes over the target
    pattern: str = ""                      # "one_way" (every pass start -> end) | "back_and_forth"
    place_back: bool | None = None         # the object goes back to where it was picked up
    after: str = ""                        # what happens to the object at the end
    summary: str = ""


class VideoUnderstanding(Strict):
    robot: RobotGuess
    phases: list[Phase] = []
    motions: list[ArmMotion] = []
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

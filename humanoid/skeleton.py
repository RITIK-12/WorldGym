"""Keypoint conventions: MHR-70 (SAM 3D Body output), Penn Action 13 (eval GT), and the matching points on the G1.

MHR-70 order follows facebookresearch/sam-3d-body `metadata/mhr70.py` (body 0-14 in COCO order without wrists,
feet 15-20, right hand 21-41 ending at the wrist, left hand 42-62 ending at the wrist, extras 63-69).
Penn Action joints: head, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r ankle.
"""
from __future__ import annotations

import numpy as np

MHR = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4, "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7,
       "r_elbow": 8, "l_hip": 9, "r_hip": 10, "l_knee": 11, "r_knee": 12, "l_ankle": 13, "r_ankle": 14,
       "l_big_toe": 15, "l_small_toe": 16, "l_heel": 17, "r_big_toe": 18, "r_small_toe": 19, "r_heel": 20,
       "r_wrist": 41, "l_wrist": 62, "l_acromion": 67, "r_acromion": 68, "neck": 69}
FOOT = [MHR[k] for k in ("l_big_toe", "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel")]

# MHR-70 left/right swap (sam-3d-body metadata MHR70_PERMUTATION): mirrors a skeleton when x is negated
MHR_LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 18, 19, 20, 15, 16, 17] + list(range(42, 63)) + \
    list(range(21, 42)) + [64, 63, 66, 65, 68, 67, 69]

PENN = ["head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee",
        "r_knee", "l_ankle", "r_ankle"]
PENN_LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
PENN_LIMBS = [(1, 3), (3, 5), (2, 4), (4, 6), (7, 9), (9, 11), (8, 10), (10, 12), (1, 2), (7, 8)]


def mhr_to_penn(kp: np.ndarray) -> np.ndarray:
    """(..., 70, D) MHR keypoints -> (..., 13, D) Penn joints; head = mean of nose and ears."""
    head = kp[..., [MHR["nose"], MHR["l_ear"], MHR["r_ear"]], :].mean(axis=-2, keepdims=True)
    rest = kp[..., [MHR[k] for k in PENN[1:]], :]
    return np.concatenate([head, rest], axis=-2)


def named(kp: np.ndarray, name: str) -> np.ndarray:
    """A named point from (..., 70, D) MHR keypoints, including derived ones (mid_hip, mid_shoulder, toes)."""
    if name == "head":
        return kp[..., [MHR["nose"], MHR["l_ear"], MHR["r_ear"]], :].mean(axis=-2)
    if name == "mid_hip":
        return kp[..., [MHR["l_hip"], MHR["r_hip"]], :].mean(axis=-2)
    if name == "mid_shoulder":
        return kp[..., [MHR["l_shoulder"], MHR["r_shoulder"]], :].mean(axis=-2)
    if name in ("l_toe", "r_toe"):
        s = name[0]
        return kp[..., [MHR[f"{s}_big_toe"], MHR[f"{s}_small_toe"]], :].mean(axis=-2)
    return kp[..., MHR[name], :]


# G1 point for each human point: (body, offset in the body frame). Joint centres are body origins; the head centre
# is the head collision sphere; toe and heel are the ends of the foot sole box (z = -0.037 in the ankle roll frame).
G1_POINTS = {
    "mid_hip": ("pelvis", (0.0, 0.0, -0.1027)),
    "l_hip": ("left_hip_roll_link", (0.0, 0.0, 0.0)), "r_hip": ("right_hip_roll_link", (0.0, 0.0, 0.0)),
    "l_knee": ("left_knee_link", (0.0, 0.0, 0.0)), "r_knee": ("right_knee_link", (0.0, 0.0, 0.0)),
    "l_ankle": ("left_ankle_roll_link", (0.0, 0.0, 0.0)), "r_ankle": ("right_ankle_roll_link", (0.0, 0.0, 0.0)),
    "l_toe": ("left_ankle_roll_link", (0.13, 0.0, -0.037)), "r_toe": ("right_ankle_roll_link", (0.13, 0.0, -0.037)),
    "l_heel": ("left_ankle_roll_link", (-0.05, 0.0, -0.037)), "r_heel": ("right_ankle_roll_link", (-0.05, 0.0, -0.037)),
    "l_shoulder": ("left_shoulder_roll_link", (0.0, 0.0, 0.0)), "r_shoulder": ("right_shoulder_roll_link", (0.0, 0.0, 0.0)),
    "l_elbow": ("left_elbow_link", (0.0, 0.0, 0.0)), "r_elbow": ("right_elbow_link", (0.0, 0.0, 0.0)),
    "l_wrist": ("left_wrist_roll_link", (0.0, 0.0, 0.0)), "r_wrist": ("right_wrist_roll_link", (0.0, 0.0, 0.0)),
    "head": ("torso_link", (0.0, 0.0, 0.43)),
}
GROUP = {"mid_hip": "legs", "l_hip": "legs", "r_hip": "legs", "l_knee": "legs", "r_knee": "legs",
         "l_ankle": "feet", "r_ankle": "feet", "l_toe": "feet", "r_toe": "feet", "l_heel": "feet", "r_heel": "feet",
         "l_shoulder": "arms", "r_shoulder": "arms", "l_elbow": "arms", "r_elbow": "arms", "l_wrist": "arms",
         "r_wrist": "arms", "head": "head"}
# Bone chains from the pelvis outward: (parent, child). bone_directions retargeting walks them with G1 bone lengths.
BONES = [("mid_hip", "l_hip"), ("mid_hip", "r_hip"), ("l_hip", "l_knee"), ("r_hip", "r_knee"), ("l_knee", "l_ankle"),
         ("r_knee", "r_ankle"), ("l_ankle", "l_toe"), ("r_ankle", "r_toe"), ("l_ankle", "l_heel"),
         ("r_ankle", "r_heel"), ("mid_hip", "l_shoulder"), ("mid_hip", "r_shoulder"), ("l_shoulder", "l_elbow"),
         ("r_shoulder", "r_elbow"), ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist"), ("mid_hip", "head")]
# Penn joints on the G1 for projection into the clip (same order as PENN)
G1_PENN = ["head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee",
           "r_knee", "l_ankle", "r_ankle"]

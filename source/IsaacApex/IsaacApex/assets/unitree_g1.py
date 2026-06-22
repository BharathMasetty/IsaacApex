# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Unitree G1 humanoid robot configurations for Isaac Lab.

This module holds **two distinct G1 variants** — they are different robots and
are not interchangeable:

1. ``G1_CFG`` — the 37-DoF Nucleus USD used by the **locomotion** task:
     1  torso joint              12 leg joints (6 per side)
     10 upper-arm joints         14 finger joints (passive)
   The locomotion policy controls only the 12 leg joints; torso, arms, and
   fingers are held at their default positions.

2. ``G1_29DOF_CFG`` — the full 29-DoF G1 (3-DoF waist + 7-DoF arms *with wrists*)
   used by the **motion-tracking** task. Spawned from the open-source Isaac Sim
   Nucleus USD, whose 29 body joints + body-link names line up exactly with the
   LAFAN1 retargeted reference motions. The same USD also carries 14 dexterous
   finger joints; the tracking task does not use them, so they are locked at their
   folded default by a dedicated ``hands`` actuator and excluded from the policy's
   action space (see ``G1_29DOF_JOINTS`` / the env's ``ActionsCfg``). Body gains
   and the per-joint action scale are computed from each motor's armature
   (BeyondMimic recipe) rather than hand-tuned.

The 37-DoF model lacks the 3-DoF waist and the wrist joints/links that the dance
motions drive, which is why tracking needs its own articulation.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"

# Command delay: 0–4 physics steps at dt=0.005 s → 0–20 ms
_MIN_DELAY = 0
_MAX_DELAY = 4

# ── Joint ordering ────────────────────────────────────────────────────────
# 23 non-finger joints used for observations and the symmetry augmentation.
# Finger joints (.*_(zero|one|two|three|four|five|six)_joint) are passive
# and excluded from the observation space.

JOINTS: list[str] = [
    # Torso (1)
    "torso_joint",
    # Left leg (6)
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Left upper arm (5)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",    "left_elbow_roll_joint",
    # Right upper arm (5)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",    "right_elbow_roll_joint",
]  # 23 total

BODIES: list[str] = [
    "pelvis",
    "torso_link",
    # Left leg
    "left_hip_pitch_link",  "left_hip_roll_link",  "left_hip_yaw_link",
    "left_knee_link",       "left_ankle_pitch_link", "left_ankle_roll_link",
    # Right leg
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link",      "right_ankle_pitch_link", "right_ankle_roll_link",
    # Left upper arm
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_pitch_link",    "left_elbow_roll_link",
    # Right upper arm
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_pitch_link",    "right_elbow_roll_link",
]

# Joints driven by the locomotion policy (legs only)
ACTION_JOINTS: list[str] = [
    "left_hip_roll_joint",  "left_hip_pitch_joint",  "left_hip_yaw_joint",
    "left_knee_joint",      "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint",
    "right_knee_joint",     "right_ankle_pitch_joint", "right_ankle_roll_joint",
]  # 12

# Non-policy joints held at default by HoldDefaultJointsAction (torso + upper arms)
NON_ACTION_JOINTS: list[str] = [j for j in JOINTS if j not in ACTION_JOINTS]
# = 1 torso + 10 arm = 11

# ── Articulation config ───────────────────────────────────────────────────

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            # Legs: slight knee bend for stable initialisation
            "left_hip_pitch_joint":    -0.10,  "right_hip_pitch_joint":    -0.10,
            "left_knee_joint":          0.30,  "right_knee_joint":          0.30,
            "left_ankle_pitch_joint":  -0.20,  "right_ankle_pitch_joint":  -0.20,
            # Arms: relaxed natural pose
            "left_shoulder_pitch_joint":   0.30,  "right_shoulder_pitch_joint":   0.30,
            "left_shoulder_roll_joint":    0.25,  "right_shoulder_roll_joint":   -0.25,
            "left_elbow_pitch_joint":      0.97,  "right_elbow_pitch_joint":      0.97,
            "left_elbow_roll_joint":       0.15,  "right_elbow_roll_joint":      -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            stiffness=100.0, damping=2.0,
            effort_limit_sim=71.0, velocity_limit_sim=35.52,
            armature=0.010, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "hip_yaw": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            stiffness=100.0, damping=2.0,
            effort_limit_sim=71.0, velocity_limit_sim=35.52,
            armature=0.010, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "hip_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            stiffness=100.0, damping=2.0,
            effort_limit_sim=111.0, velocity_limit_sim=22.70,
            armature=0.025, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            stiffness=150.0, damping=4.0,
            effort_limit_sim=111.0, velocity_limit_sim=22.70,
            armature=0.025, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "ankle_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            stiffness=40.0, damping=2.0,
            effort_limit_sim=24.8, velocity_limit_sim=40.13,
            armature=0.004, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "ankle_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            stiffness=40.0, damping=2.0,
            effort_limit_sim=24.8, velocity_limit_sim=40.13,
            armature=0.004, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "torso": DelayedPDActuatorCfg(
            joint_names_expr=["torso_joint"],
            stiffness=200.0, damping=5.0,
            effort_limit_sim=71.0, velocity_limit_sim=35.52,
            armature=0.010, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "shoulder": DelayedPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
            stiffness=40.0, damping=1.0,
            effort_limit_sim=24.8, velocity_limit_sim=40.13,
            armature=0.004, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "shoulder_yaw_elbow_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_pitch_joint"],
            stiffness=40.0, damping=1.0,
            effort_limit_sim=24.8, velocity_limit_sim=40.13,
            armature=0.004, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        "elbow_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*_elbow_roll_joint"],
            stiffness=40.0, damping=1.0,
            effort_limit_sim=24.8, velocity_limit_sim=40.13,
            armature=0.004, min_delay=_MIN_DELAY, max_delay=_MAX_DELAY,
        ),
        # Finger joints held passive with light stiffness
        "fingers": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_zero_joint", ".*_one_joint",   ".*_two_joint",
                ".*_three_joint", ".*_four_joint", ".*_five_joint", ".*_six_joint",
            ],
            stiffness=5.0, damping=0.1,
            effort_limit_sim=2.0, velocity_limit_sim=10.0,
            armature=0.001, min_delay=0, max_delay=0,
        ),
    },
)

# ── Per-joint action scale ────────────────────────────────────────────────
# scale ≈ effort_limit / stiffness / 4 so max residual command ≈ effort_limit

_EFFORT_SCALE = 4.0

ACTION_SCALE: dict[str, float] = {
    "left_hip_roll_joint":     111.0 / 100.0 / _EFFORT_SCALE,
    "left_hip_pitch_joint":     71.0 / 100.0 / _EFFORT_SCALE,
    "left_hip_yaw_joint":       71.0 / 100.0 / _EFFORT_SCALE,
    "left_knee_joint":         111.0 / 150.0 / _EFFORT_SCALE,
    "left_ankle_pitch_joint":   24.8 /  40.0 / _EFFORT_SCALE,
    "left_ankle_roll_joint":    24.8 /  40.0 / _EFFORT_SCALE,
    "right_hip_roll_joint":    111.0 / 100.0 / _EFFORT_SCALE,
    "right_hip_pitch_joint":    71.0 / 100.0 / _EFFORT_SCALE,
    "right_hip_yaw_joint":      71.0 / 100.0 / _EFFORT_SCALE,
    "right_knee_joint":        111.0 / 150.0 / _EFFORT_SCALE,
    "right_ankle_pitch_joint":  24.8 /  40.0 / _EFFORT_SCALE,
    "right_ankle_roll_joint":   24.8 /  40.0 / _EFFORT_SCALE,
}


# ═══════════════════════════════════════════════════════════════════════════
# G1 29-DoF (motion tracking) — full waist + wrists, spawned from Nucleus USD
# ═══════════════════════════════════════════════════════════════════════════
#
# Actuator gains follow the BeyondMimic / whole_body_tracking recipe: rather than
# hand-tuning, stiffness/damping are derived from each motor's rotor armature and a
# target closed-loop natural frequency (DAMPING_RATIO=2). Per-joint action scale is
# computed as 0.25 * effort_limit / stiffness.
#
# Reference: https://github.com/HybridRobotics/whole_body_tracking (robots/g1.py)

# Open-source 29-DoF G1 USD (also carries dexterous finger joints, which we lock).
G1_29DOF_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"
# Regex matching the 14 finger joints to lock (left/right index/middle/thumb).
G1_HAND_JOINT_REGEX = ".*_hand_.*_joint"

# Canonical 29-DoF joint order — the column order of the LAFAN1 retargeted CSVs.
G1_29DOF_JOINTS: list[str] = [
    # Left leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # 29 total

# Bodies tracked by the motion command (one frame target per body). The anchor
# (root reference frame) is the torso; the end-effectors drive the strict
# z-height terminations. These names must exist as links in the USD.
G1_ANCHOR_BODY_NAME: str = "torso_link"
G1_TRACKED_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
G1_END_EFFECTOR_BODY_NAMES: list[str] = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]

# ── Motor parameters → computed gains ─────────────────────────────────────
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425

_NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz target closed-loop bandwidth
_DAMPING_RATIO = 2.0

_STIFF_5020 = _ARMATURE_5020 * _NATURAL_FREQ**2
_STIFF_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ**2
_STIFF_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ**2
_STIFF_4010 = _ARMATURE_4010 * _NATURAL_FREQ**2

_DAMP_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMP_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMP_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMP_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ

G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_29DOF_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": _STIFF_7520_14, ".*_hip_roll_joint": _STIFF_7520_22,
                ".*_hip_yaw_joint": _STIFF_7520_14, ".*_knee_joint": _STIFF_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": _DAMP_7520_14, ".*_hip_roll_joint": _DAMP_7520_22,
                ".*_hip_yaw_joint": _DAMP_7520_14, ".*_knee_joint": _DAMP_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": _ARMATURE_7520_14, ".*_hip_roll_joint": _ARMATURE_7520_22,
                ".*_hip_yaw_joint": _ARMATURE_7520_14, ".*_knee_joint": _ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _STIFF_5020,
            damping=2.0 * _DAMP_5020,
            armature=2.0 * _ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _STIFF_5020,
            damping=2.0 * _DAMP_5020,
            armature=2.0 * _ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=_STIFF_7520_14,
            damping=_DAMP_7520_14,
            armature=_ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                ".*_elbow_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0, ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0, ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0, ".*_wrist_pitch_joint": 5.0, ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0, ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0, ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0, ".*_wrist_pitch_joint": 22.0, ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _STIFF_5020, ".*_shoulder_roll_joint": _STIFF_5020,
                ".*_shoulder_yaw_joint": _STIFF_5020, ".*_elbow_joint": _STIFF_5020,
                ".*_wrist_roll_joint": _STIFF_5020, ".*_wrist_pitch_joint": _STIFF_4010,
                ".*_wrist_yaw_joint": _STIFF_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": _DAMP_5020, ".*_shoulder_roll_joint": _DAMP_5020,
                ".*_shoulder_yaw_joint": _DAMP_5020, ".*_elbow_joint": _DAMP_5020,
                ".*_wrist_roll_joint": _DAMP_5020, ".*_wrist_pitch_joint": _DAMP_4010,
                ".*_wrist_yaw_joint": _DAMP_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": _ARMATURE_5020, ".*_shoulder_roll_joint": _ARMATURE_5020,
                ".*_shoulder_yaw_joint": _ARMATURE_5020, ".*_elbow_joint": _ARMATURE_5020,
                ".*_wrist_roll_joint": _ARMATURE_5020, ".*_wrist_pitch_joint": _ARMATURE_4010,
                ".*_wrist_yaw_joint": _ARMATURE_4010,
            },
        ),
        # Finger joints carried by the USD but unused for tracking — held folded at
        # their default (0). The policy does not actuate them (see G1_29DOF_JOINTS).
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[G1_HAND_JOINT_REGEX],
            effort_limit_sim=5.0,
            velocity_limit_sim=10.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.001,
        ),
    },
)


def _compute_29dof_action_scale(cfg: ArticulationCfg) -> dict[str, float]:
    """Per-joint action scale = 0.25 * effort_limit / stiffness (BeyondMimic recipe).

    Only the 29 body joints are included — the finger joints are not policy-actuated,
    so emitting a scale entry for them would leave an unmatched action-scale regex.
    """
    scale: dict[str, float] = {}
    for actuator in cfg.actuators.values():
        eff = actuator.effort_limit_sim
        stiff = actuator.stiffness
        names = actuator.joint_names_expr
        if not isinstance(eff, dict):
            eff = {n: eff for n in names}
        if not isinstance(stiff, dict):
            stiff = {n: stiff for n in names}
        for n in names:
            if "_hand_" in n:
                continue
            if n in eff and n in stiff and stiff[n]:
                scale[n] = 0.25 * eff[n] / stiff[n]
    return scale


# Keyed by the regex used in the actuator cfg; ``JointPositionActionCfg`` resolves
# these regex keys against the matched joints, same as the reference.
G1_29DOF_ACTION_SCALE: dict[str, float] = _compute_29dof_action_scale(G1_29DOF_CFG)

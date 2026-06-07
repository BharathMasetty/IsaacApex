# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Unitree G1 humanoid robot configuration for Isaac Lab.

The Nucleus USD has 37 actuated DoF:
  1  torso joint
  12 leg joints (6 per side)
  10 upper-arm joints (5 per side: shoulder×3 + elbow×2)
  14 finger joints (7 per side, kept passive)

The locomotion policy controls only the 12 leg joints; torso, arms, and
fingers are held at their default positions.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

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

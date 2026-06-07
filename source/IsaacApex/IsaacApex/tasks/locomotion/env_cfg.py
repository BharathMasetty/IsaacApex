# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""G1 locomotion environment configuration.

Standard ManagerBasedRLEnv with:
  - Gait command (velocity + height + gait clock)
  - Leg-only policy (12 joints), arms/waist held at default
  - Asymmetric actor-critic (proprioceptive policy, privileged critic)
  - Left-right symmetry augmentation hook
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
import isaaclab.envs.mdp as mdp

from IsaacApex.assets.unitree_g1 import (
    ACTION_JOINTS,
    ACTION_SCALE,
    G1_CFG,
    JOINTS,
    NON_ACTION_JOINTS,
)
import IsaacApex.tasks.locomotion.mdp as apex_mdp


# ── Scene ─────────────────────────────────────────────────────────────────

@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Flat-ground scene with contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Full-body contact sensor (used to detect illegal contacts)
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Height scanner for critic (optional, uses raycaster)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


# ── Commands ─────────────────────────────────────────────────────────────

@configclass
class CommandsCfg:
    motion_command = apex_mdp.GaitCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 10.0),
        ranges={
            "vel_x":       (-0.5,  0.5),
            "vel_y":       (-0.3,  0.3),
            "omega":       (-0.5,  0.5),
            "height":      (0.60,  0.78),
            "gait_freq":   (1.0,   2.0),
            "step_height": (0.05,  0.12),
            "heading":     (-math.pi, math.pi),
        },
        standing_prob=0.05,
        turning_prob=0.10,
        heading_prob=0.30,
        use_heading=True,
        heading_gain=1.0,
        min_walk_speed=0.15,
        debug_vis=True,
    )


# ── Actions ───────────────────────────────────────────────────────────────

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTION_JOINTS,
        scale=ACTION_SCALE,
        use_default_offset=True,
    )
    hold_default = apex_mdp.HoldDefaultJointsActionCfg(
        joint_names=NON_ACTION_JOINTS,
    )


# ── Observations ─────────────────────────────────────────────────────────

@configclass
class PolicyObsCfg(ObservationGroupCfg):
    """Proprioceptive observations for the actor.

    Observation dim: 3+3+3+6+23+23+12+4 = 77.
    This flat layout is assumed by the sagittal-mirror augmentation in symmetry.py.
    If you add history or new terms, update the slice constants there too.
    """
    enable_corruption: bool = True
    concatenate_terms: bool = True

    base_lin_vel = ObservationTermCfg(
        func=mdp.base_lin_vel,
        noise=GaussianNoiseCfg(mean=0.0, std=0.1),
    )
    base_ang_vel = ObservationTermCfg(
        func=mdp.base_ang_vel,
        noise=GaussianNoiseCfg(mean=0.0, std=0.2),
    )
    projected_gravity = ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=GaussianNoiseCfg(mean=0.0, std=0.05),
    )
    velocity_command = ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "motion_command"},
    )
    joint_pos = ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)},
        noise=GaussianNoiseCfg(mean=0.0, std=0.01),
    )
    joint_vel = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)},
        noise=GaussianNoiseCfg(mean=0.0, std=1.5),
    )
    last_action = ObservationTermCfg(
        func=mdp.last_action,
    )
    gait_phase = ObservationTermCfg(
        func=apex_mdp.gait_phase_obs,
        params={"command_name": "motion_command"},
    )


@configclass
class CriticObsCfg(ObservationGroupCfg):
    """Privileged observations for the critic (no noise, adds foot state)."""
    enable_corruption: bool = False
    concatenate_terms: bool = True

    base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
    base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
    projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
    velocity_command = ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "motion_command"},
    )
    joint_pos = ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)},
    )
    joint_vel = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)},
    )
    last_action = ObservationTermCfg(func=mdp.last_action)
    gait_phase = ObservationTermCfg(
        func=apex_mdp.gait_phase_obs,
        params={"command_name": "motion_command"},
    )
    # Privileged extras
    foot_contact = ObservationTermCfg(
        func=apex_mdp.contact_state,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"])},
    )
    foot_air_time = ObservationTermCfg(
        func=apex_mdp.contact_air_time,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"])},
    )
    foot_forces = ObservationTermCfg(
        func=apex_mdp.log_contact_forces,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"])},
    )
    foot_height = ObservationTermCfg(
        func=apex_mdp.foot_height,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])},
    )


@configclass
class ObservationsCfg:
    policy: PolicyObsCfg = PolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


# ── Events ────────────────────────────────────────────────────────────────

@configclass
class EventCfg:
    reset_scene = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )
    reset_root = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    randomize_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.1, 0.1),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ── Rewards ───────────────────────────────────────────────────────────────

_left_foot_cfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link"])
_right_foot_cfg = SceneEntityCfg("robot", body_names=["right_ankle_roll_link"])
_all_feet_cfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])
_left_sensor_cfg = SceneEntityCfg("contact_sensor", body_names=["left_ankle_roll_link"])
_right_sensor_cfg = SceneEntityCfg("contact_sensor", body_names=["right_ankle_roll_link"])
_all_feet_sensor_cfg = SceneEntityCfg("contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])
_torso_sensor_cfg = SceneEntityCfg(
    "contact_sensor",
    body_names=[
        "pelvis", "torso_link",
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
    ],
)
_all_joint_cfg = SceneEntityCfg("robot", joint_names=JOINTS)

# Per-joint weights when walking — matches source POSE_WEIGHTS for leg joints.
# Non-action joints (torso, arms) use a uniform moderate weight in both modes.
_WALKING_POSE_WEIGHTS = {
    # Leg joints — per-joint tuning from source
    "left_hip_roll_joint":    0.5,  "right_hip_roll_joint":    0.5,
    "left_hip_pitch_joint":   0.1,  "right_hip_pitch_joint":   0.1,
    "left_hip_yaw_joint":     1.0,  "right_hip_yaw_joint":     1.0,
    "left_knee_joint":        0.1,  "right_knee_joint":        0.1,
    "left_ankle_pitch_joint": 0.1,  "right_ankle_pitch_joint": 0.1,
    "left_ankle_roll_joint":  1.0,  "right_ankle_roll_joint":  1.0,
    # Non-action joints held at default with uniform weight
    **{j: 0.5 for j in NON_ACTION_JOINTS},
}


@configclass
class RewardsCfg:
    # ── Task ─────────────────────────────────────────────────────────────────
    track_vel_xy = RewardTermCfg(
        func=apex_mdp.track_lin_vel_xy,
        weight=8.0,
        params={"std": math.sqrt(0.15), "command_name": "motion_command"},
    )
    track_vel_z = RewardTermCfg(
        func=apex_mdp.TrackAngVelZ,
        weight=6.0,
        params={"std": math.sqrt(0.15), "command_name": "motion_command"},
    )
    track_height = RewardTermCfg(
        func=apex_mdp.track_height,
        weight=3.0,
        params={"std": math.sqrt(0.05), "command_name": "motion_command"},
    )
    vel_progress = RewardTermCfg(
        func=apex_mdp.vel_tracking_progress,
        weight=3.0,
        params={"command_name": "motion_command", "stance_threshold": 0.1},
    )
    # ── Gait ─────────────────────────────────────────────────────────────────
    gait_sync = RewardTermCfg(
        func=apex_mdp.gait_phase_sync,
        weight=2.0,
        params={
            "left_sensor_cfg": _left_sensor_cfg,
            "right_sensor_cfg": _right_sensor_cfg,
            "command_name": "motion_command",
        },
    )
    foot_clearance = RewardTermCfg(
        func=apex_mdp.foot_clearance,
        weight=2.0,
        params={
            "left_foot_cfg": _left_foot_cfg,
            "right_foot_cfg": _right_foot_cfg,
            "command_name": "motion_command",
        },
    )
    # ── Posture ───────────────────────────────────────────────────────────────
    upright = RewardTermCfg(
        func=apex_mdp.body_upright,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "std": math.sqrt(0.1),
        },
    )
    default_pose = RewardTermCfg(
        func=apex_mdp.default_pose,
        weight=0.5,
        params={
            "asset_cfg": _all_joint_cfg,
            "weights": _WALKING_POSE_WEIGHTS,
            "std": math.sqrt(0.1),
        },
    )
    # ── Foot quality ─────────────────────────────────────────────────────────
    foot_flat = RewardTermCfg(
        func=apex_mdp.foot_flat_at_contact,
        weight=1.0,
        params={"asset_cfg": _all_feet_cfg, "sensor_cfg": _all_feet_sensor_cfg, "std": math.sqrt(0.1)},
    )
    foot_slip = RewardTermCfg(
        func=apex_mdp.foot_slip,
        weight=1.0,
        params={"asset_cfg": _all_feet_cfg, "sensor_cfg": _all_feet_sensor_cfg, "std": math.sqrt(0.1)},
    )
    foot_impact = RewardTermCfg(
        func=apex_mdp.foot_impact_vel,
        weight=0.5,
        params={"asset_cfg": _all_feet_cfg, "sensor_cfg": _all_feet_sensor_cfg, "std": math.sqrt(0.1)},
    )
    # ── Regularisation ────────────────────────────────────────────────────────
    action_rate = RewardTermCfg(
        func=apex_mdp.action_rate,
        weight=1.0,
        params={"std": math.sqrt(0.1)},
    )
    joint_vel = RewardTermCfg(
        func=apex_mdp.joint_vel_penalty,
        weight=1.0,
        params={"std": math.sqrt(0.1), "asset_cfg": _all_joint_cfg},
    )
    joint_torque = RewardTermCfg(
        func=apex_mdp.joint_torque_penalty,
        weight=1.0,
        params={"std": math.sqrt(0.1), "asset_cfg": _all_joint_cfg},
    )
    joint_limits = RewardTermCfg(
        func=apex_mdp.joint_pos_limits_penalty,
        weight=1.0,
        params={"std": math.sqrt(0.1), "asset_cfg": _all_joint_cfg},
    )
    # angular_momentum = RewardTermCfg(
        # func=apex_mdp.angular_momentum_penalty,
        # weight=0.01,
        # params={"std": math.sqrt(0.1), "asset_cfg": SceneEntityCfg("robot")},
    # )
    # Termination
    termination = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-50.0,
    )


# ── Terminations ──────────────────────────────────────────────────────────

@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fell_over = TerminationTermCfg(
        func=apex_mdp.fell_over,
        params={"min_height": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
    illegal_contact = TerminationTermCfg(
        func=apex_mdp.illegal_contact,
        params={"threshold": 1.0, "sensor_cfg": _torso_sensor_cfg},
    )


# ── Curriculum ───────────────────────────────────────────────────────────

@configclass
class CurriculumCfg:
    """Two-track curriculum: performance-gated velocity ranges + time-based reg annealing."""

    vel_range = CurriculumTermCfg(
        func=apex_mdp.VelRangeCurriculum,
        params={
            "command_name": "motion_command",
            "stages": [
                {"vel_x": (-0.5, 0.5), "vel_y": (-0.3, 0.3), "omega": (-0.5, 0.5)},
                {"vel_x": (-0.8, 0.8), "vel_y": (-0.4, 0.4), "omega": (-0.8, 0.8)},
                {"vel_x": (-1.0, 1.0), "vel_y": (-0.5, 0.5), "omega": (-1.0, 1.0)},
            ],
            "advance_thresh": 0.65,
            "regress_thresh": 0.25,
            "ema_alpha": 0.1,
        },
    )

    reg_weights = CurriculumTermCfg(
        func=apex_mdp.RegWeightCurriculum,
        params={
            "terms": {
                "action_rate":  0.1,   
                "joint_vel":    0.1,   
                "joint_torque": 0.1,  
                "default_pose": 0.1,  
            },
            "warmup_iters": 3000,
        },
    )


# ── Environment Config ────────────────────────────────────────────────────

@configclass
class G1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Full configuration for the G1 velocity-tracking locomotion task."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22

# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""G1 whole-body motion-tracking environment configuration.

A ManagerBasedRLEnv that drives the 29-DoF G1 to imitate a reference motion
(LAFAN1 retargeted, expressive dance by default). The policy receives the
reference joint targets + anchor-relative pose error and is rewarded (DeepMimic
style) for matching per-body position/orientation/velocity. Reset uses Reference
State Initialisation with an adaptive start-frame sampler (see
[mdp/commands.py](mdp/commands.py)).

Ported from BeyondMimic / whole_body_tracking, adapted to the IsaacApex layout.
The default ``motion_file`` points at the bundled .npz so ``train.py`` runs with
no extra flags. Generate other motions with
[scripts/tracking/csv_to_npz.py](../../../../../scripts/tracking/csv_to_npz.py).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from IsaacApex.assets.unitree_g1 import (
    G1_29DOF_ACTION_SCALE,
    G1_29DOF_CFG,
    G1_29DOF_JOINTS,
    G1_ANCHOR_BODY_NAME,
    G1_END_EFFECTOR_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
import IsaacApex.tasks.tracking.mdp as mdp

# Default reference motion bundled with the task (expressive LAFAN1 dance).
MOTIONS_DIR = os.path.join(os.path.dirname(__file__), "motions")
DEFAULT_MOTION_FILE = os.path.join(MOTIONS_DIR, "dance1_subject2.npz")

# Root velocity perturbations — used both for random pushes and for the random
# velocity offset applied at reference-state initialisation.
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# All bodies except the 4 end-effectors — contact on any of these is penalised.
_UNDESIRED_CONTACT_REGEX = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)


# ── Scene ─────────────────────────────────────────────────────────────────


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene with a full-body contact sensor."""

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
    )

    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


# ── Commands ─────────────────────────────────────────────────────────────


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=DEFAULT_MOTION_FILE,
        anchor_body_name=G1_ANCHOR_BODY_NAME,
        body_names=G1_TRACKED_BODY_NAMES,
        resampling_time_range=(1.0e9, 1.0e9),  # resample only at motion end / failure
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


# ── Actions ──────────────────────────────────────────────────────────────


@configclass
class ActionsCfg:
    # Policy actuates the 29 body joints; the USD's finger joints are excluded and
    # held folded by the robot's ``hands`` actuator (see assets/unitree_g1.py).
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_JOINTS,
        scale=G1_29DOF_ACTION_SCALE,
        use_default_offset=True,
    )


# ── Observations ─────────────────────────────────────────────────────────


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive + reference observations for the actor (order preserved)."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic observations: same as policy (noise-free) plus per-body frames."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


# ── Events ───────────────────────────────────────────────────────────────


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


# ── Rewards ──────────────────────────────────────────────────────────────


@configclass
class RewardsCfg:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        # Only the 29 policy-driven body joints. The USD's locked finger joints rest
        # at their (hard-limit) default, which lies *outside* their soft limits — so
        # including them here would charge a large constant penalty every step
        # (this is what pinned the reward at ~-0.28). Fingers are not actuated, so
        # excluding them is correct, not just a workaround.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINTS)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[_UNDESIRED_CONTACT_REGEX]),
            "threshold": 1.0,
        },
    )


# ── Terminations ─────────────────────────────────────────────────────────


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": G1_END_EFFECTOR_BODY_NAMES,
        },
    )


@configclass
class CurriculumCfg:
    pass


# ── Environment ──────────────────────────────────────────────────────────


@configclass
class G1TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Full configuration for the G1 whole-body motion-tracking task."""

    scene: TrackingSceneCfg = TrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer follows the robot root
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

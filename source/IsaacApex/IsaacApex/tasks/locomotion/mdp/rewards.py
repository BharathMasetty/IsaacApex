# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Reward functions for the G1 locomotion task."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Reward kernels ────────────────────────────────────────────────────────

def _exp_l1(x: torch.Tensor, std: float) -> torch.Tensor:
    """exp(-mean(|x|) / std²)  — bounded [0, 1]."""
    return torch.exp(-x.abs().mean(dim=-1) / std**2)


def _quad(x: torch.Tensor, std: float) -> torch.Tensor:
    """1 - mean((x/std)²)  — value 1 at zero, decreasing quadratically."""
    return 1.0 - (x / std).square().mean(dim=-1)


def _qabs(x: torch.Tensor, std: float) -> torch.Tensor:
    """0.5 * (exp(-mean|x|/std²) + 1 - mean((x/std)²))  — smooth blended kernel."""
    return 0.5 * (_exp_l1(x, std) + _quad(x, std))


# ── Velocity and height tracking ──────────────────────────────────────────

def track_lin_vel_xy(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "motion_command",
    stance_threshold: float = 0.1,
) -> torch.Tensor:
    """Exponential reward for matching commanded xy velocity.

    Uses sum-of-squared errors (not mean) to match source kernel:
    exp(-(||v_xy_err||² + vz²) / std²).
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    moving = (cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()) >= stance_threshold
    target_xy = torch.where(moving.unsqueeze(1), cmd[:, :2], torch.zeros_like(cmd[:, :2]))
    vel_b = asset.data.root_lin_vel_b
    err_xy = target_xy - vel_b[:, :2]
    total_sq = err_xy.square().sum(dim=-1) + vel_b[:, 2].square()
    return torch.exp(-total_sq / std**2)


def track_ang_vel_z(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "motion_command",
    stance_threshold: float = 0.1,
) -> torch.Tensor:
    """Exponential reward for yaw tracking and pitch/roll suppression.

    Uses sum-of-squared errors: exp(-(||ang_xy||² + z_err²) / std²).
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    moving = (cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()) >= stance_threshold
    target_z = torch.where(moving, cmd[:, 2], torch.zeros_like(cmd[:, 2]))
    ang_b = asset.data.root_ang_vel_b
    xy_sq = ang_b[:, :2].square().sum(dim=-1)
    z_sq = (target_z - ang_b[:, 2]).square()
    return torch.exp(-(xy_sq + z_sq) / std**2)


def track_height(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "motion_command",
) -> torch.Tensor:
    """Exponential L1 reward for matching the commanded pelvis height."""
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    err = (cmd[:, 3] - asset.data.root_pos_w[:, 2]).unsqueeze(-1)
    return _exp_l1(err, std)


# ── Posture ───────────────────────────────────────────────────────────────

def body_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward keeping a link's z-axis aligned with world up."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).view(1, 1, 3).expand(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    proj_g = quat_apply_inverse(quat, g_world)
    tilt = (1.0 + proj_g[..., 2])  # 0 = upright, positive = tilted
    return _exp_l1(tilt, std)


def default_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    weights: dict[str, float],
    std: float = 0.1,
) -> torch.Tensor:
    """Reward per-joint proximity to default pose with motion-adaptive per-joint weights.

    Uses `weights` when the robot is moving and `standing_weights` when stationary,
    allowing tighter stance enforcement during standing without penalising walking gait.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, joint_names = asset.find_joints(list(weights.keys()))
    _, _, w_walk = resolve_matching_names_values(weights, joint_names)
    w_walk_t = torch.tensor(w_walk, device=env.device, dtype=torch.float32)
    deviation = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return _exp_l1(deviation * w_walk_t, std)


# ── Gait shape ────────────────────────────────────────────────────────────

def gait_phase_sync(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    command_name: str = "motion_command",
    stance_threshold: float = 0.05,
    ds_fraction: float = 0.05,
) -> torch.Tensor:
    """Reward foot contact/air states that match the commanded gait clock."""
    left:  ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    speed = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
    is_standing = speed < stance_threshold

    ct_l = left.data.current_contact_time[:, left_sensor_cfg.body_ids].sum(dim=1)
    ct_r = right.data.current_contact_time[:, right_sensor_cfg.body_ids].sum(dim=1)
    at_l = left.data.current_air_time[:, left_sensor_cfg.body_ids].sum(dim=1)
    at_r = right.data.current_air_time[:, right_sensor_cfg.body_ids].sum(dim=1)

    both_grounded = torch.minimum(ct_l, ct_r).clamp(max=1.0)

    gait_term = env.command_manager.get_term(command_name)
    phase = (gait_term.get_gait_phase()[:, 0] / (2.0 * math.pi)) % 1.0
    freq = cmd[:, 4].clamp(min=0.1)
    T = 1.0 / freq
    swing = 0.5 - ds_fraction

    sw1 = (phase >= 0.0)       & (phase < swing)
    st1 = (phase >= swing)     & (phase < 0.5)
    sw2 = (phase >= 0.5)       & (phase < 0.5 + swing)
    st2 =  phase >= (0.5 + swing)

    in_c_l = (ct_l > 0).float(); in_a_l = (at_l > 0).float()
    in_c_r = (ct_r > 0).float(); in_a_r = (at_r > 0).float()
    
    binary = (
        sw1 * (in_a_l * in_c_r)
        + st1 * (in_c_l * in_c_r)
        + sw2 * (in_c_l * in_a_r)
        + st2 * (in_c_l * in_c_r)
    )

    lct_l = left.data.last_contact_time[:, left_sensor_cfg.body_ids].sum(dim=1)
    lct_r = right.data.last_contact_time[:, right_sensor_cfg.body_ids].sum(dim=1)
    lat_l = left.data.last_air_time[:, left_sensor_cfg.body_ids].sum(dim=1)
    lat_r = right.data.last_air_time[:, right_sensor_cfg.body_ids].sum(dim=1)
    sw_T = (swing * T).clamp(min=1e-6)
    ds_T = (ds_fraction * T).clamp(min=1e-6)
    st_T = (0.5 * T).clamp(min=1e-6)

    timing = (
        sw1 * (lat_l / sw_T + lct_r / sw_T)
        + st1 * (lct_l / ds_T + lct_r / st_T)
        + sw2 * (lct_l / sw_T + lat_r / sw_T)
        + st2 * (lct_l / st_T + lct_r / ds_T)
    ).clamp(max=1.0) * 0.5

    walk_reward = (binary + timing) * 0.5
    return torch.where(is_standing, both_grounded, walk_reward)


def foot_clearance(
    env: ManagerBasedRLEnv,
    left_foot_cfg: SceneEntityCfg,
    right_foot_cfg: SceneEntityCfg,
    command_name: str = "motion_command",
    stance_threshold: float = 0.1,
    ds_fraction: float = 0.05,
    min_height: float = 0.035,
) -> torch.Tensor:
    """Reward each foot reaching the commanded step height during its swing phase."""
    asset: Articulation = env.scene[left_foot_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    step_h = cmd[:, 5]
    moving = (cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()) >= stance_threshold

    gait_term = env.command_manager.get_term(command_name)
    phase = (gait_term.get_gait_phase()[:, 0] / (2.0 * math.pi)) % 1.0
    swing = 0.5 - ds_fraction

    in_l_swing = moving & (phase < swing)
    in_r_swing = moving & (phase >= 0.5) & (phase < swing + 0.5)

    target_l = torch.where(
        in_l_swing,
        step_h * torch.sin(math.pi * phase / swing) + min_height,
        torch.zeros_like(step_h),
    )
    target_r = torch.where(
        in_r_swing,
        step_h * torch.sin(math.pi * (phase - 0.5) / swing) + min_height,
        torch.zeros_like(step_h),
    )

    z_l = asset.data.body_link_pos_w[:, left_foot_cfg.body_ids, 2].mean(dim=-1)
    z_r = asset.data.body_link_pos_w[:, right_foot_cfg.body_ids, 2].mean(dim=-1)

    err_l = (z_l - target_l).abs()
    err_r = (z_r - target_r).abs()
    mean_error = (err_l + err_r) * 0.5
    std = step_h.clamp(min=1e-3)*0.5
    return _exp_l1(mean_error, std)


# ── Foot quality ──────────────────────────────────────────────────────────

def foot_flat_at_contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward foot flatness during ground contact (1 = perfectly flat, 0 = max tilt)."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 0.0).float()
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).view(1, 1, 3).expand(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    proj_g = quat_apply_inverse(quat, g_w)
    lateral_tilt = proj_g[..., :2].norm(dim=-1) * in_contact
    return _exp_l1(lateral_tilt, std)


def foot_slip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward low foot slip during ground contact (1 = no slip, 0 = high slip)."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 0.0).float()
    xy_speed = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return _exp_l1(xy_speed * in_contact, std)


def foot_impact_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward soft foot landings (1 = gentle touchdown, →0 = high-speed impact).

    At no-contact frames first_contact=0, so error=0 and reward=1 (max).
    At touchdown with high velocity, reward approaches 0.
    Use with positive weight to incentivise gentle landing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    speed = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids].norm(dim=-1)
    return _exp_l1(speed * first_contact, std)


# ── Velocity progress ────────────────────────────────────────────────────

def vel_tracking_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "motion_command",
    stance_threshold: float = 0.1,
) -> torch.Tensor:
    """Linear-decay reward: 1 - ||vel_error|| / ||cmd_vel||, clamped to [0, 1].
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
    moving = cmd_speed >= stance_threshold

    vel_b = asset.data.root_lin_vel_b
    ang_b = asset.data.root_ang_vel_b
    err_xy = (cmd[:, :2] - vel_b[:, :2]).norm(dim=1)
    err_z = (cmd[:, 2] - ang_b[:, 2]).abs()
    progress = (1.0 - (err_xy + err_z) / cmd_speed.clamp(min=1e-3)).clamp(min=0.0)
    return torch.where(moving, progress, torch.ones_like(progress))


# ── Regularisation ────────────────────────────────────────────────────────

def action_rate(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward smooth actions — penalises large consecutive action differences."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return _qabs(delta, std)


def joint_vel_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward low joint velocities normalised by soft limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    lim = asset.data.joint_vel_limits[:, asset_cfg.joint_ids].clamp(min=1e-6)
    return _qabs(vel / lim, std)


def joint_torque_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward low joint torques normalised by effort limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    lim = asset.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp(min=1e-6)
    return _qabs(tau / lim, std)


def joint_pos_limits_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward staying away from position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lims = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    rng = (lims[..., 1] - lims[..., 0]).clamp(min=1e-6)
    lo = (lims[..., 0] - pos).clamp(min=0.0)
    hi = (pos - lims[..., 1]).clamp(min=0.0)
    violation = (lo + hi) / rng
    return _qabs(violation, std)


def angular_momentum_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward low whole-body angular momentum."""
    asset: Articulation = env.scene[asset_cfg.name]
    mass = asset.data.default_mass[:, :asset.num_bodies].unsqueeze(-1).to(env.device)
    ang_vel = asset.data.body_ang_vel_w[:, :asset.num_bodies]
    L = (mass * ang_vel).sum(dim=1)
    return _qabs(L, std)

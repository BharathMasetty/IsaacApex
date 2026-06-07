"""Custom observation functions for the G1 locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Sin/cos encoding of the left and right gait phases.

    Returns [sin_L, sin_R, cos_L, cos_R], shape (N, 4).
    """
    term = env.command_manager.get_term(command_name)
    phase = term.get_gait_phase()  # (N, 2) in [0, 2π]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


def foot_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame z-position of the specified foot bodies, shape (N, num_bodies)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]


def contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Binary contact indicator (0/1) for each foot, shape (N, num_bodies)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    assert sensor.data.net_forces_w is not None
    return (sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0).float()


def contact_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current air time for each foot, shape (N, num_bodies)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    assert sensor.data.current_air_time is not None
    return sensor.data.current_air_time[:, sensor_cfg.body_ids]


def log_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Log-magnitude-compressed contact forces, shape (N, num_bodies * 3)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    assert sensor.data.net_forces_w is not None
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, B, 3)
    flat = forces.flatten(start_dim=1)
    return flat.sign() * flat.abs().log1p()

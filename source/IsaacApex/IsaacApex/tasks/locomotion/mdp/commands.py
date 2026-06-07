"""Gait-aware locomotion command generator.

Produces velocity + height + gait-frequency commands and maintains per-foot
gait phase used by gait-shaping rewards and observations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GaitCommand(CommandTerm):
    """Samples locomotion commands (vx, vy, ω, height, gait_freq, step_height) and
    advances per-foot gait phases for reward shaping.

    Environment subpopulations (mutually exclusive per episode):
      - standing:      zero velocity, squat/stand height
      - turn-in-place: zero linear vel, nonzero ω
      - heading:       ω computed from heading error (proportional)
      - walking:       full (vx, vy, ω) sampled uniformly
    """

    cfg: GaitCommandCfg

    def __init__(self, cfg: GaitCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if cfg.use_heading and "heading" not in cfg.ranges:
            raise ValueError("use_heading=True but 'heading' not in cfg.ranges.")

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Command vector: [vx, vy, ω, height, gait_freq, step_height]
        self.cmd = torch.zeros(self.num_envs, 6, device=self.device)

        # Environment type flags
        self.is_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_turning = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_heading = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)

        # Gait phase: [left_phase, right_phase] in [0, 2π]; right starts at π (out of phase)
        self.gait_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self.gait_phase[:, 1] = math.pi

        # Tracking metrics
        self.metrics["err_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["err_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["err_omega"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["err_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.cmd

    def get_gait_phase(self) -> torch.Tensor:
        """Current gait phases [left, right] in [0, 2π]."""
        return self.gait_phase

    # ── Internal updates ─────────────────────────────────────────────────

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)

        # Sample vel_x with a minimum magnitude to avoid near-zero walking commands.
        # Uniformly pick a magnitude in [min_walk_speed, max] then randomly negate.
        vx_lo, vx_hi = self.cfg.ranges["vel_x"]
        min_walk = self.cfg.min_walk_speed
        vx_mag = r.uniform_(min_walk, max(abs(vx_lo), abs(vx_hi)))
        vx_sign = torch.where(r.uniform_(0.0, 1.0) < 0.5, torch.ones_like(vx_mag), -torch.ones_like(vx_mag))
        self.cmd[env_ids, 0] = vx_mag * vx_sign

        self.cmd[env_ids, 1] = r.uniform_(*self.cfg.ranges["vel_y"])
        self.cmd[env_ids, 2] = r.uniform_(*self.cfg.ranges["omega"])
        self.cmd[env_ids, 3] = r.uniform_(*self.cfg.ranges["height"])
        self.cmd[env_ids, 4] = r.uniform_(*self.cfg.ranges["gait_freq"])
        self.cmd[env_ids, 5] = r.uniform_(*self.cfg.ranges["step_height"])

        # Standing environments (highest priority)
        self.is_standing[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.standing_prob

        # Turn-in-place (from remaining)
        not_standing = ~self.is_standing[env_ids]
        turn_prob = self.cfg.turning_prob / (1.0 - self.cfg.standing_prob + 1e-8)
        self.is_turning[env_ids] = not_standing & (r.uniform_(0.0, 1.0) < turn_prob)

        # Heading-based envs
        if self.cfg.use_heading:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges["heading"])
            self.is_heading[env_ids] = (
                ~self.is_standing[env_ids] & (r.uniform_(0.0, 1.0) < self.cfg.heading_prob)
            )

        # Reset gait phases on episode start
        self.gait_phase[env_ids, 0] = 0.0
        self.gait_phase[env_ids, 1] = math.pi

    def _update_command(self):
        # Heading environments: compute ω from heading error
        if self.cfg.use_heading:
            heading_ids = self.is_heading.nonzero(as_tuple=False).flatten()
            if len(heading_ids):
                err = math_utils.wrap_to_pi(
                    self.heading_target[heading_ids] - self.robot.data.heading_w[heading_ids]
                )
                self.cmd[heading_ids, 2] = torch.clamp(
                    self.cfg.heading_gain * err,
                    self.cfg.ranges["omega"][0],
                    self.cfg.ranges["omega"][1],
                )

        # Zero lateral/forward velocity for standing envs
        stand_ids = self.is_standing.nonzero(as_tuple=False).flatten()
        self.cmd[stand_ids, :3] = 0.0

        # Zero linear velocity for turn-in-place envs (keep ω)
        turn_ids = self.is_turning.nonzero(as_tuple=False).flatten()
        self.cmd[turn_ids, :2] = 0.0

        # Advance gait phase only when robot has a nonzero motion command
        dt = self._env.step_dt
        speed = self.cmd[:, :2].norm(dim=1) + self.cmd[:, 2].abs()
        moving = speed >= 0.1
        delta = 2.0 * math.pi * self.cmd[:, 4] * dt
        self.gait_phase[:, 0] = torch.where(
            moving, (self.gait_phase[:, 0] + delta) % (2.0 * math.pi), torch.zeros_like(delta)
        )
        self.gait_phase[:, 1] = torch.where(
            moving,
            (self.gait_phase[:, 1] + delta) % (2.0 * math.pi),
            torch.full_like(delta, math.pi),
        )

    def _update_metrics(self):
        dt = self._env.step_dt
        vel_b = self.robot.data.root_lin_vel_b
        ang_b = self.robot.data.root_ang_vel_b
        self.metrics["err_vel_x"] += (self.cmd[:, 0] - vel_b[:, 0]).abs() * dt
        self.metrics["err_vel_y"] += (self.cmd[:, 1] - vel_b[:, 1]).abs() * dt
        self.metrics["err_omega"] += (self.cmd[:, 2] - ang_b[:, 2]).abs() * dt
        self.metrics["err_height"] += (self.cmd[:, 3] - self.robot.data.root_pos_w[:, 2]).abs() * dt

    # ── Debug visualization ───────────────────────────────────────────────

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_vis"):
                self._goal_vis = VisualizationMarkers(self.cfg.goal_vel_vis_cfg)
                self._curr_vis = VisualizationMarkers(self.cfg.curr_vel_vis_cfg)
                self._height_vis = VisualizationMarkers(self.cfg.height_vis_cfg)
            self._goal_vis.set_visibility(True)
            self._curr_vis.set_visibility(True)
            self._height_vis.set_visibility(True)
        elif hasattr(self, "_goal_vis"):
            self._goal_vis.set_visibility(False)
            self._curr_vis.set_visibility(False)
            self._height_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # Place both arrows at the commanded height so goal vs. current is easy to compare
        pos = self.robot.data.root_pos_w.clone()
        pos[:, 2] = self.cmd[:, 3]  # commanded height

        goal_scale, goal_quat = self._vel_to_arrow(self.cmd[:, :2], self._goal_vis)
        curr_scale, curr_quat = self._vel_to_arrow(self.robot.data.root_lin_vel_b[:, :2], self._curr_vis)
        self._goal_vis.visualize(pos, goal_quat, goal_scale)
        self._curr_vis.visualize(pos, curr_quat, curr_scale)
        # Height cube at commanded height for every environment
        self._height_vis.visualize(pos)

    def _vel_to_arrow(self, xy: torch.Tensor, marker: VisualizationMarkers):
        default = marker.cfg.markers["arrow"].scale
        scale = torch.tensor(default, device=self.device).expand(xy.shape[0], -1).clone()
        scale[:, 0] *= xy.norm(dim=1) * 3.0
        angle = torch.atan2(xy[:, 1], xy[:, 0])
        zeros = torch.zeros_like(angle)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, angle)
        quat = math_utils.quat_mul(self.robot.data.root_quat_w, quat)
        return scale, quat


@configclass
class GaitCommandCfg(CommandTermCfg):
    """Configuration for the gait locomotion command generator."""

    class_type: type = GaitCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    ranges: dict[str, tuple[float, float]] = {
        "vel_x":      (-1.0, 1.0),
        "vel_y":      (-0.5, 0.5),
        "omega":      (-1.0, 1.0),
        "height":     (0.60, 0.78),
        "gait_freq":  (1.0, 2.0),
        "step_height":(0.05, 0.12),
        "heading":    (-math.pi, math.pi),
    }
    """Uniform sampling ranges for each command dimension."""

    standing_prob: float = 0.20
    """Fraction of environments that receive zero-velocity commands."""

    min_walk_speed: float = 0.1
    """Minimum |vel_x| magnitude for non-standing walking commands.
    Prevents near-zero commands that let standing be optimal."""

    turning_prob: float = 0.10
    """Fraction (of non-standing envs) that receive zero-linear-velocity commands."""

    heading_prob: float = 0.30
    """Fraction (of non-standing envs) that use heading-based ω tracking."""

    use_heading: bool = True
    """Compute ω from heading error rather than sampling it directly."""

    heading_gain: float = 1.0
    """Proportional gain: ω = heading_gain * heading_error."""

    resampling_time_range: tuple[float, float] = (8.0, 10.0)

    # Visualization configs
    goal_vel_vis_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/GaitCmd/goal_vel")
    curr_vel_vis_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/GaitCmd/curr_vel")
    height_vis_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/GaitCmd/height")

    goal_vel_vis_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    curr_vel_vis_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    height_vis_cfg.markers["cuboid"].scale = (0.15, 0.15, 0.15)
    height_vis_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)

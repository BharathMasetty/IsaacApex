# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Custom action terms for the G1 locomotion task."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HoldDefaultJointsAction(ActionTerm):
    """Sends the default joint positions to a specified subset of joints every step.

    This is used to keep non-action joints (waist, arms) at their default
    positions without coupling them to the policy output.
    """

    cfg: HoldDefaultJointsActionCfg

    def __init__(self, cfg: HoldDefaultJointsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty(self.num_envs, 0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.empty(self.num_envs, 0, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        defaults = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(defaults, joint_ids=self._joint_ids)


@configclass
class HoldDefaultJointsActionCfg(ActionTermCfg):
    """Configuration for HoldDefaultJointsAction."""

    class_type: type = HoldDefaultJointsAction

    asset_name: str = "robot"
    """Name of the robot asset in the scene. Defaults to 'robot'."""

    joint_names: list[str] = MISSING
    """Joint names (or regex patterns) to hold at their default positions."""

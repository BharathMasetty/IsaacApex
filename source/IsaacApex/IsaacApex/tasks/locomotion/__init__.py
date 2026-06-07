# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""G1 locomotion task — velocity + height tracking with gait clock."""

import gymnasium as gym

from .env_cfg import G1LocomotionEnvCfg

gym.register(
    id="IsaacApex-G1-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1LocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": "IsaacApex.tasks.locomotion.agents.rsl_rl_cfg:G1LocomotionPPOCfg",
    },
)

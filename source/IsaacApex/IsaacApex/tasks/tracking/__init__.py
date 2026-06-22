# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""G1 whole-body motion-tracking task (BeyondMimic-style) on a LAFAN1 dance."""

import gymnasium as gym

from .env_cfg import G1TrackingEnvCfg

gym.register(
    id="IsaacApex-G1-Tracking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1TrackingEnvCfg,
        "rsl_rl_cfg_entry_point": "IsaacApex.tasks.tracking.agents.rsl_rl_cfg:G1TrackingPPOCfg",
        "fpo_cfg_entry_point": "IsaacApex.tasks.tracking.agents.fpo_cfg:G1TrackingFPOCfg",
    },
)

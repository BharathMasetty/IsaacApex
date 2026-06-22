# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""MDP components for the G1 motion-tracking task.

Re-exports Isaac Lab's stock ``mdp`` plus the tracking-specific terms, so the env
config can refer to either as ``mdp.<name>``.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .commands import MotionCommand, MotionCommandCfg, MotionLoader  # noqa: F401
from .events import randomize_joint_default_pos, randomize_rigid_body_com  # noqa: F401
from .observations import (  # noqa: F401
    motion_anchor_ori_b,
    motion_anchor_pos_b,
    robot_body_ori_b,
    robot_body_pos_b,
)
from .rewards import (  # noqa: F401
    feet_contact_time,
    motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
)
from .terminations import (  # noqa: F401
    bad_anchor_ori,
    bad_anchor_pos,
    bad_anchor_pos_z_only,
    bad_motion_body_pos,
    bad_motion_body_pos_z_only,
)

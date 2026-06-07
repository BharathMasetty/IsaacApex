# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""MDP components for the G1 locomotion task."""

from .actions import HoldDefaultJointsAction, HoldDefaultJointsActionCfg
from .commands import GaitCommand, GaitCommandCfg
from .curriculum import RegWeightCurriculum, VelRangeCurriculum
from .observations import contact_air_time, contact_state, foot_height, gait_phase_obs, log_contact_forces
from .rewards import (
    action_rate,
    angular_momentum_penalty,
    body_upright,
    foot_clearance,
    foot_flat_at_contact,
    foot_impact_vel,
    foot_slip,
    gait_phase_sync,
    joint_pos_limits_penalty,
    joint_torque_penalty,
    joint_vel_penalty,
    default_pose,
    TrackAngVelZ,
    track_height,
    track_lin_vel_xy,
    vel_tracking_progress,
)
from .terminations import fell_over, illegal_contact

# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Robot asset configurations for IsaacApex."""

from .unitree_g1 import (
    # 37-DoF Nucleus USD — locomotion
    G1_CFG,
    JOINTS,
    BODIES,
    ACTION_JOINTS,
    NON_ACTION_JOINTS,
    ACTION_SCALE,
    # 29-DoF USD — motion tracking
    G1_29DOF_USD_PATH,
    G1_29DOF_CFG,
    G1_29DOF_JOINTS,
    G1_29DOF_ACTION_SCALE,
    G1_HAND_JOINT_REGEX,
    G1_ANCHOR_BODY_NAME,
    G1_TRACKED_BODY_NAMES,
    G1_END_EFFECTOR_BODY_NAMES,
)

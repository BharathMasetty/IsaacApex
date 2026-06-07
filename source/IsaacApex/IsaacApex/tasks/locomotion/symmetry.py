# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Sagittal-plane reflection for the G1 locomotion policy.

Appending mirrored samples to every training mini-batch improves sample
efficiency and encourages gait symmetry without requiring a separate reward.

The bilateral joint table drives all index and sign mappings, so adding a
new joint pair or changing the action set only requires updating that table.

RSL-RL integration
------------------
Pass ``g1_locomotion_augmentation`` as the ``data_augmentation_func`` in
``RslRlSymmetryCfg``.  The function signature matches what RSL-RL's
``Symmetry`` extension expects::

    (env, obs: TensorDict | None, actions: Tensor | None)
        -> (TensorDict | None, Tensor | None)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from IsaacApex.assets.unitree_g1 import ACTION_JOINTS, JOINTS


# ---------------------------------------------------------------------------
# Bilateral joint table
# Each entry: (left_joint, right_joint, reflection_sign)
# +1 → value keeps its sign after L↔R swap (pitch/symmetric dofs)
# -1 → value reverses sign (roll/yaw/lateral dofs)
# ---------------------------------------------------------------------------

_BILATERAL_PAIRS: list[tuple[str, str, int]] = [
    # Hips
    ("left_hip_pitch_joint",        "right_hip_pitch_joint",        +1),
    ("left_hip_roll_joint",         "right_hip_roll_joint",         -1),
    ("left_hip_yaw_joint",          "right_hip_yaw_joint",          -1),
    # Knee
    ("left_knee_joint",             "right_knee_joint",             +1),
    # Ankles
    ("left_ankle_pitch_joint",      "right_ankle_pitch_joint",      +1),
    ("left_ankle_roll_joint",       "right_ankle_roll_joint",       -1),
    # Shoulders
    ("left_shoulder_pitch_joint",   "right_shoulder_pitch_joint",   +1),
    ("left_shoulder_roll_joint",    "right_shoulder_roll_joint",    -1),
    ("left_shoulder_yaw_joint",     "right_shoulder_yaw_joint",     -1),
    # Elbows
    ("left_elbow_pitch_joint",      "right_elbow_pitch_joint",      +1),
    ("left_elbow_roll_joint",       "right_elbow_roll_joint",       -1),
]

# Midline joints — no left/right partner, but some change sign under reflection.
# torso_joint is assumed to be a pitch (sagittal-plane) dof → sign +1.
_MIDLINE_SIGNS: dict[str, int] = {
    "torso_joint": +1,
}


# ---------------------------------------------------------------------------
# Build index/sign arrays from the table (done once at import)
# ---------------------------------------------------------------------------

def _derive_mapping(joint_list: list[str]) -> tuple[list[int], list[float]]:
    """Return (swap_indices, sign_mask) for a joint vector ordered by joint_list."""
    n = len(joint_list)
    position = {j: i for i, j in enumerate(joint_list)}
    swap_to = list(range(n))
    sign_mask = [1.0] * n

    for name, s in _MIDLINE_SIGNS.items():
        if name in position:
            sign_mask[position[name]] = float(s)

    for left, right, s in _BILATERAL_PAIRS:
        if left in position and right in position:
            li, ri = position[left], position[right]
            swap_to[li], swap_to[ri] = ri, li
            sign_mask[li] = float(s)
            sign_mask[ri] = float(s)

    return swap_to, sign_mask


_ALL_SWAP, _ALL_SIGN = _derive_mapping(JOINTS)
_ACT_SWAP, _ACT_SIGN = _derive_mapping(ACTION_JOINTS)

# Per-device tensor cache (populated lazily on first use)
_cache: dict[str, dict[torch.device, torch.Tensor]] = {
    k: {} for k in ("all_swap", "all_sign", "act_swap", "act_sign")
}


def _load_cache(device: torch.device) -> None:
    if device not in _cache["all_swap"]:
        _cache["all_swap"][device] = torch.tensor(_ALL_SWAP, device=device)
        _cache["all_sign"][device] = torch.tensor(_ALL_SIGN, dtype=torch.float32, device=device)
        _cache["act_swap"][device] = torch.tensor(_ACT_SWAP, device=device)
        _cache["act_sign"][device] = torch.tensor(_ACT_SIGN, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Per-vector reflection utilities (exported for standalone use)
# ---------------------------------------------------------------------------

def reflect_joint_state(x: torch.Tensor) -> torch.Tensor:
    """Reflect a batch of 23-joint state vectors (positions or velocities).

    Shape: (N, 23) -> (N, 23)
    """
    _load_cache(x.device)
    return x[:, _cache["all_swap"][x.device]] * _cache["all_sign"][x.device]


def reflect_leg_commands(a: torch.Tensor) -> torch.Tensor:
    """Reflect a batch of 12-DoF leg action vectors.

    Shape: (N, 12) -> (N, 12)
    """
    _load_cache(a.device)
    return a[:, _cache["act_swap"][a.device]] * _cache["act_sign"][a.device]


# ---------------------------------------------------------------------------
# Observation layout (77-dim policy obs, no group-level history)
#   base_lin_vel       [0 : 3]
#   base_ang_vel       [3 : 6]
#   projected_gravity  [6 : 9]
#   velocity_command   [9 :15]
#   joint_pos          [15:38]   (23 joints)
#   joint_vel          [38:61]   (23 joints)
#   last_action        [61:73]   (12 action joints)
#   gait_phase         [73:77]
# ---------------------------------------------------------------------------

_OBS_LIN_VEL  = slice(0,  3)
_OBS_ANG_VEL  = slice(3,  6)
_OBS_GRAVITY  = slice(6,  9)
_OBS_CMD      = slice(9,  15)
_OBS_JPOS     = slice(15, 38)
_OBS_JVEL     = slice(38, 61)
_OBS_LAST_ACT = slice(61, 73)
_OBS_PHASE    = slice(73, 77)


def _reflect_single_frame(obs: torch.Tensor) -> torch.Tensor:
    """Apply the sagittal reflection to one un-historied 77-dim obs frame."""
    out = obs.clone()

    # Linear velocity: lateral component flips
    out[:, 1] = -obs[:, 1]

    # Angular velocity: roll (x) and yaw (z) flip
    out[:, 3] = -obs[:, 3]
    out[:, 5] = -obs[:, 5]

    # Projected gravity: lateral component flips
    out[:, 7] = -obs[:, 7]

    # Command: lateral vel (idx 10) and yaw rate (idx 11) flip
    out[:, 10] = -obs[:, 10]
    out[:, 11] = -obs[:, 11]

    # Joint state — swap and sign via bilateral table
    out[:, _OBS_JPOS] = reflect_joint_state(obs[:, _OBS_JPOS])
    out[:, _OBS_JVEL] = reflect_joint_state(obs[:, _OBS_JVEL])

    # Last leg action
    out[:, _OBS_LAST_ACT] = reflect_leg_commands(obs[:, _OBS_LAST_ACT])

    # Gait clocks: shifting each foot's phase by π negates all four sinusoids
    out[:, _OBS_PHASE] = -obs[:, _OBS_PHASE]

    return out


# ---------------------------------------------------------------------------
# RSL-RL entry point
# ---------------------------------------------------------------------------

def g1_locomotion_augmentation(
    env,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Sagittal-plane reflection for RSL-RL's Symmetry extension.

    Doubles each mini-batch by appending the reflected counterpart of every
    sample.  The ``"policy"`` group is reflected using the joint bilateral
    table; the ``"critic"`` group is copied unchanged.

    Args:
        env:     RSL-RL VecEnv wrapper (not used directly; present for API compat).
        obs:     TensorDict produced by the env, or None when only actions are needed.
        actions: Leg action tensor of shape (N, 12), or None.

    Returns:
        (aug_obs, aug_actions) each with 2x the original batch dimension.
    """
    aug_obs = None
    aug_actions = None

    if obs is not None:
        policy_obs = obs["policy"]                                       # (N, 77)
        reflected  = _reflect_single_frame(policy_obs)                   # (N, 77)
        doubled_policy = torch.cat([policy_obs, reflected], dim=0)       # (2N, 77)

        # Build a new TensorDict with batch_size=[2N]; cloning the original would
        # retain batch_size=[N] and reject the doubled tensor.
        new_dict = {}
        for key in obs.keys():
            if key == "policy":
                new_dict[key] = doubled_policy
            else:
                new_dict[key] = torch.cat([obs[key], obs[key]], dim=0)
        aug_obs = TensorDict(new_dict, batch_size=[doubled_policy.shape[0]], device=policy_obs.device)

    if actions is not None:
        aug_actions = torch.cat([actions, reflect_leg_commands(actions)], dim=0)

    return aug_obs, aug_actions

# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Dynamic velocity curriculum for the G1 locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VelRangeCurriculum(ManagerTermBase):
    """Expands / contracts command velocity ranges based on rolling tracking quality.

    Performance is measured at each episode reset using the time-integrated velocity
    errors accumulated in the GaitCommand metrics.  For each resetting environment with
    a non-zero movement command the tracking quality is:

        quality = 1  -  (err_x + err_y + err_omega)  /  (cmd_speed * episode_s)

    where the denominator is the worst-case error (robot stationary for the full
    episode).  An exponential moving average (EMA) smooths the per-reset estimates
    into a stable performance signal, which is compared against advance / regress
    thresholds to step through the provided ``stages``.

    Args:
        command_name: Name of the GaitCommand term in the command manager.
        stages: List of dicts (ascending difficulty), each mapping command-range
            keys to ``(lo, hi)`` tuples.  Stage 0 must match the initial ranges set
            in CommandsCfg so the curriculum starts correctly.
        advance_thresh: EMA quality above which the curriculum advances one stage.
        regress_thresh: EMA quality below which the curriculum regresses one stage.
        ema_alpha: Smoothing factor for the performance EMA (lower = slower).
        min_moving_frac: Minimum fraction of resetting envs that must have a nonzero
            movement command for the update to count.  Avoids noisy updates when most
            envs are standing.
        log_interval: Print a status line every this many env steps (0 = stage changes only).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._perf_ema: float = 0.0  # pessimistic start — forces early progression
        self._stage: int = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        stages: list[dict],
        advance_thresh: float = 0.65,
        regress_thresh: float = 0.25,
        ema_alpha: float = 0.05,
        min_moving_frac: float = 0.2,
    ) -> float:
        cmd_term = env.command_manager.get_term(command_name)
        cmd = cmd_term.command[env_ids]  # (B, 6)
        cmd_speed = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
        moving = cmd_speed > 0.1

        # Skip update if too few moving envs (avoids noise from standing-only resets)
        if moving.float().mean().item() < min_moving_frac:
            return self._perf_ema

        # Episode length in seconds — read BEFORE _reset_idx zeroes episode_length_buf
        ep_len_s = env.episode_length_buf[env_ids].float() * env.step_dt  # (B,)

        # Time-integrated absolute errors accumulated by GaitCommand over the episode
        err_x = cmd_term.metrics["err_vel_x"][env_ids]   # (B,)
        err_y = cmd_term.metrics["err_vel_y"][env_ids]
        err_w = cmd_term.metrics["err_omega"][env_ids]
        total_err = (err_x + err_y + err_w).clamp(min=0.0)

        # Quality: 1 - actual_err / worst_case_err; 0 = stationary, 1 = perfect track
        worst_err = cmd_speed * ep_len_s.clamp(min=1e-3)
        quality = (1.0 - total_err / worst_err.clamp(min=1e-3)).clamp(0.0, 1.0)

        mean_q = quality[moving].mean().item()
        self._perf_ema = (1.0 - ema_alpha) * self._perf_ema + ema_alpha * mean_q

        # Advance or regress stage
        new_stage = self._stage
        if self._perf_ema >= advance_thresh and self._stage < len(stages) - 1:
            new_stage = self._stage + 1
        elif self._perf_ema < regress_thresh and self._stage > 0:
            new_stage = self._stage - 1

        if new_stage != self._stage:
            self._stage = new_stage
            self._apply_stage(env, command_name, stages[new_stage])

        log = env.extras.setdefault("log", {})
        log["curriculum/vel/ema"] = self._perf_ema
        log["curriculum/vel/stage"] = float(self._stage)

        return self._perf_ema

    def _apply_stage(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        stage: dict,
    ) -> None:
        """Hot-patch the command config so subsequent resamples use the new ranges."""
        cmd_cfg = getattr(env.command_manager.cfg, command_name)
        updated = dict(cmd_cfg.ranges)
        updated.update(stage)
        cmd_cfg.ranges = updated


class RegWeightCurriculum(ManagerTermBase):
    """Linearly anneals regularization reward weights from high start values to final values.

    On the first call, reads the final weight for each named term directly from the
    reward manager config (so RewardsCfg is always the source of truth for converged
    behavior) and overrides it with the provided start weight.  Subsequent calls
    interpolate linearly toward the final value over ``warmup_steps`` env steps.

    Start weights should be higher than final weights so early training is conservative
    (smooth actions, near-default pose), then the constraint relaxes as the policy matures.

    Args:
        terms: Mapping of reward term name → start weight (high regularization).
            The final weight is read from RewardsCfg at initialization time.
        warmup_steps: Number of ``env.step()`` calls over which to anneal.
            With RSL-RL collecting 24 steps/iter, 20 000 steps ≈ 833 PPO updates.
        log_interval: Print a weight summary every this many env steps (0 = off).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._final_weights: dict[str, float] = {}
        self._initialized = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        terms: dict[str, float],
        warmup_steps: int = 20_000,
    ) -> float:
        if not self._initialized:
            for term_name, start_weight in terms.items():
                self._final_weights[term_name] = getattr(env.reward_manager.cfg, term_name).weight
                getattr(env.reward_manager.cfg, term_name).weight = start_weight
            self._initialized = True

        step = env.common_step_counter
        alpha = min(1.0, step / max(warmup_steps, 1))

        log = env.extras.setdefault("log", {})
        log["curriculum/reg/alpha"] = alpha
        for term_name, start_weight in terms.items():
            new_weight = start_weight + alpha * (self._final_weights[term_name] - start_weight)
            getattr(env.reward_manager.cfg, term_name).weight = new_weight
            log[f"curriculum/reg/{term_name}"] = new_weight

        return alpha

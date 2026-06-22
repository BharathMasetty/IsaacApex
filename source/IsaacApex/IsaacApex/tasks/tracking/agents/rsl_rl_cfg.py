# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""RSL-RL PPO configuration for the G1 motion-tracking task.

Hyperparameters follow BeyondMimic / whole_body_tracking's tuned G1 setup:
asymmetric actor-critic (privileged ``critic`` obs group), empirical normalization,
adaptive-KL learning-rate schedule. This is the primary, validated training path.
"""

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


@configclass
class G1TrackingPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 30000
    save_interval: int = 500
    experiment_name: str = "g1_tracking"
    empirical_normalization: bool = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

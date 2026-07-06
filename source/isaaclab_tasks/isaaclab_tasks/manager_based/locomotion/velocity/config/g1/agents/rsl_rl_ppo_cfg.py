# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]


@configclass
class G1FlatVxOnlyPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_full_gait"
        self.max_iterations = 5000


@configclass
class G1FlatVxFullPeriod070PPORunnerCfg(G1FlatVxOnlyPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_full_period070"


@configclass
class G1FlatVxFullEffortPPORunnerCfg(G1FlatVxOnlyPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_full_effort"


@configclass
class G1FlatVxFullPeriodEffortPPORunnerCfg(G1FlatVxOnlyPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_full_period070_effort"


@configclass
class G1FlatVxYawSmallPeriodEffortPPORunnerCfg(G1FlatVxFullPeriodEffortPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_yaw03_period070_effort"


@configclass
class G1FlatVxYawAntiHopAPPORunnerCfg(G1FlatVxYawSmallPeriodEffortPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_yaw03_antihop_a"
        self.max_iterations = 1000


@configclass
class G1FlatVxYawAntiHopBPPORunnerCfg(G1FlatVxYawAntiHopAPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_yaw03_antihop_b"


@configclass
class G1FlatVxYawAntiHopPeriod065PPORunnerCfg(G1FlatVxYawAntiHopBPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_yaw03_antihop_period065"
        self.max_iterations = 800


@configclass
class G1FlatVxYawAntiHopPeriod075PPORunnerCfg(G1FlatVxYawAntiHopBPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_yaw03_antihop_period075"
        self.max_iterations = 800


@configclass
class G1FlatVxYawAntiHopPeriod080PPORunnerCfg(G1FlatVxYawAntiHopBPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_yaw03_antihop_period080"
        self.max_iterations = 800


@configclass
class G1FlatVxYawAntiHopWeakClockPPORunnerCfg(G1FlatVxYawAntiHopPeriod075PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_vx_yaw03_antihop_weak_clock"
        self.max_iterations = 1000


@configclass
class G1FlatOmniHumanPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_omni_human_arm_simple"
        self.max_iterations = 500
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.003


@configclass
class G1FlatOmniHumanFullPPORunnerCfg(G1FlatOmniHumanPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_omni_human_full"
        self.max_iterations = 4000


@configclass
class G1FlatOmniHumanSimToRealPPORunnerCfg(G1FlatOmniHumanPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_omni_human_sim2real_mlp"
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.max_iterations = 4000


@configclass
class G1FlatOmniHumanGRUPPORunnerCfg(G1FlatOmniHumanSimToRealPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_omni_human_sim2real_gru"
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=0.8,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            rnn_type="gru",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
        )

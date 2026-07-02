# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.35
        self.rewards.torso_height_l2.weight = -1.0
        self.rewards.joint_deviation_arms.weight = -0.18
        self.rewards.joint_vel_arms.weight = -0.012
        self.rewards.dof_pos_limits_arms.weight = -0.75
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.0e-7
        self.rewards.feet_air_time.weight = 1.25
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time_symmetry.weight = -0.75
        self.rewards.feet_contact_count.weight = -0.2
        self.rewards.feet_slide.weight = -0.15
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1FlatVxOnlyEnvCfg(G1FlatEnvCfg):
    """Flat G1 velocity tracking variant focused on forward velocity."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Rewards: prioritize forward tracking while keeping yaw stable and actions smooth.
        self.rewards.track_lin_vel_xy_exp.weight = 2.4
        # self.rewards.track_lin_vel_xy_error.weight = -0.35
        # self.rewards.track_lin_vel_xy_error.log_only = False
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        # self.rewards.track_ang_vel_z_error.weight = -0.15
        # self.rewards.track_ang_vel_z_error.log_only = False
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.joint_deviation_torso.weight = -0.3
        self.rewards.joint_deviation_arms.weight = -0.22
        self.rewards.joint_vel_torso.weight = -0.04
        self.rewards.joint_vel_hip_yaw.weight = -0.01
        self.rewards.joint_vel_arms.weight = -0.015
        self.rewards.dof_pos_limits_arms.weight = -0.9
        self.rewards.torso_height_l2.weight = -1.25
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.feet_contact_count.weight = -0.25

        # Keep the vx-first bootstrap task inside the 8 GB VRAM training budget.
        self.scene.num_envs = 512
        self.actions.joint_pos.scale = 0.35

        # Commands: first curriculum stage from PLANS.md, vx only.
        self.commands.base_velocity.resampling_time_range = (12.0, 12.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.rewards.track_lin_vel_xy_exp.weight = 2.6
        self.rewards.lin_vel_z_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.45
        self.rewards.feet_slide.weight = -0.18
        self.rewards.feet_gait_clock.params["period"] = 0.7 #短周期增强高速下步幅表现
        # Commands: expand vx after the medium-speed gait is stable.
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, -0.0)


@configclass
class G1FlatVxFullPeriod070EnvCfg(G1FlatVxOnlyEnvCfg):
    """Flat full-vx comparison that uses the shorter gait period only."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.feet_gait_clock.params["period"] = 0.7


@configclass
class G1FlatVxFullEffortEnvCfg(G1FlatVxOnlyEnvCfg):
    """Flat full-vx comparison that keeps the original gait period and adds effort smoothness terms."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Compare against the short-period result by restoring the original clock.
        self.rewards.feet_gait_clock.params["period"] = 0.8

        # Penalize torque jumps and mechanical power without changing the gait clock.
        self.rewards.dof_torque_rate_l2.weight = -2.5e-7
        self.rewards.dof_power_abs.weight = -2.0e-5


@configclass
class G1FlatVxFullPeriodEffortEnvCfg(G1FlatVxFullPeriod070EnvCfg):
    """Flat full-vx comparison that combines the shorter gait period with effort smoothness terms."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.dof_torque_rate_l2.weight = -2.5e-7
        self.rewards.dof_power_abs.weight = -2.0e-5


@configclass
class G1FlatVxYawSmallPeriodEffortEnvCfg(G1FlatVxFullPeriodEffortEnvCfg):
    """Flat full-vx gait variant with small yaw commands."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Commands: introduce yaw gradually after the full-vx gait is stable.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Give yaw tracking a little more weight, but keep forward tracking dominant.
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.joint_vel_hip_yaw.weight = -0.015
        self.rewards.feet_gait_clock.weight = 1.0


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


class G1FlatVxOnlyEnvCfg_PLAY(G1FlatVxOnlyEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # fixed forward walking command for visual gait inspection
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

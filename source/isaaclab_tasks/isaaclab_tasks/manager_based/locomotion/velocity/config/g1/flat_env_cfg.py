# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

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
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None

        # Give yaw tracking a little more weight, but keep forward tracking dominant.
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.joint_vel_hip_yaw.weight = -0.015
        self.rewards.feet_gait_clock.weight = 1.0

        #Yaw-in-Place Experiment:小线速度下施加yaw转向
        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        # self.commands.base_velocity.heading_command = False
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Give yaw tracking a little more weight, but keep forward tracking dominant.
        # self.rewards.track_ang_vel_z_exp.weight = 1.2
        # self.rewards.joint_vel_hip_yaw.weight = -0.015
        # self.rewards.feet_gait_clock.weight = 0.7
        # self.rewards.feet_slide.weight = -0.25
        # self.rewards.ang_vel_xy_l2.weight = -0.55
        # self.rewards.flat_orientation_l2.weight = -1.8


@configclass
class G1FlatVxYawAntiHopAEnvCfg(G1FlatVxYawSmallPeriodEffortEnvCfg):
    """Anti-hop experiment A: reduce air-time reward and strengthen contact-pattern penalty."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.feet_air_time.weight = 0.8
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_contact_count.weight = -0.35
        self.rewards.feet_gait_clock.params["period"] = 0.7
        self.rewards.feet_gait_clock.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.45
        self.rewards.torso_height_l2.weight = -1.25


@configclass
class G1FlatVxYawAntiHopBEnvCfg(G1FlatVxYawAntiHopAEnvCfg):
    """Anti-hop experiment B: add stronger vertical and roll/pitch stabilization."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.55
        self.rewards.torso_height_l2.weight = -1.8
        self.rewards.flat_orientation_l2.weight = -1.5


@configclass
class G1FlatVxYawAntiHopPeriod065EnvCfg(G1FlatVxYawAntiHopAEnvCfg):
    """Anti-hop experiment C: gait-clock period 0.65 s."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.feet_gait_clock.params["period"] = 0.65


@configclass
class G1FlatVxYawAntiHopPeriod075EnvCfg(G1FlatVxYawAntiHopAEnvCfg):
    """Anti-hop experiment C: gait-clock period 0.75 s."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.feet_gait_clock.params["period"] = 0.75


@configclass
class G1FlatVxYawAntiHopPeriod080EnvCfg(G1FlatVxYawAntiHopAEnvCfg):
    """Anti-hop experiment C: gait-clock period 0.80 s."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.feet_gait_clock.params["period"] = 0.8


@configclass
class G1FlatVxYawAntiHopWeakClockEnvCfg(G1FlatVxYawAntiHopAEnvCfg):
    """Anti-hop experiment D: weaken clock and air-time shaping around the 0.75 s period."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.feet_gait_clock.weight = 0.6
        self.rewards.feet_air_time.weight = 0.6
        self.rewards.feet_contact_count.weight = -0.4
        self.rewards.feet_air_time_symmetry.weight = -0.5


@configclass
class G1FlatOmniHumanEnvCfg(G1FlatEnvCfg):
    """Flat G1 omni-directional task with human-like arm and gait shaping."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Observations: expose the gait phase so the policy does not have to infer the reward clock.
        self.observations.policy.phase_clock = ObsTerm(func=mdp.gait_phase_clock, params={"period": 0.75})

        # Commands: phase-1 omni range. Expand to (-0.6, 1.2), (-0.5, 0.5), (-0.6, 0.6) after stable.
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None

        # Training scale.
        self.scene.num_envs = 4096
        self.actions.joint_pos.scale = 0.35

        # Tracking.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.2

        # Anti-hop and omni contact routing.
        self.rewards.feet_air_time.weight = 0.3
        self.rewards.feet_air_time.params["threshold"] = 0.2
        self.rewards.feet_contact_count.weight = -0.35
        self.rewards.feet_contact_count_yaw.weight = -0.25
        self.rewards.feet_gait_clock.weight = 0.0
        self.rewards.feet_gait_clock_omni.weight = 0.6
        self.rewards.feet_gait_clock_omni.params["period"] = 0.75
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_close.weight = -0.4
        self.rewards.feet_close.params["distance_threshold"] = 0.08
        self.rewards.feet_touchdown_velocity.weight = -0.14
        self.rewards.feet_swing_height_trajectory.weight = 0.45
        self.rewards.feet_swing_vertical_velocity.weight = -0.02

        # Posture.
        self.rewards.lin_vel_z_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.45
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.torso_height_l2.weight = -1.25

        # Effort and smoothness.
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_torque_rate_l2.weight = -2.5e-7
        self.rewards.dof_power_abs.weight = -2.0e-5
        self.rewards.stand_still.weight = -0.35

        # Human-like arms: shoulder pitch swings; shoulder roll/yaw and elbows stay near the default pose.
        self.rewards.joint_deviation_arms.weight = 0.0
        self.rewards.joint_vel_arms.weight = 0.0
        self.rewards.joint_deviation_arm_swing.weight = -0.003
        self.rewards.joint_vel_arm_swing.weight = 0.0
        self.rewards.joint_deviation_arm_aux.weight = -0.18
        self.rewards.joint_vel_arm_aux.weight = -0.01
        self.rewards.dof_pos_limits_arms.weight = -0.5
        self.rewards.arm_swing_coordination.weight = 0.0
        self.rewards.arm_swing_clocked_shoulder_pitch.weight = 1.2
        self.rewards.arm_swing_clocked_shoulder_pitch.params["period"] = 0.75
        self.rewards.arm_swing_clocked_shoulder_pitch.params["min_amplitude"] = 0.05
        self.rewards.arm_swing_clocked_shoulder_pitch.params["max_amplitude"] = 0.38
        self.rewards.arm_swing_clocked_shoulder_pitch.params["phase_sign"] = -1.0
        self.rewards.arm_swing_clocked_shoulder_pitch.params["std"] = 0.16
        self.rewards.arm_swing_clocked_shoulder_pitch.params["side_scale"] = 0.25
        self.rewards.arm_swing_clocked_shoulder_pitch.params["turn_scale"] = 0.15
        self.rewards.arm_swing_clocked_shoulder_pitch.params["mixed_turn_scale"] = 0.8
        self.rewards.arm_swing_clocked_shoulder_velocity.weight = 0.25
        self.rewards.arm_swing_clocked_shoulder_velocity.params["period"] = 0.75
        self.rewards.arm_swing_clocked_shoulder_velocity.params["min_amplitude"] = 0.05
        self.rewards.arm_swing_clocked_shoulder_velocity.params["max_amplitude"] = 0.38
        self.rewards.arm_swing_clocked_shoulder_velocity.params["max_velocity"] = 1.2
        self.rewards.arm_swing_clocked_shoulder_velocity.params["phase_sign"] = -1.0
        self.rewards.arm_swing_clocked_shoulder_velocity.params["std"] = 0.45
        self.rewards.arm_swing_clocked_shoulder_velocity.params["side_scale"] = 0.25
        self.rewards.arm_swing_clocked_shoulder_velocity.params["turn_scale"] = 0.15
        self.rewards.arm_swing_clocked_shoulder_velocity.params["mixed_turn_scale"] = 0.8
        self.rewards.arm_swing_opposite_leg_phase.weight = 0.0
        self.rewards.arm_swing_opposite_leg_phase.params["min_amplitude"] = 0.12
        self.rewards.arm_swing_opposite_leg_phase.params["max_amplitude"] = 0.55
        self.rewards.arm_swing_opposite_leg_phase.params["leg_phase_scale"] = 0.22
        self.rewards.arm_swing_opposite_leg_phase.params["leg_velocity_scale"] = 0.8
        self.rewards.arm_swing_opposite_leg_phase.params["leg_velocity_weight"] = 0.6
        self.rewards.arm_swing_opposite_leg_phase.params["min_phase_magnitude"] = 0.45
        self.rewards.arm_swing_opposite_leg_phase.params["phase_sign"] = 1.0
        self.rewards.arm_swing_opposite_leg_phase.params["std"] = 0.32
        self.rewards.arm_swing_opposite_foot_phase.weight = 0.0
        self.rewards.arm_swing_opposite_foot_phase.params["min_amplitude"] = 0.16
        self.rewards.arm_swing_opposite_foot_phase.params["max_amplitude"] = 0.75
        self.rewards.arm_swing_opposite_foot_phase.params["foot_phase_scale"] = 0.16
        self.rewards.arm_swing_opposite_foot_phase.params["min_phase_magnitude"] = 0.5
        self.rewards.arm_swing_opposite_foot_phase.params["phase_sign"] = 1.0
        self.rewards.arm_swing_opposite_foot_phase.params["std"] = 0.36
        self.rewards.arm_swing_amplitude_schedule.weight = 0.0
        self.rewards.arm_swing_sagittal_velocity.weight = 0.0
        self.rewards.arm_swing_pose_envelope.weight = 0.0


@configclass
class G1FlatOmniHumanEnvCfg_PLAY(G1FlatOmniHumanEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, -0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, -0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None

@configclass
class G1FlatOmniHumanFullEnvCfg(G1FlatOmniHumanEnvCfg):
    """Flat G1 omni-directional task with the full command range."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)


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


class G1FlatVxYawSmallPeriodEffortEnvCfg_PLAY(G1FlatVxYawSmallPeriodEffortEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # fixed forward walking command for visual gait inspection
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, -0.0)
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

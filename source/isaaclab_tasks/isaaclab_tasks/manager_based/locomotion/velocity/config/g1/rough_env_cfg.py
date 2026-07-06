# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip


G1_ARM_JOINTS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_roll_joint",
]

G1_ARM_SWING_JOINTS = [
    ".*_shoulder_pitch_joint",
]

G1_ARM_AUX_JOINTS = [
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_roll_joint",
]


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # track_lin_vel_xy_error = RewTerm(
    #     func=mdp.track_lin_vel_xy_yaw_frame_error,
    #     weight=0.0,
    #     log_only=True,
    #     params={"command_name": "base_velocity"},
    # )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # track_ang_vel_z_error = RewTerm(
    #     func=mdp.track_ang_vel_z_world_error,
    #     weight=0.0,
    #     log_only=True,
    #     params={"command_name": "base_velocity"},
    # )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_air_time_symmetry = RewTerm(
        func=mdp.feet_air_time_symmetry_biped,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "max_err": 0.25,
        },
    )
    feet_air_time_symmetry_error = RewTerm(
        func=mdp.feet_air_time_symmetry_biped,
        weight=0.0,
        log_only=True,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "max_err": 0.25,
        },
    )
    feet_contact_count = RewTerm(
        func=mdp.feet_contact_count_biped,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_threshold": 0.1,
        },
    )
    feet_contact_count_yaw = RewTerm(
        func=mdp.feet_contact_count_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_threshold": 0.1,
            "activation": "omni",
        },
    )
    feet_gait_clock = RewTerm(
        func=mdp.feet_gait_clock_biped,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_threshold": 0.1,
        },
    )
    feet_gait_clock_omni = RewTerm(
        func=mdp.feet_gait_clock_biped,
        weight=0.0,
        params={
            "period": 0.75,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_threshold": 0.1,
            "activation": "omni",
            "turn_scale": 0.45,
            "side_scale": 0.55,
            "mode_routing": True,
        },
    )
    feet_close = RewTerm(
        func=mdp.feet_close_biped,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "distance_threshold": 0.08,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_touchdown_velocity = RewTerm(
        func=mdp.feet_touchdown_velocity_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "command_threshold": 0.1,
            "activation": "omni",
            "max_downward_velocity": 1.5,
        },
    )
    feet_swing_height_trajectory = RewTerm(
        func=mdp.feet_swing_height_trajectory,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "period": 0.75,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "min_clearance": 0.035,
            "max_clearance": 0.09,
            "max_speed": 1.5,
            "std": 0.035,
            "command_threshold": 0.1,
            "activation": "omni",
            "side_clearance_scale": 0.75,
            "turn_clearance_scale": 0.55,
        },
    )
    feet_swing_vertical_velocity = RewTerm(
        func=mdp.feet_swing_vertical_velocity_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "period": 0.75,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_threshold": 0.1,
            "activation": "omni",
            "deadband": 0.2,
            "max_velocity": 1.2,
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    dof_pos_limits_arms = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_deadband,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS), "deadband": 0.08},
    )
    joint_deviation_arm_swing = RewTerm(
        func=mdp.joint_deviation_l1_deadband,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_SWING_JOINTS), "deadband": 0.12},
    )
    joint_deviation_arm_aux = RewTerm(
        func=mdp.joint_deviation_l1_deadband,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_AUX_JOINTS), "deadband": 0.06},
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )
    joint_vel_torso = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )
    joint_vel_hip_yaw = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_yaw_joint")},
    )
    joint_vel_arms = RewTerm(
        func=mdp.joint_vel_l2_deadband,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS), "deadband": 0.35},
    )
    joint_vel_arm_swing = RewTerm(
        func=mdp.joint_vel_l2_deadband,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_SWING_JOINTS), "deadband": 0.45},
    )
    joint_vel_arm_aux = RewTerm(
        func=mdp.joint_vel_l2_deadband,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_AUX_JOINTS), "deadband": 0.18},
    )
    stand_still = RewTerm(
        func=mdp.stand_regularization,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            "command_threshold": 0.08,
            "base_weight": 1.0,
            "joint_vel_weight": 0.05,
            "joint_dev_weight": 0.02,
        },
    )
    arm_swing_coordination = RewTerm(
        func=mdp.arm_swing_coordination,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
            "leg_cfg": SceneEntityCfg(
                "robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"], preserve_order=True
            ),
            "command_threshold": 0.15,
            "std": 0.35,
            "side_scale": 0.35,
            "turn_scale": 0.45,
        },
    )
    arm_swing_clocked_shoulder_pitch = RewTerm(
        func=mdp.arm_swing_clocked_shoulder_pitch,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
            "period": 0.75,
            "min_amplitude": 0.05,
            "max_amplitude": 0.38,
            "max_speed": 1.5,
            "phase_sign": 1.0,
            "std": 0.16,
            "command_threshold": 0.08,
            "side_scale": 0.25,
            "turn_scale": 0.15,
            "mixed_turn_scale": 0.8,
        },
    )
    arm_swing_clocked_shoulder_velocity = RewTerm(
        func=mdp.arm_swing_clocked_shoulder_velocity,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
            "period": 0.75,
            "min_amplitude": 0.05,
            "max_amplitude": 0.38,
            "max_speed": 1.5,
            "max_velocity": 1.2,
            "phase_sign": 1.0,
            "std": 0.45,
            "command_threshold": 0.08,
            "side_scale": 0.25,
            "turn_scale": 0.15,
            "mixed_turn_scale": 0.8,
        },
    )
    arm_swing_shoulder_pitch_rms = RewTerm(
        func=mdp.arm_swing_shoulder_pitch_rms,
        weight=0.0,
        log_only=True,
        params={
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
        },
    )
    arm_swing_clocked_shoulder_pitch_error = RewTerm(
        func=mdp.arm_swing_clocked_shoulder_pitch_error,
        weight=0.0,
        log_only=True,
        params={
            "command_name": "base_velocity",
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
            "period": 0.75,
            "min_amplitude": 0.05,
            "max_amplitude": 0.38,
            "max_speed": 1.5,
            "phase_sign": 1.0,
            "command_threshold": 0.08,
            "side_scale": 0.25,
            "turn_scale": 0.15,
            "mixed_turn_scale": 0.8,
        },
    )
    arm_swing_clocked_shoulder_velocity_error = RewTerm(
        func=mdp.arm_swing_clocked_shoulder_velocity_error,
        weight=0.0,
        log_only=True,
        params={
            "command_name": "base_velocity",
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
            "period": 0.75,
            "min_amplitude": 0.05,
            "max_amplitude": 0.38,
            "max_speed": 1.5,
            "max_velocity": 1.2,
            "phase_sign": 1.0,
            "command_threshold": 0.08,
            "side_scale": 0.25,
            "turn_scale": 0.15,
            "mixed_turn_scale": 0.8,
        },
    )
    arm_swing_opposite_leg_phase = RewTerm(
        func=mdp.arm_swing_opposite_leg_phase,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
            "leg_cfg": SceneEntityCfg(
                "robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"], preserve_order=True
            ),
            "command_threshold": 0.15,
            "min_amplitude": 0.08,
            "max_amplitude": 0.45,
            "max_speed": 1.5,
            "leg_phase_deadband": 0.03,
            "leg_phase_scale": 0.35,
            "leg_velocity_deadband": 0.08,
            "leg_velocity_scale": 1.2,
            "leg_velocity_weight": 0.35,
            "min_phase_magnitude": 0.25,
            "phase_sign": 1.0,
            "std": 0.20,
            "side_scale": 0.35,
            "turn_scale": 0.45,
        },
    )
    arm_swing_opposite_foot_phase = RewTerm(
        func=mdp.arm_swing_opposite_foot_phase,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
            "foot_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "command_threshold": 0.15,
            "min_amplitude": 0.12,
            "max_amplitude": 0.65,
            "max_speed": 1.5,
            "foot_phase_deadband": 0.015,
            "foot_phase_scale": 0.22,
            "min_phase_magnitude": 0.35,
            "phase_sign": 1.0,
            "std": 0.30,
            "side_scale": 0.35,
            "turn_scale": 0.45,
        },
    )
    arm_swing_amplitude_schedule = RewTerm(
        func=mdp.arm_swing_amplitude_schedule,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"], preserve_order=True
            ),
            "period": 0.75,
            "min_amplitude": 0.05,
            "max_amplitude": 0.45,
            "max_speed": 1.5,
            "std": 0.18,
            "phase_offset": 0.5,
            "command_threshold": 0.08,
        },
    )
    arm_swing_sagittal_velocity = RewTerm(
        func=mdp.arm_swing_sagittal_velocity,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "body_cfg": SceneEntityCfg(
                "robot", body_names=["left_elbow_roll_link", "right_elbow_roll_link"], preserve_order=True
            ),
            "period": 0.75,
            "min_velocity": 0.10,
            "max_velocity": 0.65,
            "max_speed": 1.5,
            "std": 0.35,
            "lateral_weight": 2.0,
            "vertical_weight": 0.25,
            "phase_offset": 0.5,
            "command_threshold": 0.12,
            "side_scale": 0.35,
            "turn_scale": 0.45,
        },
    )
    arm_swing_pose_envelope = RewTerm(
        func=mdp.arm_swing_pose_envelope,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "body_cfg": SceneEntityCfg(
                "robot", body_names=["left_elbow_roll_link", "right_elbow_roll_link"], preserve_order=True
            ),
            "torso_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "min_lateral": 0.08,
            "max_lateral": 0.34,
            "max_vertical": 0.05,
            "min_vertical": -0.55,
            "max_sagittal": 0.50,
            "std": 0.18,
            "command_threshold": 0.08,
        },
    )
    dof_torque_rate_l2 = RewTerm(
        func=mdp.joint_torque_rate_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_power_abs = RewTerm(
        func=mdp.joint_power_abs,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    torso_height_l2 = RewTerm(
        func=mdp.body_height_l2_deadband,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "target_height": 0.74, "deadband": 0.035},
    )


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        }
        self.events.base_com = None

        # Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # self.rewards.track_lin_vel_xy_error.weight = -0.25
        # self.rewards.track_lin_vel_xy_error.log_only = False
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.25
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.joint_deviation_hip.weight = -0.15
        self.rewards.joint_deviation_torso.weight = -0.2
        self.rewards.joint_deviation_arms.weight = -0.15
        self.rewards.joint_vel_arms.weight = -0.01
        self.rewards.dof_pos_limits_arms.weight = -0.5
        self.rewards.torso_height_l2.weight = 0.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

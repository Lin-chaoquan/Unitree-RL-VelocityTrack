# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent Rough/Flat velocity tasks for the repository-local G1 29-DOF model."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from isaaclab_assets import G1_29DOF_LOCOMOTION_CFG, G1_29DOF_POLICY_JOINT_NAMES


G1_29DOF_ARM_JOINTS = [
    ".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_(roll|pitch|yaw))_joint",
]
G1_29DOF_TORSO_JOINTS = ["waist_(yaw|roll|pitch)_joint"]


@configclass
class G129DOFRewardsCfg(RewardsCfg):
    """Minimal rewards defined specifically for the G1 29-DOF locomotion tasks."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
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
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_deadband,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_ARM_JOINTS), "deadband": 0.08},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_TORSO_JOINTS)},
    )
    dof_power_abs = RewTerm(
        func=mdp.joint_power_abs,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )


@configclass
class G1Rough29DOFEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough-terrain task built directly for the local G1 29-DOF model."""

    rewards: G129DOFRewardsCfg = G129DOFRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_29DOF_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Four fixed sensor frames have no URDF inertials. Replace PhysX's 1 kg fallback
        # masses with the minimum safe mass at startup.
        self.events.normalize_sensor_link_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["imu_in_pelvis", "d435_link", "imu_in_torso", "mid360_link"]
                ),
                "mass_distribution_params": (1.0e-6, 1.0e-6),
                "operation": "abs",
                "recompute_inertia": True,
                "min_mass": 1.0e-6,
            },
        )

        # Explicit policy order makes actions, observations, export, and sim-to-sim
        # independent of the joint order authored inside the USD.
        policy_joint_names = list(G1_29DOF_POLICY_JOINT_NAMES)
        self.actions.joint_pos.joint_names = policy_joint_names
        self.actions.joint_pos.preserve_order = True
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=policy_joint_names, preserve_order=True)
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=policy_joint_names, preserve_order=True)
        }

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
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
            },
        }

        self.rewards.track_lin_vel_xy_exp.weight = 1.5
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

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1Rough29DOFEnvCfg_PLAY(G1Rough29DOFEnvCfg):
    """Lightweight visual-evaluation configuration for the 29-DOF Rough task."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-0.8, 0.8)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1Flat29DOFEnvCfg(G1Rough29DOFEnvCfg):
    """Flat-terrain task derived from the independent 29-DOF Rough task."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.35
        self.rewards.joint_deviation_arms.weight = -0.18
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.0e-7
        self.rewards.feet_air_time.weight = 1.25
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_slide.weight = -0.15
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1Flat29DOFEnvCfg_PLAY(G1Flat29DOFEnvCfg):
    """Visual-evaluation configuration for the 29-DOF Flat task."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG  # isort:skip  


##
# Scene definition
##


@configclass
class CartpoleLearnSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(150.0, 150.0)),
    )

    # robot
    robot: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=200.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        joint_effort = ObsTerm(func=mdp.joint_effort)
        # joint_last_effort = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "joint_pose_target"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        joint_effort = ObsTerm(func=mdp.joint_effort)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "joint_pose_target"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.0, 0.0),
        },
    )

    reset_pendulum_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["pole_to_pendulum"]),
            "position_range": (0.75 * math.pi, 1.25 * math.pi),
            "velocity_range": (-0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: track the commanded pole and pendulum pose
    cart_pos = RewTerm(
        func=mdp.joint_pos_cart_target_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "command_name": "joint_pose_target"},
    )
    pole_pos = RewTerm(
        func=mdp.joint_pos_pole_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "command_name": "joint_pose_target"},
    )
    # pendulum_pos = RewTerm(func=mdp.joint_pos_pendulum_target_l2,
    #     weight=-5.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["pole_to_pendulum"]), "command_name": "joint_pose_target"}
    # )
    pendulum_abs_pos = RewTerm(func=mdp.double_pendulum_angle_l2,
        weight=-1.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole", "pole_to_pendulum"]), "command_name": "joint_pose_target"}
    )
    # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_target_l2,
    #     weight=-1.0e-6,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.near_target_vel_pole_l2,
    #     weight=-1.0,
    #     params={"command_name": "joint_pose_target", "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )
    # pendulum_vel = RewTerm(
    #     func=mdp.near_target_vel_pendulum_l2,
    #     weight=-1.0,
    #     params={"command_name": "joint_pose_target", "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole", "pole_to_pendulum"])},
    # )
    # cart_vel = RewTerm(
    #     func=mdp.near_target_vel_cart_l2,
    #     weight=-1.0,
    #     params={"command_name": "joint_pose_target", "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    stationary = RewTerm(
        func=mdp.near_target_stationary_l2,
        weight=-0.1,
        params={
            "command_name": "joint_pose_target",
            "std": 0.35,
            "cart_vel_scale": 1.5,
            "cart_pos_scale": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=["slider_to_cart", "cart_to_pole", "pole_to_pendulum"],
                preserve_order=True,
            )
        },
    )
    # (6) Shaping tasks: lower joint effort
    # joint_effort = RewTerm(
    #     func=mdp.joint_effort_l2,
    #     weight=-1.0e-6,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # (7) Shaping tasks: keep cart near target position


    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-5)
    near_target_action_rate = RewTerm(
        func=mdp.near_target_action_rate_l2,
        weight=-0.9,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["slider_to_cart", "cart_to_pole", "pole_to_pendulum"],
                preserve_order=True,
            ),
            "command_name": "joint_pose_target",
            "std": 0.35,
        },
    )

    near_target_action = RewTerm(
        func=mdp.near_target_action_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["slider_to_cart", "cart_to_pole", "pole_to_pendulum"],
                preserve_order=True,
            ),
            "command_name": "joint_pose_target",
            "std": 0.35,
        },
    )

    pos_keep_pole = RewTerm(
        func=mdp.pos_keep_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "command_name": "joint_pose_target",
            "std": 0.5,
            "target_indices": (1,),
        },
    )

    pos_keep_pendulum = RewTerm(
        func=mdp.absolute_angle_keep_reward,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole", "pole_to_pendulum"]),
            "command_name": "joint_pose_target",
            "std": 0.5,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-5.0, 5.0)},
    )


@configclass
class CommandsCfg:
    joint_pose_target = mdp.UniformJointPoseCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        ranges=mdp.UniformJointPoseCommandCfg.Ranges(
            pos_cart=(-0.0, 0.0),
            pos_pole=(0.0 * math.pi, 0.0 * math.pi),
            pos_pendulum=(0.0 * math.pi, 0.0 * math.pi),
        ),
    )
##
# Environment configuration
##


@configclass
class CartpoleLearnEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: CartpoleLearnSceneCfg = CartpoleLearnSceneCfg(num_envs=16, env_spacing=6.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation

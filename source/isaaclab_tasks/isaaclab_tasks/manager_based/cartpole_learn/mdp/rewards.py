# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_pole_target_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # compute the reward
    target = env.command_manager.get_command(command_name)[:, 1:2]
    error = wrap_to_pi(joint_pos - target)
    return torch.sum(torch.square(error), dim=1)


def double_pendulum_angle_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    pendulum_angle = torch.sum(q, dim=1, keepdim=True)

    target = env.command_manager.get_command(command_name)
    target_angle = target[:, 1:2] + target[:, 2:3]
    pendulum_err = wrap_to_pi(pendulum_angle - target_angle)

    return torch.sum(torch.square(pendulum_err), dim=1)


def joint_pos_pendulum_target_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize pendulum joint position deviation from the commanded relative joint target."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target = env.command_manager.get_command(command_name)[:, 2:3]
    error = wrap_to_pi(joint_pos - target)
    return torch.sum(torch.square(error), dim=1)


def joint_vel_target_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocity deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_effort_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint effort."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_cart_target_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize root position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    target = env.command_manager.get_command(command_name)[:, 0:1]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target), dim=1)


def pos_keep_reward(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_indices: tuple[int, ...],
) -> torch.Tensor:
    """Reward keeping the selected joint positions close to the commanded pole and pendulum targets."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    target_pos = env.command_manager.get_command(command_name)[:, list(target_indices)]
    error = torch.sum(torch.square(wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids] - target_pos)), dim=1)
    return torch.exp(-error / std**2)


def absolute_angle_keep_reward(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward keeping the summed joint angle close to the commanded absolute angle."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    angle = torch.sum(joint_pos, dim=1, keepdim=True)

    target = env.command_manager.get_command(command_name)
    target_angle = target[:, 1:2] + target[:, 2:3]
    error = wrap_to_pi(angle - target_angle)
    return torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)


def near_target_vel_pole_l2(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize pole joint velocity deviation closely near the commanded target."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, 1:2]
    near_target = torch.exp(-torch.sum(torch.square(wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids] - target_pos)), dim=1) / std**2)
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]) * near_target.unsqueeze(1), dim=1)


def near_target_vel_pendulum_l2(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize pendulum joint velocity deviation closely near the commanded target."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_pos = env.command_manager.get_command(command_name)[:, 1:3]
    error = joint_pos[:, 0:1] + joint_pos[:, 1:2] - target_pos[:, 0:1] - target_pos[:, 1:2]
    near_target = torch.exp(-torch.sum(torch.square(wrap_to_pi(error)), dim=1) / std**2)
    absolute_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids[0]] + asset.data.joint_vel[
        :, asset_cfg.joint_ids[1]
    ]
    return near_target * torch.square(absolute_velocity)


def near_target_vel_cart_l2(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize cart joint velocity deviation closely near the commanded target."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, 0:1]
    near_target = torch.exp(-torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target_pos), dim=1) / std**2)
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]) * near_target.unsqueeze(1), dim=1)


def near_target_stationary_l2(
    env: ManagerBasedRLEnv,
    std: float,
    cart_pos_scale: float,
    cart_vel_scale: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward keeping the joint positions close to the commanded target."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    cart_error = joint_pos[:, 0] - target_pos[:, 0]
    pole_error = wrap_to_pi(joint_pos[:, 1] - target_pos[:, 1])
    pendulum_error = wrap_to_pi(joint_pos[:, 1] + joint_pos[:, 2] - target_pos[:, 1] - target_pos[:, 2])
    near_target = torch.exp(-(torch.square(pole_error) + torch.square(pendulum_error)) / std**2)
    pole_vel = joint_vel[:, 1]
    pendulum_vel = joint_vel[:, 1] + joint_vel[:, 2]
    cart_vel = joint_vel[:, 0]
    stationary_error = (
        torch.square(pole_vel)
        + torch.square(pendulum_vel)
        + cart_vel_scale * torch.square(cart_vel)
        + cart_pos_scale * torch.square(cart_error)
    )
    return near_target * stationary_error


def near_target_action_rate_l2(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize changes in the executed action, weighted by proximity to the upright target."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pole_error = wrap_to_pi(joint_pos[:, 1] - target_pos[:, 1])
    pendulum_error = wrap_to_pi(joint_pos[:, 1] + joint_pos[:, 2] - target_pos[:, 1] - target_pos[:, 2])
    near_target = torch.exp(-(torch.square(pole_error) + torch.square(pendulum_error)) / std**2)

    # Reduce the action dimension before applying the per-environment gate. Multiplying
    # (num_envs,) by (num_envs, action_dim) directly would broadcast across environments.
    action_rate_l2 = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )
    return near_target * action_rate_l2


def near_target_action_l2(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize action magnitude near the commanded upright target."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    pole_error = wrap_to_pi(joint_pos[:, 1] - target_pos[:, 1])
    pendulum_error = wrap_to_pi(
        joint_pos[:, 1]
        + joint_pos[:, 2]
        - target_pos[:, 1]
        - target_pos[:, 2]
    )

    near_target = torch.exp(
        -(torch.square(pole_error) + torch.square(pendulum_error)) / std**2
    )

    action_l2 = torch.sum(
        torch.square(env.action_manager.action),
        dim=1,
    )
    return near_target * action_l2
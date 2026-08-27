# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions used by the maintained locomotion velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command_speed(command: torch.Tensor) -> torch.Tensor:
    """Return a scalar locomotion demand including planar and yaw commands."""
    return torch.norm(command[:, :2], dim=1) + 0.5 * torch.abs(command[:, 2])


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward completed foot swing durations above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward alternating single-stance phases up to a maximum duration."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal foot motion while the corresponding foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def joint_power_abs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute mechanical joint power."""
    asset = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward planar velocity tracking in the gravity-aligned robot yaw frame."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward world-frame yaw-rate tracking with an exponential kernel."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def joint_deviation_l1_deadband(
    env: ManagerBasedRLEnv,
    deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joint deviation from default outside a tolerance band."""
    asset = env.scene[asset_cfg.name]
    joint_error = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.clamp(joint_error - deadband, min=0.0), dim=1)


def stand_regularization(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.08,
    base_weight: float = 1.0,
    joint_vel_weight: float = 0.05,
    joint_dev_weight: float = 0.0,
) -> torch.Tensor:
    """Penalize base and joint motion when the commanded velocity is near zero."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
    standing = _command_speed(command) < command_threshold
    base_motion = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    base_motion += torch.square(asset.data.root_ang_vel_b[:, 2])
    joint_motion = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    joint_deviation = torch.mean(
        torch.square(
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    penalty = base_weight * base_motion + joint_vel_weight * joint_motion + joint_dev_weight * joint_deviation
    return penalty * standing


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Maintain Digit compatibility by penalizing joint offsets at near-zero commands."""
    return stand_regularization(
        env,
        command_name,
        asset_cfg=asset_cfg,
        command_threshold=command_threshold,
        base_weight=0.0,
        joint_vel_weight=0.0,
        joint_dev_weight=1.0,
    )

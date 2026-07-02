# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_symmetry_biped(
    env, command_name: str, sensor_cfg: SceneEntityCfg, max_err: float = 0.25
) -> torch.Tensor:
    """Penalize left/right foot gait timing asymmetry for bipeds.

    The term compares the active mode time of the two feet: contact time for stance feet and air time for
    swing feet. During a regular alternating gait, the swing foot air time should stay close to the stance
    foot contact time, which helps avoid one foot taking consistently shorter or longer steps.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if len(sensor_cfg.body_ids) != 2:
        raise ValueError("feet_air_time_symmetry_biped expects exactly two feet in sensor_cfg.body_ids.")

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    mode_time = torch.where(in_contact, contact_time, air_time)
    asymmetry = torch.square(mode_time[:, 0] - mode_time[:, 1])
    asymmetry = torch.clamp(asymmetry, max=max_err**2)
    asymmetry *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return asymmetry


def feet_contact_count_biped(
    env, command_name: str, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.1
) -> torch.Tensor:
    """Penalize non-alternating biped contact patterns while walking.

    The term is zero when exactly one foot is in contact. It softly penalizes double support and flight
    phases under non-zero velocity commands, which discourages stop-and-go solutions without prescribing
    a fixed gait phase schedule.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if len(sensor_cfg.body_ids) != 2:
        raise ValueError("feet_contact_count_biped expects exactly two feet in sensor_cfg.body_ids.")

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact_count = torch.sum((contact_time > 0.0).int(), dim=1)
    penalty = torch.abs(contact_count - 1).float()
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def feet_gait_clock_biped(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward biped feet for matching an alternating clocked contact pattern.

    The clock defines each foot's stance phase as ``phase < threshold``. With offsets ``[0.0, 0.5]``
    and a threshold slightly above ``0.5``, the target gait becomes an alternating walk with a small
    double-support overlap.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if len(sensor_cfg.body_ids) != 2:
        raise ValueError("feet_gait_clock_biped expects exactly two feet in sensor_cfg.body_ids.")
    if len(offset) != 2:
        raise ValueError("feet_gait_clock_biped expects exactly two phase offsets.")
    if period <= 0.0:
        raise ValueError("feet_gait_clock_biped expects period to be positive.")

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    command = env.command_manager.get_command(command_name)[:, :3]
    phase_offset = torch.tensor(offset, device=env.device, dtype=torch.float32).unsqueeze(0)
    elapsed_time = env.episode_length_buf.float().unsqueeze(1) * env.step_dt
    global_phase = torch.remainder(elapsed_time, period) / period
    phase = torch.remainder(global_phase + phase_offset, 1.0)
    expected_contact = phase < threshold
    reward = torch.mean((in_contact == expected_contact).float(), dim=1)
    reward *= torch.norm(command, dim=1) > command_threshold
    return reward


def feet_close_biped(
    env,
    asset_cfg: SceneEntityCfg,
    distance_threshold: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize the two feet getting close enough to collide or trip each other."""
    asset = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet_close_biped expects exactly two feet in asset_cfg.body_ids.")

    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_distance = torch.norm(feet_pos[:, 0, :] - feet_pos[:, 1, :], dim=1)
    penalty = torch.clamp(distance_threshold - feet_distance, min=0.0) / distance_threshold
    if command_name is not None:
        penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


class joint_torque_rate_l2(ManagerTermBase):
    """Penalize step-to-step changes in applied joint torques."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset = env.scene[asset_cfg.name]
        self._asset_name = asset_cfg.name
        self._joint_ids = asset_cfg.joint_ids
        self._previous_torque = asset.data.applied_torque[:, self._joint_ids].clone()

    def reset(self, env_ids: torch.Tensor | None = None):
        asset = self._env.scene[self._asset_name]
        if env_ids is None:
            self._previous_torque[:] = asset.data.applied_torque[:, self._joint_ids]
        else:
            self._previous_torque[env_ids] = asset.data.applied_torque[env_ids][:, self._joint_ids]

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        current_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
        torque_rate = current_torque - self._previous_torque
        self._previous_torque[:] = current_torque
        return torch.sum(torch.square(torque_rate), dim=1)


def joint_power_abs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize absolute mechanical joint power."""
    asset = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


# def track_lin_vel_xy_yaw_frame_error(
#     env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Compute the tracking error of linear velocity commands (xy axes) in the gravity aligned
#     robot frame.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
#     lin_vel_error = torch.pow(torch.sum(
#         torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
#     ), 0.5)
#     return lin_vel_error


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


# def track_ang_vel_z_world_error(
#     env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Compute squared yaw-rate tracking error in world frame."""
#     asset = env.scene[asset_cfg.name]
#     return torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])


def body_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize selected body height deviation from a target world-frame z value."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height), dim=1)


def body_height_l2_deadband(
    env, target_height: float, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize selected body height only after it leaves a small acceptable z band."""
    asset = env.scene[asset_cfg.name]
    height_error = torch.abs(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    return torch.sum(torch.square(torch.clamp(height_error - deadband, min=0.0)), dim=1)


def joint_deviation_l1_deadband(
    env, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize selected joint deviation from default only outside a small tolerance band."""
    asset = env.scene[asset_cfg.name]
    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.clamp(joint_error - deadband, min=0.0), dim=1)


def joint_vel_l2_deadband(env, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize selected joint speed only when it exceeds a small tolerance."""
    asset = env.scene[asset_cfg.name]
    joint_speed = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(torch.clamp(joint_speed - deadband, min=0.0)), dim=1)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

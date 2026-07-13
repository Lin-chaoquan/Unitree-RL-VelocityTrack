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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command_speed(command: torch.Tensor) -> torch.Tensor:
    """Return a scalar locomotion demand that includes planar and yaw commands."""
    return torch.norm(command[:, :2], dim=1) + 0.5 * torch.abs(command[:, 2])


def _command_mask(command: torch.Tensor, command_threshold: float, activation: str) -> torch.Tensor:
    """Return a command activation mask for reward routing."""
    if activation == "xy":
        demand = torch.norm(command[:, :2], dim=1)
    elif activation == "omni":
        demand = _command_speed(command)
    elif activation == "yaw":
        demand = torch.abs(command[:, 2])
    else:
        raise ValueError(f"Unsupported command activation mode: {activation}.")
    return demand > command_threshold


def _vectors_in_base_yaw_frame(asset, vectors_w: torch.Tensor) -> torch.Tensor:
    """Rotate a batch of world-frame vectors into the robot yaw frame."""
    if vectors_w.ndim == 2:
        return quat_apply_inverse(yaw_quat(asset.data.root_quat_w), vectors_w)
    num_vectors = vectors_w.shape[1]
    root_yaw = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, num_vectors, -1)
    vectors_b = quat_apply_inverse(root_yaw.reshape(-1, 4), vectors_w.reshape(-1, 3))
    return vectors_b.view(vectors_w.shape)


def _biped_clock_phase(env: ManagerBasedRLEnv, period: float, offset: list[float]) -> torch.Tensor:
    """Return per-foot clock phases for two-foot gait rewards."""
    if len(offset) != 2:
        raise ValueError("Biped clock terms expect exactly two phase offsets.")
    if period <= 0.0:
        raise ValueError("Biped clock terms expect period to be positive.")
    phase_offset = torch.tensor(offset, device=env.device, dtype=torch.float32).unsqueeze(0)
    elapsed_time = env.episode_length_buf.float().unsqueeze(1) * env.step_dt
    global_phase = torch.remainder(elapsed_time, period) / period
    return torch.remainder(global_phase + phase_offset, 1.0)


def gait_phase_clock(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """Observation term for a cyclic gait phase represented as sin/cos."""
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt, period) / period
    return torch.stack((torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)), dim=1)


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
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
    activation: str = "xy",
) -> torch.Tensor:
    """Penalize non-alternating biped contact patterns while walking.

    The term is zero when exactly one foot is in contact. It softly penalizes double support and flight
    phases under active commands, which discourages stop-and-go solutions without prescribing a fixed gait
    phase schedule. The ``activation`` parameter can be ``"xy"``, ``"yaw"``, or ``"omni"``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if len(sensor_cfg.body_ids) != 2:
        raise ValueError("feet_contact_count_biped expects exactly two feet in sensor_cfg.body_ids.")

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact_count = torch.sum((contact_time > 0.0).int(), dim=1)
    penalty = torch.abs(contact_count - 1).float()
    command = env.command_manager.get_command(command_name)[:, :3]
    penalty *= _command_mask(command, command_threshold, activation)
    return penalty


def feet_contact_count_biped_omni(
    env, command_name: str, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.1
) -> torch.Tensor:
    """Compatibility wrapper for omni-directional contact-count activation."""
    return feet_contact_count_biped(env, command_name, sensor_cfg, command_threshold, activation="omni")


def feet_gait_clock_biped(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float = 0.1,
    activation: str = "omni",
    turn_scale: float = 1.0,
    side_scale: float = 1.0,
    mode_routing: bool = False,
) -> torch.Tensor:
    """Reward biped feet for matching an alternating clocked contact pattern.

    The clock defines each foot's stance phase as ``phase < threshold``. With offsets ``[0.0, 0.5]``
    and a threshold slightly above ``0.5``, the target gait becomes an alternating walk with a small
    double-support overlap. ``mode_routing`` can reduce clock pressure for side-stepping and turning.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if len(sensor_cfg.body_ids) != 2:
        raise ValueError("feet_gait_clock_biped expects exactly two feet in sensor_cfg.body_ids.")
    if len(offset) != 2:
        raise ValueError("feet_gait_clock_biped expects exactly two phase offsets.")
    if period <= 0.0:
        raise ValueError("feet_gait_clock_biped expects period to be positive.")

    command = env.command_manager.get_command(command_name)[:, :3]
    abs_vx = torch.abs(command[:, 0])
    abs_vy = torch.abs(command[:, 1])
    abs_wz = torch.abs(command[:, 2])
    planar_speed = torch.norm(command[:, :2], dim=1)

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    phase = _biped_clock_phase(env, period, offset)
    expected_contact = phase < threshold
    reward = torch.mean((in_contact == expected_contact).float(), dim=1)
    if mode_routing:
        route_scale = torch.ones_like(reward)
        turn_in_place = (planar_speed < 0.15) & (abs_wz > command_threshold)
        side_step = (abs_vy > abs_vx) & (abs_vy > command_threshold)
        route_scale = torch.where(turn_in_place, torch.full_like(route_scale, turn_scale), route_scale)
        route_scale = torch.where(side_step, torch.full_like(route_scale, side_scale), route_scale)
        reward *= route_scale
    reward *= _command_mask(command, command_threshold, activation)
    return reward


def feet_gait_clock_biped_omni(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float = 0.1,
    turn_scale: float = 0.45,
    side_scale: float = 0.55,
) -> torch.Tensor:
    """Compatibility wrapper for omni-directional mode-routed gait-clock activation."""
    return feet_gait_clock_biped(
        env,
        command_name,
        sensor_cfg,
        period,
        offset,
        threshold,
        command_threshold,
        activation="omni",
        turn_scale=turn_scale,
        side_scale=side_scale,
        mode_routing=True,
    )


def stand_regularization(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.08,
    base_weight: float = 1.0,
    joint_vel_weight: float = 0.05,
    joint_dev_weight: float = 0.0,
) -> torch.Tensor:
    """Penalize base motion and optional joint motion/deviation while standing."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
    standing = _command_speed(command) < command_threshold
    base_motion = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    base_motion += torch.square(asset.data.root_ang_vel_b[:, 2])
    joint_motion = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    joint_deviation = torch.mean(
        torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    penalty = base_weight * base_motion + joint_vel_weight * joint_motion + joint_dev_weight * joint_deviation
    return penalty * standing


def stand_still_biped(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.08,
) -> torch.Tensor:
    """Compatibility wrapper for stand regularization with base and joint-velocity penalties."""
    return stand_regularization(env, command_name, asset_cfg, command_threshold)


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


def feet_lateral_order_biped(
    env,
    asset_cfg: SceneEntityCfg,
    min_width: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    activation: str = "omni",
) -> torch.Tensor:
    """Penalize left/right foot order reversals in the robot yaw frame.

    ``asset_cfg`` must preserve the order ``[left_foot, right_foot]``. This term catches cross-over
    steps that can still have a large Euclidean foot distance and therefore pass ``feet_close_biped``.
    """
    asset = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet_lateral_order_biped expects exactly two feet in asset_cfg.body_ids.")

    foot_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    foot_rel_b = _vectors_in_base_yaw_frame(asset, foot_rel_w)
    lateral_gap = foot_rel_b[:, 0, 1] - foot_rel_b[:, 1, 1]
    penalty = torch.clamp(min_width - lateral_gap, min=0.0) / max(min_width, 1.0e-6)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)[:, :3]
        penalty *= _command_mask(command, command_threshold, activation)
    return penalty


def feet_lateral_width_biped(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    min_width: float = 0.16,
    max_width: float = 0.28,
    max_lateral_speed: float = 1.8,
    command_threshold: float = 0.1,
    activation: str = "omni",
) -> torch.Tensor:
    """Penalize a too-narrow biped stance, with a wider target for larger lateral commands."""
    asset = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet_lateral_width_biped expects exactly two feet in asset_cfg.body_ids.")

    command = env.command_manager.get_command(command_name)[:, :3]
    foot_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    foot_rel_b = _vectors_in_base_yaw_frame(asset, foot_rel_w)
    lateral_gap = foot_rel_b[:, 0, 1] - foot_rel_b[:, 1, 1]
    lateral_ratio = torch.clamp(torch.abs(command[:, 1]) / max_lateral_speed, 0.0, 1.0)
    target_width = min_width + (max_width - min_width) * lateral_ratio
    penalty = torch.clamp(target_width - lateral_gap, min=0.0) / torch.clamp(target_width, min=1.0e-6)
    penalty *= _command_mask(command, command_threshold, activation)
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


def feet_touchdown_velocity_l2(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
    activation: str = "omni",
    max_downward_velocity: float = 1.5,
) -> torch.Tensor:
    """Penalize fast downward foot velocity at first ground contact.

    This targets the "hold the foot in swing, then stamp down" failure mode without punishing normal
    swing motion away from touchdown.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    if len(sensor_cfg.body_ids) != len(asset_cfg.body_ids):
        raise ValueError("feet_touchdown_velocity_l2 expects matching sensor and asset feet.")

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    foot_vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    downward_velocity = torch.clamp(-foot_vz, min=0.0, max=max_downward_velocity)
    penalty = torch.sum(torch.square(downward_velocity) * first_contact, dim=1)
    command = env.command_manager.get_command(command_name)[:, :3]
    penalty *= _command_mask(command, command_threshold, activation)
    return penalty


def feet_swing_height_trajectory(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    threshold: float,
    min_clearance: float = 0.035,
    max_clearance: float = 0.09,
    max_speed: float = 1.5,
    std: float = 0.035,
    command_threshold: float = 0.1,
    activation: str = "omni",
    side_clearance_scale: float = 0.75,
    turn_clearance_scale: float = 0.55,
) -> torch.Tensor:
    """Reward swing feet for following a smooth phase-based clearance arc.

    The target is zero near lift-off and touchdown, with maximum clearance at mid-swing. Foot height is
    measured relative to the lower foot, which keeps the term usable on flat terrain without hard-coding
    the ankle link's absolute contact height.
    """
    asset = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet_swing_height_trajectory expects exactly two feet in asset_cfg.body_ids.")
    if threshold >= 1.0:
        raise ValueError("feet_swing_height_trajectory expects threshold to be less than 1.0.")

    command = env.command_manager.get_command(command_name)[:, :3]
    active = _command_mask(command, command_threshold, activation)
    abs_vx = torch.abs(command[:, 0])
    abs_vy = torch.abs(command[:, 1])
    abs_wz = torch.abs(command[:, 2])
    planar_speed = torch.norm(command[:, :2], dim=1)
    side_step = (abs_vy > abs_vx) & (abs_vy > command_threshold)
    turn_in_place = (planar_speed < 0.15) & (abs_wz > command_threshold)

    speed_ratio = torch.clamp(_command_speed(command) / max_speed, 0.0, 1.0)
    target_clearance = min_clearance + (max_clearance - min_clearance) * speed_ratio
    target_clearance = torch.where(side_step, target_clearance * side_clearance_scale, target_clearance)
    target_clearance = torch.where(turn_in_place, target_clearance * turn_clearance_scale, target_clearance)

    phase = _biped_clock_phase(env, period, offset)
    swing_mask = phase >= threshold
    swing_phase = torch.clamp((phase - threshold) / (1.0 - threshold), 0.0, 1.0)
    target_height = target_clearance.unsqueeze(1) * torch.sin(torch.pi * swing_phase)

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    relative_height = foot_z - torch.min(foot_z, dim=1, keepdim=True)[0]
    height_error = torch.square(relative_height - target_height) * swing_mask
    reward = torch.exp(-torch.sum(height_error, dim=1) / std**2)
    return reward * active


def feet_swing_vertical_velocity_l2(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float = 0.1,
    activation: str = "omni",
    deadband: float = 0.2,
    max_velocity: float = 1.2,
) -> torch.Tensor:
    """Penalize excessive vertical foot speed while the foot is in the clocked swing phase."""
    asset = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet_swing_vertical_velocity_l2 expects exactly two feet in asset_cfg.body_ids.")
    phase = _biped_clock_phase(env, period, offset)
    swing_mask = phase >= threshold
    foot_vz = torch.clamp(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], min=-max_velocity, max=max_velocity)
    excess_vz = torch.clamp(torch.abs(foot_vz) - deadband, min=0.0)
    penalty = torch.sum(torch.square(excess_vz) * swing_mask, dim=1)
    command = env.command_manager.get_command(command_name)[:, :3]
    penalty *= _command_mask(command, command_threshold, activation)
    return penalty


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


def centroidal_angular_momentum_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axes: str = "xyz",
    include_spin: bool = True,
    normalize_by_mass: bool = True,
) -> torch.Tensor:
    """Return squared centroidal angular momentum around the robot COM in the base-yaw frame."""
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    state_device = asset.data.body_com_pos_w.device
    masses = asset.data.default_mass[:, body_ids].to(device=state_device).unsqueeze(-1)
    num_bodies = masses.shape[1]
    total_mass = torch.clamp(torch.sum(masses, dim=1), min=1.0e-6)
    com_pos = torch.sum(asset.data.body_com_pos_w[:, body_ids, :] * masses, dim=1) / total_mass
    com_vel = torch.sum(asset.data.body_com_lin_vel_w[:, body_ids, :] * masses, dim=1) / total_mass

    rel_pos = asset.data.body_com_pos_w[:, body_ids, :] - com_pos.unsqueeze(1)
    rel_vel = asset.data.body_com_lin_vel_w[:, body_ids, :] - com_vel.unsqueeze(1)
    momentum = torch.sum(torch.cross(rel_pos, masses * rel_vel, dim=-1), dim=1)

    if include_spin:
        inertia = asset.data.default_inertia[:, body_ids, :].to(device=state_device).view(masses.shape[0], num_bodies, 3, 3)
        omega_w = asset.data.body_com_ang_vel_w[:, body_ids, :]
        quat_w = asset.data.body_com_quat_w[:, body_ids, :]
        omega_b = quat_apply_inverse(quat_w.reshape(-1, 4), omega_w.reshape(-1, 3)).view_as(omega_w)
        spin_b = torch.matmul(inertia, omega_b.unsqueeze(-1)).squeeze(-1)
        spin_w = quat_apply(quat_w.reshape(-1, 4), spin_b.reshape(-1, 3)).view_as(spin_b)
        momentum = momentum + torch.sum(spin_w, dim=1)

    momentum_b = _vectors_in_base_yaw_frame(asset, momentum)
    components = []
    if "x" in axes:
        components.append(momentum_b[:, 0])
    if "y" in axes:
        components.append(momentum_b[:, 1])
    if "z" in axes:
        components.append(momentum_b[:, 2])
    if len(components) == 0:
        raise ValueError("centroidal_angular_momentum_l2 expects axes to include at least one of 'x', 'y', or 'z'.")
    selected = torch.stack(components, dim=1)
    if normalize_by_mass:
        selected = selected / total_mass
    return torch.sum(torch.square(selected), dim=1)


def centroidal_yaw_angular_momentum_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    include_spin: bool = True,
    normalize_by_mass: bool = True,
) -> torch.Tensor:
    """Return squared yaw-axis centroidal angular momentum around the robot COM."""
    return centroidal_angular_momentum_l2(env, asset_cfg, axes="z", include_spin=include_spin, normalize_by_mass=normalize_by_mass)


def _arm_swing_clock_targets(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    period: float,
    min_amplitude: float,
    max_amplitude: float,
    max_speed: float,
    phase_sign: float,
    command_threshold: float,
    side_scale: float,
    turn_scale: float,
    mixed_turn_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clocked shoulder-pitch position/velocity targets and mode scales."""
    if period <= 0.0:
        raise ValueError("Clocked arm swing rewards expect period to be positive.")
    if len(arm_cfg.joint_ids) != 2:
        raise ValueError("Clocked arm swing rewards expect two shoulder pitch joints.")

    command = env.command_manager.get_command(command_name)[:, :3]
    speed = _command_speed(command)
    moving = speed > command_threshold
    abs_vx = torch.abs(command[:, 0])
    abs_vy = torch.abs(command[:, 1])
    abs_wz = torch.abs(command[:, 2])
    planar_speed = torch.norm(command[:, :2], dim=1)

    side_step = (abs_vy > abs_vx) & (abs_vy > command_threshold)
    turn_in_place = (planar_speed < 0.15) & (abs_wz > command_threshold)
    mixed_turn = (planar_speed >= 0.15) & (abs_wz > command_threshold)

    route_scale = torch.ones_like(speed)
    route_scale = torch.where(side_step, torch.full_like(route_scale, side_scale), route_scale)
    route_scale = torch.where(mixed_turn, torch.full_like(route_scale, mixed_turn_scale), route_scale)
    route_scale = torch.where(turn_in_place, torch.full_like(route_scale, turn_scale), route_scale)
    route_scale = torch.where(moving, route_scale, torch.zeros_like(route_scale))

    speed_ratio = torch.clamp(speed / max_speed, 0.0, 1.0)
    amplitude = min_amplitude + (max_amplitude - min_amplitude) * speed_ratio
    amplitude = torch.where(moving, amplitude, torch.zeros_like(amplitude))

    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt, period) / period
    sin_phase = torch.sin(2.0 * torch.pi * phase)
    cos_phase = torch.cos(2.0 * torch.pi * phase)
    target = phase_sign * amplitude.unsqueeze(1) * torch.stack((sin_phase, -sin_phase), dim=1)
    target_vel = phase_sign * (2.0 * torch.pi / period) * amplitude.unsqueeze(1) * torch.stack(
        (cos_phase, -cos_phase), dim=1
    )
    return target, target_vel, route_scale, amplitude


def arm_swing_clocked_shoulder_pitch(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    period: float = 0.75,
    min_amplitude: float = 0.05,
    max_amplitude: float = 0.38,
    max_speed: float = 1.5,
    phase_sign: float = 1.0,
    std: float = 0.16,
    command_threshold: float = 0.08,
    side_scale: float = 0.25,
    turn_scale: float = 0.15,
    mixed_turn_scale: float = 0.8,
) -> torch.Tensor:
    """Reward clocked, anti-phase shoulder pitch swing with command-scaled amplitude."""
    asset = env.scene[arm_cfg.name]
    target, _, route_scale, _ = _arm_swing_clock_targets(
        env,
        command_name,
        arm_cfg,
        period,
        min_amplitude,
        max_amplitude,
        max_speed,
        phase_sign,
        command_threshold,
        side_scale,
        turn_scale,
        mixed_turn_scale,
    )
    arm = asset.data.joint_pos[:, arm_cfg.joint_ids] - asset.data.default_joint_pos[:, arm_cfg.joint_ids]
    error = torch.sum(torch.square(arm - target), dim=1)
    return torch.exp(-error / std**2) * route_scale


def arm_swing_clocked_shoulder_velocity(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    period: float = 0.75,
    min_amplitude: float = 0.05,
    max_amplitude: float = 0.38,
    max_speed: float = 1.5,
    max_velocity: float = 1.2,
    phase_sign: float = 1.0,
    std: float = 0.45,
    command_threshold: float = 0.08,
    side_scale: float = 0.25,
    turn_scale: float = 0.15,
    mixed_turn_scale: float = 0.8,
) -> torch.Tensor:
    """Reward shoulder pitch velocity following the clocked swing target."""
    asset = env.scene[arm_cfg.name]
    _, target_vel, route_scale, _ = _arm_swing_clock_targets(
        env,
        command_name,
        arm_cfg,
        period,
        min_amplitude,
        max_amplitude,
        max_speed,
        phase_sign,
        command_threshold,
        side_scale,
        turn_scale,
        mixed_turn_scale,
    )
    target_vel = torch.clamp(target_vel, min=-max_velocity, max=max_velocity)
    arm_vel = asset.data.joint_vel[:, arm_cfg.joint_ids]
    error = torch.sum(torch.square(arm_vel - target_vel), dim=1)
    return torch.exp(-error / std**2) * route_scale


def arm_swing_shoulder_pitch_rms(env, arm_cfg: SceneEntityCfg) -> torch.Tensor:
    """Log shoulder pitch RMS relative to default pose."""
    asset = env.scene[arm_cfg.name]
    if len(arm_cfg.joint_ids) != 2:
        raise ValueError("arm_swing_shoulder_pitch_rms expects two shoulder pitch joints.")
    arm = asset.data.joint_pos[:, arm_cfg.joint_ids] - asset.data.default_joint_pos[:, arm_cfg.joint_ids]
    return torch.sqrt(torch.mean(torch.square(arm), dim=1))


def arm_swing_target_amplitude(
    env,
    command_name: str,
    min_amplitude: float = 0.12,
    max_amplitude: float = 0.75,
    max_speed: float = 2.8,
    command_threshold: float = 0.08,
) -> torch.Tensor:
    """Log the command-scaled arm swing target amplitude."""
    command = env.command_manager.get_command(command_name)[:, :3]
    speed = _command_speed(command)
    speed_ratio = torch.clamp(speed / max_speed, 0.0, 1.0)
    amplitude = min_amplitude + (max_amplitude - min_amplitude) * speed_ratio
    return torch.where(speed > command_threshold, amplitude, torch.zeros_like(amplitude))


def arm_swing_clocked_shoulder_pitch_error(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    period: float = 0.75,
    min_amplitude: float = 0.05,
    max_amplitude: float = 0.38,
    max_speed: float = 1.5,
    phase_sign: float = 1.0,
    command_threshold: float = 0.08,
    side_scale: float = 0.25,
    turn_scale: float = 0.15,
    mixed_turn_scale: float = 0.8,
) -> torch.Tensor:
    """Log mean shoulder pitch target tracking error for the clocked arm swing."""
    asset = env.scene[arm_cfg.name]
    target, _, route_scale, _ = _arm_swing_clock_targets(
        env,
        command_name,
        arm_cfg,
        period,
        min_amplitude,
        max_amplitude,
        max_speed,
        phase_sign,
        command_threshold,
        side_scale,
        turn_scale,
        mixed_turn_scale,
    )
    arm = asset.data.joint_pos[:, arm_cfg.joint_ids] - asset.data.default_joint_pos[:, arm_cfg.joint_ids]
    return torch.mean(torch.square(arm - target), dim=1) * route_scale


def arm_swing_clocked_shoulder_velocity_error(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    period: float = 0.75,
    min_amplitude: float = 0.05,
    max_amplitude: float = 0.38,
    max_speed: float = 1.5,
    max_velocity: float = 1.2,
    phase_sign: float = 1.0,
    command_threshold: float = 0.08,
    side_scale: float = 0.25,
    turn_scale: float = 0.15,
    mixed_turn_scale: float = 0.8,
) -> torch.Tensor:
    """Log mean shoulder pitch velocity target tracking error for the clocked arm swing."""
    asset = env.scene[arm_cfg.name]
    _, target_vel, route_scale, _ = _arm_swing_clock_targets(
        env,
        command_name,
        arm_cfg,
        period,
        min_amplitude,
        max_amplitude,
        max_speed,
        phase_sign,
        command_threshold,
        side_scale,
        turn_scale,
        mixed_turn_scale,
    )
    target_vel = torch.clamp(target_vel, min=-max_velocity, max=max_velocity)
    arm_vel = asset.data.joint_vel[:, arm_cfg.joint_ids]
    return torch.mean(torch.square(arm_vel - target_vel), dim=1) * route_scale


def arm_swing_coordination(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    leg_cfg: SceneEntityCfg,
    command_threshold: float = 0.15,
    std: float = 0.35,
    side_scale: float = 0.35,
    turn_scale: float = 0.45,
) -> torch.Tensor:
    """Reward shoulder pitch motion coordinated with the opposite hip pitch motion."""
    asset = env.scene[arm_cfg.name]
    if len(arm_cfg.joint_ids) != 2 or len(leg_cfg.joint_ids) != 2:
        raise ValueError("arm_swing_coordination expects two arm joints and two leg joints.")

    command = env.command_manager.get_command(command_name)[:, :3]
    moving = _command_speed(command) > command_threshold
    abs_vx = torch.abs(command[:, 0])
    abs_vy = torch.abs(command[:, 1])
    abs_wz = torch.abs(command[:, 2])
    planar_speed = torch.norm(command[:, :2], dim=1)
    side_step = (abs_vy > abs_vx) & (abs_vy > command_threshold)
    turn_in_place = (planar_speed < 0.15) & (abs_wz > command_threshold)
    arm = asset.data.joint_pos[:, arm_cfg.joint_ids] - asset.data.default_joint_pos[:, arm_cfg.joint_ids]
    leg = asset.data.joint_pos[:, leg_cfg.joint_ids] - asset.data.default_joint_pos[:, leg_cfg.joint_ids]

    # Accept either joint-sign convention while enforcing left/right opposite-limb coordination.
    err_same_sign = torch.square(arm[:, 0] - leg[:, 1]) + torch.square(arm[:, 1] - leg[:, 0])
    err_opposite_sign = torch.square(arm[:, 0] + leg[:, 1]) + torch.square(arm[:, 1] + leg[:, 0])
    reward = torch.exp(-torch.minimum(err_same_sign, err_opposite_sign) / std**2)
    route_scale = torch.ones_like(reward)
    route_scale = torch.where(side_step, torch.full_like(route_scale, side_scale), route_scale)
    route_scale = torch.where(turn_in_place, torch.full_like(route_scale, turn_scale), route_scale)
    return reward * route_scale * moving


def arm_swing_opposite_leg_phase(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    leg_cfg: SceneEntityCfg,
    command_threshold: float = 0.15,
    min_amplitude: float = 0.08,
    max_amplitude: float = 0.45,
    max_speed: float = 1.5,
    leg_phase_deadband: float = 0.03,
    leg_phase_scale: float = 0.35,
    leg_velocity_deadband: float = 0.08,
    leg_velocity_scale: float = 1.2,
    leg_velocity_weight: float = 0.35,
    min_phase_magnitude: float = 0.25,
    phase_sign: float = 1.0,
    std: float = 0.20,
    side_scale: float = 0.35,
    turn_scale: float = 0.45,
) -> torch.Tensor:
    """Reward shoulder pitch swing from the real opposite-leg hip phase.

    The term does not use a fixed clock. Instead, the left shoulder follows the right hip pitch phase and
    the right shoulder follows the left hip pitch phase with a fixed sign convention. Keeping the sign
    fixed prevents a static anti-symmetric arm pose from satisfying both gait half-cycles.
    """
    asset = env.scene[arm_cfg.name]
    if len(arm_cfg.joint_ids) != 2 or len(leg_cfg.joint_ids) != 2:
        raise ValueError("arm_swing_opposite_leg_phase expects two arm joints and two leg joints.")

    command = env.command_manager.get_command(command_name)[:, :3]
    speed = _command_speed(command)
    moving = speed > command_threshold
    abs_vx = torch.abs(command[:, 0])
    abs_vy = torch.abs(command[:, 1])
    abs_wz = torch.abs(command[:, 2])
    planar_speed = torch.norm(command[:, :2], dim=1)
    side_step = (abs_vy > abs_vx) & (abs_vy > command_threshold)
    turn_in_place = (planar_speed < 0.15) & (abs_wz > command_threshold)

    arm = asset.data.joint_pos[:, arm_cfg.joint_ids] - asset.data.default_joint_pos[:, arm_cfg.joint_ids]
    leg_pos = asset.data.joint_pos[:, leg_cfg.joint_ids] - asset.data.default_joint_pos[:, leg_cfg.joint_ids]
    leg_vel = asset.data.joint_vel[:, leg_cfg.joint_ids]
    opposite_leg_pos = torch.stack((leg_pos[:, 1], leg_pos[:, 0]), dim=1)
    opposite_leg_vel = torch.stack((leg_vel[:, 1], leg_vel[:, 0]), dim=1)

    leg_pos_phase = torch.clamp(opposite_leg_pos / leg_phase_scale, min=-1.0, max=1.0)
    leg_vel_phase = torch.clamp(opposite_leg_vel / leg_velocity_scale, min=-1.0, max=1.0)
    leg_phase = torch.clamp(leg_pos_phase + leg_velocity_weight * leg_vel_phase, min=-1.0, max=1.0)
    phase_abs = torch.abs(leg_phase)
    leg_phase_direction = torch.sign(leg_phase)
    phase_active = (torch.abs(opposite_leg_pos) > leg_phase_deadband) | (torch.abs(opposite_leg_vel) > leg_velocity_deadband)
    phase_abs = torch.where(phase_active, torch.clamp(phase_abs, min=min_phase_magnitude), torch.zeros_like(phase_abs))
    leg_phase = leg_phase_direction * phase_abs
    speed_ratio = torch.clamp(speed / max_speed, 0.0, 1.0)
    target_amplitude = min_amplitude + (max_amplitude - min_amplitude) * speed_ratio
    target = phase_sign * target_amplitude.unsqueeze(1) * leg_phase

    # Use a fixed sign convention. Dynamically accepting both signs lets a static anti-symmetric arm pose
    # match both gait half-cycles, which is exactly the "right arm forward, left arm back" failure mode.
    reward = torch.exp(-torch.sum(torch.square(arm - target), dim=1) / std**2)
    phase_confidence = torch.mean(phase_abs, dim=1)

    route_scale = torch.ones_like(reward)
    route_scale = torch.where(side_step, torch.full_like(route_scale, side_scale), route_scale)
    route_scale = torch.where(turn_in_place, torch.full_like(route_scale, turn_scale), route_scale)
    return reward * phase_confidence * route_scale * moving


def arm_swing_opposite_foot_phase(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    command_threshold: float = 0.15,
    min_amplitude: float = 0.12,
    max_amplitude: float = 0.65,
    max_speed: float = 1.5,
    foot_phase_deadband: float = 0.015,
    foot_phase_scale: float = 0.22,
    min_phase_magnitude: float = 0.35,
    phase_sign: float = 1.0,
    std: float = 0.30,
    side_scale: float = 0.35,
    turn_scale: float = 0.45,
) -> torch.Tensor:
    """Reward shoulder pitch swing from the real opposite-foot fore-aft phase.

    This uses the feet positions in the robot yaw frame, centered by the mean foot x position. The left
    shoulder follows the right foot phase and the right shoulder follows the left foot phase with a fixed
    sign convention. It is a stronger fallback than hip phase when hip pitch motion is small.
    """
    asset = env.scene[arm_cfg.name]
    if len(arm_cfg.joint_ids) != 2 or len(foot_cfg.body_ids) != 2:
        raise ValueError("arm_swing_opposite_foot_phase expects two arm joints and two foot bodies.")

    command = env.command_manager.get_command(command_name)[:, :3]
    speed = _command_speed(command)
    moving = speed > command_threshold
    abs_vx = torch.abs(command[:, 0])
    abs_vy = torch.abs(command[:, 1])
    abs_wz = torch.abs(command[:, 2])
    planar_speed = torch.norm(command[:, :2], dim=1)
    side_step = (abs_vy > abs_vx) & (abs_vy > command_threshold)
    turn_in_place = (planar_speed < 0.15) & (abs_wz > command_threshold)

    arm = asset.data.joint_pos[:, arm_cfg.joint_ids] - asset.data.default_joint_pos[:, arm_cfg.joint_ids]
    foot_rel_w = asset.data.body_pos_w[:, foot_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    foot_rel_b = _vectors_in_base_yaw_frame(asset, foot_rel_w)
    foot_x = foot_rel_b[:, :, 0]
    foot_x = foot_x - torch.mean(foot_x, dim=1, keepdim=True)
    opposite_foot_phase = torch.stack((foot_x[:, 1], foot_x[:, 0]), dim=1)

    phase = torch.clamp(opposite_foot_phase / foot_phase_scale, min=-1.0, max=1.0)
    phase_abs = torch.abs(phase)
    foot_phase_direction = torch.sign(phase)
    phase_active = torch.abs(opposite_foot_phase) > foot_phase_deadband
    phase_abs = torch.where(phase_active, torch.clamp(phase_abs, min=min_phase_magnitude), torch.zeros_like(phase_abs))
    phase = foot_phase_direction * phase_abs

    speed_ratio = torch.clamp(speed / max_speed, 0.0, 1.0)
    target_amplitude = min_amplitude + (max_amplitude - min_amplitude) * speed_ratio
    target = phase_sign * target_amplitude.unsqueeze(1) * phase

    # Keep the sign fixed for the full run. Per-step sign selection rewards a frozen anti-symmetric arm pose.
    reward = torch.exp(-torch.sum(torch.square(arm - target), dim=1) / std**2)
    phase_confidence = torch.mean(phase_abs, dim=1)

    route_scale = torch.ones_like(reward)
    route_scale = torch.where(side_step, torch.full_like(route_scale, side_scale), route_scale)
    route_scale = torch.where(turn_in_place, torch.full_like(route_scale, turn_scale), route_scale)
    return reward * phase_confidence * route_scale * moving


def arm_swing_amplitude_schedule(
    env,
    command_name: str,
    arm_cfg: SceneEntityCfg,
    period: float,
    min_amplitude: float = 0.05,
    max_amplitude: float = 0.45,
    max_speed: float = 1.5,
    std: float = 0.18,
    phase_offset: float = 0.5,
    command_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward phase-varying arm swing amplitude that grows with commanded locomotion speed."""
    asset = env.scene[arm_cfg.name]
    if len(arm_cfg.joint_ids) != 2:
        raise ValueError("arm_swing_amplitude_schedule expects two arm joints.")
    if period <= 0.0:
        raise ValueError("arm_swing_amplitude_schedule expects period to be positive.")

    command = env.command_manager.get_command(command_name)[:, :3]
    speed = _command_speed(command)
    speed_ratio = torch.clamp(speed / max_speed, 0.0, 1.0)
    target_amplitude = min_amplitude + (max_amplitude - min_amplitude) * speed_ratio
    target_amplitude = torch.where(speed > command_threshold, target_amplitude, torch.zeros_like(target_amplitude))
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt, period) / period
    left_target = target_amplitude * torch.sin(2.0 * torch.pi * phase)
    right_target = target_amplitude * torch.sin(2.0 * torch.pi * torch.remainder(phase + phase_offset, 1.0))
    arm = asset.data.joint_pos[:, arm_cfg.joint_ids] - asset.data.default_joint_pos[:, arm_cfg.joint_ids]
    err_same_sign = torch.square(arm[:, 0] - left_target) + torch.square(arm[:, 1] - right_target)
    err_opposite_sign = torch.square(arm[:, 0] + left_target) + torch.square(arm[:, 1] + right_target)
    return torch.exp(-torch.minimum(err_same_sign, err_opposite_sign) / std**2)


def arm_swing_sagittal_velocity(
    env,
    command_name: str,
    body_cfg: SceneEntityCfg,
    period: float,
    min_velocity: float = 0.10,
    max_velocity: float = 0.65,
    max_speed: float = 1.5,
    std: float = 0.35,
    lateral_weight: float = 2.0,
    vertical_weight: float = 0.25,
    phase_offset: float = 0.5,
    command_threshold: float = 0.12,
    side_scale: float = 0.35,
    turn_scale: float = 0.45,
) -> torch.Tensor:
    """Reward forearm/hand end links for sagittal swing velocity and low lateral flailing.

    Unlike joint-angle arm rewards, this term checks the visible end-link velocity in the robot yaw frame:
    x is forward/backward swing, y is lateral swing. This keeps the reward aligned with human-readable
    motion even when joint axis conventions differ across G1 assets.
    """
    asset = env.scene[body_cfg.name]
    if len(body_cfg.body_ids) != 2:
        raise ValueError("arm_swing_sagittal_velocity expects two arm end-link bodies.")
    if period <= 0.0:
        raise ValueError("arm_swing_sagittal_velocity expects period to be positive.")

    command = env.command_manager.get_command(command_name)[:, :3]
    speed = _command_speed(command)
    moving = speed > command_threshold
    abs_vx = torch.abs(command[:, 0])
    abs_vy = torch.abs(command[:, 1])
    abs_wz = torch.abs(command[:, 2])
    planar_speed = torch.norm(command[:, :2], dim=1)
    side_step = (abs_vy > abs_vx) & (abs_vy > command_threshold)
    turn_in_place = (planar_speed < 0.15) & (abs_wz > command_threshold)

    relative_vel_w = asset.data.body_lin_vel_w[:, body_cfg.body_ids, :] - asset.data.root_lin_vel_w.unsqueeze(1)
    arm_vel_b = _vectors_in_base_yaw_frame(asset, relative_vel_w)

    speed_ratio = torch.clamp(speed / max_speed, 0.0, 1.0)
    target_velocity = min_velocity + (max_velocity - min_velocity) * speed_ratio
    target_velocity = torch.where(moving, target_velocity, torch.zeros_like(target_velocity))
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt, period) / period
    left_target = target_velocity * torch.cos(2.0 * torch.pi * phase)
    right_target = target_velocity * torch.cos(2.0 * torch.pi * torch.remainder(phase + phase_offset, 1.0))

    sagittal_error = torch.square(arm_vel_b[:, 0, 0] - left_target) + torch.square(arm_vel_b[:, 1, 0] - right_target)
    lateral_error = torch.square(arm_vel_b[:, 0, 1]) + torch.square(arm_vel_b[:, 1, 1])
    vertical_error = torch.square(arm_vel_b[:, 0, 2]) + torch.square(arm_vel_b[:, 1, 2])
    reward = torch.exp(-(sagittal_error + lateral_weight * lateral_error + vertical_weight * vertical_error) / std**2)

    route_scale = torch.ones_like(reward)
    route_scale = torch.where(side_step, torch.full_like(route_scale, side_scale), route_scale)
    route_scale = torch.where(turn_in_place, torch.full_like(route_scale, turn_scale), route_scale)
    return reward * route_scale * moving


def arm_swing_pose_envelope(
    env,
    command_name: str,
    body_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg,
    min_lateral: float = 0.08,
    max_lateral: float = 0.34,
    max_vertical: float = 0.05,
    min_vertical: float = -0.55,
    max_sagittal: float = 0.50,
    std: float = 0.18,
    command_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward arm end links for staying in a human-like envelope around the torso.

    This complements sagittal swing velocity: the velocity term makes the arms swing forward/backward,
    while this pose envelope keeps the arms near the body sides and avoids high, wide, or overextended
    poses.
    """
    asset = env.scene[body_cfg.name]
    if len(body_cfg.body_ids) != 2:
        raise ValueError("arm_swing_pose_envelope expects two arm end-link bodies.")
    if len(torso_cfg.body_ids) != 1:
        raise ValueError("arm_swing_pose_envelope expects one torso body.")

    torso_pos_w = asset.data.body_pos_w[:, torso_cfg.body_ids[0], :].unsqueeze(1)
    arm_rel_w = asset.data.body_pos_w[:, body_cfg.body_ids, :] - torso_pos_w
    arm_pos_b = _vectors_in_base_yaw_frame(asset, arm_rel_w)

    lateral = torch.abs(arm_pos_b[:, :, 1])
    lateral_error = torch.square(torch.clamp(min_lateral - lateral, min=0.0))
    lateral_error += torch.square(torch.clamp(lateral - max_lateral, min=0.0))
    vertical_error = torch.square(torch.clamp(arm_pos_b[:, :, 2] - max_vertical, min=0.0))
    vertical_error += torch.square(torch.clamp(min_vertical - arm_pos_b[:, :, 2], min=0.0))
    sagittal_error = torch.square(torch.clamp(torch.abs(arm_pos_b[:, :, 0]) - max_sagittal, min=0.0))

    pose_error = torch.sum(lateral_error + vertical_error + sagittal_error, dim=1)
    return torch.exp(-pose_error / std**2)


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


def track_lin_vel_y_yaw_frame_abs_error(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Log absolute lateral velocity tracking error in the robot yaw frame."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    return torch.abs(env.command_manager.get_command(command_name)[:, 1] - vel_yaw[:, 1])


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
    return stand_regularization(
        env,
        command_name,
        asset_cfg=asset_cfg,
        command_threshold=command_threshold,
        base_weight=0.0,
        joint_vel_weight=0.0,
        joint_dev_weight=1.0,
    )

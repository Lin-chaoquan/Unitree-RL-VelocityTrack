# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class high_speed_velocity_command_curriculum(ManagerTermBase):
    """Performance-gated curriculum for expanding flat-ground velocity commands."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._stages = cfg.params["stages"]
        self._xy_error_thresholds = cfg.params["xy_error_thresholds"]
        self._yaw_error_threshold = cfg.params.get("yaw_error_threshold", 0.3)
        self._survival_threshold = cfg.params.get("survival_threshold", 0.8)
        self._failure_survival_threshold = cfg.params.get("failure_survival_threshold", 0.5)
        self._failure_xy_error_scale = cfg.params.get("failure_xy_error_scale", 1.8)
        self._min_steps_per_stage = cfg.params.get("min_steps_per_stage", 25_000)
        self._min_episodes_per_stage = cfg.params.get("min_episodes_per_stage", 2048)
        self._failure_episodes = cfg.params.get("failure_episodes", 4096)
        self._ema_alpha = cfg.params.get("ema_alpha", 0.08)
        self._posture_thresholds = cfg.params.get("posture_thresholds", {})
        self._extra_reward_metrics = cfg.params.get("extra_reward_metrics", [])
        self._stage = 0
        self._stage_start_step = env.common_step_counter
        self._stage_episodes = 0
        self._failure_episode_counter = 0
        self._ema: dict[str, float] = {}
        self._posture_gate = False
        self._last_upgrade_gate = False
        self._step_gate = False
        self._episode_gate = False
        self._xy_error_gate = False
        self._yaw_error_gate = False
        self._survival_gate = False
        self._apply_stage(env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: list[dict],
        xy_error_thresholds: list[float],
        yaw_error_threshold: float = 0.3,
        survival_threshold: float = 0.8,
        failure_survival_threshold: float = 0.5,
        failure_xy_error_scale: float = 1.8,
        min_steps_per_stage: int = 25_000,
        min_episodes_per_stage: int = 2048,
        failure_episodes: int = 4096,
        ema_alpha: float = 0.08,
        posture_thresholds: dict | None = None,
        extra_reward_metrics: list[str] | None = None,
    ) -> dict[str, float]:
        env_ids_tensor = self._resolve_env_ids(env, env_ids)
        env_ids_tensor = env_ids_tensor[env.episode_length_buf[env_ids_tensor] > 0]
        if env_ids_tensor.numel() == 0:
            return self._state()

        self._stage_episodes += int(env_ids_tensor.numel())
        self._update_ema("error_vel_xy", self._command_metric(env, env_ids_tensor, "error_vel_xy"))
        self._update_ema("error_vel_yaw", self._command_metric(env, env_ids_tensor, "error_vel_yaw"))
        self._update_ema("survival", self._survival_rate(env, env_ids_tensor))
        for name in self._posture_thresholds:
            value = self._reward_episode_metric(env, env_ids_tensor, name)
            if value is not None:
                self._update_ema(name, value)
        for name in self._extra_reward_metrics:
            value = self._reward_episode_metric(env, env_ids_tensor, name)
            if value is not None:
                self._update_ema(name, value)

        self._posture_gate = self._compute_posture_gate()
        self._last_upgrade_gate = self._compute_upgrade_gate(env)
        if self._last_upgrade_gate:
            self._stage += 1
            self._stage_start_step = env.common_step_counter
            self._stage_episodes = 0
            self._failure_episode_counter = 0
            self._apply_stage(env)
        else:
            self._update_failure_guard(env_ids_tensor)
            if self._should_downgrade():
                self._stage -= 1
                self._stage_start_step = env.common_step_counter
                self._stage_episodes = 0
                self._failure_episode_counter = 0
                self._apply_stage(env)

        return self._state()

    def reset(self, env_ids: Sequence[int] | None = None):
        self._apply_stage(self._env)

    def _resolve_env_ids(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=env.device, dtype=torch.long).flatten()
        return torch.tensor(env_ids, device=env.device, dtype=torch.long).flatten()

    def _apply_stage(self, env: ManagerBasedRLEnv):
        stage = self._stages[self._stage]
        command_term = env.command_manager.get_term("base_velocity")
        command_term.cfg.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
        command_term.cfg.ranges.lin_vel_y = tuple(stage["lin_vel_y"])
        command_term.cfg.ranges.ang_vel_z = tuple(stage["ang_vel_z"])
        period = stage.get("period")
        if period is not None:
            self._set_reward_period(env, "feet_gait_clock_omni", period)
            self._set_reward_period(env, "feet_swing_height_trajectory", period)
            self._set_reward_period(env, "feet_swing_vertical_velocity", period)
            self._set_reward_period(env, "arm_swing_clocked_shoulder_pitch", period)
            self._set_reward_period(env, "arm_swing_clocked_shoulder_velocity", period)
            self._set_reward_period(env, "arm_swing_clocked_shoulder_pitch_error", period)
            self._set_reward_period(env, "arm_swing_clocked_shoulder_velocity_error", period)
            self._set_reward_period(env, "arm_swing_sagittal_velocity", period)
            if hasattr(env.observation_manager.cfg.policy, "phase_clock"):
                env.observation_manager.cfg.policy.phase_clock.params["period"] = period
                self._set_observation_period(env, "policy", "phase_clock", period)

    def _set_reward_period(self, env: ManagerBasedRLEnv, term_name: str, period: float):
        if term_name in env.reward_manager.active_terms:
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            term_cfg.params["period"] = period
            env.reward_manager.set_term_cfg(term_name, term_cfg)

    def _set_observation_period(self, env: ManagerBasedRLEnv, group_name: str, term_name: str, period: float):
        if group_name not in env.observation_manager._group_obs_term_names:
            return
        term_names = env.observation_manager._group_obs_term_names[group_name]
        if term_name not in term_names:
            return
        term_idx = term_names.index(term_name)
        env.observation_manager._group_obs_term_cfgs[group_name][term_idx].params["period"] = period

    def _command_metric(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, metric_name: str) -> float:
        metrics = env.command_manager.get_term("base_velocity").metrics
        if metric_name not in metrics:
            return float("inf")
        return float(torch.mean(metrics[metric_name][env_ids]).item())

    def _survival_rate(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> float:
        min_length = int(0.9 * env.max_episode_length)
        return float(torch.mean((env.episode_length_buf[env_ids] >= min_length).float()).item())

    def _reward_episode_metric(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, term_name: str) -> float | None:
        if term_name not in env.reward_manager.active_terms:
            return None
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        episode_sum = env.reward_manager._episode_sums[term_name][env_ids]
        if term_cfg.log_only:
            episode_lengths_s = env.episode_length_buf[env_ids].to(dtype=torch.float) * env.step_dt
            episode_lengths_s = torch.clamp(episode_lengths_s, min=env.step_dt)
            value = torch.mean(episode_sum / episode_lengths_s)
        else:
            value = torch.mean(episode_sum) / env.max_episode_length_s
        return float(value.item())

    def _update_ema(self, name: str, value: float):
        if name not in self._ema:
            self._ema[name] = value
        else:
            self._ema[name] = (1.0 - self._ema_alpha) * self._ema[name] + self._ema_alpha * value

    def _compute_posture_gate(self) -> bool:
        for name, threshold_cfg in self._posture_thresholds.items():
            value = self._ema.get(name)
            if value is None:
                continue
            if "max" in threshold_cfg and value > threshold_cfg["max"]:
                return False
            if "min" in threshold_cfg and value < threshold_cfg["min"]:
                return False
        return True

    def _compute_upgrade_gate(self, env: ManagerBasedRLEnv) -> bool:
        self._step_gate = env.common_step_counter - self._stage_start_step >= self._min_steps_per_stage
        self._episode_gate = self._stage_episodes >= self._min_episodes_per_stage
        if self._stage >= len(self._stages) - 1:
            return False
        xy_error = self._ema.get("error_vel_xy", float("inf"))
        yaw_error = self._ema.get("error_vel_yaw", float("inf"))
        survival = self._ema.get("survival", 0.0)
        xy_threshold = self._xy_error_thresholds[min(self._stage, len(self._xy_error_thresholds) - 1)]
        self._xy_error_gate = xy_error <= xy_threshold
        self._yaw_error_gate = yaw_error <= self._yaw_error_threshold
        self._survival_gate = survival >= self._survival_threshold
        return all(
            (
                self._step_gate,
                self._episode_gate,
                self._xy_error_gate,
                self._yaw_error_gate,
                self._survival_gate,
                self._posture_gate,
            )
        )

    def _update_failure_guard(self, env_ids: torch.Tensor):
        if self._stage == 0:
            self._failure_episode_counter = 0
            return
        xy_threshold = self._xy_error_thresholds[min(self._stage - 1, len(self._xy_error_thresholds) - 1)]
        failed = (
            self._ema.get("survival", 1.0) < self._failure_survival_threshold
            or self._ema.get("error_vel_xy", 0.0) > xy_threshold * self._failure_xy_error_scale
        )
        if failed:
            self._failure_episode_counter += int(env_ids.numel())
        else:
            self._failure_episode_counter = 0

    def _should_downgrade(self) -> bool:
        return self._stage > 0 and self._failure_episode_counter >= self._failure_episodes

    def _state(self) -> dict[str, float]:
        stage_cfg = self._stages[self._stage]
        state = {
            "stage": float(self._stage),
            "stage_episodes": float(self._stage_episodes),
            "vx_max": float(stage_cfg["lin_vel_x"][1]),
            "vy_abs_max": float(max(abs(stage_cfg["lin_vel_y"][0]), abs(stage_cfg["lin_vel_y"][1]))),
            "wz_abs_max": float(max(abs(stage_cfg["ang_vel_z"][0]), abs(stage_cfg["ang_vel_z"][1]))),
            "period": float(stage_cfg.get("period", 0.0)),
            "ema_error_vel_xy": float(self._ema.get("error_vel_xy", 0.0)),
            "ema_error_vel_yaw": float(self._ema.get("error_vel_yaw", 0.0)),
            "ema_survival": float(self._ema.get("survival", 0.0)),
            "posture_gate": float(self._posture_gate),
            "upgrade_gate": float(self._last_upgrade_gate),
            "step_gate": float(self._step_gate),
            "episode_gate": float(self._episode_gate),
            "xy_error_gate": float(self._xy_error_gate),
            "yaw_error_gate": float(self._yaw_error_gate),
            "survival_gate": float(self._survival_gate),
            "failure_episodes": float(self._failure_episode_counter),
        }
        for name in self._posture_thresholds:
            state[f"ema_{name}"] = float(self._ema.get(name, 0.0))
        for name in self._extra_reward_metrics:
            state[f"ema_{name}"] = float(self._ema.get(name, 0.0))
        return state


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

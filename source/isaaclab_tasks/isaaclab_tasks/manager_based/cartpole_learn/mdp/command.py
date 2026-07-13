from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformJointPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions and orientations uniformly within specified ranges. The
    pose commands are generated in the base frame of the robot, and not the simulation world frame. Users therefore need
    to handle the transformation from the base frame to the simulation world frame themselves.
    """

    cfg: UniformJointPoseCommandCfg

    def __init__(self, cfg: UniformJointPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.jointposecommand = torch.zeros(env.num_envs, 3, device=env.device)

        # Resolve and preserve the joint order used by all metric calculations:
        # [cart position, pole relative angle, pendulum relative angle].
        self._joint_ids, _ = self.robot.find_joints(
            ["slider_to_cart", "cart_to_pole", "pole_to_pendulum"], preserve_order=True
        )

        # Accumulate episode statistics per environment. CommandTerm metrics are emitted when an
        # environment resets, so accumulating here gives episode-level values instead of only the
        # state observed at the final step.
        self._metric_count = torch.zeros(env.num_envs, device=env.device)
        self._metric_square_sums = {
            "cart_pos_rms": torch.zeros(env.num_envs, device=env.device),
            "cart_vel_rms": torch.zeros(env.num_envs, device=env.device),
            "action_rms": torch.zeros(env.num_envs, device=env.device),
            "action_rate_rms": torch.zeros(env.num_envs, device=env.device),
            "applied_force_rms": torch.zeros(env.num_envs, device=env.device),
        }
        self._near_target_metric_count = torch.zeros(env.num_envs, device=env.device)
        self._near_target_metric_square_sums = {
            "near_target_cart_pos_rms": torch.zeros(env.num_envs, device=env.device),
            "near_target_cart_vel_rms": torch.zeros(env.num_envs, device=env.device),
            "near_target_action_rms": torch.zeros(env.num_envs, device=env.device),
            "near_target_action_rate_rms": torch.zeros(env.num_envs, device=env.device),
            "near_target_applied_force_rms": torch.zeros(env.num_envs, device=env.device),
        }
        self._metric_linear_sums = {
            "action_saturation": torch.zeros(env.num_envs, device=env.device),
            "pole_error": torch.zeros(env.num_envs, device=env.device),
            "pendulum_abs_error": torch.zeros(env.num_envs, device=env.device),
        }
        self._near_target_metric_linear_sums = {
            "near_target_action_saturation": torch.zeros(env.num_envs, device=env.device),
        }

    def __str__(self) -> str:
        msg = f"UniformPoseCommand: {self.cfg.asset_name}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.jointposecommand

    def _update_metrics(self):
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._joint_ids]

        cart_pos_error = joint_pos[:, 0] - self.jointposecommand[:, 0]
        pole_error = wrap_to_pi(joint_pos[:, 1] - self.jointposecommand[:, 1])
        pendulum_abs_error = wrap_to_pi(
            joint_pos[:, 1]
            + joint_pos[:, 2]
            - self.jointposecommand[:, 1]
            - self.jointposecommand[:, 2]
        )

        action = self._env.action_manager.action
        action_rate = action - self._env.action_manager.prev_action
        applied_force = self.robot.data.applied_torque[:, self._joint_ids[0]]
        action_saturation = torch.mean((torch.abs(action) >= 0.99).to(dtype=torch.float), dim=1)

        # Square sums are converted to RMS values in reset(). For a multi-dimensional action,
        # average over action dimensions first so the metric remains independent of action count.
        self._metric_square_sums["cart_pos_rms"] += torch.square(cart_pos_error)
        self._metric_square_sums["cart_vel_rms"] += torch.square(joint_vel[:, 0])
        self._metric_square_sums["action_rms"] += torch.mean(torch.square(action), dim=1)
        self._metric_square_sums["action_rate_rms"] += torch.mean(torch.square(action_rate), dim=1)
        self._metric_square_sums["applied_force_rms"] += torch.square(applied_force)

        # Conditional metrics use their own sample count. Dividing these sums by the full
        # episode length would mix target occupancy with the behavior observed near the target.
        near_target = (torch.abs(pole_error) < self.cfg.near_target_pole_threshold) & (
            torch.abs(pendulum_abs_error) < self.cfg.near_target_pendulum_threshold
        )
        near_target_float = near_target.to(dtype=cart_pos_error.dtype)
        self._near_target_metric_count += near_target_float
        self._near_target_metric_square_sums["near_target_cart_pos_rms"] += (
            near_target_float * torch.square(cart_pos_error)
        )
        self._near_target_metric_square_sums["near_target_cart_vel_rms"] += (
            near_target_float * torch.square(joint_vel[:, 0])
        )
        self._near_target_metric_square_sums["near_target_action_rms"] += near_target_float * torch.mean(
            torch.square(action), dim=1
        )
        self._near_target_metric_square_sums["near_target_action_rate_rms"] += near_target_float * torch.mean(
            torch.square(action_rate), dim=1
        )
        self._near_target_metric_square_sums["near_target_applied_force_rms"] += (
            near_target_float * torch.square(applied_force)
        )
        self._near_target_metric_linear_sums["near_target_action_saturation"] += (
            near_target_float * action_saturation
        )

        # Saturation is the fraction of executed normalized actions at the wrapper clip boundary.
        self._metric_linear_sums["action_saturation"] += action_saturation
        # Report mean absolute wrapped angle error in radians.
        self._metric_linear_sums["pole_error"] += torch.abs(pole_error)
        self._metric_linear_sums["pendulum_abs_error"] += torch.abs(pendulum_abs_error)
        self._metric_count += 1.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Log episode metrics, reset their accumulators, and resample commands."""
        if env_ids is None:
            env_ids = slice(None)

        metric_count = self._metric_count[env_ids]
        count = torch.clamp(metric_count, min=1.0)
        near_target_count = self._near_target_metric_count[env_ids]
        valid_near_target = near_target_count > 0.0
        extras: dict[str, float] = {}

        for name, square_sum in self._metric_square_sums.items():
            # Compute a single RMS over the resetting environments and their episode samples.
            mean_square = torch.mean(square_sum[env_ids] / count)
            extras[name] = torch.sqrt(mean_square).item()
            square_sum[env_ids] = 0.0

        # Only environments that reached the target contribute to conditional RMS values.
        # The sample ratio below distinguishes a true zero from an episode with no target samples.
        for name, square_sum in self._near_target_metric_square_sums.items():
            if torch.any(valid_near_target):
                mean_square = torch.mean(
                    square_sum[env_ids][valid_near_target] / near_target_count[valid_near_target]
                )
                extras[name] = torch.sqrt(mean_square).item()
            else:
                extras[name] = 0.0
            square_sum[env_ids] = 0.0

        for name, linear_sum in self._metric_linear_sums.items():
            extras[name] = torch.mean(linear_sum[env_ids] / count).item()
            linear_sum[env_ids] = 0.0

        for name, linear_sum in self._near_target_metric_linear_sums.items():
            if torch.any(valid_near_target):
                extras[name] = torch.mean(
                    linear_sum[env_ids][valid_near_target] / near_target_count[valid_near_target]
                ).item()
            else:
                extras[name] = 0.0
            linear_sum[env_ids] = 0.0

        extras["near_target_sample_ratio"] = torch.mean(near_target_count / count).item()

        self._metric_count[env_ids] = 0.0
        self._near_target_metric_count[env_ids] = 0.0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the given environment IDs.

        Args:
            env_ids: The environment IDs for which to resample the command.
        """
        # Use the indexed tensor shape instead of len(env_ids), since env_ids may be a slice.
        sample_count = self.jointposecommand[env_ids].shape[0]
        self.jointposecommand[env_ids, 0] = torch.empty(sample_count, device=self._env.device).uniform_(
            self.cfg.ranges.pos_cart[0], self.cfg.ranges.pos_cart[1]
        )
        self.jointposecommand[env_ids, 1] = torch.empty(sample_count, device=self._env.device).uniform_(
            self.cfg.ranges.pos_pole[0], self.cfg.ranges.pos_pole[1]
        )
        self.jointposecommand[env_ids, 2] = torch.empty(sample_count, device=self._env.device).uniform_(
            self.cfg.ranges.pos_pendulum[0], self.cfg.ranges.pos_pendulum[1]
        )

    def _update_command(self):
        """Update the command for the given environment IDs.

        Args:
            env_ids: The environment IDs for which to update the command.
        """
        # update the command for the given environment IDs
        pass


@configclass
class UniformJointPoseCommandCfg(CommandTermCfg):
    """Command generator for generating joint pose commands uniformly.

    The command generator generates joint poses by sampling joint positions uniformly within specified
    ranges. The joint position commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.
    """

    class_type: type = UniformJointPoseCommand

    """Configuration for the command generator."""
    asset_name: str = MISSING

    @configclass
    class Ranges:
        """Uniform distribution ranges for the joint pose commands."""

        pos_cart: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_pole: tuple[float, float] = MISSING
        """Range for the pole position (in rad)."""
        pos_pendulum: tuple[float, float] = MISSING
        """Range for the pendulum position (in rad)."""

    ranges: Ranges = MISSING

    near_target_pole_threshold: float = 0.2
    """Maximum absolute pole error included in near-target metrics, in radians."""

    near_target_pendulum_threshold: float = 0.2
    """Maximum absolute pendulum error included in near-target metrics, in radians."""

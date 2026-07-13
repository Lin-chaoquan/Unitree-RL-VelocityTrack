# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train Cartpole with configurable checkpoint loading for RSL-RL.

This script extends the standard ``train.py`` workflow with model warm-start
support. In particular, ``--checkpoint_load_mode weights_only`` loads the
actor and critic from a checkpoint while keeping the optimizer created from
the current agent configuration. This makes a newly configured learning rate
and fresh optimizer state effective.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Train Cartpole with RSL-RL checkpoint warm-start support.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used by the environment and agent.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of learning iterations to run.")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=None,
    help="Override the PPO learning rate. This is especially useful with a fresh optimizer warm-start.",
)
parser.add_argument("--distributed", action="store_true", default=False, help="Enable distributed training.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Process id supplied by Ray integration."
)
parser.add_argument(
    "--checkpoint_load_mode",
    choices=("full", "weights_only", "weights_and_iteration"),
    default="full",
    help=(
        "Checkpoint loading mode. 'full' restores model, optimizer, and iteration; "
        "'weights_only' restores actor/critic with a new optimizer and iteration 0; "
        "'weights_and_iteration' restores actor/critic and iteration with a new optimizer."
    ),
)
parser.add_argument(
    "--reset_policy_std",
    type=float,
    default=None,
    help="Reset the Gaussian policy standard deviation to this value after loading the checkpoint.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.checkpoint_load_mode != "full" and not args_cli.resume:
    parser.error("--checkpoint_load_mode requires --resume when it is not 'full'.")
if args_cli.reset_policy_std is not None and args_cli.reset_policy_std <= 0.0:
    parser.error("--reset_policy_std must be greater than zero.")
if args_cli.learning_rate is not None and args_cli.learning_rate <= 0.0:
    parser.error("--learning_rate must be greater than zero.")

# Clear arguments handled by this script before Hydra processes its overrides.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for the minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    raise SystemExit(1)

"""Everything below runs after Isaac Sim has launched."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _checkpoint_load_cfg(mode: str) -> dict[str, bool] | None:
    """Build the RSL-RL load configuration for the requested mode."""
    if mode == "full":
        return None
    return {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": mode == "weights_and_iteration",
        "rnd": False,
    }


def _reset_gaussian_policy_std(runner: OnPolicyRunner, std: float) -> None:
    """Reset scalar/log Gaussian exploration std after loading actor weights."""
    distribution = runner.alg.actor.distribution
    with torch.no_grad():
        if hasattr(distribution, "std_param"):
            distribution.std_param.fill_(std)
        elif hasattr(distribution, "log_std_param"):
            distribution.log_std_param.fill_(math.log(std))
        else:
            raise RuntimeError(
                "The actor distribution does not expose std_param or log_std_param; "
                "--reset_policy_std is unsupported for this distribution."
            )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train an RSL-RL policy, optionally warm-starting with a new optimizer."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported on a CPU device.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager-based environments.")
    env_cfg.log_dir = log_dir

    # Resolve the source checkpoint before the new timestamped log directory is created.
    resume_path = None
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        if args_cli.checkpoint_load_mode != "full" or args_cli.reset_policy_std is not None:
            raise ValueError("Custom checkpoint loading modes are only supported for OnPolicyRunner.")
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.add_git_repo_to_log(__file__)

    load_cfg = _checkpoint_load_cfg(args_cli.checkpoint_load_mode)
    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        print(f"[INFO] Checkpoint load mode: {args_cli.checkpoint_load_mode}")
        runner.load(resume_path, load_cfg=load_cfg)
        if args_cli.reset_policy_std is not None:
            if not isinstance(runner, OnPolicyRunner):
                raise TypeError("--reset_policy_std requires OnPolicyRunner.")
            _reset_gaussian_policy_std(runner, args_cli.reset_policy_std)
            print(f"[INFO] Reset policy standard deviation to: {args_cli.reset_policy_std}")

    optimizer_lrs = []
    if isinstance(runner, OnPolicyRunner):
        optimizer_lrs = [group["lr"] for group in runner.alg.optimizer.param_groups]
        print(f"[INFO] Effective optimizer learning rate(s): {optimizer_lrs}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "checkpoint_load.yaml"),
        {
            "source_checkpoint": resume_path,
            "mode": args_cli.checkpoint_load_mode,
            "load_cfg": load_cfg,
            "reset_policy_std": args_cli.reset_policy_std,
            "effective_optimizer_learning_rates": optimizer_lrs,
        },
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

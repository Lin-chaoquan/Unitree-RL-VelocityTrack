# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


def _register_task(
    task_id: str,
    env_cfg: str,
    runner_cfg: str,
    *,
    skrl_cfg: str | None = None,
):
    """Register a G1 velocity task with its supported training backends."""
    kwargs = {
        "env_cfg_entry_point": f"{__name__}.{env_cfg}",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
    }
    if skrl_cfg is not None:
        kwargs["skrl_cfg_entry_point"] = f"{agents.__name__}:{skrl_cfg}"
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs=kwargs,
    )


# Stable 37-DOF base tasks. Previous experimental Flat variants were removed pending redesign.
_register_task(
    "Isaac-Velocity-Rough-G1-v0",
    "rough_env_cfg:G1RoughEnvCfg",
    "G1RoughPPORunnerCfg",
    skrl_cfg="skrl_rough_ppo_cfg.yaml",
)
_register_task(
    "Isaac-Velocity-Rough-G1-Play-v0",
    "rough_env_cfg:G1RoughEnvCfg_PLAY",
    "G1RoughPPORunnerCfg",
    skrl_cfg="skrl_rough_ppo_cfg.yaml",
)
_register_task(
    "Isaac-Velocity-Flat-G1-v0",
    "flat_env_cfg:G1FlatEnvCfg",
    "G1FlatPPORunnerCfg",
    skrl_cfg="skrl_flat_ppo_cfg.yaml",
)
_register_task(
    "Isaac-Velocity-Flat-G1-Play-v0",
    "flat_env_cfg:G1FlatEnvCfg_PLAY",
    "G1FlatPPORunnerCfg",
    skrl_cfg="skrl_flat_ppo_cfg.yaml",
)

# Local 29-DOF base tasks. Their runners always start fresh and use separate log directories.
_register_task(
    "Isaac-Velocity-Rough-G1-29DOF-v0",
    "g1_29dof_env_cfg:G1Rough29DOFEnvCfg",
    "G1Rough29DOFPPORunnerCfg",
)
_register_task(
    "Isaac-Velocity-Rough-G1-29DOF-Play-v0",
    "g1_29dof_env_cfg:G1Rough29DOFEnvCfg_PLAY",
    "G1Rough29DOFPPORunnerCfg",
)
_register_task(
    "Isaac-Velocity-Flat-G1-29DOF-v0",
    "g1_29dof_env_cfg:G1Flat29DOFEnvCfg",
    "G1Flat29DOFPPORunnerCfg",
)
_register_task(
    "Isaac-Velocity-Flat-G1-29DOF-Play-v0",
    "g1_29dof_env_cfg:G1Flat29DOFEnvCfg_PLAY",
    "G1Flat29DOFPPORunnerCfg",
)

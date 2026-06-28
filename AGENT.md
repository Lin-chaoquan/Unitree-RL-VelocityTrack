# AGENT — G1 Training Quick Reference

Purpose
- Provide a compact, developer-focused reference for training the Unitree `G1` robot in this repository.
- Include: training entry points, environment/MDP/reward locations, wrappers, run & test commands, evaluation metrics, and tuning suggestions.

Quick Start

Prerequisites
- Launch Isaac Sim via the provided helper or run scripts with `./isaaclab.sh -p <script>`.
- Ensure Python dependencies (see `requirements.txt` / `pyproject.toml`) are installed and `rsl-rl-lib` meets the minimum version used by scripts.

Run examples (headless)
```bash
# RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 64 --max_iterations 1000

# RL-Games
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 64 --max_iterations 1000

# Stable-Baselines3
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 64 --max_iterations 1000
```

Important Files (entry references)
- Training scripts: [scripts/reinforcement_learning/rsl_rl/train.py](scripts/reinforcement_learning/rsl_rl/train.py), [scripts/reinforcement_learning/rl_games/train.py](scripts/reinforcement_learning/rl_games/train.py), [scripts/reinforcement_learning/sb3/train.py](scripts/reinforcement_learning/sb3/train.py), [scripts/reinforcement_learning/skrl/train.py](scripts/reinforcement_learning/skrl/train.py)
- G1 task registration: [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py)
- G1 env configs: [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py), [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py)
- MDP base config: [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
- Manager env & config: [source/isaaclab/isaaclab/envs/manager_based_rl_env.py](source/isaaclab/isaaclab/envs/manager_based_rl_env.py), [source/isaaclab/isaaclab/envs/manager_based_env.py](source/isaaclab/isaaclab/envs/manager_based_env.py)
- Reward / MDP functions: [source/isaaclab/isaaclab/envs/mdp/rewards.py](source/isaaclab/isaaclab/envs/mdp/rewards.py)
- Reward manager: [source/isaaclab/isaaclab/managers/reward_manager.py](source/isaaclab/isaaclab/managers/reward_manager.py)
- RL wrappers: [source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py](source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py), [source/isaaclab_rl/isaaclab_rl/rl_games/rl_games.py](source/isaaclab_rl/isaaclab_rl/rl_games/rl_games.py), [source/isaaclab_rl/isaaclab_rl/sb3.py](source/isaaclab_rl/isaaclab_rl/sb3.py)

Training flow (high level)
1. Start training script (one of the `scripts/reinforcement_learning/*/train.py` files).
2. CLI args parsed; `@hydra_task_config` loads `env_cfg` and `agent_cfg` from registry.
3. `gym.make(task, cfg=env_cfg)` creates `ManagerBasedRLEnv` (task-specific env CFG injected).
4. Env wrapped by library wrapper (`RslRlVecEnvWrapper` / `RlGamesVecEnvWrapper` / `Sb3VecEnvWrapper`).
5. Runner/agent created and optionally resumed from checkpoint.
6. `runner.learn(...)` / `runner.run(...)` executes training loop; logs and checkpoints written to `logs/<lib>/<experiment>`.

Rewards / Observations mapping
- Core reward implementations live in [source/isaaclab/isaaclab/envs/mdp/rewards.py](source/isaaclab/isaaclab/envs/mdp/rewards.py). Task configs (e.g., `G1RoughEnvCfg` in rough_env_cfg.py) assemble reward terms and set weights.
- Observations are defined in `velocity_env_cfg.py` under `ObservationsCfg.PolicyCfg` (terms like `base_lin_vel`, `joint_pos`, `height_scan`).
- To tune behavior: adjust weights in `G1RoughEnvCfg.rewards` and observation noise in `ObservationsCfg`.

Hyperparameters & tuning checklist
- Environment: `scene.num_envs`, `episode_length_s`, `sim.dt`, `decimation` (see `velocity_env_cfg.py` and G1 overrides).
- Commands: `commands.base_velocity.ranges` (lin/ang velocities) in `G1*EnvCfg`.
- Rewards: weights inside `G1RoughEnvCfg` / `G1FlatEnvCfg`.
- Agent: `agent_cfg.max_iterations`, `agent_cfg.seed`, `agent_cfg.device`, learning rate, batch size (agent config depends on RL library).

Testing & smoke checks
- Unit tests for wrappers: run `pytest source/isaaclab_rl/test/test_rsl_rl_wrapper.py` and similar files under `source/isaaclab_rl/test/`.
- Smoke training (small): use reduced `--num_envs` and `--max_iterations` and confirm logs and `params/env.yaml` and `params/agent.yaml` are written.

Evaluation metrics
- Primary: `Train/mean_reward` (increasing, stable) and `Episode/mean_length`.
- Secondary: reward breakdown terms (e.g., `TrackVel`, `Penalty/torque`), action rate, and contact/termination counts.

Troubleshooting
- If policy fails to learn: reduce observation noise, simplify reward (start with velocity tracking only), or reduce randomization.
- If simulation instability: check `sim.dt`, `decimation`, and articulation `velocity_limit_sim` in robot asset configs.
- If actions saturate: verify wrapper action scaling and `clip_actions` in agent config.

Next steps
- Run a smoke training run and wrapper unit tests (I can run these if you want). Share this file if you want additions (threshold examples, CI job snippets).

---

File generated by automation to aid development. For edits, update this file in-place.

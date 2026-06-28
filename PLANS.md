# PLANS - G1 Rough Velocity Tracking

## Goal

Use the current Isaac Lab repository to learn and improve reinforcement learning for Unitree G1 locomotion, starting from `Isaac-Velocity-Rough-G1-v0`.

Primary objective:
- Improve velocity tracking.
- Reduce commanded vs actual velocity error.
- Reach useful performance in fewer training iterations.

Current priority:
1. Optimize forward linear velocity tracking, `vx`.
2. After `vx` is stable, add yaw angular velocity tracking, `wz`.

Scope:
- Simulation only.
- Still care about action smoothness and visually stable motion.
- No immediate sim-to-real requirement.

## User Constraints

- RL backend: `rsl_rl`.
- GPU: RTX 5070, 8 GB VRAM.
- Recommended starting `num_envs`: conservative because of 8 GB VRAM.
  - Smoke/debug: `64` or `128`.
  - Small training: `256`.
  - Main training candidate: `512`.
  - Try `1024` only if memory is stable.
- Desired behavior order:
  - First learn accurate forward tracking.
  - Then learn turning/yaw tracking.
  - Keep action smoothness penalties in the loop.

## Key Files

- Task registration:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py`
- G1 rough environment:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py`
- Shared velocity locomotion MDP:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- RSL-RL PPO config:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`
- Training entry point:
  - `scripts/reinforcement_learning/rsl_rl/train.py`
- Play/evaluation entry point:
  - `scripts/reinforcement_learning/rsl_rl/play.py`

## Baseline Commands

Smoke run:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 \
  --headless \
  --num_envs 64 \
  --max_iterations 100
```

Small baseline:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 300
```

Main baseline candidate for 8 GB VRAM:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 1000
```

If out of memory occurs, reduce `--num_envs` to `256`.

## Current G1 Rough Starting Point

In `rough_env_cfg.py`:

- `track_lin_vel_xy_exp`
  - function: `mdp.track_lin_vel_xy_yaw_frame_exp`
  - weight: `1.0`
  - std: `0.5`
- `track_ang_vel_z_exp`
  - function: `mdp.track_ang_vel_z_world_exp`
  - weight: `2.0`
  - std: `0.5`
- `action_rate_l2.weight = -0.005`
- `dof_acc_l2.weight = -1.25e-7`
- `dof_torques_l2.weight = -1.5e-7`
- Commands:
  - `lin_vel_x = (0.0, 1.0)`
  - `lin_vel_y = (0.0, 0.0)`
  - `ang_vel_z = (-1.0, 1.0)`

In `rsl_rl_ppo_cfg.py`:

- `num_steps_per_env = 24`
- `max_iterations = 3000`
- `init_noise_std = 1.0`
- `actor_obs_normalization = False`
- `critic_obs_normalization = False`
- network: `[512, 256, 128]`
- `entropy_coef = 0.008`
- `learning_rate = 1e-3`
- `schedule = "adaptive"`
- `desired_kl = 0.01`

## Plan

### Phase 0 - Reproducible Baseline

Run the unmodified task first.

Record:
- `Train/mean_reward`
- `Train/mean_episode_length`
- reward breakdown terms
- termination counts if available
- checkpoint path
- exact command
- seed
- `num_envs`
- iteration count

Do not tune before a baseline exists.

### Phase 1 - Add Direct Velocity Error Evaluation

Reward alone is not enough. Add or use an evaluation workflow that reports:

- mean absolute `vx` error: `abs(command_vx - actual_vx)`
- mean squared `vx` error
- mean absolute `wz` error after yaw tracking is enabled
- error split by command bins:
  - `vx` in `0.0-0.3`
  - `vx` in `0.3-0.7`
  - `vx` in `0.7-1.0`
- episode length
- fall/termination rate
- qualitative action smoothness from play videos or visual inspection

Preferred fixed-command play tests:

- `vx=0.3, wz=0.0`
- `vx=0.6, wz=0.0`
- `vx=1.0, wz=0.0`
- later: `vx=0.6, wz=0.5`
- later: `vx=0.6, wz=-0.5`

### Phase 2 - Forward Velocity Curriculum

Create an easier `vx`-first training variant before full rough velocity tracking.

Recommended command curriculum:

1. `vx` only, medium speed:
   - `lin_vel_x = (0.2, 0.8)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (0.0, 0.0)`
2. `vx` only, full forward range:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (0.0, 0.0)`
3. Add small yaw:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (-0.5, 0.5)`
4. Restore full yaw:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (-1.0, 1.0)`

### Phase 3 - Reward Tuning

Tune one group at a time.

Candidate experiments:

- Increase forward tracking importance:
  - `track_lin_vel_xy_exp.weight: 1.0 -> 2.0`
  - if stable, try `3.0`
- Reduce yaw tracking during `vx`-first phase:
  - `track_ang_vel_z_exp.weight: 2.0 -> 0.5` or `1.0`
- Make velocity tracking sharper:
  - `track_lin_vel_xy_exp.params["std"]: 0.5 -> 0.4`
  - later try `0.35`
- Preserve smoothness:
  - keep `action_rate_l2.weight` enabled
  - if actions are visibly jittery, try `-0.005 -> -0.01`
  - avoid making smoothness too strong before the robot can reliably walk

Evaluation rule:
- Prefer lower direct velocity error over higher reward if the two disagree.

### Phase 4 - PPO Efficiency Tuning

After the environment/reward variant has a useful learning signal, tune PPO.

Candidate experiments:

- `num_steps_per_env: 24 -> 32`
- enable observation normalization:
  - `actor_obs_normalization = True`
  - `critic_obs_normalization = True`
- keep network `[512, 256, 128]` initially
- keep `learning_rate = 1e-3` initially
- if learning is unstable, lower LR to `5e-4`
- if exploration remains too noisy late in training, reduce:
  - `entropy_coef: 0.008 -> 0.005`

For 8 GB VRAM, change only one of:
- `num_envs`
- network size
- `num_steps_per_env`

This makes memory and performance effects easier to interpret.

### Phase 5 - Experiment Matrix

Use explicit experiment names.

Suggested names:

- `g1_rough_baseline`
- `g1_rough_vx_only_cmd_02_08`
- `g1_rough_vx_only_cmd_00_10`
- `g1_rough_vx_track_w2`
- `g1_rough_vx_track_std04`
- `g1_rough_obsnorm`
- `g1_rough_vx_then_wz_05`
- `g1_rough_vx_then_wz_10`

For each run, record:

- git diff or exact code changes
- command used
- seed
- `num_envs`
- max iterations
- best checkpoint
- final checkpoint
- mean `vx` error
- mean `wz` error if applicable
- mean reward
- mean episode length
- observed smoothness issues

## Decision Rules

- If the robot falls often:
  - prioritize stability, termination rate, and episode length before chasing lower velocity error.
- If it walks but velocity lags:
  - increase linear tracking weight or reduce tracking `std`.
- If it tracks speed but jitters:
  - strengthen `action_rate_l2` moderately or inspect action scale.
- If it learns slowly but steadily:
  - try more envs if VRAM allows, or `num_steps_per_env = 32`.
- If reward improves but `vx` error does not:
  - treat the reward design as misaligned and tune direct tracking terms.

## Open Questions

- Decide whether to create separate config classes for curriculum experiments or use CLI/Hydra overrides.
- Decide the preferred logging method for direct velocity error metrics.
- Confirm maximum stable `num_envs` on RTX 5070 8 GB.


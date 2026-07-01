# PLANS - G1 Progressive Velocity Tracking

## Goal

Use the current Isaac Lab repository to learn and improve reinforcement learning for Unitree G1 locomotion.
Start from the simplest useful experiment, `Isaac-Velocity-Flat-G1-VxOnly-SymBiped-v0`, then progressively add speed range, yaw, and rough terrain.

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

First smoke run: flat map, `vx` only, reward/config validation.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-VxOnly-SymBiped-v0 \
  --headless \
  --num_envs 64 \
  --max_iterations 100
```

First learning run: flat map, `vx` only.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-VxOnly-SymBiped-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 1000
```

Rough map candidate only after flat `vx` passes:

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
  - base weight: `1.0`
  - rough override: `1.5`
  - std: `0.5`
- `track_ang_vel_z_exp`
  - function: `mdp.track_ang_vel_z_world_exp`
  - weight: `2.0`
  - std: `0.5`
- `termination_penalty.weight = -200.0`
- gait/contact shaping:
  - `feet_air_time.weight = 0.25`
  - `feet_air_time_symmetry.weight = -0.5`
  - `feet_contact_count.weight = -0.15`
  - `feet_gait_clock.weight = 0.5`
  - `feet_close.weight = -0.4`
  - `feet_slide.weight = -0.1`
- stability/smoothness:
  - `lin_vel_z_l2.weight = -0.5`
  - `flat_orientation_l2.weight = -1.0`
  - `ang_vel_xy_l2.weight = -0.25`
  - `action_rate_l2.weight = -0.01`
  - `dof_acc_l2.weight = -2.0e-7`
- `dof_torques_l2.weight = -1.5e-7`
- Commands:
  - `lin_vel_x = (0.0, 1.0)`
  - `lin_vel_y = (0.0, 0.0)`
  - `ang_vel_z = (-1.0, 1.0)`

## Current Flat Vx Starting Point

The first experiment should use `Isaac-Velocity-Flat-G1-VxOnly-SymBiped-v0`.

In `flat_env_cfg.py`, `G1FlatVxOnlyEnvCfg` currently sets:

- Terrain:
  - flat plane
  - no height scan
  - no terrain curriculum
- Commands:
  - `lin_vel_x = (0.2, 0.8)`
  - `lin_vel_y = (0.0, 0.0)`
  - `ang_vel_z = (0.0, 0.0)`
  - `resampling_time_range = (12.0, 12.0)`
- Key rewards:
  - `track_lin_vel_xy_exp.weight = 2.4`
  - `track_ang_vel_z_exp.weight = 0.5`
  - `feet_air_time.weight = 1.5`
  - `feet_contact_count.weight = -0.25`
  - `feet_gait_clock.weight = 0.5`
  - `feet_close.weight = -0.4`
  - `action_rate_l2.weight = -0.012`
  - `torso_height_l2.weight = -1.25`
- Training scale:
  - `scene.num_envs = 512`
  - `actions.joint_pos.scale = 0.35`

In `rsl_rl_ppo_cfg.py`:

- `num_steps_per_env = 32`
- `max_iterations = 3000`
- `init_noise_std = 0.8`
- `actor_obs_normalization = False`
- `critic_obs_normalization = False`
- network: `[512, 256, 128]`
- `entropy_coef = 0.004`
- `learning_rate = 5e-4`
- `schedule = "adaptive"`
- `desired_kl = 0.008`

## Optimized Experiment Strategy

The experiment must move from simple to complex so reward effects are visible before terrain and yaw add noise.
The optimized approach is:

1. Start with flat terrain and `vx` only to validate whether the reward set produces clean forward walking.
2. Freeze PPO while validating reward behavior; only change reward groups one at a time.
3. Increase task complexity in this order: flat medium `vx`, flat full `vx`, flat small yaw, rough `vx`, rough yaw.
4. Add rough terrain only after fixed-command flat `vx` tests are reliable.
5. Tune command curriculum first, reward weights second, PPO last.
6. Treat direct velocity error, fall rate, gait quality, and foot crossing as the decision metrics, not total reward alone.

## Plan

### Phase 0 - Flat Vx Reward Smoke

Run the flat `vx`-only task at small scale to catch config/reward mistakes before spending GPU time.

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
- any NaN, reward explosion, or immediate termination

Suggested first command:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-VxOnly-SymBiped-v0 \
  --headless \
  --num_envs 64 \
  --max_iterations 100
```

Pass criteria:
- no runtime errors from reward terms
- mean episode length is not near zero
- `feet_gait_clock` and `feet_close` reward magnitudes are visible but not dominating total reward

### Phase 1 - Flat Vx Reward Validation

Train the existing flat `vx`-only variant long enough to see whether the reward set learns stable forward walking.
Do not add rough terrain or yaw yet.

Suggested command:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-VxOnly-SymBiped-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 1000
```

Decision:
- If it cannot walk on flat, simplify or retune reward before any curriculum expansion.
- If it walks but shuffles or hops, tune gait/contact terms in Phase 1A.
- If it walks and tracks `vx`, continue to Phase 2.

### Phase 1A - Flat Reward Ablations

Use flat `vx` only to understand whether each gait reward helps.
Run short paired experiments, changing one thing at a time:

1. `feet_gait_clock.weight`: `0.5 -> 0.25`
2. `feet_gait_clock.weight`: `0.5 -> 0.75`
3. reduce redundant gait penalties:
   - `feet_air_time_symmetry.weight: -0.5 -> -0.25`
   - `feet_contact_count.weight: -0.15 -> -0.05`
4. anti-trip sensitivity:
   - `feet_close.weight: -0.4 -> -0.6` if feet still cross
   - `distance_threshold: 0.08 -> 0.10` only after visual confirmation

Keep the best flat `vx` reward set as the baseline for all later phases.

### Phase 1.5 - Direct Velocity Error Evaluation

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

Pass criteria before adding yaw:
- `vx=0.3`: stable for most episodes, no persistent foot crossing
- `vx=0.6`: mean absolute `vx` error below a chosen threshold, initially target `< 0.20 m/s`
- `vx=1.0`: does not collapse into hopping or excessive torso pitch

### Phase 2 - Flat Command Curriculum

After flat medium-speed `vx` works, expand commands on flat terrain first.

Recommended command stages:

1. Flat `vx` only, medium speed:
   - `lin_vel_x = (0.2, 0.8)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (0.0, 0.0)`
2. Flat `vx` only, full forward range:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (0.0, 0.0)`
3. Flat `vx` with small yaw:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (-0.5, 0.5)`
4. Flat `vx` with full yaw:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (-1.0, 1.0)`

Only advance when the current stage passes fixed-command evaluation.

### Phase 3 - Rough Curriculum

Move to rough terrain only after flat command curriculum succeeds.

Recommended rough stages:

1. Rough `vx` only, medium speed:
   - `lin_vel_x = (0.2, 0.8)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (0.0, 0.0)`
2. Rough `vx` only, full forward range:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (0.0, 0.0)`
3. Rough `vx` with small yaw:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (-0.5, 0.5)`
4. Rough full task:
   - `lin_vel_x = (0.0, 1.0)`
   - `lin_vel_y = (0.0, 0.0)`
   - `ang_vel_z = (-1.0, 1.0)`

### Phase 4 - Reward Simplification and Tuning

Tune one group at a time, preferably on flat `vx` first. Only retune on rough if the issue appears only on rough terrain.

Priority order:

1. Task tracking:
   - keep `track_lin_vel_xy_exp.weight` in `1.5-2.5`
   - try `std: 0.5 -> 0.4` only after the robot already walks
2. Gait shaping:
   - keep `feet_gait_clock.weight = 0.5` initially
   - if steps become too clock-bound or unnatural, try `0.25`
   - if gait is irregular but stable, try `0.75`
3. Redundant contact terms:
   - try reducing `feet_air_time_symmetry.weight` from `-0.5` to `-0.25`
   - try reducing `feet_contact_count.weight` from `-0.15` to `-0.05`
   - remove only after the reward breakdown shows gait clock is working
4. Anti-trip:
   - keep `feet_close.weight = -0.4`
   - adjust `distance_threshold` after visual inspection; `0.08 m` is a starting point, not a truth
5. Smoothness:
   - keep `action_rate_l2.weight = -0.01`
   - if actions are visibly jittery, try `-0.015`
   - avoid stronger smoothness before the robot can reliably walk
6. Low-value regularizers:
   - consider disabling `joint_deviation_fingers`
   - consider reducing arm penalties once arms are visually stable
   - leave `torso_height_l2` disabled on rough terrain unless posture collapses

Evaluation rule:
- Prefer lower direct velocity error over higher reward if the two disagree.

### Phase 5 - PPO Efficiency Tuning

After the environment/reward variant has a useful learning signal, tune PPO.

Candidate experiments:

- keep current `num_steps_per_env = 32` as baseline
- enable observation normalization:
  - `actor_obs_normalization = True`
  - `critic_obs_normalization = True`
- keep network `[512, 256, 128]` initially
- keep current `learning_rate = 5e-4` initially
- if learning is too slow and KL remains low, try `1e-3`
- if exploration remains too noisy late in training, reduce:
  - `entropy_coef: 0.004 -> 0.002`

For 8 GB VRAM, change only one of:
- `num_envs`
- network size
- `num_steps_per_env`

This makes memory and performance effects easier to interpret.

### Phase 6 - Experiment Matrix

Use explicit experiment names.

Suggested names:

- `g1_flat_vx_smoke_reward`
- `g1_flat_vx_reward_baseline`
- `g1_flat_vx_gait_clock_w025`
- `g1_flat_vx_gait_clock_w075`
- `g1_flat_vx_reduce_redundant_gait`
- `g1_flat_vx_full_00_10`
- `g1_flat_vx_yaw_05`
- `g1_rough_vx_only_cmd_02_08`
- `g1_rough_vx_only_cmd_00_10`
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
- If the robot survives but shuffles in place:
  - increase velocity tracking priority or sharpen `std`; reduce excessive gait/contact regularization.
- If the robot hops:
  - reduce air-time pressure; keep gait clock, contact count, and `lin_vel_z_l2` active.
- If feet cross or collide:
  - increase `feet_close.weight` magnitude or threshold slightly, then verify visually.
- If reward improves but fixed-command velocity error worsens:
  - reject the change.
- If a change only improves one command speed but hurts others:
  - keep it only for curriculum bootstrap, not final rough config.
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
- Decide whether to add explicit registered flat curriculum variants for full `vx` and small-yaw stages.

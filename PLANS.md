# PLANS - G1 Flat Omni-Human Locomotion

## Goal

Train Unitree G1 to perform human-like omni-directional locomotion on flat terrain:

- forward and backward walking
- left and right side-stepping
- turning in place
- mixed translation and yaw commands
- speed-dependent motion style:
  - low speed: small arm and leg amplitude
  - high speed: larger, coordinated arm and leg amplitude
- final validation through Isaac Lab to MuJoCo sim-to-sim

The previous `vx`-only optimization path is no longer the main objective. `vx` tests remain only as evaluation points inside the omni-directional task.

## Current Lessons

- AntiHop A is the current best contact baseline:
  - `feet_air_time.weight = 0.8`
  - `feet_air_time.threshold = 0.3`
  - `feet_contact_count.weight = -0.35`
  - jump/hop behavior is visibly reduced
- Strong vertical/posture penalties from AntiHop B can improve scalar logs while making the visual gait worse.
- Weakening the gait clock too much makes contact timing loose and reintroduces hopping.
- Turn-in-place needs yaw-aware contact shaping because xy-only contact rewards do not activate for `vx=0, vy=0, wz!=0`.

## Key Files

- Task registration:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py`
- Flat G1 configs:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py`
- Shared G1 rewards:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py`
- Custom MDP rewards and observations:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- RSL-RL PPO config:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`

## New Tasks

Phase-1 conservative omni task:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-OmniHuman-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 3000
```

Full-range omni task:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-OmniHuman-Full-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 4000
```

Sim-to-real policy comparison tasks:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-OmniHuman-Sim2Real-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 4000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-OmniHuman-GRU-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 4000
```

Use `256` envs if 8 GB VRAM is unstable. Try `1024` only after memory is known to be safe.

## Reward Strategy

Tracking:

- `track_lin_vel_xy_exp`: all planar commands
- `track_ang_vel_z_exp`: yaw-rate tracking
- `heading_command = False` for true yaw-rate control

Contact and anti-hop:

- keep AntiHop-A-style air-time/contact balance
- use parameterized contact-count activation:
  - `xy` for legacy planar-only walking
  - `omni` for planar plus turn-in-place commands
- use parameterized omni gait clock routing:
  - full clock weight for forward/backward and mixed walking
  - reduced clock pressure for side-step
  - reduced clock pressure for turn-in-place

Human-like style:

- expose `sin(phase), cos(phase)` to the policy
- relax arm default penalties
- reward coordinated shoulder pitch with opposite hip pitch, with lower weight for side-step and turn-in-place
- reward phase-varying arm swing amplitude that grows with command speed, avoiding static arm-offset solutions

Stability and sim-to-sim readiness:

- keep moderate base vertical/roll-pitch penalties
- keep torque-rate and power penalties small
- use observation normalization for sim-to-real comparison runners
- compare MLP and GRU policies before MuJoCo export
- add domain randomization only after stable flat omni behavior is learned

## Training Phases

### Phase 0 - Config Smoke

Run a short smoke test:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-OmniHuman-v0 \
  --headless \
  --num_envs 64 \
  --max_iterations 100
```

Pass criteria:

- task registers correctly
- phase clock observation does not break observation construction
- custom reward terms do not throw shape or joint-name errors
- episode length is not near zero

### Phase 1 - Conservative Omni Base

Train `Isaac-Velocity-Flat-G1-OmniHuman-v0`.

Command range:

```python
vx = (-0.3, 0.8)
vy = (-0.25, 0.25)
wz = (-0.4, 0.4)
```

Success criteria:

- no obvious hopping
- no episode-end fall during turn-in-place
- stable forward/backward/side-step/turn behavior
- arms begin to move with speed-dependent amplitude without violent flailing

### Phase 2 - Style and Routing Tuning

Tune only one group at a time:

1. arm swing coordination
2. arm amplitude schedule
3. omni gait clock routing
4. yaw-aware contact penalty

Avoid increasing base tracking weights before contact and style are visually acceptable.

### Phase 3 - Full Command Expansion

Train or fine-tune `Isaac-Velocity-Flat-G1-OmniHuman-Full-v0`.

Command range:

```python
vx = (-0.6, 1.2)
vy = (-0.5, 0.5)
wz = (-0.6, 0.6)
```

Success criteria:

- all fixed evaluation points survive at least 20 seconds
- mixed commands remain stable
- high-speed commands show larger but smooth arm/leg amplitude

### Phase 4 - MuJoCo Sim-to-Sim

Export the trained policy and validate in MuJoCo.

If MuJoCo fails while Isaac Lab succeeds, tune in this order:

1. foot slide and contact penalties
2. power and torque-rate penalties
3. action delay / randomization
4. friction and motor strength randomization

Do not solve MuJoCo mismatch by blindly increasing velocity tracking reward.

## Evaluation Set

Use fixed commands:

```text
stand:      vx=0.0,  vy=0.0,   wz=0.0
forward:    vx=0.3,  vx=0.8,   vx=1.2
backward:   vx=-0.3, vx=-0.6
side-left:  vy=0.25, vy=0.5
side-right: vy=-0.25, vy=-0.5
turn-left:  wz=0.3,  wz=0.6
turn-right: wz=-0.3, wz=-0.6
mixed:      vx=0.6, vy=±0.25, wz=±0.3
```

Record:

- fall or no fall
- time to fall
- visible hopping or double-flight
- `error_vel_xy`
- `error_vel_yaw`
- foot slide
- arm swing amplitude at low and high speed
- Isaac Lab vs MuJoCo behavior difference

## References

- Humanoid-Gym: reward/contact design, domain randomization, Isaac Gym to MuJoCo sim-to-sim.
- Gait-Conditioned RL: reward routing, human-inspired natural walking, Unitree G1 validation.
- HiLo: human-like locomotion needs style/motion guidance beyond pure velocity tracking.
- Arm-motion RL: arm swing can regulate angular momentum and improve whole-body balance.

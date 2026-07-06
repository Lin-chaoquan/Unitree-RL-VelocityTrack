# G1 OmniHuman Session Export

Date: 2026-07-05

## Current Task

Task id:

```text
Isaac-Velocity-Flat-G1-OmniHuman-v0
```

Goal:

- Flat terrain omni-directional G1 locomotion.
- Commands: forward/backward, lateral motion, yaw turn, mixed commands.
- Human-like style: low-speed small arm/leg motion, faster commands with larger swing.
- Prepare for later MuJoCo sim-to-sim and possible sim-to-real.

## Current Problem

Latest training result:

- Basic omni commands can be followed.
- Lateral arm swing issue has already been removed.
- Remaining issue: arms stay in a static asymmetric pose, usually right arm forward and left arm backward.
- Legs still have mild stomp-down behavior, but the current patch focuses only on the frozen arm-pose failure.

Root cause found in reward:

- `arm_swing_opposite_leg_phase` and `arm_swing_opposite_foot_phase` previously accepted both signs every step.
- That allowed a static anti-symmetric arm pose to receive reward in both gait half-cycles.
- The fix is to use one fixed sign convention through a configurable `phase_sign`.

## Files Changed In This Session

Main reward implementation:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

Main task config:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py
```

Shared G1 reward config:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py
```

## Reward Patch Summary

`arm_swing_opposite_leg_phase`:

- Added/kept parameter:

```python
phase_sign: float = 1.0
```

- Internal gait direction variable was renamed to avoid shadowing `phase_sign`.
- Target now uses fixed sign:

```python
target = phase_sign * target_amplitude.unsqueeze(1) * leg_phase
reward = torch.exp(-torch.sum(torch.square(arm - target), dim=1) / std**2)
```

`arm_swing_opposite_foot_phase`:

- Added/kept parameter:

```python
phase_sign: float = 1.0
```

- Internal foot direction variable was renamed to avoid shadowing `phase_sign`.
- Target now uses fixed sign:

```python
target = phase_sign * target_amplitude.unsqueeze(1) * phase
reward = torch.exp(-torch.sum(torch.square(arm - target), dim=1) / std**2)
```

Important behavior:

- Static right-front/left-back arm pose should no longer satisfy both half-cycles.
- If arms start swinging with the wrong left/right phase, flip both task-level `phase_sign` values from `1.0` to `-1.0`.

## Current OmniHuman Arm Config

Current task-level arm settings in `G1FlatOmniHumanEnvCfg`:

```python
self.rewards.joint_deviation_arms.weight = 0.0
self.rewards.joint_vel_arms.weight = 0.0
self.rewards.joint_deviation_arm_swing.weight = -0.005
self.rewards.joint_vel_arm_swing.weight = 0.0
self.rewards.joint_deviation_arm_aux.weight = -0.22
self.rewards.joint_vel_arm_aux.weight = -0.012
self.rewards.dof_pos_limits_arms.weight = -0.5

self.rewards.arm_swing_coordination.weight = 0.0

self.rewards.arm_swing_opposite_leg_phase.weight = 0.35
self.rewards.arm_swing_opposite_leg_phase.params["min_amplitude"] = 0.12
self.rewards.arm_swing_opposite_leg_phase.params["max_amplitude"] = 0.55
self.rewards.arm_swing_opposite_leg_phase.params["leg_phase_scale"] = 0.22
self.rewards.arm_swing_opposite_leg_phase.params["leg_velocity_scale"] = 0.8
self.rewards.arm_swing_opposite_leg_phase.params["leg_velocity_weight"] = 0.6
self.rewards.arm_swing_opposite_leg_phase.params["min_phase_magnitude"] = 0.45
self.rewards.arm_swing_opposite_leg_phase.params["phase_sign"] = 1.0
self.rewards.arm_swing_opposite_leg_phase.params["std"] = 0.32

self.rewards.arm_swing_opposite_foot_phase.weight = 0.9
self.rewards.arm_swing_opposite_foot_phase.params["min_amplitude"] = 0.16
self.rewards.arm_swing_opposite_foot_phase.params["max_amplitude"] = 0.75
self.rewards.arm_swing_opposite_foot_phase.params["foot_phase_scale"] = 0.16
self.rewards.arm_swing_opposite_foot_phase.params["min_phase_magnitude"] = 0.5
self.rewards.arm_swing_opposite_foot_phase.params["phase_sign"] = 1.0
self.rewards.arm_swing_opposite_foot_phase.params["std"] = 0.36

self.rewards.arm_swing_amplitude_schedule.weight = 0.0
self.rewards.arm_swing_sagittal_velocity.weight = 0.0
self.rewards.arm_swing_pose_envelope.weight = 0.0
```

Arm joint split:

```python
G1_ARM_SWING_JOINTS = [".*_shoulder_pitch_joint"]

G1_ARM_AUX_JOINTS = [
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_roll_joint",
]
```

## Current OmniHuman Locomotion Config

Command range:

```python
self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.8)
self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
self.commands.base_velocity.heading_command = False
```

Contact/gait shaping:

```python
self.rewards.feet_air_time.weight = 0.3
self.rewards.feet_air_time.params["threshold"] = 0.2
self.rewards.feet_contact_count.weight = -0.35
self.rewards.feet_contact_count_yaw.weight = -0.25
self.rewards.feet_gait_clock.weight = 0.0
self.rewards.feet_gait_clock_omni.weight = 0.6
self.rewards.feet_gait_clock_omni.params["period"] = 0.75
self.rewards.feet_slide.weight = -0.2
self.rewards.feet_close.weight = -0.4
self.rewards.feet_touchdown_velocity.weight = -0.14
self.rewards.feet_swing_height_trajectory.weight = 0.45
self.rewards.feet_swing_vertical_velocity.weight = -0.02
```

Tracking/posture/effort:

```python
self.rewards.track_lin_vel_xy_exp.weight = 2.0
self.rewards.track_ang_vel_z_exp.weight = 1.2
self.rewards.lin_vel_z_l2.weight = -0.3
self.rewards.ang_vel_xy_l2.weight = -0.45
self.rewards.flat_orientation_l2.weight = -1.5
self.rewards.torso_height_l2.weight = -1.25
self.rewards.action_rate_l2.weight = -0.012
self.rewards.dof_acc_l2.weight = -2.5e-7
self.rewards.dof_torque_rate_l2.weight = -2.5e-7
self.rewards.dof_power_abs.weight = -2.0e-5
self.rewards.stand_still.weight = -0.35
```

## Recommended Next Experiment

Start with resume training from the latest OmniHuman checkpoint for 500 to 800 iterations.

Suggested command template:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-OmniHuman-v0 \
  --resume \
  --load_run <latest_omnihuman_run> \
  --checkpoint <checkpoint_name>
```

Evaluation points:

```text
stand: vx=0.0, vy=0.0, wz=0.0
forward: vx=0.3/0.8, vy=0.0, wz=0.0
backward: vx=-0.3, vy=0.0, wz=0.0
side: vx=0.0, vy=+/-0.25, wz=0.0
turn: vx=0.0, vy=0.0, wz=+/-0.4
mixed: vx=0.5, vy=+/-0.2, wz=+/-0.3
```

Pass criteria for this patch:

- Arms no longer remain fixed as right-front/left-back.
- Shoulder pitch shows visible periodic forward/backward swing during forward walking.
- Elbow and shoulder roll/yaw remain relatively quiet.
- Basic omni tracking does not collapse.
- Stomp-down does not become worse.

## If The Next Result Is Still Bad

If arms swing but same-side phase looks wrong:

```python
self.rewards.arm_swing_opposite_leg_phase.params["phase_sign"] = -1.0
self.rewards.arm_swing_opposite_foot_phase.params["phase_sign"] = -1.0
```

If arms are still static:

- Increase `arm_swing_opposite_foot_phase.weight` from `0.9` to `1.2`.
- Reduce `joint_deviation_arm_aux.weight` from `-0.22` to `-0.16` only if auxiliary joints look over-constrained.
- Keep `joint_deviation_arm_swing.weight` small, around `-0.005` to `-0.01`.
- Do not re-enable clock-only arm swing first; prefer real foot/leg phase.

If stomp-down remains the main issue after arms improve:

- Keep arm rewards fixed.
- Tune foot terms next:

```python
self.rewards.feet_touchdown_velocity.weight = -0.18
self.rewards.feet_swing_height_trajectory.weight = 0.35
self.rewards.feet_swing_vertical_velocity.weight = -0.03
```

The intent is to reduce hard touchdown without asking the robot to over-lift the feet.

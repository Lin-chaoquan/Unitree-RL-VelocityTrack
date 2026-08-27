#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/workspace/code
ISAACLAB_ROOT=/workspace/isaaclab
ISAACLAB_PYTHON="$ISAACLAB_ROOT/_isaac_sim/python.sh"
LOCAL_OUTPUT=/workspace/local_output
OSS_OUTPUT=/workspace/output

export ISAACLAB_PATH="$ISAACLAB_ROOT"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TZ="${TZ:-UTC}"

G1_TASK="${G1_TASK:-Isaac-Velocity-Flat-G1-29DOF-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-aliyun_g1_29dof_flat}"

echo "[g1-flat] starting"
echo "[g1-flat] task=$G1_TASK num_envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS seed=$SEED"
"$ISAACLAB_PYTHON" --version
nvidia-smi || true

# DLC may inject these variables even for a single-GPU job. IsaacLab would then
# incorrectly enter distributed mode, so remove them for this configuration.
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE

mkdir -p "$LOCAL_OUTPUT" "$OSS_OUTPUT"
cd "$CODE_ROOT"

# Keep the image's train.py, isaaclab_rl, and dependency versions together, but
# overlay the repository-local G1 29-DOF model and task definitions. These
# copies disappear with the Pod and never alter the ACR image.
ASSETS_SRC="$CODE_ROOT/source/isaaclab_assets/isaaclab_assets/robots"
ASSETS_DST="$ISAACLAB_ROOT/source/isaaclab_assets/isaaclab_assets/robots"
TASKS_SRC="$CODE_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
TASKS_DST="$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"

required_files=(
  "$ASSETS_SRC/unitree.py"
  "$ASSETS_SRC/g1_29dof/g1_29dof.usd"
  "$TASKS_SRC/config/g1/__init__.py"
  "$TASKS_SRC/config/g1/g1_29dof_env_cfg.py"
  "$TASKS_SRC/config/g1/agents/rsl_rl_ppo_cfg.py"
)
for file in "${required_files[@]}"; do
  if [[ ! -s "$file" ]]; then
    echo "[g1-flat] ERROR: required G1 29-DOF file is missing or empty: $file" >&2
    exit 1
  fi
done

mkdir -p "$ASSETS_DST/g1_29dof" "$TASKS_DST/config/g1"
cp "$ASSETS_SRC/unitree.py" "$ASSETS_DST/unitree.py"
cp -a "$ASSETS_SRC/g1_29dof/." "$ASSETS_DST/g1_29dof/"
cp -a "$TASKS_SRC/config/g1/." "$TASKS_DST/config/g1/"
cp "$TASKS_SRC/mdp/rewards.py" "$TASKS_DST/mdp/rewards.py"

cmp -s "$ASSETS_SRC/unitree.py" "$ASSETS_DST/unitree.py"
cmp -s "$ASSETS_SRC/g1_29dof/g1_29dof.usd" "$ASSETS_DST/g1_29dof/g1_29dof.usd"
cmp -s "$TASKS_SRC/config/g1/g1_29dof_env_cfg.py" "$TASKS_DST/config/g1/g1_29dof_env_cfg.py"
cmp -s "$TASKS_SRC/config/g1/agents/rsl_rl_ppo_cfg.py" "$TASKS_DST/config/g1/agents/rsl_rl_ppo_cfg.py"
cmp -s "$TASKS_SRC/mdp/rewards.py" "$TASKS_DST/mdp/rewards.py"
echo "[g1-flat] local G1 29-DOF model and task files overlaid into job container"

# RSL-RL writes logs relative to the current directory. Keep iteration-time
# logging and checkpoints on the container filesystem, then upload them to the
# OSS mount once when the process exits. The trap also preserves partial logs
# when training fails normally.
sync_output() {
  local training_status=$?
  trap - EXIT

  echo "[g1-flat] syncing local output to OSS (training_status=$training_status)"
  if ! cp -a "$LOCAL_OUTPUT/." "$OSS_OUTPUT/"; then
    echo "[g1-flat] ERROR: failed to sync output to OSS" >&2
    exit 1
  fi
  echo "[g1-flat] output sync complete"
  exit "$training_status"
}
trap sync_output EXIT

cd "$LOCAL_OUTPUT"

# Use the image's matching train.py and isaaclab_rl implementation, just like the
# known-good bctrain IsaacLab example.
"$ISAACLAB_PYTHON" "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task "$G1_TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --seed "$SEED" \
  --run_name "$RUN_NAME"

# Some Isaac Sim python.sh wrappers can return zero even after an internal Python
# traceback. Treat a run with no RSL-RL artifacts as failed.
FIRST_RSL_RL_OUTPUT="$(find "$LOCAL_OUTPUT/logs/rsl_rl" -type f -print -quit 2>/dev/null || true)"
if [[ -z "$FIRST_RSL_RL_OUTPUT" ]]; then
  echo "[g1-flat] ERROR: training produced no files under $LOCAL_OUTPUT/logs/rsl_rl" >&2
  exit 1
fi

echo "[g1-flat] done"

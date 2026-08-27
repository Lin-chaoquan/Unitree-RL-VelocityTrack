# G1 Flat on bctrain

This directory submits the repository's current G1 Flat task implementation to
Alibaba Cloud PAI DLC through `bctrainctl`.

## Before submitting

The Alibaba Cloud image, quota, resources, OSS mounts, and NAS mount in `job.yaml`
are copied directly from `bctrain/bctrainctl-v0.2.0-main.7/examples/isaaclab/job.yaml`.
Only the uploaded code, entrypoint, and G1 Flat training parameters differ.

The checked-in defaults run a small smoke test with 256 environments and 10 PPO
iterations. The launcher uses the image's matching `train.py`, `isaaclab_rl`, and
RSL-RL versions, just like the known-good bctrain example. Before training, it
overlays only the repository's modified G1 files (`flat_env_cfg.py`,
`rough_env_cfg.py`, and velocity `mdp/rewards.py`) into the current job container.
The copies are ephemeral and do not alter the registry image or other DLC jobs.

Do not import `isaaclab_tasks` from a standalone Python check before launching the
training script. IsaacLab modules require Isaac Sim's `AppLauncher` to initialize
Omniverse modules such as `omni.timeline`; the official `train.py` performs that
initialization in the required order.

## Submit the smoke test

Run from the IsaacLab repository root:

```bash
source .venv-bctrainctl/bin/activate
bctrainctl submit -f bctrain_g1_flat/job.yaml --dry-run
bctrainctl submit -f bctrain_g1_flat/job.yaml
bctrainctl logs <job_id> --follow --lines 200
```

The job succeeds when training iterations run and files appear under:

```text
/workspace/output/logs/rsl_rl/g1_flat/
```

That directory is persisted to the OSS path configured in `job.yaml`.

## Full training

After the smoke test succeeds, update `spec.runtime.env` in `job.yaml`, for example:

```yaml
NUM_ENVS: "2048"
MAX_ITERATIONS: "1500"
RUN_NAME: aliyun_full_01
```

The copied example output prefix is shared with its Cartpole example. If GPU memory
is sufficient, increase `NUM_ENVS` to `4096`; otherwise reduce it to `1024` or `512`.

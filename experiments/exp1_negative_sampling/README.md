# Experiment 1: Negative Sampling Ablation

**Objective:**
Test whether proof-based corruption (corrupting a single leaf node in a proof tree to invalidate the chain) removes shallow statistical biases and improves the robustness (lowers False Positive Rate) over random or simply constrained negative sampling.

**Datasets Used:**

- Built using `family-tree.ttl`.
- Data yields are housed in `data/exp1/random`, `data/exp1/constrained`, and `data/exp1/proof_based`.
- And a frozen target evaluation testing dataset at `data/exp1/test_set`.

**Configurations:**

For generation, Hydra overrides append `generator.negative_sampling.strategy={strategy}` and `dataset.output_dir=data/exp1/{strategy}`.
For RRN training, `data.dataset.path=cwd/data/exp1/{strategy}` overrides point it to appropriately generated sets. `WandB` is structured with `logger.name`, `logger.group`, and `logger.tags`.

**Execution Commands:**

1. `uv run invoke exp1-generate-trainval-sets`
2. `uv run invoke exp1-generate-test-set` _(Generates the frozen "near-miss" hard negative test set)_
3. `uv run invoke exp1-train-rrn --strategy="random"`
4. `uv run invoke exp1-train-rrn --strategy="constrained"`
5. `uv run invoke exp1-train-rrn --strategy="proof_based"`

## Running on HPC (LSF)

This section provides concrete commands to run Exp 1 on the DTU-style LSF cluster (queue `gpua100`).

### 1. Prepare environment and generate datasets (run once)

```bash
cd /path/to/synthology

module load python3/3.9.19
module load cuda/11.7

source .env
source .venv/bin/activate

uv sync

# Generate all train/val variants (random, constrained, proof_based)
uv run invoke exp1-generate-trainval-sets

# Generate frozen hard-negative test set
uv run invoke exp1-generate-test-set
```

### 2. Submit RRN training jobs (one job per strategy)

The repository contains reusable LSF scripts in `jobscripts/` with the `expx-...` prefix:

```bash
# Submit from repo root
bsub < jobscripts/expx-exp1-train-random.sh
bsub < jobscripts/expx-exp1-train-constrained.sh
bsub < jobscripts/expx-exp1-train-proof-based.sh
```

Each script already sets:

- queue/resources (`gpua100`, 1 GPU, 32 GB RAM, 24h)
- module loading and environment activation
- `uv sync`
- dedicated Hydra config for the strategy

Hydra configs used by the scripts:

- `configs/rrn/exp1_random_hpc.yaml`
- `configs/rrn/exp1_constrained_hpc.yaml`
- `configs/rrn/exp1_proof_based_hpc.yaml`

### 3. Monitor and inspect jobs

```bash
# Show your queued/running jobs
bjobs

# Follow a specific job's stdout
tail -f logs/exp1_random_<JOB_ID>.out
```

### 4. Notes

- The jobscripts run `rrn.train` with strategy-specific Hydra config files (`--config-name=...`).
- If your cluster blocks outbound access to W&B, set `logger.offline: true` in the corresponding Hydra config file.
- If your cluster requires a different queue/account/email policy, adjust the `bsub` options accordingly.

**Expected Results:**

- Tracking False Positive Rate (FPR) on near-misses.
- Proof-based strategy should demonstrate improved robustness vs. the random baseline.
- All metrics are logged to Weights & Biases under the group `exp1_negative_sampling`.

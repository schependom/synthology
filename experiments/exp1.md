# Experiment 1: Negative Sampling Ablation

## Objective

This experiment isolates the effect of negative sampling strategy on RRN learning quality,
holding all other dataset properties fixed. Using the Family Tree ontology, four Synthology
variants are generated that differ **only** in how negative samples are produced:

| Strategy | Description |
|---|---|
| `random` | Subject or object corrupted with an arbitrary individual. |
| `constrained` | Corruption restricted to individuals satisfying the predicate's declared domain/range. |
| `proof_based` | A base fact in the proof tree is perturbed and the error propagated upward, producing a structurally near-perfect near-miss negative. |
| `mixed` | 50 % constrained + 50 % proof-based corruption blended within each batch. |

Each variant consists of 2 000 training, 500 validation, and 500 test knowledge graphs.
A **single frozen hard-negative test set** (generated with elevated proof-depth limits,
rich in near-miss targets) is shared across all four strategies to ensure a fair comparison.

**Expected outcome:** Proof-based negatives should produce the best PR-AUC, AUC-ROC, F1,
and positive-class accuracy, because they force the model to trace full reasoning chains
rather than rejecting triples on surface features. The mixed strategy is expected to
underperform due to training-signal inconsistency: exposing the model to qualitatively
different negatives within a single batch prevents a stable decision boundary from forming.

**Primary metrics:** PR-AUC (↑) and FPR (↓).

---

## Preflight

```bash
uv sync
uv run invoke --list
```

Exp 1 is CPU-only (no Jena). No Java/Maven setup required.

---

## Artifacts and Logging

- Canonical run records are written to `reports/experiment_runs/<YYYY-MM-DD>/exp1/<task>/<timestamp>/`.
- Each archive includes `manifest.json`, `run.log`, copied configs, and copied artifacts.
- RRN runs also write to Weights & Biases and the archive's `wandb/`, `lightning_logs/`,
  and `checkpoints/` subdirectories.

---

## Datasets Used

- Ontology: `ontologies/family.ttl` (OWL 2 RL translation used throughout the pipeline).
- Train/val datasets (one per strategy):
  - `data/exp1/random/`
  - `data/exp1/constrained/`
  - `data/exp1/proof_based/`
- Frozen hard-negative test set (shared): `data/exp1/test_set/`
- Per-split output format:
  - `facts.csv` — base context facts
  - `targets.csv` — positive and negative targets

---

## Configurations

**Generation configs:**

- `configs/ont_generator/exp1_random.yaml`
- `configs/ont_generator/exp1_constrained.yaml`
- `configs/ont_generator/exp1_proof_based.yaml`
- `configs/ont_generator/exp1_test.yaml`

**RRN dataset configs:**

- `configs/rrn/data/dataset/exp1_random.yaml`
- `configs/rrn/data/dataset/exp1_constrained.yaml`
- `configs/rrn/data/dataset/exp1_proof_based.yaml`

**WandB logging convention:**

- `logger.group=exp1_negative_sampling`
- `logger.name=exp1_<strategy>`
- `logger.tags=[exp1, <strategy>]`

---

## Execution Commands

### 1. Generate all train/val variants

```bash
uv run invoke exp1-generate-trainval-sets
```

### 2. Generate and freeze the hard-negative test set

```bash
uv run invoke exp1-generate-test-set
```

### 3. Train one model per strategy

```bash
uv run invoke exp1-train-rrn --strategy="random"
uv run invoke exp1-train-rrn --strategy="constrained"
uv run invoke exp1-train-rrn --strategy="proof_based"
```

**Expected outputs:**

- Datasets in `data/exp1/{random,constrained,proof_based}/` and `data/exp1/test_set/`
- Training and evaluation metrics in W&B and run-archive logs

---

## WandB Exports

### Figures

Export a single metric (e.g. PR-AUC) across all three strategies:

```bash
uv run python scripts/plot_wandb_to_latex.py \
  --runs "exp1_random_hpc" "exp1_constrained_hpc" "exp1_proof_based_hpc" \
  --labels "Random Negative Sampling" "Constrained Negative Sampling" "Proof-Based Negative Sampling" \
  --section "val" \
  --metric "triple_pr_auc" \
  --smooth 0.6 \
  --output "paper/figures/exp1_validation_auc.pdf"
```

Export all metrics at once:

```bash
uv run python scripts/plot_wandb_to_latex.py \
  --runs "avj2zcvr" "exp1_constrained_hpc" "exp1_proof_based_hpc" \
  --labels "Mixed" "Constrained" "Proof-Based" \
  --section "val" \
  --metric "all" \
  --smooth 0.6
```

Output is written automatically to:

```
wandb/graphs/<date>/<section>/<metric>.pdf
```

### CSV

```bash
uv run python scripts/export_wandb_to_csv.py \
  --runs "exp1_mixed_hpc" "exp1_constrained_hpc" "exp1_proof_based_hpc" \
  --labels "Mixed Negative Sampling" "Constrained Negative Sampling" "Proof-Based Negative Sampling" \
  --section "val" \
  --metric "all"
```

---

## Visual Inspection

Generate a Family Tree sample from the ont-generator and render it as a graph:

```bash
uv run invoke gen-ft-ont
uv run --package kgvisualiser python -m kgvisualiser.visualize \
  io.input_csv=data/ont/family_tree/train/targets.csv \
  io.sample_id=1000 \
  output.dir=visual-verification/ont_generator \
  output.name_template=ont_sample_1000
```

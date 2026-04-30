# Experiment 1: Negative Sampling Ablation

## Objective

...

## Datasets Used

- Ontology: `ontologies/family.ttl` (OWL2 RL translation used throughout the thesis pipeline).
- Train/val datasets:
    - `data/exp1/random`
    - `data/exp1/constrained`
    - `data/exp1/proof_based`
- Frozen hard-negative test set:
    - `data/exp1/test_set`
- Shared output format per split:
    - `facts.csv` (base context facts)
    - `targets.csv` (positive and negative targets)

## Configurations

Generation configs:

- `configs/ont_generator/exp1_random.yaml`
- `configs/ont_generator/exp1_constrained.yaml`
- `configs/ont_generator/exp1_proof_based.yaml`
- `configs/ont_generator/exp1_test.yaml`

RRN dataset configs:

- `configs/rrn/data/dataset/exp1_random.yaml`
- `configs/rrn/data/dataset/exp1_constrained.yaml`
- `configs/rrn/data/dataset/exp1_proof_based.yaml`

RRN logging convention:

- `logger.group=exp1_negative_sampling`
- `logger.name=exp1_<strategy>`
- `logger.tags=[exp1,<strategy>]`

## Execution Commands

1. Generate all train/val variants:

```bash
uv run invoke exp1-generate-trainval-sets
```

2. Generate and freeze near-miss test set:

```bash
uv run invoke exp1-generate-test-set
```

3. Train one model per strategy:

```bash
uv run invoke exp1-train-rrn --strategy="random"
uv run invoke exp1-train-rrn --strategy="constrained"
uv run invoke exp1-train-rrn --strategy="proof_based"
```

## WandB exports

### Figures

Single metric:

```bash
uv run python scripts/plot_wandb_to_latex.py \
  --runs "exp1_random_hpc" "exp1_constrained_hpc" "exp1_proof_based_hpc" \
  --labels "Random Negative Sampling" "Constrained Negative Sampling" "Proof-Based Negative Sampling" \
  --section "val" \
  --metric "triple_pr_auc" \
  --smooth 0.6 \
  --output "paper/figures/exp1_validation_auc.pdf"
```

All metrics:

```bash
uv run python scripts/plot_wandb_to_latex.py \
  --runs "avj2zcvr" "exp1_constrained_hpc" "exp1_proof_based_hpc" \
  --labels "Mixed" "Constrained" "Proof-Based" \
  --section "val" \
  --metric "all" \
  --smooth 0.6
```

The above command handles the directory structure automatically and will output the graphs to:
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
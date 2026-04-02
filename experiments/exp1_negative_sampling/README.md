# Experiment 1: Negative Sampling Ablation

## Objective

Test the paper hypothesis that proof-based corruption produces harder negatives than random or constrained corruption, forcing stricter logical reasoning and reducing false positives on near-miss triples.

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

## Tracked Metrics

Primary metrics (paper-aligned for hard-negative evaluation):

- `PR-AUC` (primary discrimination metric for near-miss negatives)
- `FPR` (false positive robustness on logically false near-miss triples)

Secondary metrics:

- `AUC-ROC`
- `F1` at threshold `P > 0.5`

All runs must log metrics to W&B under `exp1_negative_sampling`.

## Required Artifacts For The Paper

- Table: strategy vs `PR-AUC`, `FPR`, `AUC-ROC`, `F1`.
- Plot 1: grouped bar plot of the four metrics per strategy.
- Plot 2: Precision-Recall curves (same frozen test set).
- Plot 3: ROC curves (same frozen test set).
- Brief analysis paragraph explaining why lower FPR with high PR-AUC indicates better logical boundary learning.

## Expected Result

The `proof_based` strategy should be the winner on hard negatives, with lower `FPR` and stronger `PR-AUC` than `random` and `constrained`, then be frozen for later experiments.

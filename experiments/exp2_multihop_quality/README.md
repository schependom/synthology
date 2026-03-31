# Experiment 2: Multi-Hop Reasoning Quality

## Objective

Evaluate whether Synthology's backward-chaining data generation yields better multi-hop reasoning performance than the forward-chaining baseline when both are controlled for data budget and graph-level parity.

## Datasets Used

- Ontology: Family Tree OWL2 RL setup used in Exp 1.
- Frozen gold test set (multi-hop only): generated to contain target facts requiring proof depth `d >= 3`.
- Baseline train/val data (forward-chaining): `data/exp2/baseline/...`
- Synthology train/val data (backward-chaining): `data/exp2/synthology/...`
- Recommended frozen test location:
    - `data/exp2/frozen_test/facts.csv`
    - `data/exp2/frozen_test/targets.csv`

## Configurations

Generator configs:

- `configs/ont_generator/exp2_gold_test.yaml`
- `configs/fc_baseline/exp2_baseline.yaml`
- `configs/ont_generator/exp2_synthology.yaml`

RRN dataset configs:

- `configs/rrn/data/dataset/exp2_baseline.yaml`
- `configs/rrn/data/dataset/exp2_synthology.yaml`

Reporting config:

- `configs/data_reporter/exp2_compare.yaml`

Fairness constraints required by the paper protocol:

- Match total data budget (facts and targets)
- Match positive:negative ratio
- Match node count and edge density as closely as possible
- Enforce parity for deep (`d >= 3`) signal in training data

## Execution Commands

1. Generate and freeze multi-hop gold test set:

```bash
uv run invoke exp2-generate-gold-test
```

2. Generate baseline and synthology datasets (shared budget caps):

```bash
uv run invoke exp2-balance-datasets \
  --fact-cap=<Nf> \
  --target-cap=<Nt> \
  --baseline-base-facts=<K1> \
  --synthology-proof-roots=<K2>
```

3. Run parity/distribution reporting:

```bash
uv run invoke exp2-report-data
```

4. Train one RRN per dataset:

```bash
uv run invoke exp2-train-rrn --dataset="baseline"
uv run invoke exp2-train-rrn --dataset="synthology"
```

## Tracked Metrics

Primary metrics on the frozen `d >= 3` test set:

- `PR-AUC`
- `AUC-ROC`

Secondary diagnostics:

- `F1` at threshold `P > 0.5`
- `FPR`

Data-quality diagnostics to report alongside model metrics:

- inference depth distribution (especially share of `d >= 3`)
- node count parity
- edge density parity
- class balance parity (positive:negative)

## Required Artifacts For The Paper

- Table: baseline vs synthology on `PR-AUC`, `AUC-ROC`, `F1`, `FPR` (frozen `d >= 3` test).
- Plot 1: PR curves (baseline vs synthology).
- Plot 2: ROC curves (baseline vs synthology).
- Plot 3: inference-depth distribution comparison (train data), highlighting `d >= 3` coverage.
- Plot 4: parity dashboard (facts, targets, node count, edge density, pos:neg ratio).
- One short paragraph explicitly stating this is a quality comparison, not only a quantity comparison.

## Expected Result

The Synthology-trained model should outperform baseline on deep-chain evaluation (`d >= 3`) with stronger `PR-AUC` and `AUC-ROC`, showing that engineered path quality and integration matter beyond raw forward-chained volume.

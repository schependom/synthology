# Experiment 3: Scaling Benchmark (OWL2Bench)

## Objective

Demonstrate ontology-agnostic scalability and data-quality gains on a complex OWL2 RL ontology (`UNIV-BENCH-OWL2RL.owl`) by comparing:

- Forward-chaining materialization baseline (Nemo-based OWL2Bench pipeline)
- Synthology backward-chaining generation

under matched data budgets and shared evaluation protocol.

## Datasets Used

- Ontology: `data/OWL2Bench/input/UNIV-BENCH-OWL2RL.owl`
- Baseline pipeline package: `apps/OWL2Bench`
- Baseline output root (default): `data/OWL2Bench/output`
- Synthology OWL2Bench output root (project-specific config; freeze path in config)
- Frozen deep-test set for Exp 3 (`d >= 3`, optionally split by 3-hop and 4-hop)

Expected train/val/test format for both methods:

- `facts.csv`
- `targets.csv`

## Configurations

Baseline (Nemo + OWL2Bench pipeline):

- `configs/owl2bench/config.yaml`
- Optional quick check: `configs/owl2bench/config_toy.yaml`

RRN dataset mapping for baseline:

- `configs/rrn/data/dataset/owl2bench.yaml`

Required Synthology-side config files for this experiment (if missing, create and freeze):

- `configs/ont_generator/exp3_synthology.yaml`
- `configs/ont_generator/exp3_gold_test.yaml`
- `configs/rrn/data/dataset/exp3_synthology.yaml`
- `configs/rrn/data/dataset/exp3_baseline.yaml`

## Required Actions (Paper Protocol)

1. Generate baseline dataset A from OWL2Bench with Nemo materialization.
2. Filter baseline artifacts as needed for ML suitability (for example BNodes and non-target schema noise).
3. Partition into sample-level graphs (`sample_id`) suitable for RRN training.
4. Generate Synthology dataset B on the same ontology with over-generation.
5. Balance dataset B down to baseline budget parity (facts/targets/samples and class ratio).
6. Freeze an Exp 3 deep-test set with only complex inferences (`d >= 3`).
7. Train one RRN on dataset A and one RRN on dataset B using identical training settings.
8. Evaluate both only on the frozen deep Exp 3 test set.
9. Report model metrics and generator metrics together.

## Execution Commands

If the canonical Exp 3 invoke tasks are available, run:

```bash
uv run invoke exp3-generate-baseline --universities=50
uv run invoke exp3-generate-synthology --universities=50
uv run invoke exp3-balance-data
uv run invoke exp3-generate-gold-test
uv run invoke exp3-train-rrn --dataset="baseline"
uv run invoke exp3-train-rrn --dataset="synthology"
```

Current baseline generator command already available in this repo:

```bash
uv run --package owl2bench python -m owl2bench.pipeline dataset.universities=[50]
```

Current Synthology generation command pattern:

```bash
uv run --package ont_generator python -m ont_generator.create_data --config-name=exp3_synthology
```

## Tracked Metrics

Model metrics (frozen deep test set):

- `PR-AUC`
- `AUC-ROC`
- `F1` at threshold `P > 0.5`
- `FPR`

Generator/structural metrics (required for this scaling benchmark):

- Inference depth distribution (baseline vs synthology)
- Ontology coverage: percentage of OWL2Bench rules triggered
- Yield rate: inferred targets produced per base-fact budget
- Generation runtime (and runtime vs depth settings)

Parity checks (must be logged):

- number of samples/graphs
- fact count
- target count
- positive:negative ratio

## Required Plots And Tables

- Table 1: baseline vs synthology model metrics (`PR-AUC`, `AUC-ROC`, `F1`, `FPR`).
- Plot 1: PR curves (baseline vs synthology) on frozen Exp 3 test.
- Plot 2: ROC curves (baseline vs synthology) on frozen Exp 3 test.
- Plot 3: inference-depth histogram (baseline vs synthology).
- Plot 4: runtime vs depth/proof-budget settings for Synthology.
- Plot 5: ontology coverage and yield-rate comparison.
- Table 2: budget-parity audit (samples, facts, targets, pos:neg ratio).

## Expected Result

At equalized data budget, the Synthology-trained model should perform better on deep OWL2Bench reasoning than the forward-chaining baseline, while generation diagnostics show stronger deep-path yield and useful ontology-rule activation.

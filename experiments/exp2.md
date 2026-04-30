# Experiment 2: Multi-Hop Reasoning Quality

## Objective

...

## Datasets Used

- Ontology: `ontologies/family.ttl`.
- Exp2 Synthology train/val/test output root: `data/exp2/synthology/family_tree/`.
- Exp2 UDM baseline train/val/test output root: `data/exp2/baseline/family_tree/`.
- Frozen shared deep test set root: `data/exp2/frozen_test/`.

All task wrappers also archive canonical run artifacts under:

- `reports/experiment_runs/<date>/exp2/<task>/<timestamp>/`

## Experiment Mechanics

1. Generate a frozen deep test set once and reuse it for both models.
2. Generate baseline and Synthology training datasets under comparable budget constraints.
3. Enforce baseline parity with retry-loop generation until deep and structural criteria are satisfied.
4. If parity does not converge within practical compute budgets, run matched-budget balancing (shared train fact/target caps) and report that strict parity remained unattained.
5. Train one RRN on baseline and one RRN on Synthology with equivalent training protocol.
6. Compare model metrics and data diagnostics, with explicit deep-hop reporting.

Why baseline Jena is single-pass in Exp2:

- Standard Jena closure is fixed-point materialization and does not preserve derivation depth metadata natively.
- The current baseline pipeline materializes once per sample (single-pass closure) and reads hop-depth information from the Jena materialization output path used in the generator code.

## Configurations

- Baseline generator: `configs/udm_baseline/exp2_baseline.yaml`
- Synthology generator: `configs/ont_generator/exp2_synthology.yaml`
- Frozen test generation: `configs/ont_generator/exp2_gold_test.yaml`
- RRN baseline training: `configs/rrn/exp2_baseline_hpc.yaml`
- RRN Synthology training: `configs/rrn/exp2_synthology_hpc.yaml`
- Exp2 comparison report: `configs/data_reporter/exp2_compare.yaml`

Important fixed semantics in current config/task stack:

- Exp2 baseline uses Jena profile `owl_micro`.
- Exp2 baseline materialization is single-pass (`materialization.iterative: false` compatibility flag in config).
- Parity loop/report commands archive their outputs in run archives and keep attempt-level artifacts in the attempts root for reproducibility.

## Preconditions

Before running Exp2 baseline/parity on a fresh machine or HPC node:

```bash
uv sync
module load openjdk/21
./install-mvn.sh
which java && java -version
which mvn && mvn -v
```

Optional quick Java preflight:

```bash
cd apps/udm_baseline/java
mvn -q -DskipTests package
ls -lh target/jena-materializer-1.0.0-shaded.jar
cd ../../..
```

## Execution Commands (in order)

### 1. Freeze Exp2 deep test set

```bash
uv run invoke exp2-generate-gold-test
```

### 2. Generate baseline dataset (UDM + Jena iterative)

```bash
uv run invoke exp2-generate-baseline
```

### 3. Generate Synthology dataset

```bash
uv run invoke exp2-generate-synthology
```

### 4. Optional strict budget matching helper

Use this when you want both generators to share explicit train caps:

```bash
uv run invoke exp2-balance-datasets --fact-cap=<FACT_CAP> --target-cap=<TARGET_CAP>
```

HPC preset for reproducible fallback runs:

```bash
bsub < jobscripts/exp2-balance-datasets.sh
```

This submits:

```bash
uv run invoke exp2-balance-datasets-hpc --config-path=configs/experiments/exp2_balance_hpc.yaml
```

Default notes for `exp2-balance-datasets`:

- `fact_cap` is required.
- `target_cap` is optional; if omitted, each generator uses its own config default.
- `baseline_base_facts` overrides `generator.base_relations_per_sample` for baseline.
- `synthology_proof_roots` overrides `generator.proof_roots_per_rule` for Synthology.
- Current generator defaults differ, so explicitly setting both caps is recommended for fair comparisons.

### 7. Build Exp2 distribution/comparison report

```bash
uv run invoke exp2-report-data
```

### 8. Train RRN on baseline and Synthology

```bash
uv run invoke exp2-train-rrn --dataset=baseline
uv run invoke exp2-train-rrn --dataset=synthology
```

### 9. Optional smoke and paper plots

```bash
uv run invoke exp2-smoke-jena-visual

uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
  --out-dir=reports/paper
```
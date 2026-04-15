# Experiment 3: Generalization to Complex Ontologies (OWL2Bench)

## Objective

Experiment 3 validates the paper's generalization/scalability claim on a substantially larger ontology setting (OWL2Bench): Synthology-style structural supervision should remain advantageous when ontology complexity and graph size increase.

Paper traceability:

- Method framing: paper section "Experiment 3: Generalization to Complex Ontologies".
- Baseline construction policy: paper section "The Unguided Deductive Materialization (UDM) Baseline".
- Interpretation focus: results/discussion text on shallow-schema dominance vs useful multi-hop supervision.

## Datasets Used

- TBox: `ontologies/UNIV-BENCH-OWL2RL.owl`
- OWL2Bench raw ABox export root: `data/owl2bench/output/raw/owl2bench_<U>/`
- OWL2Bench sampled CSV dataset root: `data/owl2bench/output/`
- Exp3 baseline closure/inferred export root: `data/exp3/baseline/owl2bench_<U>/`

Where `<U>` is the number of universities (default: 20 in Exp3 tasks).

All task wrappers archive canonical evidence under:

- `reports/experiment_runs/<date>/exp3/<task>/<timestamp>/`

## Experiment Mechanics (Current Implementation)

1. Generate OWL2Bench ABox data (procedural benchmark generation).
2. Materialize with UDM/Jena baseline semantics.
3. Split/slice large graphs via BFS subgraph partitioning for train/val/test consumption.
4. Produce paper-oriented diagnostics and training artifacts.

Critical baseline semantics in Exp3:

- Exp3 uses single-pass Jena materialization (not Exp2-style iterative diffing).
- This is intentional for OWL2Bench scale and is enforced in config (`iterative: false`, `max_iterations: 1`).
- Jena profile stays fixed at `owl_mini` for reproducibility and runtime feasibility.

## Configurations (Source of Truth)

- OWL2Bench main pipeline config: `configs/owl2bench/config.yaml`
- OWL2Bench toy smoke config: `configs/owl2bench/config_toy.yaml`
- UDM materialization config path used by tasks: `configs/udm_baseline/config.yaml`
- Exp3 RRN training wrapper config: `configs/rrn/exp3_owl2bench_hpc.yaml`

Important notes for reviewers:

- The task layer provides explicit Exp3 wrappers for baseline generation/materialization.
- The task layer provides an explicit `exp3-generate-synthology` wrapper for Synthology-side OWL2Bench data generation.

## Preconditions

```bash
uv sync
module load openjdk/21
./install-mvn.sh
which java && java -version
which mvn && mvn -v
```

If Maven is only available in the repo-local install, export it explicitly:

```bash
export MAVEN_EXECUTABLE="$PWD/apache-maven-3.9.13/bin/mvn"
export PATH="$PWD/apache-maven-3.9.13/bin:$PATH"
"$MAVEN_EXECUTABLE" -v
```

## Execution Commands (Canonical Order)

For HPC runs, prefer the jobscripts below. Each script accepts environment-variable overrides.

### 1. Fast smoke run (recommended before full Exp3)

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy
```

Optional ultra-fast smoke override:

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy --args='dataset.reasoning_input_triple_cap=600 dataset.bfs.sample_count=6 dataset.inferred_target_limit=200'
```

### 2. Exp3 Synthology-side generation

HPC jobscript:

```bash
UNIVERSITIES=20 SYNTHOLOGY_JENA_XMX_MB=16384 bsub < jobscripts/exp3-generate-synthology.sh
```

Direct task equivalent:

```bash
uv run invoke exp3-generate-synthology --universities=20
```

Default Synthology-side outputs are written to:

- `data/exp3/synthology/owl2bench_20/`
- `data/exp3/synthology/raw/owl2bench_20/`

### 3. Exp3 ABox generation wrapper (explicit university count)

```bash
uv run invoke exp3-generate-owl2bench-abox --universities=20
```

### 4. Exp3 baseline chain (ABox + Jena materialization)

HPC jobscript:

```bash
UNIVERSITIES=20 SYNTHOLOGY_UDM_BASELINE_XMX_MB=16384 bsub < jobscripts/exp3-generate-baseline.sh
```

Direct task equivalent:

```bash
uv run invoke exp3-generate-baseline --universities=20
```

Paper-canonical UDM baseline defaults are used unless overridden:

- `dataset.mask_base_facts=false`
- `dataset.target_ratio=0.0`
- `dataset.negatives_per_positive=1`

Optional explicit canonical override (equivalent):

```bash
uv run invoke exp3-generate-baseline --universities=20 \
  --args='dataset.mask_base_facts=false dataset.target_ratio=0.0 dataset.negatives_per_positive=1 dataset.inferred_target_limit=250000'
```

### 5. Balance Synthology labels to baseline yield (paper comparison prep)

HPC jobscript:

```bash
UNIVERSITIES=20 bsub < jobscripts/exp3-balance-data.sh
```

Direct task equivalent:

```bash
uv run invoke exp3-balance-data --universities=20
```

What this balancing step does exactly:

- Per split (`train`, `val`, `test`), it matches Synthology target counts to baseline target counts separately for positive and negative labels.
- It samples Synthology positives and negatives with a fixed seed and writes balanced targets to `data/exp3/balanced/owl2bench_<U>/...`.
- It keeps Synthology `facts.csv` unchanged and only resamples `targets.csv`.

What this balancing step does not do:

- It does not match hop-depth distribution.
- It does not match predicate distribution.
- It does not match graph topology or entity-degree statistics.
- It does not rebalance baseline; it only down-samples Synthology targets.

### 6. Freeze Exp3 test split for reproducible evaluation

HPC jobscript:

```bash
UNIVERSITIES=20 bsub < jobscripts/exp3-generate-gold-test.sh
```

Direct task equivalent:

```bash
uv run invoke exp3-generate-gold-test --universities=20
```

### 7. Optional direct materialization for any existing ABox

```bash
uv run invoke exp3-materialize-abox \
  --abox=data/owl2bench/output/raw/owl2bench_20/OWL2RL-20.owl \
  --tbox=ontologies/UNIV-BENCH-OWL2RL.owl \
  --closure-out=data/exp3/baseline/owl2bench_20/closure.nt \
  --inferred-out=data/exp3/baseline/owl2bench_20/inferred.nt \
  --jena-profile=owl_mini
```

### 8. Paper visuals including Exp3 artifacts

HPC jobscript:

```bash
UNIVERSITIES=20 bsub < jobscripts/exp3-paper-visual-report.sh
```

Direct task equivalent:

```bash
uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
  --exp3-targets=data/owl2bench/output/owl2bench_20/train/targets.csv \
  --exp3-abox=data/owl2bench/output/raw/owl2bench_20/OWL2RL-20.owl \
  --exp3-inferred=data/exp3/baseline/owl2bench_20/inferred.nt \
  --out-dir=reports/paper
```

## RRN Training For Exp3 (Paper Wrapper)

Use the Exp3 wrapper that pins dataset roots and run labels per arm:

HPC jobscripts:

```bash
UNIVERSITIES=20 bsub < jobscripts/exp3-train-baseline.sh
UNIVERSITIES=20 bsub < jobscripts/exp3-train-synthology.sh
```

Direct task equivalents:

```bash
uv run invoke exp3-train-rrn --dataset=baseline --universities=20
uv run invoke exp3-train-rrn --dataset=synthology --universities=20
```

The synthology arm automatically prefers balanced data at
`data/exp3/balanced/owl2bench_20/` when present.

## Fair Comparison Protocol (Exp3)

Use the following protocol for paper-grade fairness and traceability:

1. Keep canonical UDM baseline semantics fixed:
  - single-pass Jena (`iterative: false`, `max_iterations: 1`)
  - `dataset.mask_base_facts=false`
  - `dataset.target_ratio=0.0`
  - `dataset.negatives_per_positive=1`
2. Generate baseline and Synthology arms independently (do not overwrite paths).
3. Run `exp3-balance-data` to match per-split positive/negative target counts.
4. Freeze one gold test split and reuse it for both models.
5. Train with identical model/training config for both arms; only dataset paths differ.
6. Run multiple seeds per arm and report mean plus confidence intervals.
7. Report both model metrics and data diagnostics:
  - metrics: `PR-AUC`, `AUC-ROC`, `F1`, `FPR`
  - diagnostics: label counts, hop distributions, predicate skew, runtime costs

Interpretation note:

- The balancing step provides fairness on label volume, not full structural parity.
- Exp3 claims should therefore be supported by both metric improvements and structural diagnostics, not by label matching alone.

## Analysis and Monitoring

Latest baseline diagnostics:

```bash
bsub < jobscripts/exp3-analyze-latest-baseline.sh
```

Queue and log monitoring:

```bash
bjobs -u "$USER"
tail -n 120 logs/exp3_generate_baseline_*.out
find reports/experiment_runs -type f -name run.log | tail -n 20
```

## Expected Results (What Must Be Reported)

At minimum, include:

- Runtime and scalability evidence (ABox generation, materialization, partitioning overhead).
- Structural diagnostics from generated datasets and exported artifacts.
- Trivial/schema-dominance analysis for baseline closure behavior.
- Downstream RRN metrics (`PR-AUC`, `AUC-ROC`, `F1`, `FPR`) for whichever Exp3 datasets are trained/evaluated.

Primary evidence locations:

- Run archives: `reports/experiment_runs/...`
- Exp3 baseline artifacts: `data/exp3/baseline/...`
- OWL2Bench baseline datasets: `data/owl2bench/output/...`
- Exp3 synthology datasets: `data/exp3/synthology/...`
- Exp3 balanced synthology datasets: `data/exp3/balanced/...`
- Paper figures: `reports/paper/...`
- Training logs: W&B + run archive logs.

## Reviewer Checklist

1. Confirm Exp3 configs keep Jena in single-pass mode (`iterative: false`, `max_iterations: 1`).
2. Confirm all run archives include `manifest.json`, `run.log`, and copied artifacts.
3. Confirm any RRN run uses explicit data path overrides if path casing differs on disk.
4. Confirm paper figures are generated from the same run outputs cited in the summary.

## Cross-Reference

For full end-to-end paper execution (Exp1-Exp3 + visuals), see `experiments/PAPER_RUNBOOK.md`.

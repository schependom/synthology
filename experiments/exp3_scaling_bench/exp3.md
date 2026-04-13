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

## Execution Commands (Canonical Order)

### 1. Fast smoke run (recommended before full Exp3)

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy
```

Optional ultra-fast smoke override:

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy --args='dataset.reasoning_input_triple_cap=600 dataset.bfs.sample_count=6 dataset.inferred_target_limit=200'
```

### 2. Exp3 Synthology-side generation

```bash
uv run invoke exp3-generate-synthology --universities=20
```

### 3. Exp3 ABox generation wrapper (explicit university count)

```bash
uv run invoke exp3-generate-owl2bench-abox --universities=20
```

### 4. Exp3 baseline chain (ABox + Jena materialization)

```bash
uv run invoke exp3-generate-baseline --universities=20
```

### 5. Optional direct materialization for any existing ABox

```bash
uv run invoke exp3-materialize-abox \
  --abox=data/owl2bench/output/raw/owl2bench_20/OWL2RL-20.owl \
  --tbox=ontologies/UNIV-BENCH-OWL2RL.owl \
  --closure-out=data/exp3/baseline/owl2bench_20/closure.nt \
  --inferred-out=data/exp3/baseline/owl2bench_20/inferred.nt \
  --jena-profile=owl_mini
```

### 6. Paper visuals including Exp3 artifacts

```bash
uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
  --exp3-targets=data/owl2bench/output/owl2bench_20/train/targets.csv \
  --exp3-abox=data/owl2bench/output/raw/owl2bench_20/OWL2RL-20.owl \
  --exp3-inferred=data/exp3/baseline/owl2bench_20/inferred.nt \
  --out-dir=reports/paper
```

## RRN Training For Exp3 (Current Task Wrapper)

Current wrapper task:

```bash
uv run invoke train-rrn-owl2bench
```

Important path caveat:

- `configs/rrn/data/dataset/owl2bench.yaml` currently points to `data/OWL2Bench/output/...` (capitalized path).
- If your generated data is under `data/owl2bench/output/...` (lowercase, default in current generation tasks), run with explicit path overrides:

```bash
uv run invoke train-rrn-owl2bench --args='data.train_path=data/owl2bench/output/owl2bench_20/train data.val_path=data/owl2bench/output/owl2bench_20/val data.test_path=data/owl2bench/output/owl2bench_20/test logger.name=exp3_owl2bench_current_paths'
```

For baseline-vs-Synthology comparative training, execute this training task separately per dataset by overriding the `data.*_path` triplet and run name/group tags.

## Expected Results (What Must Be Reported)

At minimum, include:

- Runtime and scalability evidence (ABox generation, materialization, partitioning overhead).
- Structural diagnostics from generated datasets and exported artifacts.
- Trivial/schema-dominance analysis for baseline closure behavior.
- Downstream RRN metrics (`PR-AUC`, `AUC-ROC`, `F1`, `FPR`) for whichever Exp3 datasets are trained/evaluated.

Primary evidence locations:

- Run archives: `reports/experiment_runs/...`
- Exp3 baseline artifacts: `data/exp3/baseline/...`
- OWL2Bench datasets: `data/owl2bench/output/...`
- Paper figures: `reports/paper/...`
- Training logs: W&B + run archive logs.

## Reviewer Checklist

1. Confirm Exp3 configs keep Jena in single-pass mode (`iterative: false`, `max_iterations: 1`).
2. Confirm all run archives include `manifest.json`, `run.log`, and copied artifacts.
3. Confirm any RRN run uses explicit data path overrides if path casing differs on disk.
4. Confirm paper figures are generated from the same run outputs cited in the summary.

## Cross-Reference

For full end-to-end paper execution (Exp1-Exp3 + visuals), see `experiments/PAPER_RUNBOOK.md`.

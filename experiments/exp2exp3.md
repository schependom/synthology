# Exp2/Exp3 Audit and Run Readiness

Date: 2026-04-13

## 1) Task Argument Audit (Exp2 and Exp3)

Goal checked:
- Remove unnecessary command arguments in `tasks.py` that duplicate values already defined in Exp2/Exp3 config files.

Findings:
- Exp2 task commands are mostly clean: fixed values are read from config files, while task arguments are used as optional runtime overrides (`fact_cap`, `target_cap`, `proof_roots_per_rule`, parity controls).
- Exp3 had one unnecessary duplicate override in `exp3_generate_owl2bench_abox`:
  - Removed `dataset.output_dir=data/owl2bench/output` from command overrides because it is already defined in `configs/owl2bench/config.yaml`.
- Kept necessary dynamic overrides:
  - `dataset.universities=[...]` in Exp3 task wrappers.
  - timing output/run-tag overrides (run-specific archive behavior).

Result:
- No unnecessary duplicate config argument remains in Exp2/Exp3 task commands that would cause confusion.

## 2) Configuration Feasibility Tuning for 1 Week

### Exp2 tuning applied

Files changed:
- `configs/ont_generator/exp2_synthology.yaml`
- `configs/ont_generator/exp2_gold_test.yaml`
- `configs/udm_baseline/exp2_baseline.yaml`

Changes:
- `exp2_synthology.yaml`
  - `dataset.n_train: 2000 -> 1200`
  - `dataset.n_val: 200 -> 150`
  - `dataset.n_test: 200 -> 150`
  - `generator.min_individuals: 1 -> 4`
  - `generator.max_individuals: 1000 -> 240`
- `exp2_gold_test.yaml`
  - `dataset.n_test: 600 -> 350`
  - `generator.min_individuals: 1 -> 6`
  - `generator.max_individuals: 1000 -> 300`
  - `generator.global_max_depth: 10 -> 9`
- `exp2_baseline.yaml`
  - `dataset.n_train: 2000 -> 1200`
  - `dataset.n_val: 200 -> 150`
  - `dataset.n_test: 200 -> 150`

Why this is realistic:
- Keeps datasets large enough for meaningful comparison/training.
- Reduces extreme per-sample graph explosion risk and generation time.
- Maintains parity experiment validity with practical convergence settings already in tasks (`deep-count-mode=tolerance`, default tolerance 10%).

### Exp3 tuning applied

Files changed:
- `configs/owl2bench/config.yaml`
- `configs/rrn/data/dataset/owl2bench.yaml`

Changes:
- `owl2bench/config.yaml`
  - `dataset.universities: [10] -> [20]`
  - `dataset.bfs.sample_count: 5000 -> 1800`
  - `dataset.bfs.max_individuals_per_sample: 200 -> 120`
  - `dataset.inferred_target_limit: 0 -> 250000`
  - `dataset.diagnostics.max_rows_per_file: 0 -> 200000`
- `rrn/data/dataset/owl2bench.yaml`
  - fixed path casing and aligned to U=20:
  - `data/OWL2Bench/output/owl2bench_10/... -> data/owl2bench/output/owl2bench_20/...`

Why this is realistic:
- U=20 is a balanced "not too high / not too low" default for a one-week execution window.
- BFS/output caps prevent data explosion while keeping enough training signal.
- RRN dataset path mismatch (previous blocker) is fixed, so Exp3 training config is now runnable with current outputs.

## 3) Readiness Verdict

Status: READY

Checked:
- `tasks.py` (no errors)
- tuned Exp2/Exp3 configs (no errors)
- Exp3 RRN dataset config path alignment fixed

Notes:
- `uv run invoke --list` output could not be retrieved by the terminal tool in this chat session, but task definitions exist in `tasks.py` and config references are consistent.

## 4) Exact Commands to Run Next for Exp2

Run from repository root.

### Step A: Environment preflight

```bash
source .venv/bin/activate
uv sync
module load openjdk/21 || true
./install-mvn.sh
which java && java -version
which mvn && mvn -v
```

### Step B: Freeze shared Exp2 deep test set

```bash
uv run invoke exp2-generate-gold-test
```

### Step C: Generate Synthology reference first (sets parity target)

```bash
uv run invoke exp2-generate-synthology
```

### Step D: Generate baseline once

```bash
uv run invoke exp2-generate-baseline
```

### Step E: Run parity loop with tolerance (recommended)

```bash
uv run invoke exp2-parity-loop --deep-count-mode=tolerance --tolerance-pct=10.0
uv run invoke exp2-parity-report
```

### Step F: Data diagnostics report

```bash
uv run invoke exp2-report-data
```

### Step G: Train both RRN models

```bash
uv run invoke exp2-train-rrn --dataset=baseline
uv run invoke exp2-train-rrn --dataset=synthology
```

## 5) How to Monitor Exp2 While Running

### A) Watch parity convergence

```bash
ls -lah data/exp2/baseline/parity_runs
cat data/exp2/baseline/parity_runs/parity_loop_summary.json
cat data/exp2/baseline/parity_runs/parity_report.json
```

What to check:
- `match_found` / parity success in summary/report.
- attempt count to convergence.
- deep-hop parity metrics and tolerance satisfaction.

### B) Track dataset growth and outputs

```bash
du -sh data/exp2/synthology data/exp2/baseline data/exp2/frozen_test
find data/exp2 -maxdepth 4 -type f | wc -l
```

### C) Monitor archived run logs (canonical record)

```bash
find reports/experiment_runs -type f -name run.log | tail -n 10
```

Then inspect the newest run log path shown by the command:

```bash
tail -n 120 <latest_run_log_path>
```

### D) Monitor training progress

- Weights & Biases run groups:
  - `exp2_multihop`
- Local artifacts:
  - checkpoints and logs under the corresponding `reports/experiment_runs/.../exp2/train_rrn/...` archive.

## 6) If parity is still too slow

Use one of these controlled adjustments (in order):

1. Lower Synthology generation pressure:
```bash
uv run invoke exp2-generate-synthology --proof-roots-per-rule=8
```

2. Apply shared budget cap:
```bash
uv run invoke exp2-balance-datasets --fact-cap=180000 --target-cap=140000
```

3. Increase tolerance slightly:
```bash
uv run invoke exp2-parity-loop --deep-count-mode=tolerance --tolerance-pct=12.5
```

Keep the exact command history and resulting parity report in the run archive for paper traceability.

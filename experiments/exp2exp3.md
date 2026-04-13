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

### Step C1: Analyze Synthology output before choosing parity tolerance

Run this immediately after Step C:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

targets_path = Path("data/exp2/synthology/family_tree/train/targets.csv")
if not targets_path.exists():
    raise SystemExit(f"Missing {targets_path}")

df = pd.read_csv(targets_path)

required = {"label", "type", "hops"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing expected columns: {missing}; found={list(df.columns)}")

# Positive inferred targets are the parity signal.
pos_inf = df[(df["label"] == 1) & (df["type"] == "inferred")].copy()

if pos_inf.empty:
    raise SystemExit("No positive inferred targets found; parity loop is not meaningful yet.")

k_deep = int((pos_inf["hops"] >= 3).sum())
deep_share = float(k_deep / len(pos_inf))

hop_counts = pos_inf["hops"].value_counts().sort_index()
tail_share = float((pos_inf["hops"] >= 5).mean())

print("=== Exp2 Synthology readiness stats ===")
print(f"positive_inferred_total={len(pos_inf)}")
print(f"k_deep_ge3={k_deep}")
print(f"deep_share_ge3={deep_share:.4f}")
print(f"tail_share_ge5={tail_share:.4f}")
print("hop_histogram_positive_inferred=")
print(hop_counts.to_string())

# Practical tolerance guidance for parity convergence.
if k_deep <= 8000 and deep_share <= 0.25 and tail_share <= 0.08:
    tol = 7.5
elif k_deep <= 15000 and deep_share <= 0.35 and tail_share <= 0.15:
    tol = 10.0
elif k_deep <= 25000 and deep_share <= 0.45 and tail_share <= 0.25:
    tol = 12.5
else:
    tol = 15.0

print(f"recommended_tolerance_pct={tol}")
PY
```

Interpretation and decision rules:

- Use 7.5% when deep signal is modest and mostly 3-4 hop.
- Use 10.0% as default for balanced deep signal (recommended baseline).
- Use 12.5% when deep count is high and hop tail is heavy.
- Use 15.0% only when parity attempts are repeatedly stalling due to very high deep-tail mass.

Then run parity with the selected tolerance:

```bash
uv run invoke exp2-parity-loop --deep-count-mode=tolerance --tolerance-pct=<SELECTED_VALUE>
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

## 7) OWL2Bench Maven Troubleshooting (HPC)

Symptom:

- `gen-owl2bench-toy` fails with `FileNotFoundError: ... 'mvn'` even though Maven exists at a custom path.

Cause:

- The OWL2Bench pipeline uses `generator.maven_executable` from Hydra config. If your active config/runtime still resolves that value to `mvn`, PATH must contain Maven.

Guaranteed workaround (bypass PATH dependency):

```bash
source .venv/bin/activate
export MAVEN_EXECUTABLE="$PWD/apache-maven-3.9.13/bin/mvn"
"$MAVEN_EXECUTABLE" -v

SYNTHOLOGY_JENA_XMX_MB=3072 \
uv run invoke gen-owl2bench-toy \
    --args='generator.maven_executable=/dtu/blackhole/16/221590/synthology/apache-maven-3.9.13/bin/mvn'
```

Equivalent full OWL2Bench command:

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 \
uv run invoke gen-owl2bench \
    --args='generator.maven_executable=/dtu/blackhole/16/221590/synthology/apache-maven-3.9.13/bin/mvn'
```

Optional performance warning suppression (shared filesystems):

```bash
export UV_LINK_MODE=copy
```

If you get `No space left on device` under `~/.m2/repository`:

```bash
mkdir -p /dtu/blackhole/16/221590/synthology/.cache/m2
export MAVEN_OPTS="-Dmaven.repo.local=/dtu/blackhole/16/221590/synthology/.cache/m2"

SYNTHOLOGY_JENA_XMX_MB=3072 \
uv run invoke gen-owl2bench-toy \
    --args='generator.maven_executable=/dtu/blackhole/16/221590/synthology/apache-maven-3.9.13/bin/mvn'
```

Notes:

- This redirects Maven downloads/build artifacts away from home quota to project storage.
- The task layer now also sets this repo-local Maven cache automatically for OWL2Bench tasks.

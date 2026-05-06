# Experiment 3: Generalization to Complex Ontologies (OWL2Bench)

## Objective

This experiment tests whether Synthology generalizes to a more complex ontology — the
**OWL2Bench university benchmark** — without any ontology-specific modifications. It also
validates the scalability claim: that Synthology's independent, sample-bounded backward-chaining
avoids the **Reasoning Wall** that forward-chaining materializers hit at production scale.

Unlike Experiment 2 (which uses a random ABox as a deliberately weak UDM baseline), Exp 3
uses the OWL2Bench Java generator to procedurally instantiate a realistic university ABox —
a structurally rich starting point that gives the UDM forward-chainer a head start. This is
the strongest realistic UDM configuration, so any structural advantage Synthology retains
here is a conservative lower bound.

**Hypotheses:**

- Despite the richer UDM starting ABox, Synthology will still produce a deeper hop
  distribution.
- At large scale (`u = 14` universities, ~664 K base facts), Jena's forward-chaining will
  exhaust available memory and fail to terminate — the Reasoning Wall. Synthology will
  complete the same target volume in well under 70 minutes.
- The dataset-quality and scalability advantages established in Exp 2 persist at this
  larger, more complex ontology.

**Scope:** Experiment 3 reports dataset-quality metrics, hop distributions, predicate
coverage, and scalability/timing data. Full RRN training on OWL2Bench is left for future
work; the downstream learning advantage is therefore not empirically confirmed for this
ontology.

**Primary metrics:** hop-depth distribution, predicate coverage, generation runtime,
and memory behaviour.

---

## Preflight

```bash
uv sync
uv run invoke --list
```

### HPC preflight (required for Jena-backed steps)

```bash
module load openjdk/21
./install-mvn.sh
which java && java -version
which mvn && mvn -v
```

Optional: set the global heap once for the session:

```bash
export SYNTHOLOGY_HEAP_MB=16384
```

If Maven is only available via the repo-local install, export it explicitly:

```bash
export MAVEN_EXECUTABLE="$PWD/apache-maven-3.9.13/bin/mvn"
export PATH="$PWD/apache-maven-3.9.13/bin:$PATH"
"$MAVEN_EXECUTABLE" -v
```

**HPC heap guardrail:** If your cluster profile exports `JAVA_TOOL_OPTIONS` or
`_JAVA_OPTIONS` with a `-Xmx...` flag, that can silently override the heap size
expected for Jena materialization. Keep `SYNTHOLOGY_UDM_BASELINE_XMX_MB` as the
single source of heap sizing for the baseline step; strip any inherited `-Xmx`
defaults from your job environment if this is an issue.

---

## Artifacts and Logging

- Canonical run records are written to `reports/experiment_runs/<YYYY-MM-DD>/exp3/<task>/<timestamp>/`.
- Each archive includes `manifest.json`, `run.log`, copied configs, and copied artifacts.
- RRN runs also write to Weights & Biases and the archive's `wandb/`, `lightning_logs/`,
  and `checkpoints/` subdirectories.

---

## Datasets Used

- TBox: `ontologies/UNIV-BENCH-OWL2RL.owl`
- OWL2Bench raw ABox export root: `data/owl2bench/output/raw/owl2bench_<U>/`
- OWL2Bench sampled CSV dataset root: `data/owl2bench/output/`
- Exp 3 UDM baseline closure/inferred export root: `data/exp3/baseline/owl2bench_<U>/`

`<U>` is the number of universities.

**Path-casing guardrail:** Generation tasks write lowercase paths under
`data/owl2bench/output/...`. Some training configs may still reference
`data/OWL2Bench/output/...`. Pass explicit lowercase data-path overrides when
training to avoid silently training on stale or missing data.

---

## Experiment Mechanics

1. Generate the OWL2Bench ABox (procedural benchmark generation via the Java generator).
2. Materialize the ABox with the UDM/Jena baseline semantics.
3. Partition the large materialized graph into sample-sized subgraphs via BFS, initialized
   at random individuals and capped at `k` individuals per subgraph; inferred triples that
   span subgraph boundaries are discarded.
4. Generate the Synthology dataset independently on the same TBox.
5. Run the matched-budget balancing step to align per-split positive/negative target counts.
6. Produce paper-oriented diagnostics and scalability artifacts.

---

## Configurations

| Purpose | Config path |
|---|---|
| OWL2Bench main pipeline | `configs/owl2bench/config.yaml` |
| OWL2Bench toy smoke run | `configs/owl2bench/config_toy.yaml` |
| UDM materialization | `configs/udm_baseline/config.yaml` |
| Exp 3 RRN training wrapper | `configs/rrn/exp3_owl2bench_hpc.yaml` |

---

## Choosing University Count and BFS Sample Count

### How many universities to generate

The goal is to match the number of unique training subgraphs produced by Synthology
(`n_train` in the Synthology config). The calibration constant comes from measuring unique
`sample_id` values in the train split:

| Run | Universities | `bfs.sample_count` | Unique train `sample_id`s | Unique / university |
|---|---|---|---|---|
| Full (`u=20`) | 20 | 1800 | 1440 | 72 |
| Smoke (`u=2`) | 2 | 200 | 160 | 80 |

The density converges to roughly **72 unique train sample IDs per university** at standard
settings. Use this formula:

```
universities  = ceil(n_train / 72)
sample_count  = universities × 90   # 90 total BFS draws/university; ~80% land in train
```

Example for `n_train = 1000`:

```
universities = ceil(1000 / 72) = 14
sample_count = 14 × 90 = 1260
```

These values are currently set in `configs/experiments/exp3_hpc.yaml` and
`configs/owl2bench/config.yaml`.

**Scale decision note:** The default university count in this runbook is **`u=5`** for a
practical one-week turnaround. Increase with `--universities=<N>` when compute budget
allows. The paper results use `u=20` (dataset quality) and `u=14` (Reasoning Wall
demonstration).

### Memory model

Jena LP backward-chaining builds a tabling cache of every unique derived ground goal.
Memory scales roughly as a power law with ABox size:

```
process_GB ≈ 58.9 × (universities × 48495 / 969891)^0.55
```

Calibration data points:

| Universities | Base triples | Heap (GB) | Process (GB) | Outcome |
|---|---|---|---|---|
| 20 | ~970 K | 51 | ~59 | OOM — Jena LP tabling |
| 14 | ~679 K | 51 | ~48 | Fits (estimated) |
| 5 | ~242 K | 51 | ~27 | Fits (estimated) |
| 2 | ~97 K | 8 | ~14 | Fits (measured, smoke run) |

The safe ceiling on a standard 64 GB HPC node (51 GB heap) is approximately **u = 18–19**
(process ≈ 57 GB). Leave headroom; u = 14–16 is comfortable.

For nodes with more RAM (e.g. `select[maxmem>=512000]` bigmem nodes), increase the heap
via `abox_jena_heap_mb` in `exp3_hpc.yaml`.

### Derivation logging

Full derivation logging (`derivation_logging: true`) stores complete proof trees per
inferred triple for hop-depth analysis. This is **additional** memory on top of LP tabling
and is impractical at full scale.

**Resolution:** disable derivation logging for the full-scale run; keep it enabled only for
the 2-university smoke run. The hop-depth distribution from the smoke run is a valid proxy
for the full run because OWL2Bench universities are structurally independent ABox fragments
over the same TBox — the same rules fire at the same depths regardless of university count.

| Config | `materialization.derivation_logging` |
|---|---|
| `configs/owl2bench/config.yaml` | `false` |
| `configs/owl2bench/config_smoke.yaml` | `true` |

---

## Execution Commands (Canonical Order)

For HPC runs, prefer the jobscripts below. Each script accepts environment-variable overrides.

### 1. Fast smoke run (recommended before full Exp 3)

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy
```

Optional ultra-fast override:

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy \
  --args='dataset.reasoning_input_triple_cap=600 dataset.bfs.sample_count=6 dataset.inferred_target_limit=200'
```

### 2. Generate the Exp 3 Synthology dataset

HPC jobscript:

```bash
UNIVERSITIES=20 SYNTHOLOGY_JENA_XMX_MB=16384 bsub < jobscripts/exp3-generate-synthology.sh
```

Direct equivalent:

```bash
uv run invoke exp3-generate-synthology --universities=5
```

If this OOMs on your node, re-run with explicit caps:

```bash
SYNTHOLOGY_JENA_XMX_MB=16384 uv run invoke exp3-generate-synthology --universities=5 \
  --args='dataset.reasoning_input_triple_cap=80000 dataset.inferred_target_limit=120000 dataset.bfs.sample_count=1200 dataset.bfs.max_individuals_per_sample=100'
```

Outputs are written to:

- `data/exp3/synthology/owl2bench_<U>/`
- `data/exp3/synthology/raw/owl2bench_<U>/`

**Important:** Exp 3 baseline and Synthology-side data are generated independently. Use
explicit dataset paths when training/evaluating to avoid path-casing mismatches.

### 3. Generate the OWL2Bench ABox

```bash
uv run invoke exp3-generate-owl2bench-abox --universities=5
```

### 4. Generate the UDM baseline (ABox + Jena materialization)

HPC jobscript:

```bash
UNIVERSITIES=20 SYNTHOLOGY_UDM_BASELINE_XMX_MB=16384 bsub < jobscripts/exp3-generate-baseline.sh
```

Direct equivalent:

```bash
uv run invoke exp3-generate-baseline --universities=5
```

Paper-canonical UDM defaults are active by default. Pass explicitly if overriding:

```bash
uv run invoke exp3-generate-baseline --universities=5 \
  --args='dataset.mask_base_facts=false dataset.target_ratio=0.0 dataset.negatives_per_positive=1 dataset.inferred_target_limit=250000'
```

### 5. Balance Synthology labels to baseline yield

Matches per-split positive and negative target counts between both generators by
down-sampling Synthology targets. Balanced output is written to
`data/exp3/balanced/owl2bench_<U>/...`; `facts.csv` is unchanged.

HPC jobscript:

```bash
UNIVERSITIES=20 bsub < jobscripts/exp3-balance-data.sh
```

Direct equivalent:

```bash
uv run invoke exp3-balance-data --universities=5
```

**What this step does:**
- Matches Synthology positive and negative target counts to baseline counts per split,
  sampling with a fixed seed.

**What this step does not do:**
- Does not match hop-depth distribution, predicate distribution, graph topology, or
  entity-degree statistics.
- Does not rebalance the baseline; only down-samples Synthology.

### 6. Freeze the Exp 3 test split for reproducible evaluation

HPC jobscript:

```bash
UNIVERSITIES=20 bsub < jobscripts/exp3-generate-gold-test.sh
```

Direct equivalent:

```bash
uv run invoke exp3-generate-gold-test --universities=5
```

### 7. Optional: materialize any existing ABox directly

```bash
uv run invoke exp3-materialize-abox \
  --abox=data/owl2bench/output/raw/owl2bench_20/OWL2RL-20.owl \
  --tbox=ontologies/UNIV-BENCH-OWL2RL.owl \
  --closure-out=data/exp3/baseline/owl2bench_20/closure.nt \
  --inferred-out=data/exp3/baseline/owl2bench_20/inferred.nt \
  --jena-profile=owl_mini
```

### 8. Exp 3 comparison report and baseline diagnostics

HPC jobscript:

```bash
bsub < jobscripts/exp3-report-and-analyze.sh
```

Direct equivalent:

```bash
uv run invoke exp3-report-and-analyze-hpc --config-path=configs/experiments/exp3_hpc.yaml
```

Produces:

- Exp 3 compare report (including missing inferred predicates per method):
  `reports/experiment_runs/.../exp3/report_data/.../report/`
- Latest baseline diagnostics archive:
  `reports/experiment_runs/.../exp3/analyze_latest_baseline/.../analysis/`

### 9. Train RRN on both Exp 3 arms

HPC jobscripts:

```bash
UNIVERSITIES=20 bsub < jobscripts/exp3-train-baseline.sh
UNIVERSITIES=20 bsub < jobscripts/exp3-train-synthology.sh
```

Direct equivalents:

```bash
uv run invoke exp3-train-rrn --dataset=baseline --universities=5
uv run invoke exp3-train-rrn --dataset=synthology --universities=5
```

The Synthology arm automatically prefers balanced data at
`data/exp3/balanced/owl2bench_<U>/` when present.

**Expected outputs:**

- Raw OWL2Bench ABox in `data/owl2bench/output/raw/...`
- Closure/inferred triples in `data/exp3/baseline/...`
- Split CSVs in `data/owl2bench/output/...`
- Synthology split CSVs in `data/exp3/synthology/...`
- Balanced Synthology split CSVs in `data/exp3/balanced/...`

---

## Fair Comparison Protocol

Use the following protocol for paper-grade fairness and traceability:

1. Keep canonical UDM baseline semantics fixed:
   - Single-pass Jena (`iterative: false`, `max_iterations: 1`)
   - `dataset.mask_base_facts=false`
   - `dataset.target_ratio=0.0`
   - `dataset.negatives_per_positive=1`
2. Generate baseline and Synthology arms independently; do not overwrite paths.
3. Run `exp3-balance-data` to match per-split positive/negative target counts.
4. Freeze one gold test split and reuse it for both models.
5. Train with an identical model/training config for both arms; only dataset paths differ.
6. Run multiple seeds per arm and report mean ± confidence intervals.
7. Report both model metrics and data diagnostics:
   - **Metrics:** PR-AUC, AUC-ROC, F1, FPR
   - **Diagnostics:** label counts, hop distributions, predicate skew, runtime costs

**Interpretation note:** The balancing step provides fairness on label volume, not full
structural parity. Claims should therefore be supported by both metric improvements and
structural diagnostics, not by label matching alone.

---

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

---

## Paper Plots and Figures

All plotting commands below assume Exp 2 and Exp 3 datasets are already generated.
Run them after all experiments complete, or on HPC when the data is present.

### Combined data-level figures (standard)

```bash
uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
  --exp3-targets=data/owl2bench/output/owl2bench_20/train/targets.csv \
  --exp3-abox=data/owl2bench/output/raw/owl2bench_20/OWL2RL-20.owl \
  --exp3-inferred=data/exp3/baseline/owl2bench_20/inferred.nt \
  --out-dir=reports/paper
```

Expected outputs in `reports/paper/` (each saved as both `.png` and `.pdf`):

- `exp2_base_vs_inferred.{png,pdf}`
- `exp2_hops_distribution.{png,pdf}`
- `exp2_parity_attempts.{png,pdf}`
- `exp3_base_vs_inferred.{png,pdf}`
- `exp3_hops_distribution.{png,pdf}` (optional)
- `summary.json`

### Compact reviewer figures (small PDFs)

Generate compact side-by-side Synthology vs UDM graphs for both Family Tree and OWL2Bench:

```bash
# Ensure Family Tree inputs exist
uv run invoke exp2-generate-synthology
uv run invoke exp2-generate-baseline

# Generate OWL2Bench inputs (store Synthology output separately)
uv run invoke exp3-generate-synthology --universities=1 \
  --args="dataset.output_dir=data/exp3/smoke/synth_ref dataset.inferred_target_limit=80000 dataset.bfs.sample_count=1200 dataset.bfs.max_individuals_per_sample=100"
uv run invoke exp3-generate-baseline --universities=1

# Render compact figures
uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-baseline-targets=data/exp2/baseline/family_tree/train/targets.csv \
  --exp2-parity-summary=reports/experiment_runs/2026-04-01/exp2/parity_report/210943_parity/parity_report.json \
  --exp3-synth-targets=data/exp3/smoke/synth_ref/owl2bench_1/train/targets.csv \
  --exp3-baseline-targets=data/owl2bench/output/owl2bench_1/train/targets.csv \
  --out-dir=reports/paper_small_graphs
```

Expected outputs in `reports/paper_small_graphs/`:

- `family_tree_density_small.{png,pdf}`
- `family_tree_multihop_small.{png,pdf}`
- `owl2bench_density_small.{png,pdf}`
- `owl2bench_multihop_small.{png,pdf}`
- `summary.json`

### HPC full-size plots (when data is already present)

```bash
uv run invoke exp2-report-data
uv run invoke exp3-report-data --universities=50 \
  --baseline-path=data/owl2bench/output_baseline/owl2bench_200 \
  --synthology-path=data/owl2bench/output/owl2bench_200

uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-baseline-targets=data/exp2/baseline/family_tree/train/targets.csv \
  --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_report.json \
  --exp3-synth-targets=data/owl2bench/output/owl2bench_200/train/targets.csv \
  --exp3-baseline-targets=data/owl2bench/output_baseline/owl2bench_200/train/targets.csv \
  --exp3-abox=data/owl2bench/output/raw/owl2bench_200/OWL2RL-50.owl \
  --exp3-inferred=data/exp3/baseline/owl2bench_200/inferred.nt \
  --out-dir=reports/paper_hpc
```

Adjust `--exp3-baseline-targets` and `--exp2-parity-summary` paths if your HPC run stores
them in a different location.

### MATLAB hop distributions (final figure styling)

Produces color-consistent, LaTeX-labeled hop charts from the compare reports:

```bash
cd matlab
exp23_hop_distribution
```

Outputs:

- `paper/figures/exp2_hop_distr.pdf`
- `paper/figures/exp3_hop_distr.pdf`

By default the script uses the latest Exp 2/Exp 3 compare run summaries. For strict
reproducibility, pin exact summary files via `exp2SummaryOverride` and
`exp3SummaryOverride` at the top of the MATLAB script.

### Auto-fill paper tables

Export LaTeX row snippets for all paper result tables directly from run artifacts:

```bash
uv run invoke paper-export-tables \
  --out-dir=paper/generated \
  --model-metrics=paper/metrics/model_results.json
```

Generated files:

- `paper/generated/exp1_results_rows.tex`
- `paper/generated/overall_performance_rows.tex`
- `paper/generated/generation_metrics_rows.tex`
- `paper/generated/timing_breakdown_rows.tex`

Use these snippets to populate the corresponding tables in `paper/paper.tex` reproducibly.

---

## Visual Inspection

### OWL2Bench KG sample renderings

```bash
# UDM baseline sample
uv run --package kgvisualiser python -m kgvisualiser.visualize \
  io.input_csv=data/owl2bench/output/owl2bench_1/train/facts.csv \
  io.targets_csv=data/owl2bench/output/owl2bench_1/train/targets.csv \
  io.sample_id=710367 \
  output.dir=visual-verification/graphs \
  output.name_template=owl2bench_baseline_sample_710367 \
  output.format=pdf \
  filters.include_negatives=false \
  filters.max_edges=90 \
  render.class_nodes=false \
  render.show_edge_labels=true

# Synthology sample
uv run --package kgvisualiser python -m kgvisualiser.visualize \
  io.input_csv=data/exp3/smoke/synth_ref/owl2bench_1/train/facts.csv \
  io.targets_csv=data/exp3/smoke/synth_ref/owl2bench_1/train/targets.csv \
  io.sample_id=710000 \
  output.dir=visual-verification/graphs \
  output.name_template=owl2bench_synthology_sample_710000 \
  output.format=pdf \
  filters.include_negatives=false \
  filters.max_edges=90 \
  render.class_nodes=false \
  render.show_edge_labels=true
```

---

## Notes for Paper Reproducibility

- Jena library version: `5.2.0`
- Default Jena profile: `owl_mini`
- Baseline reasoning semantics: one-shot Jena closure call (Jena performs internal fixpoint)
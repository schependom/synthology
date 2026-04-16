# PAPER_RUNBOOK.md

This runbook gives a practical, reproducible order to generate the datasets, train models, and produce the figures/artifacts referenced by the paper.

## 0. Preflight

```bash
uv sync
uv run invoke --list
```

### HPC preflight for Jena-backed steps (Exp2 baseline, OWL2Bench materialization)

```bash
# Load Java 21 (required by Jena 5.x)
module load openjdk/21

# Verify Java is available and correct version
which java && java -version

# Now install Maven
./install-mvn.sh

# Verify Maven is available
which mvn && mvn -v

# Optional but recommended: one global heap setting for all experiment commands
# (applies to Maven-backed Java runs and Jena materialization helper).
export SYNTHOLOGY_HEAP_MB=16384
```

## Logging And Artifacts

- Canonical experiment records are written to `reports/experiment_runs/<YYYY-MM-DD>/<experiment>/<task>/<timestamp>/`.
- Each run archive includes `manifest.json`, `run.log`, copied configs, and copied artifacts.
- RRN runs also write metrics and checkpoints to Weights & Biases plus the run archive's `wandb/`, `lightning_logs/`, and `checkpoints/` folders.
- Hydra-managed application outputs still appear under their app-specific `outputs/` directories when applicable, but the task archive is the primary paper record.

## 1. Experiment 1 (Negative Sampling)

1. Generate train/val datasets for all three strategies:

```bash
uv run invoke exp1-generate-trainval-sets
```

2. Generate and freeze hard-negative test set:

```bash
uv run invoke exp1-generate-test-set
```

3. Train one model per strategy:

```bash
uv run invoke exp1-train-rrn --strategy=random
uv run invoke exp1-train-rrn --strategy=constrained
uv run invoke exp1-train-rrn --strategy=proof_based
```

Expected outputs:

- datasets in `data/exp1/...`
- training/eval metrics in W&B and logs

## 2. Experiment 2 (Multi-Hop Quality)

0. Scientific guardrail (do not skip):

- Exp2 is run as a matched-budget comparison: shared fact cap, shared target cap, matched split sample counts, and matched split positive/negative target counts.
- Deep-hop distributions are not forced to parity; they are reported as diagnostics alongside model results.

1. Generate and freeze deep test set:

```bash
uv run invoke exp2-generate-gold-test
```

2. Generate matched-budget train/val data for both arms:

```bash
uv run invoke exp2-balance-datasets --fact-cap=<Nf> --target-cap=<Nt>
```

Default behavior and source of values:

- `fact_cap` is required by the task.
- `target_cap` is optional; when omitted, each generator falls back to its own config defaults.
- Baseline optional override: `baseline_base_facts` maps to `generator.base_relations_per_sample`.
- Synthology optional override: `synthology_proof_roots` maps to `generator.proof_roots_per_rule`.
- Current config defaults: baseline caps are null in `configs/udm_baseline/exp2_baseline.yaml`; synthology defaults are `train_fact_cap=180000`, `train_target_cap=140000`, `proof_roots_per_rule=8` in `configs/ont_generator/exp2_synthology.yaml`.

HPC preset (recommended for reproducible matched-budget runs):

```bash
bsub < jobscripts/exp2-balance-datasets.sh
```

The preset reads `configs/experiments/exp2_balance_hpc.yaml` and runs:

```bash
uv run invoke exp2-balance-datasets-hpc --config-path=configs/experiments/exp2_balance_hpc.yaml
```

3. Generate data comparison report (includes budget + hop diagnostics):

```bash
uv run invoke exp2-report-data
```

4. Train models:

```bash
uv run invoke exp2-train-rrn --dataset=baseline
uv run invoke exp2-train-rrn --dataset=synthology
```

Expected outputs:

- train data in `data/exp2/baseline/...` and `data/exp2/synthology/...`
- model metrics in W&B/logs

## 3. Experiment 3 (OWL2Bench Generalization)

1. Fast toy run (recommended first):

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy
```

2. Generate Exp3 Synthology-side OWL2Bench output:

```bash
SYNTHOLOGY_JENA_XMX_MB=16384 uv run invoke exp3-generate-synthology --universities=5
```

Default Synthology-side outputs:

- `data/exp3/synthology/owl2bench_5/`
- `data/exp3/synthology/raw/owl2bench_5/`

If this still OOMs on your node, re-run with explicit caps:

```bash
SYNTHOLOGY_JENA_XMX_MB=16384 uv run invoke exp3-generate-synthology --universities=5 \
  --args='dataset.reasoning_input_triple_cap=80000 dataset.inferred_target_limit=120000 dataset.bfs.sample_count=1200 dataset.bfs.max_individuals_per_sample=100'
```

Important dependency note:

- Exp3 baseline and Synthology-side data are generated independently; use explicit dataset paths when training/evaluating to avoid path-casing mismatches.

3. Generate Exp3 ABox source only:

```bash
uv run invoke exp3-generate-owl2bench-abox --universities=5
```

4. Generate Exp3 baseline chain (ABox + one-shot Jena closure):

```bash
SYNTHOLOGY_UDM_BASELINE_XMX_MB=16384 uv run invoke exp3-generate-baseline --universities=5
```

Canonical UDM baseline defaults are active by default for this command:

- `dataset.mask_base_facts=false`
- `dataset.target_ratio=0.0`
- `dataset.negatives_per_positive=1`

Or use the global heap knob once in your shell/session and omit per-command heap flags:

```bash
export SYNTHOLOGY_HEAP_MB=16384
uv run invoke exp3-generate-baseline --universities=5
```

HPC heap guardrail:

- If your cluster profile exports `JAVA_TOOL_OPTIONS` or `_JAVA_OPTIONS` with `-Xmx...`, that can force a smaller heap than expected for Jena materialization.
- Keep `SYNTHOLOGY_UDM_BASELINE_XMX_MB` as the single source of heap sizing for this step (for example by removing inherited `-Xmx` defaults in your job environment).

Scale decision note:

- Exp3 default university count in this runbook is 5 for practical one-week turnaround; increase with `--universities=<N>` when compute budget allows.

Path-casing guardrail:

- Current generation tasks write lowercase paths under data/owl2bench/output/....
- Some training configs may still reference data/OWL2Bench/output/.... If so, pass explicit lowercase data path overrides to avoid silently training on stale/missing data.

5. Optional direct materialization command:

```bash
uv run invoke exp3-materialize-abox \
  --abox=data/owl2bench/output/raw/owl2bench_5/OWL2RL-5.owl \
  --tbox=ontologies/UNIV-BENCH-OWL2RL.owl \
  --closure-out=data/exp3/baseline/owl2bench_5/closure.nt \
  --inferred-out=data/exp3/baseline/owl2bench_5/inferred.nt \
  --jena-profile=owl_mini
```

6. Match Synthology train/val/test target counts to baseline:

```bash
uv run invoke exp3-balance-data --universities=5
```

7. Freeze Exp3 test split used for evaluation:

```bash
uv run invoke exp3-generate-gold-test --universities=5
```

8. Train RRN on both Exp3 arms:

```bash
uv run invoke exp3-train-rrn --dataset=baseline --universities=5
uv run invoke exp3-train-rrn --dataset=synthology --universities=5
```

Expected outputs:

- raw OWL2Bench ABox in `data/owl2bench/output/raw/...`
- closure/inferred triples in `data/exp3/baseline/...`
- split CSVs in `data/owl2bench/output/...`
- synthology split CSVs in `data/exp3/synthology/...`
- balanced synthology split CSVs in `data/exp3/balanced/...`

## 4. Paper Plots and Graphs

Generate combined figures used for data-level comparisons:

```bash
uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
  --exp3-targets=data/owl2bench/output/owl2bench_5/train/targets.csv \
  --exp3-abox=data/owl2bench/output/raw/owl2bench_5/OWL2RL-20.owl \
  --exp3-inferred=data/exp3/baseline/owl2bench_5/inferred.nt \
  --out-dir=reports/paper
```

Expected outputs in `reports/paper/` (each plot is saved as both `.png` and `.pdf`):

- `exp2_base_vs_inferred.{png,pdf}`
- `exp2_hops_distribution.{png,pdf}`
- `exp2_parity_attempts.{png,pdf}`
- `exp3_base_vs_inferred.{png,pdf}`
- optional `exp3_hops_distribution.{png,pdf}`
- `summary.json`

### Compact Reviewer Figures (Small PDFs)

Generate compact side-by-side Synthology vs UDM baseline graphs for both Family Tree and OWL2Bench:

```bash
# Family Tree inputs
uv run invoke exp2-generate-synthology
uv run invoke exp2-generate-baseline

# OWL2Bench inputs (store synthology output separately)
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

Expected compact outputs in `reports/paper_small_graphs/`:

- `family_tree_density_small.{png,pdf}`
- `family_tree_multihop_small.{png,pdf}`
- `owl2bench_density_small.{png,pdf}`
- `owl2bench_multihop_small.{png,pdf}`
- `summary.json`

### HPC Full-Size Plots (Existing Data)

When full Exp2/Exp3 datasets are already present on HPC, run only analysis/plotting commands:

```bash
uv run invoke exp2-report-data
uv run invoke exp3-report-data --universities=50 \
  --baseline-path=data/owl2bench/output_baseline/owl2bench_50 \
  --synthology-path=data/owl2bench/output/owl2bench_50

uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-baseline-targets=data/exp2/baseline/family_tree/train/targets.csv \
  --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_report.json \
  --exp3-synth-targets=data/owl2bench/output/owl2bench_50/train/targets.csv \
  --exp3-baseline-targets=data/owl2bench/output_baseline/owl2bench_50/train/targets.csv \
  --exp3-abox=data/owl2bench/output/raw/owl2bench_50/OWL2RL-50.owl \
  --exp3-inferred=data/exp3/baseline/owl2bench_50/inferred.nt \
  --out-dir=reports/paper_hpc
```

Adjust `--exp3-baseline-targets` and `--exp2-parity-summary` paths if your HPC run stores them in a different existing location.

### Auto-fill Paper Tables

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

### MATLAB Hop Distributions (Final Figure Styling)

To produce color-consistent, LaTeX-labeled hop charts from the compare reports:

```bash
cd matlab
exp23_hop_distribution
```

Outputs:

- `paper/figures/exp2_hop_distr.pdf`
- `paper/figures/exp3_hop_distr.pdf` (if Exp3 compare summary is available)

By default the script uses the latest Exp2/Exp3 compare run summaries; for strict reproducibility you can pin exact summary files via `exp2SummaryOverride` and `exp3SummaryOverride` at the top of the MATLAB script.

### Local Small KG Visuals (Actual Graph Renderings)

```bash
# Family tree graph samples
uv run invoke synthology-visual-verification
uv run invoke udm-visual-verification --n-samples=3

# OWL2Bench graph samples
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

## 5. Visual Inspection Artifacts

1. Exp2 smoke graph:

```bash
uv run invoke exp2-smoke-jena-visual
```

Output:

- `visual-verification/exp2_smoke/exp2_jena_smoke_1000.pdf`

2. Ont-generator sample visualization:

```bash
uv run invoke gen-ft-ont
uv run --package kgvisualiser python -m kgvisualiser.visualize \
  io.input_csv=data/ont/family_tree/train/targets.csv \
  io.sample_id=1000 \
  output.dir=visual-verification/ont_generator \
  output.name_template=ont_sample_1000
```

## 6. Notes for Paper Reproducibility

- Jena library version currently used: `5.2.0`
- Default Jena profile currently used: `owl_mini`
- Baseline reasoning semantics: one-shot Jena closure call (Jena performs internal fixpoint)

TODO before camera-ready:

- freeze one final profile (`owl_micro`, `owl_mini`, or `owl_full`) for all reported UDM runs
- add small profile-sensitivity table in appendix

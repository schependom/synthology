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

- Exp2's core claim depends on parity enforcement. If parity is skipped, any Synthology gain can be attributed to deep-fact volume alone, which invalidates the main Exp2 argument.

1. Generate and freeze deep test set:

```bash
uv run invoke exp2-generate-gold-test
```

2. Generate Synthology data first (sets parity target):

```bash
uv run invoke exp2-generate-synthology
```

3. Inspect deep-hop target volume before baseline parity:

- Check the d>=3 count in the generated Synthology train targets.
- If parity appears unreachable in reasonable attempts, reduce Synthology generation pressure (for example lower proof roots per rule and/or fact cap), then regenerate.

4. Generate baseline data:

```bash
uv run invoke exp2-generate-baseline
```

5. Optional budget-matched generation helper:

```bash
uv run invoke exp2-balance-datasets --fact-cap=<Nf> --target-cap=<Nt>
```

6. Run parity loop and summarize parity attempts (recommended tolerance mode):

```bash
uv run invoke exp2-parity-loop \
  --deep-count-mode=tolerance \
  --tolerance-pct=10.0
uv run invoke exp2-parity-report
```

Tip:

- Exp2 default now uses a wider 10% tolerance band to improve practical parity convergence under compute constraints.
- Prefer tolerance mode (5-10%) over exact deep-count matching for practical convergence while preserving scientific defensibility.

7. Generate data comparison report:

```bash
uv run invoke exp2-report-data
```

8. Train models:

```bash
uv run invoke exp2-train-rrn --dataset=baseline
uv run invoke exp2-train-rrn --dataset=synthology
```

Expected outputs:

- parity artifacts in `data/exp2/baseline/parity_runs/`
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
  --exp3-targets=data/owl2bench/output/owl2bench_20/train/targets.csv \
  --exp3-abox=data/owl2bench/output/raw/owl2bench_20/OWL2RL-20.owl \
  --exp3-inferred=data/exp3/baseline/owl2bench_20/inferred.nt \
  --out-dir=reports/paper
```

Expected outputs in `reports/paper/` (each plot is saved as both `.png` and `.pdf`):

- `exp2_base_vs_inferred.{png,pdf}`
- `exp2_hops_distribution.{png,pdf}`
- `exp2_parity_attempts.{png,pdf}`
- `exp3_base_vs_inferred.{png,pdf}`
- optional `exp3_hops_distribution.{png,pdf}`
- `summary.json`

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

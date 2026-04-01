# PAPER_RUNBOOK.md

This runbook gives a practical, reproducible order to generate the datasets, train models, and produce the figures/artifacts referenced by the paper.

## 0. Preflight

```bash
uv sync
uv run invoke --list
```

Optional quick sanity checks:

```bash
uv run invoke exp2-smoke-jena-visual
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy
```

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

1. Generate and freeze deep test set:

```bash
uv run invoke exp2-generate-gold-test
```

2. Generate baseline data:

```bash
uv run invoke exp2-generate-baseline
```

3. Generate Synthology data:

```bash
uv run invoke exp2-generate-synthology
```

4. Optional budget-matched generation helper:

```bash
uv run invoke exp2-balance-datasets --fact-cap=<Nf> --target-cap=<Nt>
```

5. Run parity loop and summarize parity attempts:

```bash
uv run invoke exp2-parity-loop
uv run invoke exp2-parity-report
```

6. Generate data comparison report:

```bash
uv run invoke exp2-report-data
```

7. Train models:

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

2. Generate OWL2Bench pipeline output:

```bash
uv run invoke gen-owl2bench
```

3. Generate Exp3 ABox source only:

```bash
uv run invoke exp3-generate-owl2bench-abox --universities=50
```

4. Generate Exp3 baseline chain (ABox + one-shot Jena closure):

```bash
uv run invoke exp3-generate-baseline --universities=50
```

5. Optional direct materialization command:

```bash
uv run invoke exp3-materialize-abox \
  --abox=data/owl2bench/output/raw/owl2bench_50/OWL2RL-50.owl \
  --tbox=data/owl2bench/input/UNIV-BENCH-OWL2RL.owl \
  --closure-out=data/exp3/baseline/owl2bench_50/closure.nt \
  --inferred-out=data/exp3/baseline/owl2bench_50/inferred.nt \
  --jena-profile=owl_mini
```

Expected outputs:

- raw OWL2Bench ABox in `data/owl2bench/output/raw/...`
- closure/inferred triples in `data/exp3/baseline/...`
- split CSVs in `data/owl2bench/output/...`

## 4. Paper Plots and Graphs

Generate combined figures used for data-level comparisons:

```bash
uv run invoke paper-visual-report \
  --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
  --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
  --exp3-targets=data/owl2bench/output/owl2bench_50/train/targets.csv \
  --exp3-abox=data/owl2bench/output/raw/owl2bench_50/OWL2RL-50.owl \
  --exp3-inferred=data/exp3/baseline/owl2bench_50/inferred.nt \
  --out-dir=reports/paper
```

Expected outputs in `reports/paper/`:

- `exp2_base_vs_inferred.png`
- `exp2_hops_distribution.png`
- `exp2_parity_attempts.png`
- `exp3_base_vs_inferred.png`
- optional `exp3_hops_distribution.png`
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

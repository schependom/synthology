## Visual verification

This section groups quick visual sanity checks by generator/baseline, so you can inspect outputs before running full experiments.

Jena setup in code/configs:

- Apache Jena libraries: `5.2.0`
- Jena reasoner profile default: `owl_mini`
- Materialization semantics: one-shot closure call (internal fixpoint in Jena)

TODO (camera-ready paper reproducibility): freeze and document one final profile choice for all reported runs (`owl_micro` vs `owl_mini` vs `owl_full`) and include a profile-sensitivity appendix table.

### Category A: OWL2Bench generator checks

Use these commands to confirm the OWL2Bench pipeline generates raw OWL artifacts and parsed split files correctly.

1. **Toy end-to-end OWL2Bench run + auto-visualization**

    ```bash
    uv run invoke gen-owl2bench-toy
    ```

    This runs a small OWL2Bench configuration (`config_toy`), performs materialization, exports split CSV files, and then visualizes a sample graph for a fast end-to-end smoke check.
    If it is still too slow on your machine, reduce the reasoning subset temporarily:

    ```bash
    uv run invoke gen-owl2bench-toy --args="dataset.reasoning_input_triple_cap=3000"
    ```

2. **Full OWL2Bench pipeline run**

    ```bash
    uv run invoke gen-owl2bench
    ```

    This runs the standard OWL2Bench generation/materialization/export pipeline for larger-scale verification and stores results under `data/owl2bench/output`.

3. **Exp3-style OWL2Bench ABox generation path**

    ```bash
    uv run invoke exp3-generate-owl2bench-abox --universities=50
    ```

    This is the experiment-oriented entrypoint that generates OWL2Bench data with the requested university count, used as the baseline ABox source for Exp3.

### Category B: UDM baseline checks

Use these commands to verify Apache Jena-backed UDM materialization and inspect generated baseline samples.

1. **Visual smoke test for UDM + Jena (recommended first check)**

    ```bash
    uv run invoke exp2-smoke-jena-visual
    ```

    This generates a tiny baseline dataset with Jena reasoning and writes a rendered sample graph to `visual-verification/exp2_smoke`, which is useful for quickly checking inferred-fact presence and graph structure.

2. **Paper-ready UDM visual verification (Synthology-comparable PDFs)**


    ```bash
    uv run invoke udm-visual-verification
    ```

    This generates UDM baseline samples with visual settings aligned to `synthology-visual-verification` (same split style and explicit graph naming), then renders train-sample PDFs to `visual-verification/graphs` with names like `udm_baseline_sample_1001.pdf` for side-by-side paper figures.
    The command uses a balanced fast profile by default (relation labels enabled + moderately larger graphs + capped edges for runtime). If you want denser/heavier plots, override explicitly via `--args`, for example:

    ```bash
    uv run invoke udm-visual-verification --n-samples=1 --args="filters.include_negatives=true filters.max_edges=120 render.show_edge_labels=true"
    ```

3. **Family-tree UDM baseline generation (task wrapper)**


    ```bash
    uv run invoke gen-ft-fc
    ```

    This is the reusable UDM baseline generation command for family-tree style data and is the quickest way to validate that baseline `facts.csv` and `targets.csv` generation is healthy.

4. **Exp3 baseline chaining (OWL2Bench generation + UDM materialization)**

    ```bash
    uv run invoke exp3-generate-baseline --universities=50
    ```

    This runs the baseline chain used in Exp3: OWL2Bench ABox generation followed by UDM/Jena materialization, producing closure/inferred artifacts for benchmarking.

5. **Direct ABox materialization with UDM/Jena**

    ```bash
    uv run invoke exp3-materialize-abox \
      --abox=path/to/owl2bench_abox.ttl \
    --tbox=ontologies/UNIV-BENCH-OWL2RL.owl \
      --closure-out=outputs/exp3/closure.nt \
      --inferred-out=outputs/exp3/inferred.nt
    ```

    Use this when you already have an ABox and only want to validate the materialization layer independently from generation.

### Category C: Synthology ont_generator checks

Use these commands to visually verify the backward-chaining generator output and sample-level graph quality.

1. **Generate family-tree data with Synthology (task wrapper)**

    ```bash
    uv run invoke gen-ft-ont
    ```

    This produces standard-format outputs for `train/val/test`, which you can inspect for depth, fact types, and negative-sampling structure.

2. **Generate Exp2 Synthology dataset (experiment path)**

    ```bash
    uv run invoke exp2-generate-synthology
    ```

    This executes the Exp2-aligned Synthology generation path so you can verify the same configuration family used for parity and model comparisons.

3. **Render a selected sample graph from generated CSVs**

    ```bash
    uv run --package kgvisualiser python -m kgvisualiser.visualize \
      io.input_csv=data/ont/family_tree/train/targets.csv \
      io.sample_id=1000 \
      output.dir=visual-verification/ont_generator \
      output.name_template=ont_sample_1000
    ```

    This explicit visualization command is useful when you want to inspect one graph in detail (for example, to confirm multi-hop inferred paths and corruption patterns).

### Category D: Cross-generator paper plots

Use this report command when you want a side-by-side visual summary of baseline vs Synthology behavior.

1. **Generate paper-ready visual diagnostics**

    ```bash
    uv run invoke paper-visual-report \
      --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
      --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
      --exp3-targets=data/owl2bench/output/owl2bench_200/train/targets.csv \
      --exp3-abox=data/owl2bench/output/raw/owl2bench_200/OWL2RL-50.owl \
      --exp3-inferred=data/exp3/baseline/owl2bench_200/inferred.nt \
      --out-dir=reports/paper
    ```

    This generates consolidated inspection plots (base vs inferred, hop distributions, parity-attempt trend) so you can validate dataset behavior before or alongside model training.

2. **Generate compact reviewer-facing PDFs (family tree + OWL2Bench)**

    Use this sequence to create small side-by-side Synthology vs UDM baseline graphs in PDF format.

    ```bash
    # 1) Ensure Exp2 comparison inputs exist (family tree)
    uv run invoke exp2-generate-synthology
    uv run invoke exp2-generate-baseline

    # 2) Ensure Exp3 comparison inputs exist (OWL2Bench)
    #    (Synthology output is written to a dedicated path so it won't be overwritten.)
    uv run invoke exp3-generate-synthology --universities=1 \
      --args="dataset.output_dir=data/exp3/smoke/synth_ref dataset.inferred_target_limit=80000 dataset.bfs.sample_count=1200 dataset.bfs.max_individuals_per_sample=100"
    uv run invoke exp3-generate-baseline --universities=1

    # 3) Render compact paper graphs
    uv run invoke paper-visual-report \
      --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
      --exp2-baseline-targets=data/exp2/baseline/family_tree/train/targets.csv \
      --exp2-parity-summary=reports/experiment_runs/2026-04-01/exp2/parity_report/210943_parity/parity_report.json \
      --exp3-synth-targets=data/exp3/smoke/synth_ref/owl2bench_1/train/targets.csv \
      --exp3-baseline-targets=data/owl2bench/output/owl2bench_1/train/targets.csv \
      --out-dir=reports/paper_small_graphs
    ```

    Expected compact outputs (both `.png` and `.pdf`) in `reports/paper_small_graphs/`:
    - `family_tree_density_small.*`
    - `family_tree_multihop_small.*`
    - `owl2bench_density_small.*`
    - `owl2bench_multihop_small.*`
    - `summary.json`

3. **HPC commands for full-sized chart generation (no re-generation of data)**

    If your HPC workspace already contains generated Exp2/Exp3 datasets, run only the plotting/reporting commands below.

    ```bash
    # Exp2 distribution report (bar plots, hops, predicate/type distributions)
    uv run invoke exp2-report-data

    # Exp3 distribution report (same analyzer, Exp3 compare config)
    uv run invoke exp3-report-data --universities=50 \
      --baseline-path=data/owl2bench/output_baseline/owl2bench_200 \
      --synthology-path=data/owl2bench/output/owl2bench_200

    # Combined paper-style charts from existing full datasets
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

    Notes:
    - If your baseline split CSVs are stored under a different path than `data/owl2bench/output_baseline/...`, replace `--exp3-baseline-targets` with the correct existing location.
    - If parity summary is archived under `reports/experiment_runs/...`, pass that file instead of `data/exp2/baseline/parity_runs/parity_report.json`.

4. **Export LaTeX table rows directly from run artifacts**

    ```bash
    uv run invoke paper-export-tables \
      --out-dir=paper/generated \
      --model-metrics=paper/metrics/model_results.json
    ```

    This writes reusable row snippets for the paper tables:
    - `paper/generated/exp1_results_rows.tex`
    - `paper/generated/overall_performance_rows.tex`
    - `paper/generated/generation_metrics_rows.tex`
    - `paper/generated/timing_breakdown_rows.tex`

    Use these snippets to populate the corresponding result tables in the manuscript reproducibly.

5. **Generate publication-quality hop charts with MATLAB (KUL colors + LaTeX labels)**

    ```bash
    # Make sure compare summaries exist first
    uv run invoke exp2-report-data
    uv run invoke exp3-report-data --universities=50 \
      --baseline-path=data/owl2bench/output_baseline/owl2bench_200 \
      --synthology-path=data/owl2bench/output/owl2bench_200

    # Then run from MATLAB/Octave
    cd matlab
    exp23_hop_distribution
    ```

    The script reads method dataset paths from the latest compare run summaries and exports:
    - `paper/figures/exp2_hop_distr.pdf`
    - `paper/figures/exp3_hop_distr.pdf` (when Exp3 summary exists)

    To pin a specific run instead of using latest, set `exp2SummaryOverride` / `exp3SummaryOverride` at the top of `matlab/exp23_hop_distribution.m`.

6. **Generate small local KG PDFs (actual graph visuals)**

    These commands generate visual graph samples (not aggregate charts), so reviewers can inspect how the KGs look.

    ```bash
    # Family Tree: Synthology + UDM baseline small graph sets
    uv run invoke synthology-visual-verification
    uv run invoke udm-visual-verification --n-samples=3

    # OWL2Bench: one baseline sample and one synthology sample
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

    Expected local KG PDFs in `visual-verification/graphs/` include:
    - `train_sample_0.pdf`, `train_sample_1.pdf`, `train_sample_2.pdf` (Synthology family tree)
    - `udm_baseline_sample_*.pdf` (UDM family tree)
    - `owl2bench_baseline_sample_710367.pdf`
    - `owl2bench_synthology_sample_710000.pdf`
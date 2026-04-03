# Experiment 3: Generalization to Complex Ontologies (OWL2Bench)

## Paper-aligned goal

Validate that Synthology remains advantageous on a larger ontology (OWL2Bench), while explicitly quantifying how much UDM/Jena supervision is dominated by shallow or trivial inferences.

## Critical correction: Jena semantics for Exp3 (Single-Pass)

Unlike Experiment 2, Experiment 3 operates on massive OWL2Bench graphs. Running iterative diffing on these scales results in prohibitive computational overhead (OOM errors and timeouts). Therefore, for each OWL2Bench ABox, Jena must be run **once** to compute the full fixed-point closure.

- Jena performs the internal fixpoint iteration natively.
- All inferred facts resulting from this single-pass materialization will be assigned a default `hops=1` label.
- Because we rely on BFS subgraph sampling rather than a strict parity-retry loop, accurate hop-depth recovery is not required for data generation.

## Mandatory reporting additions

Because the single-pass baseline defaults all inferred facts to `hops=1`, direct depth-bucketed comparisons (1-hop vs 2-plus-hop) against Synthology cannot rely on the baseline's target labels. Instead, report:

- Trivial-schema bucket share: The raw volume and percentage of the baseline's closure dominated by `rdf:type` and schema-propagation inferences.
- This supports the paper's core argument regarding trivial-fact composition and shallow-signal dominance in unguided materialization.

## Practical quick start (UDM + Apache Jena)

1. Exp3 baseline chain:

```bash
uv run invoke exp3-generate-baseline --universities=50
```

2. Direct materialization for an existing ABox:

```bash
uv run invoke exp3-materialize-abox \
    --abox=path/to/owl2bench_abox.ttl \
    --tbox=ontologies/UNIV-BENCH-OWL2RL.owl \
    --closure-out=outputs/exp3/closure.nt \
    --inferred-out=outputs/exp3/inferred.nt
```

3. Generate paper visuals:

```bash
uv run invoke paper-visual-report \
    --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
    --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
    --exp3-targets=data/owl2bench/output/owl2bench_50/train/targets.csv \
    --exp3-abox=data/owl2bench/output/raw/owl2bench_50/OWL2RL-50.owl \
    --exp3-inferred=data/exp3/baseline/owl2bench_50/inferred.nt \
    --out-dir=reports/paper
```

## Complete command list for Exp3

Use this sequence to verify the refactored Exp3 pipeline and produce report artifacts.

1. Fast toy smoke run (pipeline + visualization):

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy
```

2. Optional reduced toy run for quick CI/dev checks:

```bash
SYNTHOLOGY_JENA_XMX_MB=3072 uv run invoke gen-owl2bench-toy --args='dataset.reasoning_input_triple_cap=600 dataset.bfs.sample_count=6 dataset.inferred_target_limit=200'
```

3. Full OWL2Bench pipeline run:

```bash
uv run invoke gen-owl2bench
```

4. Generate OWL2Bench ABox specifically for Exp3:

```bash
uv run invoke exp3-generate-owl2bench-abox --universities=50
```

5. Run Exp3 baseline chain (ABox generation + Jena materialization):

```bash
uv run invoke exp3-generate-baseline --universities=50
```

6. Direct one-shot materialization for an existing ABox:

```bash
uv run invoke exp3-materialize-abox \
    --abox=path/to/owl2bench_abox.ttl \
    --tbox=ontologies/UNIV-BENCH-OWL2RL.owl \
    --closure-out=outputs/exp3/closure.nt \
    --inferred-out=outputs/exp3/inferred.nt \
    --jena-profile=owl_mini
```

7. Generate cross-experiment paper plots:

```bash
uv run invoke paper-visual-report \
    --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
    --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
    --exp3-targets=data/owl2bench/output/owl2bench_50/train/targets.csv \
    --exp3-abox=data/owl2bench/output/raw/owl2bench_50/OWL2RL-50.owl \
    --exp3-inferred=data/exp3/baseline/owl2bench_50/inferred.nt \
    --out-dir=reports/paper
```

8. Run Exp3 parity loop and report:

```bash
uv run invoke exp3-parity-loop --universities=50 \
    --synth-targets=data/owl2bench/output/owl2bench_50/train/targets.csv \
    --synth-facts=data/owl2bench/output/owl2bench_50/train/facts.csv

uv run invoke exp3-parity-report \
    --synth-targets=data/owl2bench/output/owl2bench_50/train/targets.csv \
    --synth-facts=data/owl2bench/output/owl2bench_50/train/facts.csv
```

For a monorepo-wide end-to-end protocol (Exp1/2/3 + artifacts), see `experiments/PAPER_RUNBOOK.md`.

## Exact code changes needed

1. **Enforce Single-Shot Jena Closure:**
    - File: `apps/OWL2Bench/src/owl2bench/pipeline.py`
    - Change: Ensure the reasoning pipeline calls `jena.materialize(...)` exactly once per ABox.
2. **Disable Iterative Flags:**
    - Files: `configs/owl2bench/config.yaml`, `configs/owl2bench/config_toy.yaml`
    - Change: Ensure `materialization.iterative` is set to `false`.
3. **Adjust Reporting Scripts:**
    - File: `apps/data_reporter/src/data_reporter/paper_plots.py`
    - Change: When plotting Exp 3 data, acknowledge that baseline deep-hop counts are fundamentally obscured by the single-pass limitation, highlighting Synthology's native ability to provide this metadata.

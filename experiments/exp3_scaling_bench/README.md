# Experiment 3: Generalization to Complex Ontologies (OWL2Bench)

## Paper-aligned goal

Validate that Synthology remains advantageous on a larger ontology (OWL2Bench), while explicitly quantifying how much UDM/Jena supervision is dominated by shallow or trivial inferences.

## Critical correction: Jena semantics for Exp3

For each OWL2Bench ABox, Jena must be run once to compute closure.

- Jena reasoner performs internal fixpoint iteration itself.
- External repeated Jena calls on the same evolving closure are not part of intended baseline semantics.
- Experiment-level repeats are allowed only across different generated ABoxes (different seeds or sizes).

## Updated Exp3 procedure

1. Generate OWL2Bench ABox(s) with target university count.
2. For each ABox, run one-shot Jena closure with explicit reasoner profile.
3. Export base facts and inferred facts in standard CSV format.
4. Build Synthology dataset under matched data budget.
5. Train two RRN models (baseline vs Synthology) with identical settings.
6. Evaluate on shared held-out test data.
7. Report depth-stratified metrics and trivial-fact composition.

## Mandatory reporting additions

In addition to global metrics, report these breakdowns for both training data and evaluation slices:

- 1-hop bucket
- 2-plus-hop bucket
- Trivial-schema bucket share (`rdf:type` and schema-propagation-heavy subset)

This is required to support the paper argument about trivial-fact criticism and shallow-signal dominance.

## Practical quick start (UDM + Apache Jena)

1. Exp3 baseline chain:

```bash
uv run invoke exp3-generate-baseline --universities=50
```

2. Direct materialization for an existing ABox:

```bash
uv run invoke exp3-materialize-abox \
    --abox=path/to/owl2bench_abox.ttl \
    --tbox=data/owl2bench/input/UNIV-BENCH-OWL2RL.owl \
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

## Exact code changes needed

These changes are needed so Exp3 implementation matches the corrected Jena interpretation used in the paper.

1. Enforce one-shot Jena closure in OWL2Bench reasoning path.
    - File: `apps/OWL2Bench/src/owl2bench/pipeline.py`
    - Change: replace iterative external loop over `jena.materialize(...)` with a single materialization call per ABox.

2. Make Jena profile explicit and configurable.
    - Files: `configs/owl2bench/config.yaml`, `configs/owl2bench/config_toy.yaml`, `apps/rafm_baseline/java/src/main/java/org/synthology/rafm/JenaMaterializerCli.java`
    - Change: add and use `dataset.reasoning.materialization.jena_profile` (`owl_micro`, `owl_mini`, `owl_full`) and log it in artifacts.

3. Keep iterative fields as deprecated/no-op for Jena paths.
    - Files: `configs/owl2bench/config.yaml`, `configs/owl2bench/config_toy.yaml`
    - Change: remove or mark `iterative` and `max_iterations` as legacy for Jena to avoid semantic confusion.

4. Add depth-bucket and trivial-fact reporting outputs.
    - Files: `apps/data_reporter/src/data_reporter/paper_plots.py`, `apps/rafm_baseline/src/rafm_baseline/exp2_parity_report.py`
    - Change: emit 1-hop vs 2-plus-hop counts and schema-heavy share in report JSON and plots.

5. Align task defaults with one-shot baseline semantics.
    - File: `tasks.py`
    - Change: ensure official Exp3 commands do not set iterative Jena behavior by default.

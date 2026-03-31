# experiment3.md

## Experiment 3: Generalization to Complex Ontologies (OWL2Bench)

### Goal

Show that Synthology produces structurally superior training data on a modern, complex OWL2 RL ontology (OWL2Bench) where the standard forward-chaining baseline collapses to almost exclusively 1-hop inferences.

### Exact Step-by-Step Procedure

1. **Prepare OWL2Bench data**
    - Download OWL2Bench from https://github.com/kracr/owl2bench
    - Generate one RL-profile TBox + ABox of comparable size to your Synthology runs (use the same entity count / base-fact target as in Exp 2).
    - Save the generated ABox as `owl2bench_abox.ttl` (or equivalent).

2. **Baseline forward-chaining materialization**
    - Load the OWL2Bench ABox + TBox into Apache Jena.
    - Execute full materialization.
    - Run your layered forward-chaining depth tracker (same script as Exp 2).
    - Record the inference-depth distribution (expect > 90 % of inferences to be depth 1, mostly `sameAs` / `differentFrom` or trivial subclass propagations).
    - Save the materialized positive targets and base facts as `baseline_owl2bench_train.ttl`.

3. **Generate Synthology dataset on the same ontology**
    - Run Synthology on the **exact same OWL2Bench RL ontology** (extend the parser only if a required construct is missing — the paper states this is modular).
    - Use the same target number of samples / base-fact volume as the baseline.
    - Run the layered depth tracker again.
    - Save as `synthology_owl2bench_train.ttl`.

4. **Create evaluation test set**
    - Either:
        - Use OWL2Bench’s own SPARQL query set that requires reasoning, **or**
        - Generate one additional Synthology sample on OWL2Bench and extract a held-out set of deep targets (depth ≥ 2 or ≥ 3, whichever is more abundant).
    - Keep the test KB identical for both models.

5. **Train the two RRN models**
    - Train RRN A on the Synthology OWL2Bench dataset.
    - Train RRN B on the forward-chaining baseline OWL2Bench dataset.
    - Identical hyperparameters and negative-sampling ratio.

6. **Evaluate and visualize**
    - Evaluate both models on the test set.
    - Report: AUC-ROC, PR-AUC, F1-score, False Positive Rate.
    - Produce a side-by-side bar chart / histogram of inference-depth distributions for the two training sets (this is the key visual proof of the structural gap).
    - Include a short table showing % of inferences at depth 1, 2, 3+ for both generators.

### Expected outcome / interpretation

The baseline will be overwhelmingly shallow (almost all 1-hop), while Synthology will contain a healthy distribution of multi-hop chains. Any performance advantage of the Synthology-trained model (especially on PR-AUC and FPR) demonstrates that backward-chaining successfully engineers the complex validation paths that standard procedural benchmarks intrinsically lack.

## Practical Quick Start (reusing RAFM for Exp 3)

The RAFM baseline and Apache Jena materialization are reusable from `apps/rafm_baseline`.

1. Generate an Exp 3 RAFM baseline split dataset:

```bash
uv run invoke exp3-generate-baseline --universities=50
```

This now chains two steps automatically:

- Run `apps/OWL2Bench` pipeline to generate OWL2Bench data and raw OWL output.
- Use the raw OWL output as ABox input to RAFM/Jena materialization.

Interpretation:

- The generated OWL2Bench file is the **base-facts source** (ABox).
- Apache Jena materialization produces the **inferred targets**.

By convention, generated files are:

- ABox source: `data/owl2bench/output/raw/owl2bench_50/OWL2RL-50.owl`
- Jena closure: `data/exp3/baseline/owl2bench_50/closure.nt`
- Jena inferred-only: `data/exp3/baseline/owl2bench_50/inferred.nt`

2. For direct materialization of an existing OWL2Bench ABox + TBox pair:

```bash
uv run --package rafm_baseline python -m rafm_baseline.materialize \
    --tbox data/owl2bench/input/UNIV-BENCH-OWL2RL.owl \
    --abox path/to/owl2bench_abox.ttl \
    --closure-out outputs/exp3/closure.nt \
    --inferred-out outputs/exp3/inferred.nt
```

Equivalent invoke task:

```bash
uv run invoke exp3-materialize-abox \
    --abox=path/to/owl2bench_abox.ttl \
    --tbox=data/owl2bench/input/UNIV-BENCH-OWL2RL.owl \
    --closure-out=outputs/exp3/closure.nt \
    --inferred-out=outputs/exp3/inferred.nt
```

This ensures Exp 2 and Exp 3 use the same RAFM/Jena materialization pipeline.

## Paper Visuals

Generate paper-ready inspection plots (base vs inferred, hops, parity attempts):

```bash
uv run invoke paper-visual-report \
    --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
    --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
    --exp3-targets=data/owl2bench/output/owl2bench_50/train/targets.csv \
    --exp3-abox=data/owl2bench/output/raw/owl2bench_50/OWL2RL-50.owl \
    --exp3-inferred=data/exp3/baseline/owl2bench_50/inferred.nt \
    --out-dir=reports/paper
```

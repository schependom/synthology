# Experiment 2: Multi-Hop Reasoning Quality

## Paper-aligned goal

Show that an RRN trained on Synthology data outperforms an RRN trained on an Unguided Deductive Materialization (UDM) baseline, even when baseline runs are selected to match deep-signal volume as closely as possible.

Ontology: Family Tree (same as Experiment 1).

## How Jena must be used (Iterative Diffing)

To enforce strict structural parity and extract the hop-depth distributions required for our paper's narrative (Figure 3), we utilize an **Iterative Diffing** approach for the UDM baseline in this experiment.

- Standard forward-chaining (Jena) computes a fixpoint closure in one pass, destroying the step-by-step derivation history.
- To recover this metadata, our Python wrapper feeds the working graph into Jena iteratively.
- Layer $k$ consists of triples that first appear when materializing the base facts plus all layers $< k$.
- This incurs massive computational overhead, which perfectly illustrates our paper's claim: extracting deep-hop metadata from standard unguided forward-chaining pipelines is highly inefficient.

## Updated approach

1. Generate Synthology reference data and compute deep target count from proof metadata (`d >= 3`). Denote this as `K_deep`.
2. Run baseline attempts in a retry loop:
    - Sample random ABox under the same budget envelope.
    - Run **iterative** Jena closure to accurately map the `hops` column for all inferred facts.
    - Convert to `facts.csv` and `targets.csv`.
    - Compute deep-target proxy stats from exported targets.
        - Keep/discard based on structural parity criterion (`d >= 3`).
        - Track cumulative baseline wall-clock time until parity is reached (Time to Structural Parity).
3. Freeze one additional deep-only Synthology test set (`d >= 3`) for both models. This test set MUST also be generated iteratively to provide ground-truth hop labels for evaluation slicing.
4. Train baseline vs Synthology RRNs with identical hyperparameters.
5. Evaluate on the same frozen deep test set.

## Required reporting split

To match the paper discussion around trivial-fact dominance, always report metrics for each hop count. For example, for the parity loop, report:

- Total targets
- hop=0 targets (trivial)
- hop=1 targets (mostly trivial)
- hops between 2 and 3 (non-trivial, but still mostly shallow)
- hop≥3 targets (deep, the key focus of the paper)

And keep global metrics (`PR-AUC`, `AUC-ROC`, `F1`, `FPR`) beside these bucketed results.

## Time-to-Parity metric (mandatory)

Report:

- Synthology generation runtime (from generation metrics)
- Baseline cumulative runtime across parity attempts
- Runtime ratio `baseline_time_to_parity / synthology_time`

This is the key efficiency claim: random forward-chaining is measured by time-to-equivalent structural signal, not by one attempt.

## Practical quick start (UDM + Apache Jena)

### HPC prerequisite (run this first)

The Exp2 baseline generator builds and runs a Java helper for Jena materialization.
If `mvn` is missing, `exp2-generate-baseline` will fail before data generation starts.

1. Fix shell init noise first (optional but recommended):
    - If you see `/.../.bashrc: ... envexport: No such file or directory`, open your `.bashrc` and remove or guard that line.
2. Load Java and Maven on the cluster:

```bash
module avail maven
module avail java
module load java
module load maven
```

3. Verify tools are visible:

```bash
which java && java -version
which mvn && mvn -v
```

4. Build the Jena helper once (recommended preflight):

```bash
cd apps/udm_baseline/java
mvn -q -DskipTests package
ls -lh target/jena-materializer-1.0.0-shaded.jar
cd ../../..
```

5. Run Exp2 commands:

```bash
uv run invoke exp2-generate-gold-test
uv run invoke exp2-generate-baseline
uv run invoke exp2-generate-synthology
uv run invoke exp2-parity-loop
uv run invoke exp2-parity-report
```

If your cluster does not provide a Maven module, install Maven in your home directory and prepend it to `PATH` in the current shell:

```bash
export MAVEN_HOME="$HOME/apache-maven-3.9.9"
export PATH="$MAVEN_HOME/bin:$PATH"
mvn -v
```

1. Baseline generation:

```bash
uv run invoke exp2-generate-baseline
```

2. Synthology generation:

```bash
uv run invoke exp2-generate-synthology
```

3. Parity loop and reporting:

```bash
uv run invoke exp2-parity-loop
uv run invoke exp2-parity-report
```

4. Visual smoke check:

```bash
uv run invoke exp2-smoke-jena-visual
```

5. Plot paper diagnostics:

```bash
uv run invoke paper-visual-report \
    --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
    --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
    --out-dir=reports/paper
```

## Complete command list for Exp2

Use this order for a full Exp2 execution with both data and model runs.

1. Generate and freeze the shared deep test set:

```bash
uv run invoke exp2-generate-gold-test
```

2. Generate baseline dataset:

```bash
uv run invoke exp2-generate-baseline
```

3. Generate Synthology dataset:

```bash
uv run invoke exp2-generate-synthology
```

4. Optional budget matching helper (instead of 2+3 separately):

```bash
uv run invoke exp2-balance-datasets --fact-cap=<Nf> --target-cap=<Nt>
```

5. Parity loop and parity report:

```bash
uv run invoke exp2-parity-loop
uv run invoke exp2-parity-report
```

6. Data-distribution report:

```bash
uv run invoke exp2-report-data
```

7. Train both RRN models:

```bash
uv run invoke exp2-train-rrn --dataset=baseline
uv run invoke exp2-train-rrn --dataset=synthology
```

8. Visual smoke check for baseline reasoning path:

```bash
uv run invoke exp2-smoke-jena-visual
```

9. Paper-oriented plots:

```bash
uv run invoke paper-visual-report \
    --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
    --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
    --out-dir=reports/paper
```

For a monorepo-wide end-to-end protocol (Exp1/2/3 + artifacts), see `experiments/PAPER_RUNBOOK.md`.

## Exact code changes needed

These are the code-level constraints required to make Exp 2 consistent with the iterative methodology.

1. **Enforce Iterative Jena Closure:**
    - File: `apps/udm_baseline/src/udm_baseline/create_data.py`
    - Change: In the `_materialize` router, ensure `materialization.iterative=true` successfully routes to `_materialize_iterative_jena`. Remove any warnings that force single-pass behavior.
2. **Hydra Configuration:**
    - File: `configs/udm_baseline/exp2_baseline.yaml`
    - Change: Set `materialization.iterative: true` to ensure the parity loop has access to accurate hop depths.

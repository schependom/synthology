# Experiment 2: Multi-Hop Reasoning Quality

**Goal**  
Show that an RRN trained on Synthology data outperforms an RRN trained on a standard forward-chaining baseline **even when the baseline is forced to contain the same number of deep (d ≥ 3) inferences**.

**Ontology**  
Family Tree (same as Experiment 1).

**Core idea: parity-enforced baseline**  
We iterate the random ABox generator until the materialized forward-chained dataset contains (approximately) the same count of d ≥ 3 inferences as the Synthology dataset.

**Exact concrete steps**

1. **Generate the Synthology reference dataset**
    - Run Synthology on Family Tree with stratified proof selection (or random — document which).
    - Compute and record the exact number of inferred facts with proof depth d ≥ 3. Call this number `K_deep`.
    - Save the dataset (base facts + positives + negatives) and the depth histogram.

2. **Implement layered depth computation (required for parity)**
    - Write a small Python script that performs iterative forward-chaining **layer by layer**:
        - Layer 0 = base facts.
        - In each new iteration, fire rules only on facts from previous layers.
        - Assign each new inferred fact its minimum depth.
    - Use Apache Jena (not owlrl) for the actual materialization step inside the loop.

3. **Parity-enforced forward-chaining baseline loop**
    - While true:
        - Generate a random ABox (same entity pool size and same total base-fact count as Synthology).
        - Materialize with Jena.
        - Compute depth distribution using the layered method.
        - Count number of facts with d ≥ 3.
        - If count ≥ `K_deep` (or within ±10 %), break.
        - Else discard and retry.
    - If the loop exceeds 500–1000 attempts or explodes graph size, stop and document the effort (this itself is a publishable result).

4. **Create the frozen held-out deep test set**
    - Generate **one additional Synthology sample** (or small batch) and hold it completely out of training.
    - Extract **only** the inferred facts whose proof-tree depth d ≥ 3.
    - For each such deep target, keep the minimal supporting base facts from its proof tree + all other base facts in that sample.
    - This produces a clean test KB where **every positive requires at least 3 hops**.
    - Save this test set once and reuse it for both models.

5. **Training**
    - Train RRN #1 on the Synthology dataset.
    - Train RRN #2 on the parity-enforced forward-chaining baseline dataset.
    - Use identical hyperparameters, same RRN implementation, same random seed.

6. **Evaluation**
    - Evaluate **both** models **exclusively** on the frozen deep test set from step 4.
    - Report:
        - AUC-ROC
        - PR-AUC (primary)
        - F1-score
        - FPR
    - Also plot or report the inference-depth distribution of both training sets side-by-side.

**Expected outcome**  
Even with identical numbers of deep inferences, the Synthology-trained model achieves higher PR-AUC and lower FPR because its deep paths are topologically integrated inside coherent samples, whereas the baseline’s deep facts are scattered and incidental.

**Implementation notes**

- Keep all raw baseline ABoxes that were discarded (for transparency).
- Store everything in `experiments/exp2/`.
- If parity proves practically impossible (very likely), switch to a “standard” (non-parity) baseline and clearly state in the paper: “Unguided forward-chaining could not reach parity after X attempts; we therefore compare against the natural shallow distribution.”

## Practical Quick Start (RAFM + Apache Jena)

1. Ensure `java` and `mvn` are installed and available in `PATH`.
2. Generate the Exp 2 baseline with Jena materialization:

```bash
uv run invoke exp2-generate-baseline
```

This uses `configs/fc_baseline/exp2_baseline.yaml` where:

- `materialization.reasoner=jena`
- `materialization.iterative=true`

3. Generate Synthology comparison data:

```bash
uv run invoke exp2-generate-synthology
```

4. Train both RRN variants:

```bash
uv run invoke exp2-train-rrn --dataset=baseline
uv run invoke exp2-train-rrn --dataset=synthology
```

5. Dedicated parity loop + reporting:

```bash
uv run invoke exp2-parity-loop
uv run invoke exp2-parity-report
```

6. Visual smoke test for Jena-backed RAFM:

```bash
uv run invoke exp2-smoke-jena-visual
```

The smoke visualization is written to `visual-verification/exp2_smoke/`.

7. Paper-oriented parity plots:

```bash
uv run invoke paper-visual-report \
    --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
    --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
    --out-dir=reports/paper
```

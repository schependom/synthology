# Experiment 2: Multi-Hop Quality Test

## Objective

Evaluate whether Synthology's backward-chaining generator produces training data that better supports >=3-hop reasoning than a forward-chaining baseline, under controlled data budget and identical RRN training settings.

## Datasets Used

- Ontology: `data/ont/input/family.ttl` (default; can be overridden).
- Synthology data (ONT): `data/exp2/synthology/family_tree/{train,val,test}`.
- Forward-chaining baseline data (FC): `data/exp2/baseline/family_tree/{train,val,test}`.
- Optional ASP reference (sanity baseline): `data/asp/family_tree/{train,val,test}`.
- Shared/frozen Exp2 hard test set (recommended target location):
    - `data/exp2/frozen_test/{facts.csv,targets.csv}`

## Configurations

Use Hydra configs only. Do not hardcode experiment parameters inside `tasks.py`.

### Existing config files (already present)

- Generator (Synthology): `configs/ont_generator/exp2_synthology.yaml`
- Generator (FC baseline): `configs/fc_baseline/exp2_baseline.yaml`
- Frozen test generation: `configs/ont_generator/exp2_gold_test.yaml`
- RRN dataset mappings:
    - `configs/rrn/data/dataset/exp2_synthology.yaml`
    - `configs/rrn/data/dataset/exp2_baseline.yaml`
- Data reporting:
    - `configs/data_reporter/exp2_compare.yaml`

### Exp2-specific config files that should be present

These files are recommended to ensure reproducibility and avoid long command strings.

- `configs/ont_generator/exp2_gold_test.yaml`
- `configs/ont_generator/exp2_synthology.yaml`
- `configs/fc_baseline/exp2_baseline.yaml`
- `configs/data_reporter/exp2_compare.yaml`
- `configs/rrn/data/dataset/exp2_baseline.yaml`
- `configs/rrn/data/dataset/exp2_synthology.yaml`

Minimum parameter contract for Exp2 configs:

- `ontology.path`
- data volume controls (`n_train`, `n_val`, `n_test`)
- split-level fact caps (`train_fact_cap`, `val_fact_cap`, `test_fact_cap`)
- split-level target caps (`train_target_cap`, `val_target_cap`, `test_target_cap`)
- random seed (`dataset.seed`)
- output location (`dataset.output_dir`)
- method-specific depth/closure controls:
    - Synthology: `generator.max_recursion`, `generator.global_max_depth`, `generator.proof_roots_per_rule`
    - FC baseline: `materialization.*`, relation/type sampling bounds, `generator.base_relations_per_sample`
- negative sampling controls (`neg_sampling.*`)
- logging metadata (`logger.name`, `logger.group`, `logger.tags`) for RRN

## Fact and Target Caps: Detailed Explanation and Comparison

This project exposes fact/target caps through both CLI task arguments and YAML config values on purpose. They solve different problems.

### Why both CLI and YAML exist

1. CLI caps (`--fact-cap`, `--target-cap`) are for fast experiment sweeping

- Use when you run many budget points quickly.
- Example: loop over `10k`, `25k`, `50k`, `100k` from one shell script.
- In tasks, this maps to `dataset.train_fact_cap` / `dataset.train_target_cap` for each method call.

2. YAML split caps (`dataset.train_fact_cap`, `dataset.val_fact_cap`, `dataset.test_fact_cap`, `dataset.train_target_cap`, `dataset.val_target_cap`, `dataset.test_target_cap`) are for explicit reproducibility

- Use when freezing a final experiment setting.
- Keeps all cap choices versioned in config files and easy to cite in thesis appendices.

In short:

- CLI = convenience and speed.
- YAML = reproducibility and auditability.

### Precedence and interaction

- If you pass `--fact-cap` and/or `--target-cap` in invoke, it overrides train caps at runtime for that command.
- YAML values remain the baseline defaults.
- If no CLI cap is passed, YAML values are used as-is.

### What is actually being capped

- Fact cap is on `facts.csv` row count per split.
- Target cap is on `targets.csv` row count per split.

Generation flow:

1. Build samples.
2. Convert each sample to standard rows (`facts.csv` and `targets.csv` rows).
3. Keep adding full samples while under cap(s).
4. If needed, trim rows to exact cap values.
5. Reconcile on retained `sample_id`s so both files stay aligned.

Important consequence:

- If only a fact cap is used, `targets.csv` can still be much larger.
- If only a target cap is used, `facts.csv` can still be larger.
- If both are used, the exported split enforces both budgets.

### Why this behavior is intentional

- It preserves sample integrity for reasoning structure.
- It avoids cutting target rows mid-sample in a way that can create inconsistent supervision.
- It gives clean dual-budget control: context (`facts.csv`) and supervision (`targets.csv`).

### Practical recipes

1. Quick sweep mode

- Use `exp2-balance-datasets --fact-cap=<Nf> --target-cap=<Nt>`.
- Keep all other settings fixed.

2. Final reproducible mode

- Encode fact and target caps in YAML.
- Run without CLI overrides.

3. Reporting discipline

- After each generation, run `exp2-report-data`.
- Log actual produced counts, not only requested caps.

## Fair Comparison: What to Match Between FC and Synthology

Short answer: yes, your intuition is mostly right. Matching only base facts is usually not enough.

### Why matching only base facts can be unfair

- Two methods can start with equal base-fact counts but generate very different target volumes.
- RRN training is influenced by both context (`facts.csv`) and supervision volume/composition (`targets.csv`).
- If one method has many more targets, it effectively gets more training signal.

### Better fairness target

For Exp2, fairness is strongest when you align both:

1. context budget: similar `facts.csv` count
2. supervision budget: similar `targets.csv` count

This is equivalent to controlling total data budget more strictly than base-only matching.

### Recommended fairness protocol

1. First, set equal train fact/target caps (`--fact-cap=<Nf> --target-cap=<Nt>` or YAML).
2. Then tune method knobs to reduce target mismatch:

- FC: `generator.base_relations_per_sample`
- ONT: `generator.proof_roots_per_rule`

3. Keep `neg_sampling.ratio` matched.
4. Verify with `exp2-report-data` and record:

- train facts totals
- train targets totals
- positive/negative ratio
- hop distribution

### If exact matching is impossible

- Report the residual mismatch percentage explicitly.
- Keep mismatch bounded and consistent across seeds.
- Do not change test set or model hyperparameters between methods.

## Forward-Chaining Negative Sampling Strategy (Baseline)

The FC baseline uses random corruption to create negative targets. This is configured in `configs/fc_baseline/exp2_baseline.yaml` under `neg_sampling.*` and implemented in the FC generator.

### Core Rule

- For each positive target candidate, produce negatives by random corruption.
- Negative volume is controlled by:
    - `neg_sampling.ratio`
    - `neg_sampling.max_attempts_per_negative`

### How Corruption Is Done

1. Class-membership facts (`rdf:type`)

- Corrupt subject only:
    - `(s, rdf:type, C)` -> `(s', rdf:type, C)`

2. Relation facts (`(s, p, o)`)

- Randomly corrupt subject or object with roughly 50/50 chance:
    - subject corruption: `(s, p, o)` -> `(s', p, o)`
    - object corruption: `(s, p, o)` -> `(s, p, o')`

### Validity Guards

- Candidate negative must not already exist as a positive fact.
- Candidate negative must not duplicate an already sampled negative.
- Corrupted entities must remain in the same generated individual universe for that sample.

### Output Labels and Metadata

- Negatives are written in `targets.csv` with:
    - `label=0`
    - `truth_value=false`
    - `corruption_method=random`
- Type tags are set based on origin:
    - `neg_base_fact` (from base fact)
    - `neg_inf_root` (from inferred fact)

### Why This Matters for Exp2

- This strategy is intentionally simple and unguided for the FC baseline.
- It provides a clean contrast against proof-aware negative generation in Synthology.
- Keep `neg_sampling.ratio` matched between methods whenever fairness is the goal.

## Execution Commands

### Standardized command set that should be present in `tasks.py`

Follow naming convention `<experiment-id>-<action>-<target>` from `experiments/EXPERIMENT_GUIDELINES.md`.

1. `uv run invoke exp2-generate-gold-test`
2. `uv run invoke exp2-generate-baseline --fact-cap=<Nf> --target-cap=<Nt> --base-facts-per-sample=<K>`
3. `uv run invoke exp2-generate-synthology --fact-cap=<Nf> --target-cap=<Nt> --proof-roots-per-rule=<K>`
4. `uv run invoke exp2-balance-datasets --fact-cap=<Nf> --target-cap=<Nt> --baseline-base-facts=<K1> --synthology-proof-roots=<K2>`
5. `uv run invoke exp2-report-data`
6. `uv run invoke exp2-train-rrn --dataset=baseline`
7. `uv run invoke exp2-train-rrn --dataset=synthology`

### Example command sequence

1. Generate frozen test set:

```bash
uv run invoke exp2-generate-gold-test
```

2. Budget-matched generation (single cap for both methods):

```bash
uv run invoke exp2-balance-datasets --fact-cap=50000 --target-cap=120000 --baseline-base-facts=20 --synthology-proof-roots=10
```

3. Compare distributions and parity:

```bash
uv run invoke exp2-report-data
```

4. Train baseline:

```bash
uv run invoke exp2-train-rrn --dataset=baseline --args="+logger.name=exp2_baseline_seed23 +logger.tags=[exp2,baseline,seed23]"
```

5. Train synthology:

```bash
uv run invoke exp2-train-rrn --dataset=synthology --args="+logger.name=exp2_synthology_seed23 +logger.tags=[exp2,synthology,seed23]"
```

## Extensive Checklist (Runbook)

### A. Pre-Run Sanity

- [ ] Confirm environment is synced: `uv sync`
- [ ] Confirm ontology path is fixed for all methods.
- [ ] Confirm a single random seed plan (for example, `23, 42, 1337`).
- [ ] Confirm output directories are empty or versioned per run.
- [ ] Confirm report output path for this experiment (`reports/exp2_*`).

### B. Freeze Exp2 Gold Test Set

- [ ] Generate a frozen test set once (prefer Synthology depth >=4 and filter hops >=3).
- [ ] Save to `data/exp2/frozen_test/`.
- [ ] Do not regenerate this file during method comparisons.
- [ ] Record exact command and config in lab notes/W&B.

### C. Generate Baseline and Synthology Training Data

- [ ] Generate baseline (FC) with fixed ontology + seed.
- [ ] Generate Synthology (ONT) with fixed ontology + seed.
- [ ] Set shared `fact_cap` and `target_cap` for budget matching.
- [ ] Tune method volume knobs for graph sweeps:
    - FC: `base_facts_per_sample`
    - Synthology: `proof_roots_per_rule`
- [ ] Keep train/val sizes matched (`n_train`, `n_val`) before balancing.
- [ ] Keep negative sampling ratio identical across methods.

### D. Balance and Validate Data Parity

- [ ] Match total number of train facts between methods (or report exact ratio).
- [ ] Match total number of positive targets between methods.
- [ ] Run reporter and inspect:
    - method-level facts/targets totals
    - predicate distribution alignment
    - hop distribution alignment (especially >=3-hop mass)
    - type distribution alignment
- [ ] Archive parity report to `reports/exp2_compare/`.

### E. Train RRN Fairly

- [ ] Use identical model hyperparameters for FC and ONT runs.
- [ ] Use identical optimizer/scheduler/epochs/early stopping.
- [ ] Run at least 3 seeds per method.
- [ ] Use consistent W&B group/tag schema: `exp2_multihop`.

### F. Evaluate

- [ ] Evaluate both models on the same frozen Exp2 test set.
- [ ] Track at least: AUC-ROC, PR-AUC, F1, and FPR on hard negatives.
- [ ] If available, report metrics stratified by hop count (`1`, `2`, `>=3`).

### G. Reproducibility Closure

- [ ] Store final command list used (copy from shell history).
- [ ] Store exact config files/overrides used.
- [ ] Store commit hash.
- [ ] Store generated report + plots in `reports/exp2_final/`.

## Desired Graphs (for thesis)

Generate these figures for each seed-aggregated comparison (mean +/- std).

1. Main quality curve:

- x-axis: effective train fact budget
- y-axis: AUC-ROC (or PR-AUC)
- series: FC vs ONT

2. Multi-hop robustness:

- x-axis: hop bucket (`1`, `2`, `>=3`)
- y-axis: F1 (or recall)
- grouped bars: FC vs ONT

3. Hard-negative safety:

- x-axis: method
- y-axis: FPR on frozen near-miss negatives

4. Distribution parity diagnostics (from `data_reporter`):

- facts total by method
- targets total by method
- average targets per sample
- hops distribution
- type distribution (stacked)
- top predicates per method

5. Optional calibration plot:

- reliability curve or confidence histogram per method

## Expected Results

- Primary expectation: ONT (backward-chaining) outperforms FC on >=3-hop metrics at matched budget.
- Secondary expectation: ONT has lower FPR on hard negatives due to structurally grounded proofs.
- All metrics and run metadata should be logged under a single W&B group: `exp2_multihop`.

## Notes

- If Nemo/Jena baseline is reintroduced later, keep this same protocol and add it as an additional method; do not change fairness constraints.
- If strict budget matching is not possible, report mismatch percentage explicitly in the final figure captions.

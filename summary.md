# Exp3 Latest Baseline Analysis

## Run

- archive_dir: /dtu/blackhole/16/221590/synthology/reports/experiment_runs/2026-04-14/exp3/generate_baseline/205002_1
- universities: 1
- final_jena_profile: owl_mini
- final_reasoning_input_triple_cap: 15000

## Train Split

- facts rows: 27806
- targets rows: 39238
- positive rows: 35186
- negative rows: 4052
- positive/negative ratio: 8.683613030602173

## Ratio Accounting

- diagnostics summary source: /dtu/blackhole/16/221590/synthology/data/owl2bench/output/diagnostics/owl2bench_1_1776192628/summary.json
- overall eligible positive targets for negatives: 8304
- overall expected negatives: 4848
- overall generated negatives: 4848
- overall skipped negative slots: 3456
- overall rejected negative collisions with positives: 445964
- overall rejected duplicate negatives: 312586
- train eligible positives for negatives: 7380
- train expected negatives: 4052
- train generated negatives: 4052
- train skipped negative slots: 3328

## Label Skew Diagnosis

- base_fact positives (always label=1): 27806
- inferred positives: 7380
- negatives are generated only for inferred/base targets selected as positive_targets, not one-for-one against every base fact row.
- with negatives_per_positive=1 and many retained base facts, positive-heavy targets are expected.

## Integrity Checks

- unique positive triples: 35186
- unique negative triples: 4052
- positive-negative triple overlap: 0

## Materialization Timing (Latest Event)

- base_triples: 15000
- closure_triples: 664372
- inferred_triples: 648918
- reasoning_seconds: 62.84864821191877
- run_total_seconds: 97.33034064201638

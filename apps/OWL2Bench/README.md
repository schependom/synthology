# OWL2Bench Pipeline Package

This package runs an OWL 2 RL-only generation pipeline based on the Java
OWL2Bench source code in `vendor/OWL2Bench`.

Pipeline stages:

1. Run OWL2Bench Java generator (`ABoxGen.InstanceGenerator.Generator`) in RL mode.
2. Use `data/owl2bench/UNIV-BENCH-OWL2RL.owl` as the TBox.
3. Materialize with Apache Jena (iterative OWL reasoning).
4. Export standard RRN CSV format:
    - `facts.csv`
    - `targets.csv`

## Run

From the repository root:

```bash
uv run invoke gen-owl2bench
```

With overrides:

```bash
uv run invoke gen-owl2bench --args="dataset.universities=[1,5] dataset.num_samples=10"
```

## Output

- Raw generated OWL files:
    - `data/owl2bench/raw/owl2bench_<n>/OWL2RL-<n>.owl`
- CSV dataset:
    - `data/owl2bench/owl2bench_<n>/{train,val,test}/facts.csv`
    - `data/owl2bench/owl2bench_<n>/{train,val,test}/targets.csv`

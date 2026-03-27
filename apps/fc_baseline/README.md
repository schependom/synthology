# FC Baseline Generator

Random base-fact generation + forward-chaining materialization baseline.

This package generates synthetic datasets by:

1. Sampling random base facts from an ontology schema.
2. Running OWL RL materialization with `owlrl`.
3. Exporting data in the standard split format (`facts.csv`, `targets.csv`).

## Usage

```bash
uv run --package fc_baseline python -m fc_baseline.create_data
```

Configuration is loaded from `configs/fc_baseline/config.yaml`.

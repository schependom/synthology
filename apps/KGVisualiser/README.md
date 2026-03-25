# KGVisualiser

Visualize a single knowledge-graph sample from CSV files, showing:

- base facts
- inferred positives
- negative targets

This app is configured via Hydra YAML in `configs/kgvisualiser/config.yaml`.

## Run

From repository root:

```bash
uv run --package kgvisualiser python -m kgvisualiser.visualize
```

Override sample and input CSV:

```bash
uv run --package kgvisualiser python -m kgvisualiser.visualize \
  io.input_csv=data/OWL2Bench/output/owl2bench_10/train/facts.csv \
  io.sample_id=0
```

Render a sample that includes inferred facts and collapse class nodes:

```bash
uv run --package kgvisualiser python -m kgvisualiser.visualize \
  io.input_csv=data/owl2bench/output_toy/owl2bench_1/val/targets.csv \
  io.sample_id=710024 \
  render.class_nodes=false
```

If `io.input_csv` points to `facts.csv`, the app will automatically read sibling `targets.csv` unless disabled.

## Output

By default, files are written to:

- `visualizations/`

Format defaults to PDF and can be changed with `output.format`.

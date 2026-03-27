# Data Reporter

Compares generated KG datasets across methods (ASP, ONT, FC, etc.) and produces:

- summary JSON
- per-method metrics CSV
- markdown report
- distribution plots (predicate/type/hops/negatives)

Expected input layout per method path:

- `train/facts.csv`, `train/targets.csv`
- `val/facts.csv`, `val/targets.csv`
- `test/facts.csv`, `test/targets.csv`

## Run

```bash
uv run --package data_reporter python -m data_reporter.analyze
```

Use `configs/data_reporter/config.yaml` to set method names/paths and output location.

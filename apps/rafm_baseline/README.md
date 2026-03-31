# RAFM Baseline Generator

Random ABox Fact Materialization (RAFM) baseline with reusable materialization backends.

This package can:

1. Sample random base facts from an ontology schema.
2. Materialize inferred facts with either `owlrl` or Apache Jena.
3. Assign layered hop depths using iterative materialization mode.
4. Export standard split files (`facts.csv`, `targets.csv`).

## Prerequisites (Jena backend)

The Jena path uses a small Java helper in `apps/rafm_baseline/java/`.

- Java runtime in PATH (`java`)
- Maven in PATH (`mvn`)

On first Jena run, the helper JAR is built automatically with Maven.

## Generate baseline splits

```bash
uv run --package rafm_baseline python -m rafm_baseline.create_data
```

Use Exp 2 or Exp 3 configs:

```bash
uv run --package rafm_baseline python -m rafm_baseline.create_data --config-name=exp2_baseline
uv run --package rafm_baseline python -m rafm_baseline.create_data --config-name=exp3_baseline
```

Configs are loaded from `configs/fc_baseline/`.

## Reuse Jena materialization directly (for Exp 3 pipelines)

For existing TBox + ABox files, use the reusable CLI:

```bash
uv run --package rafm_baseline python -m rafm_baseline.materialize \
	--tbox data/OWL2Bench/input/UNIV-BENCH-OWL2RL.owl \
	--abox path/to/abox.ttl \
	--closure-out outputs/exp3/closure.nt \
	--inferred-out outputs/exp3/inferred.nt
```

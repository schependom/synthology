# LUBM Generator Package

This package contains scripts and configuration to generate [Lehigh University Benchmark (LUBM)](http://swat.cse.lehigh.edu/projects/lubm/) datasets and format them for use with the Recursive Reasoning Network (RRN) models.

## Usage

1. **Generate and Parse LUBM Data (recommended):**

Use the `gen_lubm` task defined in `synthology/tasks.py`.
This runs the Java vendor generator, then parses output into CSV while applying Apache Jena materialization to compute inferred facts.

```bash
uv run invoke gen-lubm
```

The Java LUBM vendor generator produces base facts only. Inferred facts are
created in the parsing stage by the configured reasoner backend.

The parser now expects Apache Jena for materialization.
Set your local Apache Jena CLI command in `configs/lubm/config.yaml` under:

- `dataset.reasoning.jena.executable`
- `dataset.reasoning.jena.command_template`

If the Jena command template is empty or invalid, parsing fails fast with a
clear setup error message.

Output goes into:

- `data/lubm/raw/lubm_{size}` (raw Turtle)
- `data/lubm/lubm_{size}/{train,val,test}/facts.csv` (base facts)
- `data/lubm/lubm_{size}/{train,val,test}/targets.csv` (base + inferred targets)

2. **Parse Existing Raw LUBM Data into Standard CSV Format:**

    Run the parser to convert raw `.ttl` data into the standard RRN-friendly structure.

    ```bash
    uv run invoke parse-lubm
    ```

    If you want targets to contain only reasoner-derived inferred positives
    (and no target-masked base facts), disable masking:

    ```bash
    uv run invoke parse-lubm --args="dataset.mask_base_facts=false"
    ```

3. **Run Configurable Reasoner Verification (subset or toy):**

    Verification is now YAML-driven via `configs/lubm/verify_reasoner.yaml`.
    By default, it samples a subset from `LUBM_1`, runs reasoning, exports CSV,
    and exports a graph with base + inferred facts.

    ```bash
    uv run invoke verify-lubm-reasoner
    ```

    Override settings for custom/future datasets with Hydra args:

    ```bash
    uv run invoke verify-lubm-reasoner --args="source.raw_dir=data/lubm/raw/lubm_10 subset.max_base_facts=800 output.dataset_label=lubm10_subset"
    ```

    Graph export size is configurable as well:

    ```bash
    uv run invoke verify-lubm-reasoner --args="output.graph.max_base_facts=150"
    ```

    Switch to the tiny deterministic toy mode:

    ```bash
    uv run invoke verify-lubm-reasoner --args="verification.mode=toy output.dataset_label=toy_case"
    ```

    Artifacts are kept (not temporary) under:
    - `data/lubm/toy_verify/raw/<dataset_label>/...`
    - `data/lubm/toy_verify/out/<dataset_label>/train/facts.csv`
    - `data/lubm/toy_verify/out/<dataset_label>/train/targets.csv`
    - `data/lubm/toy_verify/graph/verification_reasoning_graph.pdf`

## Repo-Wide Graph Export Utility

Graph export logic for base + inferred fact visualization was moved to a shared,
repo-wide utility in:

- `src/synthology/verification_visualizer.py`

This allows other generators/verifiers (current and future) to reuse identical
display semantics when exporting verification graphs.

## Timing Statistics (for reporting)

The pipeline now logs timing summaries suitable for experiments/paper reporting:

- LUBM Java generation time per dataset + total (from `lubm.orchestrator`)
- CSV parser total time
- Turtle parsing time
- Reasoning time and number of reasoning calls
- Number of inferred positive targets and observed max hops

You get these stats directly from:

```bash
uv run invoke gen-lubm
```

or parser-only mode:

```bash
uv run invoke parse-lubm
```

## Custom CSV Formats

The parsed output is located in `data/lubm/lubm_{size}/` and consists of two files per dataset, mirroring the RRN pipeline standard:

- `facts.csv`: Contains the **base facts** (known assertions) of the dataset.
    - **Schema**: `sample_id`, `subject`, `predicate`, `object`
- `targets.csv`: Contains all target facts (positive and negative) meant for training or evaluating the RRN. This includes base facts and reasoner-derived inferred facts, plus generated negatives.
    - **Schema**:
        - `sample_id`: Groups triples belonging to an independent knowledge graph sample.
        - `subject`, `predicate`, `object`: The relational triple.
        - `label`: `1` for positive, known facts. `0` for negative corruptions.
        - `truth_value`: Human-readable truth context (e.g. "True", "Unknown").
        - `type`: Target inference type (e.g., `base_fact`, `neg_inf_root`).
        - `hops`: The derivation depth (0 for base facts).
        - `corruption_method`: Metadata detailing how a negative target was generated.

    When `dataset.mask_base_facts=true`, some base facts are moved to targets-only
    as positive rows with `type=inferred` and `hops=0`.
    When `dataset.mask_base_facts=false`, inferred positives in `targets.csv`
    come only from reasoning.

_Note: The LUBM parser now applies Apache Jena materialization (optionally with the LUBM TBox from `data/ont/input/lubm.ttl`). If the TBox file is missing, materialization still runs over the sample ABox only._

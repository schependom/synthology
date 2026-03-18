# LUBM Generator Package

This package contains scripts and configuration to generate [Lehigh University Benchmark (LUBM)](http://swat.cse.lehigh.edu/projects/lubm/) datasets and format them for use with the Recursive Reasoning Network (RRN) models.

## Usage

1. **Generate and Parse LUBM Data (recommended):**

Use the `gen_lubm` task defined in `synthology/tasks.py`.
This runs the Java vendor generator, then parses output into CSV while applying an OWL RL reasoner to compute inferred facts.

```bash
uv run invoke gen-lubm
```

Output goes into:

- `data/lubm/raw/lubm_{size}` (raw Turtle)
- `data/lubm/lubm_{size}/{train,val,test}/facts.csv` (base facts)
- `data/lubm/lubm_{size}/{train,val,test}/targets.csv` (base + inferred targets)

2. **Parse Existing Raw LUBM Data into Standard CSV Format:**

    Run the parser to convert raw `.ttl` data into the standard RRN-friendly structure.

    ```bash
    uv run invoke parse-lubm
    ```

3. **Run Toy Reasoner Verification (fast sanity check):**

    This creates a tiny in-memory ontology and checks whether expected inferred
    facts and hop counts are exported correctly to `targets.csv`.

    ```bash
    uv run invoke verify-lubm-reasoner
    ```

    Artifacts are kept (not temporary) under:
    - `data/lubm/toy_verify/raw/lubm_0/toy.ttl`
    - `data/lubm/toy_verify/out/lubm_0/train/facts.csv`
    - `data/lubm/toy_verify/out/lubm_0/train/targets.csv`
    - `data/lubm/toy_verify/graph/toy_reasoning_graph.pdf`
    - `data/lubm/toy_verify/graph/toy_reasoning_graph.png`

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

_Note: The LUBM parser now applies OWL RL reasoning (optionally with the LUBM TBox from `data/ont/input/lubm.ttl`). If the TBox file is missing, closure still runs over the sample ABox only._

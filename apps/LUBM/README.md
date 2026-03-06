# LUBM Generator Package

This package contains scripts and configuration to generate [Lehigh University Benchmark (LUBM)](http://swat.cse.lehigh.edu/projects/lubm/) datasets and format them for use with the Recursive Reasoning Network (RRN) models.

## Usage

1. **Generate LUBM Raw Data:**

   Use the `gen_lubm` task defined in `synthology/tasks.py`. This uses the underlying `uba1.7` Java generator to create `.ttl` files. 

   ```bash
   uv run invoke gen-lubm
   ```
   
   Output goes into `data/lubm/raw/lubm_{size}`.

2. **Parse LUBM Raw Data into Standard CSV Format:**

   Run the parser to convert raw `.ttl` data into the standard RRN-friendly structure. 
   
   ```bash
   uv run invoke parse-lubm
   ```

## Custom CSV Formats

The parsed output is located in `data/lubm/lubm_{size}/` and consists of two files per dataset, mirroring the RRN pipeline standard:

- `facts.csv`: Contains the **base facts** (known assertions) of the dataset. These are triples where the number of inference hops is 0. 
  - **Schema**: `sample_id`, `subject`, `predicate`, `object`
  
- `targets.csv`: Contains all target facts (positive and negative) meant for training or evaluating the RRN. This includes all facts from `facts.csv` (as positive targets) and negative targets generated through sampling (when implemented).
  - **Schema**:
    - `sample_id`: Groups triples belonging to an independent knowledge graph sample.
    - `subject`, `predicate`, `object`: The relational triple.
    - `label`: `1` for positive, known facts. `0` for negative corruptions.
    - `truth_value`: Human-readable truth context (e.g. "True", "Unknown").
    - `type`: Target inference type (e.g., `base_fact`, `neg_inf_root`).
    - `hops`: The derivation depth (0 for base facts).
    - `corruption_method`: Metadata detailing how a negative target was generated.

*Note: The current LUBM generation does not yet invoke deductive reasoning or negative sampling in parity with the Ontology Generator, so the current targets.csv merely holds the base facts represented as positive targets.*

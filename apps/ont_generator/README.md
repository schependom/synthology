# Ontology Knowledge Graph Generator (ont_generator)

This application generates synthetic knowledge graphs (KGs) from OWL 2 RL ontologies using backward-chaining reasoning. It is designed to create high-quality training and testing datasets for Knowledge Graph Embedding (KGE) models.

## Features

- **Ontology-Driven Generation**: Generates data that respects the schema and logic of a provided `.ttl` ontology.
- **Backward Chaining Reasoning**: Infers new facts based on ontology rules, ensuring logical consistency.
- **Inductive Split**: Generates independent "Train" and "Test" graphs with disjoint sets of individuals (e.g., `train_Ind_0` vs `test_Ind_0`) to test inductive generalization.
- **Negative Sampling**: Supports multiple strategies (`random`, `constrained`, `type_aware`, `proof_based`) to generate challenging negative triples.
- **Explainability**: Can export proof trees for positive facts and explanations for hard negative samples.
- **Validation**: Automatically validates generated graphs against ontology constraints (disjointness, domains, ranges, etc.).

## Installation

This app is part of the `synthology` repository and uses `uv` for dependency management.

```bash
# From the root of the repo
uv sync
```

## Usage

The main entry point is `gen-ft-ont` task, which runs `apps/ont_generator/src/create_data.py`.

### Running with Defaults

```bash
uv run invoke gen-ft-ont
```

### Configuration

Configuration is managed via [Hydra](https://hydra.cc/). You can override defaults from the command line or modify `configs/ont_generator/config.yaml`.

**Key Parameters:**

- `dataset.n_train`: Number of training samples to generate.
- `dataset.n_test`: Number of testing samples to generate.
- `dataset.min_individuals` / `max_individuals`: Size range for each graph.
- `generator.max_recursion`: Max depth for recursive rules.
- `negative_sampling.strategy`: Strategy to use (`random`, `constrained`, `type_aware`, `mixed`).
- `negative_sampling.ratio`: Ratio of negative to positive triples (e.g., 0.5 for 1:2 ratio).

**Example: Generate larger dataset with type-aware negatives**

```bash
uv run invoke gen-ft-ont \
    dataset.n_train=100 \
    dataset.n_test=20 \
    dataset.min_individuals=20 \
    dataset.max_individuals=50 \
    negative_sampling.strategy=type_aware
```

## Output

Results are saved to `data/output/ont_generator/` (or as configured in `dataset.output_dir`).

- `train_*.csv`: Training samples.
- `test_*.csv`: Testing samples.
- `negatives_explanations.csv`: Explanations for negative samples (if available).
- Logs and Hydra config are saved in `outputs/`.

## CSV Format

Generated CSV files have the following columns:
`subject, predicate, object, label, proofs, metadata`

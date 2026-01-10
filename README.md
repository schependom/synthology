# Synthology

**Benchmarking Inductive Reasoning on Synthetic Knowledge Graphs**

This repository contains the source code for the "Synthology" project, which aims to benchmark the inductive reasoning capabilities of Knowledge Graph Embedding (KGE) models using synthetically generated datasets based on OWL 2 RL ontologies.

## Repository Structure

```
synthology/
├── apps/
│   ├── asp_generator/      # ASP-based data generator (adapted/reference)
│   ├── ont_generator/      # Ontology-based data generator (Main Contribution)
│   ├── RRN/                # Relational Reasoning Network (KGE Model)
│   └── TransE/             # TransE (Baseline KGE Model)
│
├── configs/                # Hydra configuration files
│   ├── ont_generator/      # Configs for ontology generator
│   └── rrn/                # Configs for RRN training
│
├── data/
│   ├── ont/                # Input ontologies (e.g., family.ttl)
│   └── output/             # Generated datasets
│
├── src/synthology/         # Shared library code (data structures, utils)
└── tests/                  # Unit and integration tests
```

## Getting Started

This project uses `uv` for dependency management and `invoke` for task automation.

### Prerequisites

-   Python 3.10+
-   `uv` installed (see [uv docs](https://github.com/astral-sh/uv))

### Installation

Sync dependencies:

```bash
uv sync
```

## Reproducing Paper Experiments

To generate the datasets used in the paper:

### 1. Generate Synthetic Data

Run the Ontology Generator to create training and testing datasets. This tool uses backward-chaining to generate valid facts and inductive constraints to split individuals between train (e.g., `train_Ind_X`) and test (e.g., `test_Ind_Y`) sets.

```bash
# Generate Standard Dataset (Family Ontology)
uv run invoke gen-ft-ont
```

**Custom Generation:**

You can override configuration parameters:

```bash
uv run invoke gen-ft-ont \
    dataset.n_train=500 \
    dataset.n_test=100 \
    negative_sampling.strategy=type_aware
```

## Development

-   **Linting**: `uv run ruff check .`
-   **Formatting**: `uv run ruff format .`
-   **Testing**: `uv run pytest`

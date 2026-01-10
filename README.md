# Synthology <!-- omit in toc -->

**Ontology-Based Synthetic Data Generation for Neuro-Symbolic Knowledge Graph Reasoning**.

This repository contains the source code for my bachelor thesis at KU Leuven.

## Introduction

### Context & Problem Statement

**Neuro-Symbolic AI** aims to bridge the gap between two paradigms: the robustness and pattern-matching capabilities of **Neural AI** (e.g., embeddings, GNNs) and the interpretable, rigorous reasoning of **Symbolic AI** (e.g., logic, ontologies). A key application domain is **Knowledge Graph Reasoning (KGR)**, which involves predicting missing links in a Knowledge Graph (KG) by performing multi-hop logical reasoning.

However, training effective Neuro-Symbolic models requires large datasets that specifically necessitate complex reasoning. Existing data generation methods - such as standard benchmarks, forward-chaining reasoners, or Answer Set Programming (ASP) - often produce datasets that are:

1.  **Biased towards "easy" logic**, allowing models to succeed via shallow heuristics (pattern recognition) rather than learning the underlying logical rules.
2.  **Limited in rule coverage**, failing to represent the full complexity of the ontology.

### Hypothesis and Approach

This project investigates the following research question:

> _How to generate high-quality data that enables a model to perform multi-hop logical reasoning rather than just pattern recognition?_

The core hypothesis is that **backward-chaining data generation** - applying deductive reasoning on ontologies (TBox) to generate synthetic data (ABox) - can create high-quality training datasets. By constructing proof trees for derived facts, we can:

1.  Ensure **logical consistency** and diverse reasoning depths.
2.  Generate **"hard" negative samples** via proof-based corruption (breaking specific links in a valid proof chain), forcing the model to distinguish between valid and invalid reasoning paths.

This repository implements this generator and evaluates the quality of the generated data by training a **Recursive Reasoning Network (RRN)**, a Neuro-Symbolic link prediction model, as well as other baseline models to benchmark performance.

## Table of Contents <!-- omit in toc -->

-   [Introduction](#introduction)
    -   [Context \& Problem Statement](#context--problem-statement)
    -   [Hypothesis and Approach](#hypothesis-and-approach)
-   [Features](#features)
-   [Installation](#installation)
    -   [macOS/Linux](#macoslinux)
    -   [Windows](#windows)
-   [Project structure](#project-structure)
-   [Generating datasets](#generating-datasets)
    -   [ASP solver](#asp-solver)
    -   [Ontology-based generator](#ontology-based-generator)
-   [Training RRN model](#training-rrn-model)
-   [Full workflow](#full-workflow)
-   [Custom configurations](#custom-configurations)
    -   [1. Edit configuration files](#1-edit-configuration-files)
    -   [2. Override configurations from command line](#2-override-configurations-from-command-line)
-   [Development](#development)
-   [Known issues](#known-issues)
    -   [1. Python output buffering](#1-python-output-buffering)

## Features

Don't worry if the repository looks a bit overwhelming :)
I value **reproducibility** of scientific experiments very highly, so:

-   I created a very sophisticated `uv` **_monorepo_**, i.e. a single repository containing multiple packages as 'subprojects'.
-   I added a **devcontainer** configuration for easy setup on any OS (including Windows).

The _subprojects_ (located in `apps/`) are:

-   `ont_generator`: The backward-chaining ontology-based data generator I created for my thesis
-   `asp_generator`: The ASP-based family tree data generator by Patrick Hohenecker (see [below](#ASP-solver))
-   `rrn`: The Recursive Reasoning Model (also by Patrick Hohenecker) is a neuro-symbolic link prediction model, used for testing the quality of the generated datasets.
-   `baselines`: A collection of baseline link prediction models (e.g., TransE, DistMult, ComplEx) to further benchmark the performance of the generated datasets.

The `uv` nature of this repo makes it possible to easily manage dependencies between these subprojects. Furthermore, it provides a task runner (`invoke`) to run common tasks (e.g., generating datasets, training models, running experiments) from the project root. Use the following command to see all available tasks:

```bash
uv run invoke --list
```

## Installation

This project uses `uv` for dependency management and `invoke` for task automation.
Make sure you have cloned the repo and are in the project root directory.

### macOS/Linux

Make sure you have `uv` installed (first-time setup), e.g. with Homebrew:

```bash
brew install uv
```

Then, install project dependencies:

```bash
uv sync
```

### Windows

You probably want to open the devcontainer included with this repository (e.g. using VS Code).
This container runs Ubuntu and auto-installs `uv`, after which it sets up the project environment.

After opening the devcontainer, you can run all Unix commands below as-is.

## Project structure

## Generating datasets

### ASP solver

Below, I describe how to generate the [`reldata`](https://github.com/phohenecker/reldata) Family Tree dataset based on the ASP solver by [Patrick Hohenecker](https://github.com/phohenecker/family-tree-data-gen)\_.

Use the provided `invoke` task to generate the dataset. This will generate the dataset in `data/asp/out-reldata`.

```bash
uv run invoke gen-ft-asp
# configs/asp_generator/config.yaml
# configs/asp_generator/dataset/family_tree.yaml
```

Alternatively, you can run the ASP generator directly:

```bash
uv run --package asp_generator python -m asp_generator.create_data
```

To tweak the generation parameters, please refer to the [configuration section](#custom-configurations)

### Ontology-based generator

To use the backward-chaining ontology-based generator (with default configurations) developed in this project, use another `invoke` task:

```bash
uv run invoke gen-ft-ont
# configs/ont_generator/config.yaml
```

Or, directly run the generator:

```bash
uv run --package ont_generator python -m ont_generator.create_data
```

## Training RRN model

To train the Recursive Reasoning Network (RRN) model on the generated family tree datasets, use the following `invoke` task:

```bash
uv run invoke train-rrn
# configs/rrn/  config.yaml
#               data/           default.yaml
#                               dataset/asp.yaml
#                               dataset/ont.yaml
#               model/          default.yaml
#               hyperparams/    default.yaml
```

## Full workflow

1. Generate a dataset using either the ASP-based (default for now) or ontology-based generator (work in progress).
2. Make sure the `data/asp/family_tree/` or `data/ont/family_tree/` folder contains 3 folders: `train/`, `val/`, and `test/`, each containing `.csv` files with triples.
3. Train the RRN model on the generated dataset

## Custom configurations

This repo uses [Hydra](https://hydra.cc/) for configuration management.

You can modify the default configurations in 2 ways:

### 1. Edit configuration files

All configurations -- for the link-prediction models _and_ the data generators -- are stored in the `configs/` folder.
You can create your own configuration files by copying and modifying the existing ones.

For example, create a `hyperparams2.yaml` file in `configs/rrn/hyperparams/` and modify `configs/rrn/config.yaml` to use it:

```yaml
defaults:
    - data: default
    - model: default
    - hyperparams: hyperparams2 # <- your custom hyperparameters
    - _self_
# rest of config...
```

### 2. Override configurations from command line

You can also override specific configuration options directly from the command line.
_(note that this only works when running the packages directly, not via `invoke`)_

```bash
uv run --package ont_generator python -m ont_generator.create_data \
    dataset.n_train=500 \
    dataset.n_val=100 \
    dataset.n_test=100
```

Another example, for training the RRN model with custom (hyper)parameters:

```bash
uv run --package rrn python -m rrn.train \
    hyperparams.num_epochs=20 \
    data/dataset=asp
```

## Development

This repo uses `ruff` for linting and formatting and `ty` for type checking.
It can be a good idea to install the VS Code extensions for these tools to get real-time feedback while coding.

You can also run the following CLI commands to check your code:

-   **Linting**: `uv run ruff check .`
-   **Formatting**: `uv run ruff format .`
-   **Testing**: `uv run pytest`

## Known issues

### 1. Python output buffering

In case the terminal doesn't show real-time updates, try setting the following environment variable:

```bash
export PYTHONUNBUFFERED=1
```

This forces Python to flush its output buffer immediately.

# Synthology <!-- omit in toc -->

**Ontology-Based Synthetic Data Generation for Neuro-Symbolic Knowledge Graph Reasoning**.

This repository contains the source code for my bachelor thesis at KU Leuven.

## Introduction

### Context & Problem Statement

**Neuro-Symbolic AI** aims to bridge the gap between two paradigms: the robustness and pattern-matching capabilities of **Neural AI** (like KG embeddings and GNNs) and the interpretable, rigorous reasoning of **Symbolic AI** (e.g. formal logic and ontologies). A key application domain is **Knowledge Graph Reasoning (KGR)**, which involves predicting missing links in a Knowledge Graph (KG) by performing multi-hop logical reasoning.

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
        -   [UV installation](#uv-installation)
        -   [DLV](#dlv)
        -   [Development tools](#development-tools)
    -   [Windows](#windows)
        -   [Activation of virtual environment](#activation-of-virtual-environment)
        -   [DLV](#dlv-1)
        -   [Development tools](#development-tools-1)
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

-   I created a sophisticated `uv` **_monorepo_**, i.e. a single repository containing multiple packages as 'subprojects', each with their own dependencies and configurations.
-   I added a **Linux devcontainer** for easy setup on any OS (including Windows, which is not Unix-based like Linux or macOS).

The _subprojects_ (located in `apps/`) are:

-   `ont_generator`: The backward-chaining ontology-based data generator I created for my thesis
-   `asp_generator`: The ASP-based family tree data generator by Patrick Hohenecker (see [below](#ASP-solver))
-   `rrn`: The Recursive Reasoning Model (also by Patrick Hohenecker) is a neuro-symbolic link prediction model, used for testing the quality of the generated datasets.
-   `baselines`: A collection of baseline link prediction models (e.g., TransE, DistMult, ComplEx) to further benchmark the performance of the generated datasets.

The `uv` nature of this repo makes it possible to easily manage **dependencies** between these subprojects. Furthermore, it provides a **task runner** (`invoke`) to run common tasks (e.g., generating datasets, training models, running experiments) from the project root. Use the following command to see all available tasks:

```bash
uv run invoke --list        # list all available tasks
uv run invoke <task-name>   # run a specific task
```

## Installation

This project uses `uv` for dependency management and `invoke` for task automation.
Make sure you have **cloned** the repo and are in the project **root directory**.

### macOS/Linux

On Unix systems, you can locally run all commands **as-is**. As an alternative, follow the [Windows](#windows) instructions to use the **devcontainer**.
Below are the steps to set up the project on your own macOS or Linux machine **without** using the devcontainer.

#### UV installation

If don't already have `uv` installed, then do so first, e.g. on macOS with Homebrew:

```bash
brew install uv
```

Or on Linux using the official installation script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, install project dependencies:

```bash
uv sync
```

As you can see, with `uv`, installing dependencies is as easy as running a single command! No contradictory `requirements.txt` files or anything like that :)

#### DLV

The family tree data generator makes use of the DLV system in order to perform symbolic reasoning over family trees by
means of the ontology mentioned above.

If you are running the project on your **own Linux machine**, you can use the provided installation script to download and set up DLV automatically:

```bash
bash install-dlv-linux.sh
```

If running the project on your **own macOS** machine, you have to download the DLV executable for your platform from the
[official website](http://www.dlvsystem.com/dlv/#1)

After you have downloaded and extracted the DLV executable, change the permissions to make it executable:

```bash
chmod +x /path/to/dlv/executable
```

Finally, update the configuration file `configs/asp_generator/config.yaml` to point to the DLV executable you just downloaded:

```yaml
# configs/asp_generator/config.yaml
# ...
dlv: /path/to/dlv/executable # <- change this!
# ...
```

#### Development tools

See the [Development](#development) section for instructions on setting up development tools like `ruff` and `ty` (using VS Code extensions is recommended).

### Windows

For the easiest use, you should open the **devcontainer**, which I included in `.devcontainer/`, for example using VS Code:

-   I assume you are in the project root directory.
-   Click the `><` icon in the bottom-left corner of VS Code.
-   Select `Reopen in Container`.

The (Linux) devcontainer will be built using `Dockerfile` and `post_create.sh` will take care of installing `uv`, as well as syncing the project dependencies and setting up the config files.

#### Activation of virtual environment

After the installation is complete, VS Code might prompt you with

> "Press any key to exit"

Once you actually press a key, a new terminal will open in the devcontainer, but the virtual environment might **not** be activated yet.

**Close the terminal and open a new one** (`CMD + J` or `Terminal > Create New Terminal`). This new terminal should now have the **virtual environment** activated _automatically_.

You should **always** see `(synthology) > ` at the beginning of the terminal prompt when working in the devcontainer, which indicates that the virtual environment is active.

#### DLV

You don't need to install DLV manually (like on macOS/Linux), as it is already installed in the devcontainer.

#### Development tools

See the [Development](#development) section for instructions on setting up development tools like `ruff` and `ty` (using VS Code extensions is recommended).

## Generating datasets

### Standard Data Format

All generators output data in a **standardized format**.
Each split (`train`, `val`, `test`) contains:

-   **`facts.csv`**: Base facts (explicit relations/memberships).
-   **`targets.csv`**: All facts (base + inferred) and negative samples.

### ASP solver (Family Tree)

Below, I describe how to generate the [`reldata`](https://github.com/phohenecker/reldata) Family Tree dataset based on the ASP solver by [Patrick Hohenecker](https://github.com/phohenecker/family-tree-data-gen).

**Quick Start (generates and converts to standard format):**
```bash
uv run invoke gen-ft-asp
```

This command generates raw `reldata` output in `data/asp/out-reldata` and then automatically converts it to the standard format (`facts.csv` and `targets.csv`) in `data/asp/family_tree/{train,val,test}`.

**Step-by-Step (for more control):**

1.  **Generate Raw Data Only**:
    ```bash
    uv run --package asp_generator python apps/asp_generator/src/asp_generator/create_data.py
    ```
    This generates raw `reldata` output in `data/asp/out-reldata` without converting.

2.  **Convert to Standard Format** (separate step):
    ```bash
    uv run invoke convert-reldata
    ```
    This converts existing data in `data/asp/out-reldata` to the standard format.

To tweak the generation parameters, please refer to the [configuration section](#custom-configurations).

### Ontology-based generator

To use the backward-chaining ontology-based generator (which outputs the standard format):

```bash
uv run invoke gen-ft-ont
```

Or run directly:

```bash
uv run --package ont_generator python -m ont_generator.create_data
```

This generates `facts.csv` and `targets.csv` in `data/ont/family/{train,val,test}`.

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

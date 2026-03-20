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

- [Introduction](#introduction)
    - [Context \& Problem Statement](#context--problem-statement)
    - [Hypothesis and Approach](#hypothesis-and-approach)
- [Features](#features)
- [Installation](#installation)
    - [macOS/Linux](#macoslinux)
        - [UV installation](#uv-installation)
        - [DLV](#dlv)
    - [Windows](#windows)
        - [Activation of virtual environment](#activation-of-virtual-environment)
        - [DLV](#dlv-1)
        - [Development tools](#development-tools)
- [Generating datasets](#generating-datasets)
    - [Standard Data Format](#standard-data-format)
    - [ASP solver (Family Tree)](#asp-solver-family-tree)
    - [Ontology-based generator](#ontology-based-generator)
- [Training RRN model](#training-rrn-model)
- [Hyperparameter Optimization (WandB Sweeps)](#hyperparameter-optimization-wandb-sweeps)
- [Full workflow](#full-workflow)
- [Custom configurations](#custom-configurations)
    - [1. Edit configuration files](#1-edit-configuration-files)
    - [2. Override configurations from command line](#2-override-configurations-from-command-line)
- [Experiments](#experiments)
    - [Descriptions](#descriptions)
        - [1. Depth test](#1-depth-test)
        - [2. Negative Sampling Quality](#2-negative-sampling-quality)
        - [3. Information Density](#3-information-density)
        - [4. Scalability](#4-scalability)
    - [Creating a 'hard test set'](#creating-a-hard-test-set)
- [Development](#development)
    - [`uv`](#uv)
- [TODO](#todo)
- [Known issues](#known-issues)
    - [1. Python output buffering](#1-python-output-buffering)

## Features

Don't worry if the repository looks a bit overwhelming :)
I value **reproducibility** of scientific experiments very highly, so:

- I created a sophisticated `uv` **_monorepo_**, i.e. a single repository containing multiple packages as 'subprojects', each with their own dependencies and configurations.
- I added a **Linux devcontainer** for easy setup on any OS (including Windows, which is not Unix-based like Linux or macOS).

The _subprojects_ (located in `apps/`) are:

- `ont_generator`: The backward-chaining ontology-based data generator I created for my thesis
- `asp_generator`: The ASP-based family tree data generator by Patrick Hohenecker (see [below](#ASP-solver))
- `rrn`: The Recursive Reasoning Model (also by Patrick Hohenecker) is a neuro-symbolic link prediction model, used for testing the quality of the generated datasets.
- `baselines`: A collection of baseline link prediction models (e.g., TransE, DistMult, ComplEx) to further benchmark the performance of the generated datasets.

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

### Windows

For the easiest use, you should open the **devcontainer**, which I included in `.devcontainer/`, for example using VS Code:

- I assume you are in the project root directory.
- Click the `><` icon in the bottom-left corner of VS Code.
- Select `Reopen in Container`.

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

- **`facts.csv`**: Base facts (explicit relations/memberships).
- **`targets.csv`**: All facts (base + inferred) and negative samples.

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

## Hyperparameter Optimization (WandB Sweeps)

You can run hyperparameter sweeps that span **both** the ontology data generation and the RRN model training. This allows you to find the optimal combination of dataset characteristics (e.g., complexity, size, negative sampling ratio) and model hyperparameters.

A wrapper script `scripts/sweep_ont_rrn.py` handles the coordination between the generator and the model.

1.  **Define your sweep configuration**:
    Create a YAML file (e.g., `configs/my_sweep.yaml`) defining the parameters to tune. Use the prefix `gen.` for generator parameters and `rrn.` for RRN parameters.

    Example (`configs/sweep_sample.yaml`):

    ```yaml
    program: scripts/sweep_ont_rrn.py
    method: bayes
    metric:
        name: val_loss
        goal: minimize
    parameters:
        # Generator Parameters
        gen.dataset.n_train:
            values: [1000, 2000]
        gen.negative_sampling.ratio:
            min: 0.5
            max: 2.0

        # Model Parameters
        rrn.hyperparams.learning_rate:
            min: 0.0001
            max: 0.01
    ```

2.  **Initialize the sweep**:

    ```bash
    uv run wandb sweep configs/sweep_sample.yaml
    ```

    This will output a sweep ID (e.g., `username/project/sweep_id`).

3.  **Start the agent**:
    ```bash
    uv run wandb agent <SWEEP_ID>
    ```

The script automatically generates a temporary dataset for each run, trains the model on it, reports metrics to WandB, and cleans up the data afterwards.

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

## Experiments

I aim to complete the first two experiments listed below. Time will probably not allow for the other two, although they are very useful for evaluating the generator.

### Descriptions

#### 1. Depth test

- **Goal**: Prove that the Synthology generator creates that that _requires deep reasoning_
- **Method**:
    - Train RRN on 500 KGs generated by OWL2Bench-RL (10 universities)
    - Train RRN on 500 KGs generated by Synthology (with high depth)
    - Test both models on **_hard_ test set** featuring 3+ hop inferences.
- **Metric**: test set accuracies
- **Expected Result**: Synthology should perform better on the hard test set.

#### 2. Negative Sampling Quality

- **Goal**: Validate the _proof-based corruption_ (hard negatives)
- **Method**:
    - Test out all different methods
        - Random
        - Constrained
        - Proof-based
        - Mixed
- **Metric**: Measure False Positive Rate (FPR) on test set that contains near-miss triples.

#### 3. Information Density

- **Goal**: Show that the data is information-dense
- **Method**:
    - Take the Synthology and OWL2Bench-RL datasets.
    - Create training **subsets** of **increasing sizes**: 10%, 25%, 50%, and 100% of the total data.
    - Train the RRN from scratch on each subset.
    - Evaluate all models on the exact same **standard evaluation/test set**.
- **Metric**: Learning curves for each subset size
- **Expected Result**:
    - An intelligent generator should reach high accuracy with _fewer_ training samples because every sample is designed to be a "lesson" in logic.
    - We expect a **steeper learning curve** for the Synthology generator.

#### 4. Scalability

- **Goal**: Show that the generator can scale to large ontologies (way more complex than family tree)
- **Method**:
    - Select 3 ontologies with increasing complexity
        - Level 1: simple hierarchy (standard subclasses, 1-hop relations)
        - Level 2: intermediate (transitivity and property chains)
        - Level 3: complex (heavy use of disjointness, recursive rules and deep property chains)
    - Generate fixed number of valid triples (10 000) for each ontology
- **Metric**:
    - Total CPU execution time for generation
    - Peak RAM usage during generation
    - Number of "dead ends" (discarded proofs)
- **Expected Result**: The generator should be able to scale to large ontologies without running out of memory or taking too long to generate.

### Creating a 'hard test set'

A standard test set is chosen at random, meaning it usually consists of "easy" 1-hop facts. To create a hard test set, we must **filter triples** based on their proof complexity:

1. Run the backward-chainer to generate a massive pool of valid inferences.
2. Filter this pool to only include triples where the shortest possible proof tree requires 3 or more hops.
3. Sample our test triples exclusively from this filtered pool.

We shall **publish** the hard test set(s) for reproducibility.

## Development

### `uv`

Creating a new subproject:

```bash
uv init apps/my-new-app --package
uv sync
```

Adding new dependencies only to a specific subproject:

```bash
uv add <dependency> --package my-new-app
```

## TODO

- [x] Add OWL2Bench RL generator pipeline
- [ ] Add experiments
- [ ] Add `invoke` commands to reproduce experiments
- [ ] Add OWL2Bench/Jena Java dependencies to devcontainer

## Known issues

### 1. Python output buffering

In case the terminal doesn't show real-time updates, try setting the following environment variable:

```bash
export PYTHONUNBUFFERED=1
```

This forces Python to flush its output buffer immediately.

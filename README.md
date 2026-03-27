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
    - [Experiment 1: The Negative Sampling Ablation Study](#experiment-1-the-negative-sampling-ablation-study)
    - [Experiment 2: The Multi-Hop Quality Test](#experiment-2-the-multi-hop-quality-test)
    - [Experiment 3: Generalization to Complex Ontologies](#experiment-3-generalization-to-complex-ontologies)
- [OWL2 RL Profile Coverage and Appendix Tables](#owl2-rl-profile-coverage-and-appendix-tables)
    - [Implemented OWL2 RL Subset](#implemented-owl2-rl-subset)
    - [Currently Missing or Partial Constructs](#currently-missing-or-partial-constructs)
    - [Appendix Table A: Configuration Parameters (1/2)](#appendix-table-a-configuration-parameters-12)
    - [Appendix Table B: Configuration Parameters (2/2)](#appendix-table-b-configuration-parameters-22)
    - [Appendix Table C: Algorithm Terminology](#appendix-table-c-algorithm-terminology)
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
        gen.neg_sampling.ratio:
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

To definitively prove the value of `Synthology`, we will conduct three rigorous experiments. All experiments compare our backward-chaining approach against standard forward-chaining baselines, utilizing the **Nemo** Datalog engine as the ground-truth reasoner.

The evaluation metric for the RRN is **AUC-ROC and F1-Score** (classification metrics), with a specific focus on the **False Positive Rate (FPR)** on hard negatives.

### Experiment 1: The Negative Sampling Ablation Study

**Goal:** Prove that proof-based corruption generates harder, more educational negative samples than random corruption.
**Steps:**

1.  **Generate Data:** Use `Synthology` on the `family-tree` ontology to generate 4 identical datasets, varying _only_ the negative sampling strategy in the `config.yaml`:
    - Dataset 1: `random`
    - Dataset 2: `constrained_random`
    - Dataset 3: `proof_based`
    - Dataset 4: `mixed`
2.  **Create the Test Set:** Generate a separate, small test set containing a high volume of "near-miss" hard negatives (e.g., swapping a parent for an uncle).
3.  **Train & Evaluate:** Train an RRN model on each of the 4 datasets using `wandb` sweeps.
4.  **Metrics:** Evaluate on the hard-negative test set. Track the **False Positive Rate (FPR)**.
5.  **Outcome:** Select the winning strategy (expected: `proof_based`) and freeze it for all subsequent experiments.

### Experiment 2: The Multi-Hop Quality Test

**Goal:** Prove that backward-chaining (Synthology) generates higher-quality multi-hop logical paths than unguided forward-chaining (Nemo).
**Steps:**

1.  **Create the Frozen Gold-Standard Test Set:**
    - Run Synthology with `max_recursion: 4` to generate a small pool of graphs.
    - Filter this pool to isolate ONLY target facts that required $\ge 3$ hops (e.g., `hasGreatGrandparent`).
    - Manually verify these facts. Freeze this as the absolute test set for both models.
2.  **Generate Baseline (Dataset A - Nemo):**
    - Write a Python script to generate random base facts (e.g., random `hasParent` links).
    - Feed these into the Nemo CLI (`nmo`) using the family tree rules to forward-chain inferred targets.
    - Apply standard random negative sampling to the outputs.
3.  **Generate Synthology (Dataset B):**
    - Use Synthology (with the winning negative sampling strategy) to generate a dataset that _exactly matches_ Dataset A in: Total KGs ($N=5000$), Entity Pool size, Total Base Facts, and Total Target Facts.
4.  **Train & Evaluate:** Train an RRN on Dataset A, and a separate RRN on Dataset B. Test BOTH strictly on the Frozen Gold-Standard Test Set.
5.  **Outcome:** A higher AUC-ROC for Dataset B proves that _how_ the base facts are engineered (backward-chaining) matters more than just randomly generating data for a standard reasoner.

### Experiment 3: Generalization to Complex Ontologies

**Goal:** Prove `Synthology` is ontology-agnostic and scales to complex schemas like `UNIV-BENCH-OWL2RL.owl`, overcoming the infrastructure bottleneck of standard forward-chainers.
**Steps:**

1.  **Setup the Baseline Engine (Nemo):**
    - Ensure `data/ont/owl2rl.rls` (the standard OWL 2 RL Datalog ruleset) is configured.
    - Run the standard `OWL2Bench` Java generator to create 50 universities (base facts).
    - Feed `UNIV-BENCH-OWL2RL.owl`, the base facts, and `owl2rl.rls` into Nemo.
    - Filter Nemo's output: Drop all `BNode` inferences and `owl:differentFrom` schema assertions.
    - Extract the University prefix (`U0...`, `U1...`) to partition the massive graph into 50 smaller `sample_id` subgraphs for the RRN. Count the exact yield of positive targets. This is **Dataset A**.
2.  **Setup Synthology (Dataset B):**
    - Feed `UNIV-BENCH-OWL2RL.owl` into Synthology.
    - Configure it to generate 50 graphs (`n_train: 50`) with an `individual_pool_size` mimicking a university.
    - Over-generate slightly (e.g., aim for 20% more targets than Dataset A).
3.  **The "Balancer" Script (Crucial for Fairness):**
    - Write a Python script to randomly downsample Synthology's target facts to _perfectly match_ the exact target count of Dataset A. Both models must train on the exact same volume of targets and graphs.
4.  **Create OWL2Bench Hard Test Set:** Generate a frozen test set of deep inferences for the university domain (e.g., 3-hop `knows` relations).
5.  **Train & Evaluate:** Train an RRN on Dataset A and Dataset B. Evaluate on the frozen test set.
6.  **Track Generator Metrics:** During generation, record Synthology's:
    - **Generation Runtime** vs. `max_proof_depth`.
    - **Yield Rate:** Ratio of base facts synthesized to inferred targets produced.
    - **Ontology Coverage:** Percentage of OWL2Bench rules triggered.

By proving that an RRN trained on a balanced Synthology dataset outperforms the Nemo baseline, we demonstrate that Synthology efficiently engineers high-quality, complex data from any arbitrary TBox.

## OWL2 RL Profile Coverage and Appendix Tables

This section documents what is currently implemented in the ontology parser/chainer and what is not yet implemented.

### Implemented OWL2 RL Subset

The current implementation supports the following core axioms and property types:

- `rdfs:subClassOf`
- `rdfs:subPropertyOf`
- `rdfs:domain`
- `rdfs:range` (object-class ranges; datatype ranges are currently skipped as inference rules)
- `owl:inverseOf`
- `owl:propertyChainAxiom` for chain lengths `1` and `2`
- `owl:disjointWith` (as a consistency constraint)
- `rdf:type` handling for:
    - `owl:SymmetricProperty`
    - `owl:TransitiveProperty`
    - `owl:ReflexiveProperty`
    - `owl:IrreflexiveProperty` (constraint)
    - `owl:AsymmetricProperty` (constraint)
    - `owl:FunctionalProperty` (constraint)

### Currently Missing or Partial Constructs

Important OWL2 RL constructs that are not yet fully supported include:

- Restriction-heavy constructs encoded with blank nodes, such as combinations of:
    - `owl:onProperty`
    - `owl:someValuesFrom`
    - `owl:allValuesFrom`
    - `owl:hasValue`
    - qualified cardinality variants
- Equivalence and identity constructs:
    - `owl:equivalentClass`
    - `owl:equivalentProperty`
    - `owl:sameAs` closure/rewrite behavior
- Set/boolean class constructors:
    - `owl:intersectionOf`
    - `owl:unionOf`
    - `owl:complementOf`
    - `owl:oneOf`
- Disjointness/group constructs such as:
    - `owl:propertyDisjointWith`
    - `owl:AllDisjointClasses`
    - `owl:AllDifferent`

Design note: this is an implementation scope choice, not an architectural limitation. New support can be added incrementally through parser handlers and rule templates.

### Appendix Table A: Configuration Parameters (1/2)

| YAML Parameter           | Symbol               | Type        | Default | Description                                                                                                                                                                      |
| ------------------------ | -------------------- | ----------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | -------------------------------------------------------------------------------------------------------- |
| `min_individuals`        | $I_{\min}$           | int         | 1       | Lower acceptance bound on sample size: graphs with fewer individuals are rejected.                                                                                               |
| `max_individuals`        | $I_{\max}$           | int         | 1000    | Upper acceptance bound on sample size: graphs with more individuals are rejected.                                                                                                |
| `min_rules`              | $R_{\min}$           | int         | 1       | Minimum number of ontology rules selected per generated sample before proof generation.                                                                                          |
| `max_rules`              | $R_{\max}$           | int         | 5       | Maximum number of ontology rules selected per generated sample.                                                                                                                  |
| `target_min_proofs_rule` | $P_{\min}$           | int         | 5       | Target lower bound on proofs kept per selected rule; effectively bounded by how many valid proofs exist.                                                                         |
| `seed`                   | $s$                  | int         | 23      | Seed for pseudorandom sampling (rule selection, proof-root counts, and corruption choices), improving reproducibility.                                                           |
| `max_recursion`          | $d_r$                | int         | 3       | Per-sample recursion cap for rule reuse in backward chaining; deeper recursion allows longer inference chains.                                                                   |
| `global_max_depth`       | $d_{\max}$           | int         | 10      | Absolute depth limit for recursive proof search; branches beyond this depth are pruned.                                                                                          |
| `max_proofs_per_atom`    | $\kappa$             | int         | 5       | Hard cap on number of proofs emitted for one goal atom, preventing combinatorial explosion.                                                                                      |
| `individual_pool_size`   | $                    | \mathcal{U} | $       | int                                                                                                                                                                              | 60  | Target size of the reusable individual pool used when instantiating variables during proof construction. |
| `individual_reuse_prob`  | $\pi_{\text{reuse}}$ | float       | 0.7     | Probability of reusing an existing individual from the pool rather than creating a new one.                                                                                      |
| `use_signature_sampling` |                      | bool        | true    | If enabled, generated proofs are grouped by structural signature and one representative per group is sampled, improving diversity and reducing redundant Cartesian combinations. |
| `min_proof_roots`        | $U_{\min}$           | int         | 5       | Minimum number of independent root-generation cycles attempted per selected rule.                                                                                                |
| `max_proof_roots`        | $U_{\max}$           | int         | 20      | Maximum number of independent root-generation cycles attempted per selected rule.                                                                                                |

### Appendix Table B: Configuration Parameters (2/2)

| YAML Parameter         | Symbol              | Type  | Default | Description                                                                                                                                              |
| ---------------------- | ------------------- | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `always_generate_base` |                     | bool  | false   | If true, emits a base proof even when derivation rules apply; if false, base proofs are mainly used when no matching rule exists.                        |
| `min_lcc_ratio`        | $\rho_{\text{lcc}}$ | float | 0.8     | Validation threshold for graph connectivity: the largest connected component must cover at least this fraction of individuals.                           |
| `strategy`             | $s_{\text{neg}}$    | enum  | `mixed` | Negative sampling mode: `random`, `constrained`, `type_aware`, `proof_based`, or `mixed`.                                                                |
| `ratio`                | $\rho_{\pm}$        | float | 1.0     | Target negative-to-positive ratio for generated examples; $\rho_{\pm}=1$ gives approximately balanced counts.                                            |
| `corrupt_base_facts`   |                     | bool  | false   | Enables corruption of proof-leaf base facts in proof-based logic; this controls whether propagated counterfactual negatives are produced in that branch. |

### Appendix Table C: Algorithm Terminology

| Algorithm Term                      | Symbol                                          | Meaning                                                                                                                      |
| ----------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Ontology                            | $\mathcal{O}$                                   | Input ontology (TBox) containing classes, properties, constraints, and executable rules.                                     |
| Split identifier                    | $S \in \{\text{train},\text{val},\text{test}\}$ | Current split being generated.                                                                                               |
| Requested split size                | $N_S$                                           | Target number of graph samples for split $S$.                                                                                |
| Accepted split dataset              | $\mathcal{D}_S$                                 | Collection of accepted generated KG samples for split $S$.                                                                   |
| Proof map                           | $\mathcal{P}$                                   | Atom-to-proofs map built during one generation attempt.                                                                      |
| Selected rule                       | $r$                                             | One ontology rule selected for backward-chaining in the current attempt.                                                     |
| Proof root attempt                  |                                                 | One independent restart of proof generation for a selected rule.                                                             |
| Base facts                          | $\mathcal{B}$                                   | Leaf atoms in proof trees; support facts not derived from deeper rule applications in the current proof instance.            |
| Inferred facts                      | $\mathcal{I}$                                   | Non-leaf atoms entailed by applying rules over base and/or previously inferred facts.                                        |
| Candidate graph                     | $\mathcal{G}$                                   | One candidate knowledge graph (KG) sample assembled from $(\mathcal{B},\mathcal{I})$ before final acceptance.                |
| Generated negatives                 | $\mathcal{N}$                                   | Negative facts created for $\mathcal{G}$ by the selected corruption strategy and ratio.                                      |
| KG sample                           |                                                 | One self-contained graph instance containing labeled positives and negatives.                                                |
| Fact-type metadata                  |                                                 | Provenance tag indicating whether a positive fact is base or inferred, and for negatives, which corruption path produced it. |
| CSV type: base fact                 | `base_fact`                                     | Positive base support fact (leaf-level fact used as observed evidence).                                                      |
| CSV type: inferred                  | `inf_root`                                      | Positive inferred fact classified as a proof root (i.e., not used as an intermediate sub-goal in another proof).             |
| CSV type: inferred intermediate     | `inf_intermediate`                              | Positive inferred fact that appears as an intermediate/sub-goal node in a deeper proof chain.                                |
| CSV type: neg from base             | `neg_base_fact`                                 | Negative generated from corruption of a base fact (proof-leaf corruption provenance).                                        |
| CSV type: neg inferred              | `neg_inf_root`                                  | Negative inferred/goal-level sample (including propagated proof-based negatives and other non-base negatives).               |
| CSV type: neg inferred intermediate | `neg_inf_intermediate`                          | Negative sample derived from corruption of inferred (non-base) support facts rather than directly from base leaves.          |

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

- [ ] Add OWL2Bench RL generator pipeline
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

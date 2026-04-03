# Synthology <!-- omit in toc -->

**Ontology-Based Synthetic Data Generation for Neuro-Symbolic Knowledge Graph Reasoning**.

_**Vincent Van Schependom**, Cas Proost, Pieter Bonte_\
_Department of Computer Science, KU Leuven campus Kulak Kortrijk_

[Read the preprint](paper/preprint.pdf)

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

1.  Ensure **multi-hop data** that requires chaining multiple reasoning steps.
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
    - [Vendor folders (OWL2Bench and Apache Jena)](#vendor-folders-owl2bench-and-apache-jena)
  - [Windows](#windows)
    - [Activation of virtual environment](#activation-of-virtual-environment)
    - [DLV](#dlv-1)
    - [Development tools](#development-tools)
  - [High Performance Cluster (HPC)](#high-performance-cluster-hpc)
    - [Dependencies](#dependencies)
- [Reproducability](#reproducability)
- [Training RRN model](#training-rrn-model)
- [Data generation](#data-generation)
  - [Ontologies](#ontologies)
  - [Standard Data Format](#standard-data-format)
  - [ASP solver (Family Tree)](#asp-solver-family-tree)
  - [Ontology-based generator (Synthology)](#ontology-based-generator-synthology)
- [Visual verification](#visual-verification)
  - [Category A: OWL2Bench generator checks](#category-a-owl2bench-generator-checks)
  - [Category B: UDM baseline checks](#category-b-udm-baseline-checks)
  - [Category C: Synthology ont\_generator checks](#category-c-synthology-ont_generator-checks)
  - [Category D: Cross-generator paper plots](#category-d-cross-generator-paper-plots)
- [Hyperparameter Optimization (WandB Sweeps)](#hyperparameter-optimization-wandb-sweeps)
- [Custom configurations](#custom-configurations)
  - [1. Edit configuration files](#1-edit-configuration-files)
  - [2. Override configurations from command line](#2-override-configurations-from-command-line)
- [Experiment Protocols](#experiment-protocols)
- [OWL2 RL Profile Coverage and Appendix Tables](#owl2-rl-profile-coverage-and-appendix-tables)
  - [Implemented OWL2 RL Subset](#implemented-owl2-rl-subset)
  - [Currently Missing or Partial Constructs](#currently-missing-or-partial-constructs)
- [Appendix](#appendix)
  - [Appendix Table A: Configuration Parameters (1/2)](#appendix-table-a-configuration-parameters-12)
  - [Appendix Table B: Configuration Parameters (2/2)](#appendix-table-b-configuration-parameters-22)
  - [Appendix Table C: Algorithm Terminology](#appendix-table-c-algorithm-terminology)
- [Development](#development)
  - [`uv`](#uv)
- [Known issues](#known-issues)
  - [1. Python output buffering](#1-python-output-buffering)

## Features

Don't worry if the repository looks a bit overwhelming :)
I value **reproducibility** of scientific experiments very highly, so:

- I created a sophisticated `uv` **_monorepo_**, i.e. a single repository containing multiple packages as 'subprojects', each with their own dependencies and configurations.
- I added a **Linux devcontainer** for easy setup on any OS (including Windows, which is not Unix-based like Linux or macOS).

The _subprojects_ (located in `apps/`) are:

[TODO]

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

#### Vendor folders (OWL2Bench and Apache Jena)

Some workflows (notably OWL2Bench generation and Jena-backed materialization) rely on files in `vendor/`.
By default, this repo keeps these folders out of git history (see `.gitignore`) to avoid committing large third-party artifacts.

From the project root, set them up as follows:

```bash
mkdir -p vendor

# OWL2Bench Java generator source (required for gen-owl2bench* tasks)
git clone https://github.com/kracr/owl2bench.git vendor/OWL2Bench

# Apache Jena distribution (required by UDM/Jena materialization helper)
curl -L -o /tmp/apache-jena-6.0.0.tar.gz \
    https://archive.apache.org/dist/jena/binaries/apache-jena-6.0.0.tar.gz
tar -xzf /tmp/apache-jena-6.0.0.tar.gz -C vendor
```

After cloning OWL2Bench, ensure the RL ontology path exists at:

- `ontologies/UNIV-BENCH-OWL2RL.owl`

If needed, copy it from the cloned vendor folder:

```bash
mkdir -p ontologies
cp vendor/OWL2Bench/UNIV-BENCH-OWL2RL.owl ontologies/
```

Can you commit `vendor/OWL2Bench` into your repo?

- Technically yes, if the upstream OWL2Bench license permits redistribution and you keep proper attribution.
- Practically, it is usually better to keep it out of git and document a setup command (or use a git submodule) so your repo stays lightweight and easier to maintain.
- Do not commit generated artifacts like `vendor/OWL2Bench/target/` or generated `.owl` outputs.

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

### High Performance Cluster (HPC)

If you want to run the experiments on an LSF cluster, you can use the provided job scripts in `jobscripts/` as templates. Make sure to adjust the resource requests and module loads according to your cluster's specifications.

#### Dependencies

The same dependencies apply as for the local installation (Python, uv, Java, Maven, OWL2Bench, Apache Jena).

If you're on an LSF cluster, you can load Java and Maven modules as follows:

```bash
# Load Java 21 (required by Jena 5.x)
module load openjdk/21

# Verify Java is available and correct version
which java && java -version

# Now install Maven
./install-mvn.sh

# Verify Maven is available
which mvn && mvn -v
```

## Reproducability

The exact sequence of `invoke` commands needed to reproduce our results are located in the 3 experiment-specific `README.md` files:

- `experiments/exp1_negative_sampling/`
- `experiments/exp2_multihop_quality/`
  `experiments/exp3_scaling_bench/`

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

To tweak the parameters, please refer to the [configuration section](#custom-configurations). This also applies to all data generation methods.

## Data generation

### Ontologies

All ontologies that were used for data generation are located in the `ontologies/` folder.

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

### Ontology-based generator (Synthology)

To use the backward-chaining ontology-based generator (which outputs the standard format):

```bash
uv run invoke gen-ft-ont
```

Or run directly:

```bash
uv run --package ont_generator python -m ont_generator.create_data
```

This generates `facts.csv` and `targets.csv` in `data/ont/family/{train,val,test}`.

## Visual verification

This section groups quick visual sanity checks by generator/baseline, so you can inspect outputs before running full experiments.

Jena setup in code/configs:

- Apache Jena libraries: `5.2.0`
- Jena reasoner profile default: `owl_mini`
- Materialization semantics: one-shot closure call (internal fixpoint in Jena)

TODO (camera-ready paper reproducibility): freeze and document one final profile choice for all reported runs (`owl_micro` vs `owl_mini` vs `owl_full`) and include a profile-sensitivity appendix table.

### Category A: OWL2Bench generator checks

Use these commands to confirm the OWL2Bench pipeline generates raw OWL artifacts and parsed split files correctly.

1. **Toy end-to-end OWL2Bench run + auto-visualization**

    ```bash
    uv run invoke gen-owl2bench-toy
    ```

    This runs a small OWL2Bench configuration (`config_toy`), performs materialization, exports split CSV files, and then visualizes a sample graph for a fast end-to-end smoke check.
    If it is still too slow on your machine, reduce the reasoning subset temporarily:

    ```bash
    uv run invoke gen-owl2bench-toy --args="dataset.reasoning_input_triple_cap=3000"
    ```

2. **Full OWL2Bench pipeline run**

    ```bash
    uv run invoke gen-owl2bench
    ```

    This runs the standard OWL2Bench generation/materialization/export pipeline for larger-scale verification and stores results under `data/owl2bench/output`.

3. **Exp3-style OWL2Bench ABox generation path**

    ```bash
    uv run invoke exp3-generate-owl2bench-abox --universities=50
    ```

    This is the experiment-oriented entrypoint that generates OWL2Bench data with the requested university count, used as the baseline ABox source for Exp3.

### Category B: UDM baseline checks

Use these commands to verify Apache Jena-backed UDM materialization and inspect generated baseline samples.

1. **Visual smoke test for UDM + Jena (recommended first check)**

    ```bash
    uv run invoke exp2-smoke-jena-visual
    ```

    This generates a tiny baseline dataset with Jena reasoning and writes a rendered sample graph to `visual-verification/exp2_smoke`, which is useful for quickly checking inferred-fact presence and graph structure.

2. **Family-tree UDM baseline generation (task wrapper)**

    ```bash
    uv run invoke gen-ft-fc
    ```

    This is the reusable UDM baseline generation command for family-tree style data and is the quickest way to validate that baseline `facts.csv` and `targets.csv` generation is healthy.

3. **Exp3 baseline chaining (OWL2Bench generation + UDM materialization)**

    ```bash
    uv run invoke exp3-generate-baseline --universities=50
    ```

    This runs the baseline chain used in Exp3: OWL2Bench ABox generation followed by UDM/Jena materialization, producing closure/inferred artifacts for benchmarking.

4. **Direct ABox materialization with UDM/Jena**

    ```bash
    uv run invoke exp3-materialize-abox \
      --abox=path/to/owl2bench_abox.ttl \
    --tbox=ontologies/UNIV-BENCH-OWL2RL.owl \
      --closure-out=outputs/exp3/closure.nt \
      --inferred-out=outputs/exp3/inferred.nt
    ```

    Use this when you already have an ABox and only want to validate the materialization layer independently from generation.

### Category C: Synthology ont_generator checks

Use these commands to visually verify the backward-chaining generator output and sample-level graph quality.

1. **Generate family-tree data with Synthology (task wrapper)**

    ```bash
    uv run invoke gen-ft-ont
    ```

    This produces standard-format outputs for `train/val/test`, which you can inspect for depth, fact types, and negative-sampling structure.

2. **Generate Exp2 Synthology dataset (experiment path)**

    ```bash
    uv run invoke exp2-generate-synthology
    ```

    This executes the Exp2-aligned Synthology generation path so you can verify the same configuration family used for parity and model comparisons.

3. **Render a selected sample graph from generated CSVs**

    ```bash
    uv run --package kgvisualiser python -m kgvisualiser.visualize \
      io.input_csv=data/ont/family_tree/train/targets.csv \
      io.sample_id=1000 \
      output.dir=visual-verification/ont_generator \
      output.name_template=ont_sample_1000
    ```

    This explicit visualization command is useful when you want to inspect one graph in detail (for example, to confirm multi-hop inferred paths and corruption patterns).

### Category D: Cross-generator paper plots

Use this report command when you want a side-by-side visual summary of baseline vs Synthology behavior.

1. **Generate paper-ready visual diagnostics**

    ```bash
    uv run invoke paper-visual-report \
      --exp2-synth-targets=data/exp2/synthology/family_tree/train/targets.csv \
      --exp2-parity-summary=data/exp2/baseline/parity_runs/parity_loop_summary.json \
      --exp3-targets=data/owl2bench/output/owl2bench_50/train/targets.csv \
      --exp3-abox=data/owl2bench/output/raw/owl2bench_50/OWL2RL-50.owl \
      --exp3-inferred=data/exp3/baseline/owl2bench_50/inferred.nt \
      --out-dir=reports/paper
    ```

    This generates consolidated inspection plots (base vs inferred, hop distributions, parity-attempt trend) so you can validate dataset behavior before or alongside model training.

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
    data/dataset=asp
```

## Experiment Protocols

The detailed, command-by-command experiment protocols now live in the experiment-specific READMEs:

- [Experiment 1: Negative Sampling Ablation](experiments/exp1_negative_sampling/README.md)
- [Experiment 2: Multi-Hop Reasoning Quality](experiments/exp2_multihop_quality/README.md)
- [Experiment 3: Scaling Benchmark](experiments/exp3_scaling_bench/README.md)
- [Paper runbook](experiments/PAPER_RUNBOOK.md)

The main README keeps the repository overview and setup instructions; the experiment folders are the canonical source for execution order, metrics, and artifact expectations.

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

## Appendix

This section contains tables with detailed descriptions of configuration parameters and algorithm terminology, supplementing the [main paper](paper/preprint.pdf) for readers who want to understand the implementation details or customize the generator.

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

| YAML Parameter         | Symbol              | Type  | Default       | Description                                                                                                                                              |
| ---------------------- | ------------------- | ----- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `always_generate_base` |                     | bool  | false         | If true, emits a base proof even when derivation rules apply; if false, base proofs are mainly used when no matching rule exists.                        |
| `min_lcc_ratio`        | $\rho_{\text{lcc}}$ | float | 0.8           | Validation threshold for graph connectivity: the largest connected component must cover at least this fraction of individuals.                           |
| `strategy`             | $s_{\text{neg}}$    | enum  | `proof_based` | Negative sampling mode used in the thesis experiments: `random`, `constrained`, `proof_based`.                                                           |
| `ratio`                | $\rho_{\pm}$        | float | 1.0           | Target negative-to-positive ratio for generated examples; $\rho_{\pm}=1$ gives approximately balanced counts.                                            |
| `corrupt_base_facts`   |                     | bool  | false         | Enables corruption of proof-leaf base facts in proof-based logic; this controls whether propagated counterfactual negatives are produced in that branch. |

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

## Known issues

### 1. Python output buffering

In case the terminal doesn't show real-time updates, try setting the following environment variable:

```bash
export PYTHONUNBUFFERED=1
```

This forces Python to flush its output buffer immediately.

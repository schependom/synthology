# Experiment Conduction Guidelines

This document outlines the standard protocol for running, tracking, and documenting all experiments for the **Synthology** bachelor thesis.

Our core philosophy is **100% Reproducibility**. Every phase of an experiment - from data synthesis to model evaluation - must be executable via a predefined `uv run invoke` command, with all parameters strictly managed by Hydra configuration files.

For the specific experiments we want to conduct, see README.md in the project repo.

## 1. Directory Structure

All experiment-specific documentation and isolated configurations must reside in the `experiments/` directory at the root of the monorepo. The structure must strictly follow this format:

```text
experiments/
├── EXPERIMENT_GUIDELINES.md         # This document
├── exp1_negative_sampling/
│   └── exp1.md                    # Details, commands, and configs for Exp 1
├── exp2_multihop_quality/
│   └── exp2.md                    # Details, commands, and configs for Exp 2
└── exp3_scaling_bench/
    └── exp3.md                    # Details, commands, and configs for Exp 3
```

## 2. The Experiment `README.md` Template

Every subfolder in the `experiments/` directory MUST contain a `README.md` file formatted with the following sections:

- **Objective:** A one-paragraph summary of what hypothesis this experiment tests.
- **Datasets Used:** The underlying ontology (e.g., `ontologies/family.ttl`, `ontologies/UNIV-BENCH-OWL2RL.owl`) and the standard output format directory.
- **Configurations:** The exact Hydra overrides or `.yaml` file names used for the generator and the RRN model.
- **Execution Commands:** The ordered list of `uv run invoke` commands required to run the experiment from start to finish.
- **Expected Results:** A brief note on what metrics are being tracked (e.g., AUC-ROC, FPR) and where they are logged (e.g., WandB).

## 3. Standardized `invoke` Command Nomenclature

To maintain consistency across the `tasks.py` file, all experiment commands must follow the `<experiment-id>-<action>-<target>` naming convention.

### Experiment 1: Negative Sampling Ablation

**Goal:** Compare random vs. constrained vs. proof-based corruption on FPR.

- `uv run invoke exp1-generate-trainval-sets`
- `uv run invoke exp1-generate-trainval`
- `uv run invoke exp1-generate-test-set` _(Generates the frozen "near-miss" hard negative test set)_
- `uv run invoke exp1-train-rrn --run-name="exp1_random"`

### Experiment 2: Multi-Hop Quality Test

**Goal:** Compare Jena forward-chaining against Synthology backward-chaining on >= 3-hop inferences.

- `uv run invoke exp2-generate-gold-test` _(Runs Synthology at depth 4, filters for >= 3 hops, freezes the test set)_
- `uv run invoke exp2-generate-baseline` _(Runs Jena on random base facts)_
- `uv run invoke exp2-generate-synthology` _(Runs Synthology to match the baseline's entity/fact volume)_
- `uv run invoke exp2-parity-loop --deep-count-mode=tolerance --tolerance-pct=10.0` _(Enforces deep-count parity with widened tolerance for convergence)_
- `uv run invoke exp2-train-rrn --dataset=baseline`
- `uv run invoke exp2-train-rrn --dataset=synthology`

### Experiment 3: Scaling Benchmark (OWL2Bench)

**Goal:** Prove ontology-agnostic scaling using `UNIV-BENCH-OWL2RL.owl`.

- `uv run invoke exp3-generate-baseline --universities=20` _(Runs OWL2Bench Java generator + Jena materialization)_
- `uv run invoke exp3-generate-synthology --universities=20` _(Runs Synthology with over-generation)_
- `uv run invoke exp3-balance-data` _(Executes the Python script to downsample Synthology targets to perfectly match the baseline yield)_
- `uv run invoke exp3-generate-gold-test` _(Generates the frozen complex LUBM-style test set)_
- `uv run invoke exp3-train-rrn --dataset=baseline`
- `uv run invoke exp3-train-rrn --dataset=synthology`

## 4. Configuration Management (Hydra)

Never hardcode variables in the `tasks.py` script. The `invoke` commands should act as thin wrappers that pass command-line arguments directly to the underlying Python modules as Hydra overrides.

**Example `tasks.py` implementation:**

```python
@task
def exp1_generate_trainval(c, strategy="proof_based"):
    """Generates train/val sets for Exp 1 with a specific negative sampling strategy."""
    print(f"Generating Exp 1 data using strategy: {strategy}")
    c.run(f"uv run --package ont_generator python -m ont_generator.create_data "
          f"generator.neg_sampling.strategy={strategy} "
          f"dataset.output_dir=data/exp1/{strategy}")
```

This ensures that the `README.md` for an experiment can simply list the command, and the exact configuration state is preserved in the terminal execution and subsequently logged to the canonical run archive.

## 5. Logging Contract

- Canonical paper-run artifacts live under `reports/experiment_runs/<YYYY-MM-DD>/<experiment>/<task>/<timestamp>/`.
- Every archived run contains `manifest.json`, `run.log`, copied configs, and any copied outputs or checkpoints.
- Hydra-managed apps may still create their own local `outputs/` directory, but those files are supplementary.
- RRN training metrics and checkpoints are additionally tracked in Weights & Biases.
- If a command is launched through a task wrapper, its stdout/stderr is tee'd into `run.log` so the console output is preserved.

# Ont Generator Configuration Reference

This document summarizes all Hydra configuration options used by the ontology-based generator (ont_generator), with paper-friendly symbols and semantics.

## Notation

- $\mathcal{O}$: ontology input
- $\mathcal{D}$: generated dataset
- $\mathcal{G}_i$: one generated KG sample
- $N_{\text{train}},N_{\text{val}},N_{\text{test}}$: number of samples per split
- $\rho_{\pm}$: negative sampling ratio (negatives per positive)
- $d_r$: recursion cap used during backward chaining
- $d_{\max}$: global proof-depth cap
- $\kappa$: max proofs stored per atom
- $\pi_{\text{reuse}}$: probability to reuse an existing individual
- $R_{\min},R_{\max}$: number of selected rules per sample
- $I_{\min},I_{\max}$: allowed number of individuals per sample
- $P_{\min}$: minimum proofs selected per chosen rule
- $U_{\min},U_{\max}$: number of proof roots attempted per rule
- $\tau_{\text{lcc}}$: minimum largest-connected-component ratio used in validation

## Top-Level Config Groups

## 1. ontology

| Key           | Symbol        | Type   | Default                   | Description                                                                          |
| ------------- | ------------- | ------ | ------------------------- | ------------------------------------------------------------------------------------ |
| ontology.path | $\mathcal{O}$ | string | data/ont/input/family.ttl | Path to OWL/Turtle ontology file parsed into classes, relations, rules, constraints. |

## 2. dataset

| Key                             | Symbol             | Type        | Default                     | Description                                                                             |
| ------------------------------- | ------------------ | ----------- | --------------------------- | --------------------------------------------------------------------------------------- |
| dataset.n_train                 | $N_{\text{train}}$ | int         | 2000                        | Number of train KG samples to generate.                                                 |
| dataset.n_val                   | $N_{\text{val}}$   | int         | 200                         | Number of validation KG samples to generate.                                            |
| dataset.n_test                  | $N_{\text{test}}$  | int         | 200                         | Number of test KG samples to generate.                                                  |
| dataset.min_individuals         | $I_{\min}$         | int         | 1                           | Minimum individuals allowed in a sample; samples outside bound are rejected.            |
| dataset.max_individuals         | $I_{\max}$         | int         | 1000                        | Maximum individuals allowed in a sample; samples outside bound are rejected.            |
| dataset.min_rules               | $R_{\min}$         | int         | 1                           | Minimum number of rules selected per sample before proof generation.                    |
| dataset.max_rules               | $R_{\max}$         | int         | 5                           | Maximum number of rules selected per sample.                                            |
| dataset.target_min_proofs_rule  | $P_{\min}$         | int         | 5                           | Minimum number of proofs to keep for each selected rule (subject to availability/caps). |
| dataset.output_dir              | -                  | string      | data/ont/output/family-tree | Output root for split folders (train/val/test), logs, and optional visual artifacts.    |
| dataset.seed                    | -                  | int or null | 23                          | Random seed for reproducibility (Python random module).                                 |
| dataset.save_individual_samples | -                  | bool        | false                       | If true, saves per-sample CSV files; otherwise saves standard split files only.         |

## 3. generator

| Key                              | Symbol               | Type           | Default        | Description                                                                                            |
| -------------------------------- | -------------------- | -------------- | -------------- | ------------------------------------------------------------------------------------------------------ |
| generator.max_recursion          | $d_r$                | int            | 3              | Max recursion for backward chaining; sampled per graph as an effective depth in $[1,d_r]$.             |
| generator.global_max_depth       | $d_{\max}$           | int            | 10             | Hard upper bound for proof expansion depth.                                                            |
| generator.max_proofs_per_atom    | $\kappa$             | int            | 5              | Upper bound on retained proofs per atom to control combinatorial growth.                               |
| generator.individual_pool_size   | -                    | int            | 60             | Target size of reusable individual pool used by the chainer.                                           |
| generator.individual_reuse_prob  | $\pi_{\text{reuse}}$ | float in [0,1] | 0.7            | Probability of reusing an existing individual instead of creating a new one.                           |
| generator.use_signature_sampling | -                    | bool           | true           | If true, sampling uses proof signatures for diversity and dedup-style control.                         |
| generator.min_proof_roots        | $U_{\min}$           | int            | 5              | Lower bound on number of proof roots attempted per selected rule.                                      |
| generator.max_proof_roots        | $U_{\max}$           | int            | 20             | Upper bound on number of proof roots attempted per selected rule.                                      |
| generator.always_generate_base   | -                    | bool           | false          | If true, forces explicit base fact generation behavior in the chainer.                                 |
| generator.min_lcc_ratio          | $\tau_{\text{lcc}}$  | float in [0,1] | 0.8 (implicit) | Optional validation threshold for connectedness; consumed via cfg.generator.get("min_lcc_ratio", 0.8). |

## 4. neg_sampling

| Key                             | Symbol       | Type  | Default | Description                                                                                              |
| ------------------------------- | ------------ | ----- | ------- | -------------------------------------------------------------------------------------------------------- |
| neg_sampling.strategy           | -            | enum  | mixed   | Strategy for generating negative examples: random, constrained, proof_based, type_aware, mixed.          |
| neg_sampling.ratio              | $\rho_{\pm}$ | float | 1.0     | Target negative-to-positive ratio. Example: $\rho_{\pm}=1$ gives balanced positives/negatives.           |
| neg_sampling.corrupt_base_facts | -            | bool  | false   | In proof-based or mixed mode, allows corruption at proof-leaf base facts for propagated counterfactuals. |

### Strategy semantics

| Strategy    | Summary                                                                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| random      | Corrupt subject or object uniformly from individuals, then validate as negative.                                   |
| constrained | Corrupt while respecting domain/range candidate filtering.                                                         |
| proof_based | Corrupt facts inside proof trees; can propagate corruption to derived conclusions.                                 |
| type_aware  | Corrupt using class-membership/type compatibility, with fallbacks.                                                 |
| mixed       | Combines multiple strategies in fractional budget; includes proof_based only when base-fact corruption is enabled. |

## 5. Export and visualization flags

| Key                 | Symbol | Type   | Default                  | Description                                                                                           |
| ------------------- | ------ | ------ | ------------------------ | ----------------------------------------------------------------------------------------------------- |
| export_proofs       | -      | bool   | false                    | Export proof-tree visualizations (positive and/or corrupted counterfactual proofs).                   |
| export_graphs       | -      | bool   | false                    | Export graph visualizations for sampled train/test subsets.                                           |
| proof_output_path   | -      | string | derived                  | Optional explicit proof export path. If omitted, defaults to <output_dir>/proofs in create_data flow. |
| graph_output_path   | -      | string | . (legacy generate flow) | Optional explicit graph visualization output path.                                                    |
| visualize_negatives | -      | bool   | false                    | Optional visualization flag used by legacy generate flow to display negative facts.                   |

## 6. logging

| Key           | Symbol | Type        | Default | Description                                                                                            |
| ------------- | ------ | ----------- | ------- | ------------------------------------------------------------------------------------------------------ |
| logging.level | -      | enum string | INFO    | Log verbosity. DEBUG enables detailed rejection/debug traces and verbose mode in generator components. |

## Acceptance and rejection behavior (important for experiments)

A sample $\mathcal{G}_i$ is accepted only if all checks pass:

1. Rule/proof generation produces atoms.
2. Individual count constraint: $I_{\min} \le |\text{Individuals}(\mathcal{G}_i)| \le I_{\max}$.
3. Contains inferred facts (not only base facts).
4. Depth quality heuristic (for deeper recursion settings).
5. Validator passes constraints/domain/range and connectedness checks.

If too many attempts fail, the generator may return fewer than requested samples. In that case split CSV files can be empty or absent depending on row counts.

## Configuration profiles currently present

| File                                                | Purpose                                                                |
| --------------------------------------------------- | ---------------------------------------------------------------------- |
| configs/ont_generator/config.yaml                   | Main/default generation config for ontology datasets.                  |
| configs/ont_generator/config_single_graph.yaml      | Small run focused on a single graph with proof/graph export enabled.   |
| configs/ont_generator/config_visualize.yaml         | Visualization-heavy run with proof exports and debug logs.             |
| configs/ont_generator/config_visual_inspection.yaml | Quick visual verification run (small train count, export enabled).     |
| configs/ont_generator/exp1_random.yaml              | Exp1 train/val generation with random negative sampling.               |
| configs/ont_generator/exp1_constrained.yaml         | Exp1 train/val generation with constrained negative sampling.          |
| configs/ont_generator/exp1_proof_based.yaml         | Exp1 train/val generation with proof-based negative sampling.          |
| configs/ont_generator/exp1_test.yaml                | Exp1 frozen test-set generation (proof-based negatives, no train/val). |

## Minimal paper citation block (copyable)

A concise way to describe settings in text:

"We generate $N_{\text{train}},N_{\text{val}},N_{\text{test}}$ KG samples from ontology $\mathcal{O}$ under size bounds $[I_{\min},I_{\max}]$, rule budget $[R_{\min},R_{\max}]$, recursion cap $d_r$, and proof budget $(P_{\min},\kappa,U_{\min},U_{\max})$. Negative sampling uses strategy $s \in \{\text{random, constrained, type\_aware, proof\_based, mixed}\}$ with ratio $\rho_{\pm}$ and optional base-fact corruption."

# Experiment 1: Negative Sampling Ablation

**Objective:**
Test whether proof-based corruption (corrupting a single leaf node in a proof tree to invalidate the chain) removes shallow statistical biases and improves the robustness (lowers False Positive Rate) over random or simply constrained negative sampling.

**Datasets Used:**
- Built using `family-tree.ttl`. 
- Data yields are housed in `data/exp1/random`, `data/exp1/constrained`, and `data/exp1/proof_based`.
- And a frozen target evaluation testing dataset at `data/exp1/test_set`.

**Configurations:**
For generation, Hydra overrides append `generator.negative_sampling.strategy={strategy}` and `dataset.output_dir=data/exp1/{strategy}`.
For RRN training, `data.dataset.path=cwd/data/exp1/{strategy}` overrides point it to appropriately generated sets. `WandB` is structured with `logger.name`, `logger.group`, and `logger.tags`.

**Execution Commands:**
1. `uv run invoke exp1-generate-trainval-sets`
2. `uv run invoke exp1-generate-test-set` *(Generates the frozen "near-miss" hard negative test set)*
3. `uv run invoke exp1-train-rrn --strategy="random"`
4. `uv run invoke exp1-train-rrn --strategy="constrained"`
5. `uv run invoke exp1-train-rrn --strategy="proof_based"`

**Expected Results:**
- Tracking False Positive Rate (FPR) on near-misses.
- Proof-based strategy should demonstrate improved robustness vs. the random baseline.
- All metrics are logged to Weights & Biases under the group `exp1_negative_sampling`.

# OWL-RL Materialization: Singlepass vs. Iterative Analysis

## Summary

We've implemented both **singlepass** and **iterative** forward-chaining modes. Testing reveals they produce **identical results** on the family-tree ontology with random base facts. This document explains why and how to proceed with the experiment.

## Test Results

Both modes were tested with identical seeds and configurations:

```bash
# Singlepass (default)
uv run invoke exp2-generate-baseline \
  --args='dataset.n_train=2 dataset.n_val=1 dataset.n_test=1 \
          dataset.output_dir=data/exp2/baseline/family_tree_singlepass_test \
          materialization.iterative=false'

# Iterative
uv run invoke exp2-generate-baseline \
  --args='dataset.n_train=2 dataset.n_val=1 dataset.n_test=1 \
          dataset.output_dir=data/exp2/baseline/family_tree_iterative_test \
          materialization.iterative=true materialization.max_iterations=10'
```

**Output Comparison** (train split, 2 samples):

| Mode       | Total Targets | hop=0     | hop=1     | hop≥2 | inf_root |
| ---------- | ------------- | --------- | --------- | ----- | -------- |
| Singlepass | 276           | 140 (51%) | 136 (49%) | 0     | 62       |
| Iterative  | 276           | 140 (51%) | 136 (49%) | 0     | 62       |

**Conclusion**: Identical distributions → Iterative mode reached fixpoint in iteration 1.

## Why No Multi-Hop Reasoning?

The family-tree ontology contains rules like:

- `?x parentOf ?y . ?y parentOf ?z → ?x grandparentOf ?z` (2-hop rule)
- `?x grandparentOf ?y . ?y parentOf ?z → ?x greatgrandparentOf ?z` (3-hop rule)

**However**, these rules require full derivation chains to exist in the base facts. With **random sampling**:

- We sample ~15-20 base individuals per sample
- We sample ~8-40 base relations uniformly at random
- Probability of having `[a parentOf b, b parentOf c, ...]` chains is **extremely low**

**Example**: To infer `a greatgrandparentOf d`, we need:

- `a parentOf b` AND `b parentOf c` AND `c parentOf d` all present in base
- With 20 individuals and ≤40 random relations, this is rare

Therefore, **iterative materialization discovers no new facts beyond iteration 1** because the base ABox lacks the connectivity chains needed.

## Implications for Experiment Design

### ❌ What WON'T Work

1. **Switching reasoners won't help**: Apache Jena is also forward-chaining and subject to the same sparse-connectivity problem.
2. **Iterative mode on random bases won't produce multi-hop**: You'll still get mostly hop {0, 1}.
3. **Cannot fairly compare "baseline vs. Synthology on multi-hop depth"** if the baseline can't produce multi-hop reasoning.

### ✅ What's the Solution?

You have three options:

#### Option A: Restructure Experiment (RECOMMENDED)

**Reframe Exp2** to compare on a different metric aligned with random-base forwards-chaining:

- **Objective**: Compare Synthology vs. FC baseline on **shallow reasoning quality** (within constraints of hop≤2)
- **Test set**: Instead of requiring `d >= 3`, require `d = 1` or `d ≤ 2`
- **Metrics**:
    - Coverage of all single-hop implications per sample
    - Fact explosion ratio (base facts → inferred facts)
    - Accuracy on relation type prediction (which relations are correctly inferred)

**Why this works**: Both methods will produce hop≤2 on random bases, enabling a fair quality comparison.

#### Option B: Structured Base Sampling

**Generate base facts that form dependency chains** instead of purely random:

- Use backward-chaining-like reasoning to **select base facts that form proof paths**
- Example: Sample `[a parentOf b, b parentOf c, c parentOf d]` as a unit so multi-hop inferences are guaranteed
- This enables hop≥3 reasoning for both baseline and Synthology

**Effort**: Medium (requires new base-fact generation algorithm)

#### Option C: Hybrid Approach

**Create two baseline variants**:

1. **Baseline-Random** (current): Random base facts → hop≤2 maximum
2. **Baseline-Structured**: Structured dependency chains → hop≤5+ possible

Compare Synthology against both to show:

- On random: Synthology ≈ baseline quality (both limited by sparsity)
- On structured: Synthology > baseline (better coverage of deep paths)

**Effort**: Low (use existing code for Baseline-Random, add chain-generation logic for Baseline-Structured)

## Recommended Path Forward

1. **Immediate**: Stick with singlepass mode (`materialization.iterative=false`) for Exp2. Iterative is identical on random bases.

2. **Experiment redesign**: Choose one of the options above (Option A is safest for thesis narrative)

3. **Future work**: Implement Jena reasoner as a bonus (can be added without blocking the experiment)

## Configuration Reference

```yaml
materialization:
    reasoner: owlrl # 'owlrl' (only option currently; Jena coming soon)
    iterative: false # Use true for iterative (no benefit on random bases)
    max_iterations: 10 # Ignored if iterative=false
    rdfs_closure: true # RDFS entailment rules
    axiomatic_triples: false # Skip OWL vocabulary self-descriptions
    datatype_axioms: false # Skip datatype reasoning
```

### Switch Modes via Command Line

```bash
# Singlepass (default, recommended for current Exp2)
uv run invoke exp2-generate-baseline \
  --args='materialization.iterative=false'

# Iterative (no practical benefit on random bases)
uv run invoke exp2-generate-baseline \
  --args='materialization.iterative=true materialization.max_iterations=10'
```

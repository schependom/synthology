# Experiment 1 - Known Problems

---

## 1. Constrained class head collapse

**Severity:** 🚨 Critical  
**Affected run:** `exp1_constrained_hpc`  
**Observed symptoms:**

- `val/class_loss` stays flat at ~0.83 throughout entire training (never decreases)
- `val/class_fpr` = 1.0 throughout (100% of class negatives classified as positive)
- `val/class_acc_neg` = 0 from ~1k steps onward
- `val/class_f1` = 0.5 (degenerate constant)
- `val/class_auc_roc` drops below 0.5 (worse than random guessing)
- `val/total_loss` stuck at ~1.45 with no convergence (inflated by the stuck class loss)
- All `val/class_acc_type_*` metrics collapse to 0 for constrained

**What this is not affecting:**

- Relation-triple metrics (FPR, F1, AUC-ROC, hop-stratified accuracy) are unaffected
    - the relation head is learning normally

**Possible causes:**

1. **No class membership negatives in the constrained dataset (most likely)**  
   The constrained negative sampling strategy may only corrupt relation triples
   (substituting subject/object) without ever generating negative `rdf:type`
   membership assertions. If the class head receives only positive supervision
   signals, BCE loss stays at max uncertainty (~0.693 for balanced, ~0.83 for
   skewed) and the head learns to always predict positive.  
   → Check `configs/ont_generator/exp1_constrained.yaml`: is there a
   `class_negative_ratio` or equivalent field? Is it zero or missing?

2. **Constrained corruption silently rejects all class membership candidates**  
   If the constrained strategy enforces type constraints so strictly that all
   candidate class negatives are rejected (e.g., a constraint that prevents
   assigning an individual to a class it is already a member of, recursively),
   the fallback may generate zero class negatives. The dataset would then contain
   only positive class memberships, collapsing the head.

3. **Dataloader bug for the constrained split**  
   A bug in the constrained dataset YAML or dataloader could cause class
   membership negatives to be silently dropped, mislabeled as positive, or
   excluded from batches, even if the generation phase produced them correctly.
   → Compare the raw `targets.csv` for the constrained split against random
   and proof-based: does it contain rows where the class membership label = 0?

4. **Class membership negatives present but masked by the loss weight**  
   If there is a `class_loss_weight` or `neg_weight` parameter set to 0 for
   the constrained run in the RRN config, the gradient from class negatives
   would be zeroed out even if the data is correct.

---

## 2. Sharp periodic metric drops in the random run

**Severity:** ⚠️ Moderate  
**Affected run:** `exp1_random_hpc`  
**Observed symptoms:**

- `val/triple_acc_pos` drops sharply from ~0.88 to ~0.61 at ~4.5k steps, then partially recovers
- `val/triple_acc_hops_1` drops similarly at ~2k and ~4.5k steps
- `val/triple_acc_type_inf_intermediate` drops at ~4.5k and ~5k steps
- Drops are sudden (one step) and partial recovery takes many steps

**Possible causes:**

1. **Learning rate scheduler with aggressive decay or restart (most likely)**  
   A cosine annealing with warm restarts, or a ReduceLROnPlateau with a large
   factor, could spike the effective step size at regular intervals, temporarily
   destabilizing the model. The periodicity (roughly every 2–2.5k steps) is
   consistent with a scheduler cycle.  
   → Check `cfg.hyperparams` for `scheduler`, `lr_scheduler`, or `step_size`.

2. **Gradient explosion event at specific steps**  
   Without gradient clipping, a bad batch can cause a gradient spike that
   overwrites learned weights. The partial recovery suggests the optimizer
   corrects over subsequent steps.  
   → Add `gradient_clip_val=1.0` to the `pl.Trainer` constructor and monitor
   `trainer/grad_norm` in W&B.

3. **Validation set composition changes at epoch boundaries**  
   If the validation dataloader reshuffles or regenerates graphs at certain
   epoch boundaries, the effective difficulty of the validation set could spike.
   The random model - being the most sensitive to negative type - would show this
   most clearly.

4. **Checkpoint callback causing model state interference**  
   If a `ModelCheckpoint` restores an earlier checkpoint mid-training (e.g., due
   to a misconfigured `restore_best_weights` equivalent), it could reset the
   model to an earlier state, causing a temporary performance drop.

---

## 3. proof_based absent from class-level metrics

**Severity:** ✅ Expected - not a bug, but must be documented  
**Affected metrics:** `val/class_recall`, `val/class_precision`, `val/class_fpr`,
`val/class_auc_roc`, `val/class_acc_neg`, `val/class_acc_hops_*`,
`val/class_acc_type_neg_*`

**Explanation:**  
Proof-based corruption operates at the base-fact level and propagates a substitution
up through the proof tree to produce a corrupted _root-level inferred triple_ as the
negative target. Class membership (`rdf:type`) assertions are not targeted by this
corruption strategy. If the proof-based split contains no class membership negatives,
all class-negative metrics are undefined (division by zero / empty set) and W&B
simply produces no curve.

**Why this matters for the paper:**  
You cannot report class-level metrics for the proof-based strategy in Table 1 or any
figure. This is a real limitation: your proof-based corruption is designed for relation
triples, not for class membership reasoning. If class membership prediction quality is
important for the paper's claims, you need either:

- A separate class membership corruption mode in the proof-based strategy, or
- Explicit acknowledgment that the experiment covers relation link prediction only.

---

## 4. proof_based absent from triple_acc_type_neg_inf_intermediate and neg_base_fact

**Severity:** ✅ Expected - not a bug  
**Explanation:**  
Same structural reason as above. Proof-based corruption produces negatives of type
`neg_inf_root` (corrupted inferred goals). There are no negatives of type
`neg_inf_intermediate` (corrupted intermediate derivation steps) or `neg_base_fact`
(corrupted base facts treated as targets) in the proof-based split.

---

## 5. Extreme noise in val/triple_acc_type_neg_base_fact

**Severity:** ⚠️ Minor / informational  
**Affected runs:** random, constrained (proof_based absent by design)  
**Observed symptoms:** Random oscillates between 0.25 and 0.75+ within single
evaluation steps. Constrained is flatter but still noisy.

**Cause:**  
The denominator for this metric (number of base-fact-type negatives in the validation
set) is very small. One or two misclassifications per validation pass causes large
swings. This is not a model instability; it is a statistical artifact of a small
validation subset.  
→ Do not report this metric in the paper without a confidence interval, or aggregate
it over multiple validation steps.

---

## 6. Test set distributional bias toward the proof-based strategy

**Severity:** ⚠️ Design-level - must be acknowledged in threats to validity  
**Description:**  
The frozen test set (`data/exp1/test_set`) is generated with proof-based corruption,
meaning its negatives are structurally similar to those in the proof-based _training_
set. The proof-based model therefore has a distributional advantage on the test set
that random and constrained models do not share.

**Implications:**

- The experiment cannot distinguish between "proof-based training teaches better
  logical reasoning" and "proof-based training simply matches the test distribution."
- Random and constrained models are evaluated out-of-distribution relative to their
  training negatives.

**Possible mitigations:**

1. Add a secondary test set using random or constrained corruption to check whether
   proof-based training _also_ generalizes to other negative types (it should, if the
   claim is about logical reasoning and not distribution matching).
2. Frame the test set explicitly as "the gold standard for hard-negative evaluation"
   and argue that any model that truly understands the ontology should score well on
   it regardless of training distribution - then acknowledge this as an assumption
   in Threats to Validity.
3. Report results on _both_ a strategy-matched validation set (each model on its own
   val negatives) and the shared proof-based test set, separating in-distribution
   from out-of-distribution performance.

---

## Summary table

| #   | Problem                                                        | Severity    | Action required                           |
| --- | -------------------------------------------------------------- | ----------- | ----------------------------------------- |
| 1   | Constrained class head collapse                                | 🚨 Critical | Fix dataset config; rerun constrained     |
| 2   | Sharp periodic drops in random                                 | ⚠️ Moderate | Investigate LR schedule + grad norm       |
| 3   | proof_based absent from class metrics                          | ✅ Expected | Document as limitation in paper           |
| 4   | proof_based absent from neg_intermediate/neg_base_fact metrics | ✅ Expected | Document; expected by design              |
| 5   | Noisy neg_base_fact metric                                     | ⚠️ Minor    | Do not report without CI; ignore in paper |
| 6   | Test set distributional bias                                   | ⚠️ Design   | Add to Threats to Validity section        |

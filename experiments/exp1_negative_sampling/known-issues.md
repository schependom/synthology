# Experiment 1: W&B Training Analysis & Issue Tracker

Based on a detailed review of the provided Weights & Biases plots, the experiment documentation (`exp1.md`), and the model implementation (`rrn_batched.py`), several critical issues are affecting the training runs. 

Here is the breakdown of the anomalies, their likely causes, and actionable steps to resolve them.

---

## Issue 1: Missing Primary Metric (`val/triple_pr_auc`) for the Proof-Based Run
**Description:** The `val/triple_pr_auc` plot completely lacks the pink line corresponding to `exp1_proof_based_hpc`. Given that your `exp1.md` document explicitly states that PR-AUC is the "primary discrimination metric for near-miss negatives" required for the paper, its absence is a showstopper.

**Possible Causes:**
* **NaN Values in Predictions:** If the RRN is outputting extreme logits that result in exact 0s or 1s after the sigmoid activation, the precision/recall calculations might be dividing by zero, resulting in `NaN`s. W&B will silently drop `NaN` metric points from line plots.
* **Metric Logging Configuration:** There might be a condition in the PyTorch Lightning module where PR-AUC is skipped or errors out exclusively for the `proof_based` batch structures.

**Points of Action:**
* Check the raw W&B tables (not the plots) to see if `val/triple_pr_auc` for the `proof_based` run is logged as `NaN` or `None`.
* Inspect the validation loop in your `LightningModule`. Add numerical stability checks (e.g., `torch.clamp`) before passing predictions to your AUROC/PRAUC metric functions.

---

## Issue 2: Catastrophic Mode Collapse in the Constrained Model
**Description:**
The `exp1_constrained_hpc` (blue line) experiences a severe, perfectly vertical drop across almost all class-related metrics at exactly ~2.5k global steps. For example, `val/class_acc_pos` plummets from 1.0 to a flat ~0.58, and `val/class_recall` drops identically. After this drop, the metrics flatline completely for the remainder of the run.

**Possible Causes:**
* **Exploding Gradients:** The model likely encountered a bad batch, causing gradients to explode and updating the weights to `NaN` or pushing them into a dead zone (e.g., dying ReLUs in the `ClassesMLP` or `RelationMLP`).
* **Mode Collapse:** The model learned a degenerate heuristic (like predicting all 0s or a constant probability) to minimize loss, effectively guessing the majority class.

**Points of Action:**
* Implement Gradient Clipping in your PyTorch Lightning Trainer (e.g., `trainer = Trainer(gradient_clip_val=1.0)`).
* Log gradient norms to W&B to verify if a spike occurs right before the 2.5k step mark.
* Lower the learning rate for the `constrained` run, or introduce a learning rate warmup phase.

---

## Issue 3: Contrary Performance and High Loss for the Proof-Based Model
**Description:**
According to `exp1.md`, the hypothesis is that the `proof_based` strategy should be the winner on hard negatives, yielding a *lower* False Positive Rate (FPR). However, the plots show that `val/triple_fpr` for `exp1_proof_based_hpc` (pink line) is the highest of all three runs (~0.7). Additionally, `val/total_loss` and `val/relation_loss` are significantly higher and plateauing early compared to the random and constrained runs.

**Possible Causes:**
* **Task Difficulty (Underfitting):** Proof-based hard negatives might be *too* difficult for the current network capacity or learning rate. If the model cannot find a generalizing pattern to separate positive multi-hop paths from structurally identical corrupted paths, it will default to a high false-positive rate.
* **Distribution Shift:** As noted in your Threats to Validity (`exp1.md`), the test set is generated using the `proof_based` strategy. While this gives a distributional advantage to the pink run, it also means the training set is saturated with incredibly difficult negatives, potentially halting convergence.

**Points of Action:**
* Review the model capacity (e.g., `embedding_size`, `num_hidden_layers` in MLPs). You may need to scale up the RRN to handle the stricter logical boundaries.
* Consider a curriculum learning approach: start training with random/constrained negatives and transition to proof-based negatives as the loss stabilizes.

---

## Issue 4: Premature Run Terminations
**Description:**
The three runs do not reach the same number of global steps. 
* `exp1_proof_based_hpc` (pink) terminates around 5.5k steps.
* `exp1_constrained_hpc` (blue) terminates around 6.5k steps.
* `exp1_random_hpc` (orange) continues past 8k steps.

**Possible Causes:**
* **HPC Walltime Limits:** The batched tensor operations might have different memory/compute footprints depending on the dataset structure, causing the `proof_based` and `constrained` runs to hit SLURM/HPC time limits before the `random` run.
* **Early Stopping:** If PyTorch Lightning's EarlyStopping callback is tied to a metric like `val_loss`, the higher losses in the pink and blue runs might be triggering a halt.

**Points of Action:**
* Check the standard output (`stdout`/`stderr`) of the HPC jobs to confirm if they were killed due to `TimeLimit`.
* If using Early Stopping, review the `patience` parameter and ensure it's not killing the harder datasets prematurely.
* Standardize the `max_steps` or `max_epochs` in the Lightning Trainer and ensure walltime requests accommodate the slowest dataset.

---

## Issue 5: Dashboard Clutter from Expected Metric Sparsity
**Description:**
Several plots look broken or incomplete, but this is actually expected behavior based on your dataset design. For instance, `val/triple_acc_type_neg_inf_root` only has a pink line, while pink is missing from `val/class_acc_type_neg_base_fact`. 

**Possible Causes:**
* As explicitly stated in `exp1.md`, proof-based generation produces no negative class targets and no `neg_base_fact` targets. Therefore, these metrics are physically impossible to calculate for the `proof_based` run. 

**Points of Action:**
* **No Code Fix Required:** This is mathematically correct. 
* **Dashboard Fix:** To make the W&B dashboard presentation-ready for the paper, group these specific "type-based" accuracy plots under a separate W&B section with a markdown note explaining the sparsity, or filter the legends so reviewers/collaborators aren't confused by the missing lines.
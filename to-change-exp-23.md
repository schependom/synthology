## What to Match and Why — RRN-Specific Analysis

The critical correction to the previous analysis: the RRN is not a neighbourhood-aggregation model. It iterates over **all triples** in a knowledge base N times, updating individual embeddings globally on every pass. This changes which confounds are real.

---

### The Actual RRN Confounds

**Confound 1: Number of training sample KBs**
More KBs = more gradient updates during training, period. Both arms must train on the same number of sample knowledge bases. This is a model-level concern, not addressed by `exp2-balance-datasets`, and must be verified explicitly.

**Confound 2: Total facts per sample KB (the fact-cap)**
The RRN performs N iterations over all facts in a KB. More facts = more embedding update steps per training iteration = more information propagation opportunities, regardless of their semantic depth. A Synthology KB with 5k facts and a UDM KB with 50k facts expose the model to fundamentally different amounts of update computation per sample. The `--fact-cap` in `exp2-balance-datasets` directly addresses this. ✅

**Confound 3: Total targets per sample KB (the target-cap)**
Targets are what the cross-entropy loss is computed over. More targets per KB = more gradient signal per sample. The `--target-cap` directly addresses this. ✅

**Confound 4: Positive/negative ratio within targets**
The cross-entropy loss treats positives and negatives asymmetrically under class imbalance — a model trained on 1:5 pos:neg learns a different decision boundary than one on 1:50, entirely independent of data quality. You must verify whether `exp2-balance-datasets` matches **positive and negative counts separately**, not just the total. If it only caps total targets, this is uncontrolled. Looking at the Exp3 balancing, it explicitly matches pos and neg per split — Exp2 should do the same.

Status: Implemented in Exp2 via split-level pos/neg alignment after generation, with strict mismatch checks in `exp2-report-data`. ✅

**Confound 5: Deep-hop distribution (now a reported diagnostic, not a matched criterion)**
Without the parity loop, you cannot match this. This demotes Exp2's claim from "under identical deep-hop distributions, Synthology wins" to "under matched budget (facts + targets), Synthology produces a richer deep-hop profile and better downstream metrics." You must report the hop distributions for both arms and frame the claim accordingly.

Status: Implemented in reporting via hop-bucket diagnostics (`d=1`, `d=2`, `d>=3`) and markdown/CSV outputs. ✅

---

### Experiment 2 — What to Match

| What | How | Status |
|---|---|---|
| Number of sample KBs (train/val/test) | Same count both arms | ✅ Enforced via post-generation split alignment |
| Total facts per KB | `--fact-cap` | ✅ `exp2-balance-datasets` |
| Total targets per KB | `--target-cap` | ✅ `exp2-balance-datasets` |
| Positive count per split | Downsample separately | ✅ Implemented in Exp2 balancing |
| Negative count per split | Downsample separately | ✅ Implemented in Exp2 balancing |
| Hyperparameters (N iterations, embedding size, lr) | Identical config | ✅ Verified for Exp2 configs |
| Test set | Frozen shared gold test | ✅ already frozen |
| Deep-hop distribution | **Report only** | ✅ Diagnostic table/report outputs |
| Predicate distribution | **Report only** | ✅ Existing reporter outputs |

### Experiment 3 — What to Match

Exp3 uses `exp3-balance-data` which explicitly matches positive and negative target counts per split separately — this is actually stricter than Exp2 on that dimension. However, there is no fact-cap equivalent for Exp3. The table:

| What | How | Status |
|---|---|---|
| Positive targets per split | Downsample Synthology | ✅ `exp3-balance-data` |
| Negative targets per split | Downsample Synthology | ✅ `exp3-balance-data` |
| Total facts | **Report ratio only** | ✅ Not matched — explicitly stated |
| Deep-hop distribution | **Report only** | ✅ Not matched — explicitly stated |
| Hyperparameters | Identical config | ✅ Verified (both inherit `exp3_owl2bench_hpc` base config) |
| Test set | Frozen shared gold test | ✅ |

The Exp3 claim is therefore necessarily weaker: **"under matched target volume, Synthology-trained models generalize better at scale"**, not a structural quality claim.

---

### What to Fix in the Paper

**In the Methodology section**, add a dedicated paragraph titled something like *"Dataset Matching Protocol"* that states precisely:

- For Exp2: both arms are generated under the same fact budget (`fact-cap = X`) and target budget (`target-cap = Y`), with positive and negative targets matched separately per split. Hyperparameters are fixed identically. The frozen test set is shared. Deep-hop distributions are reported as diagnostics but not matched.
- For Exp3: positive and negative target counts per split are matched via downsampling. Total facts and hop distributions are not matched due to scale constraints and are reported as diagnostics. The comparison is therefore a **matched-budget comparison**, not a structural parity comparison.

**In the Threats to Validity section**, you currently have a paragraph about the parity loop and non-convergence. Since you're not using the parity loop at all, replace it with: Exp2 enforces budget parity on facts and targets; deep-hop structure is not equalized, so observed gains could partly reflect a richer hop profile in Synthology rather than purely per-example quality. Exp3 enforces only target-count parity; fact volume and hop profile differences are uncontrolled and represent an acknowledged limitation of the scalability evaluation.

**In the Results section**, for both experiments add a small diagnostic table reporting: total facts, positive targets, negative targets, and hop-bucket distribution (d=1, d=2, d≥3) for both arms. This is what allows readers to evaluate the claim independently of which controls you could enforce.

Status: Added to paper as a dedicated diagnostics table template and integrated into reporter markdown/CSV outputs. ✅
"""
Subsample neg_inf_root rows in the ASP neutral test set to match the exp2
positive rate (~17.3%), making evaluation protocols comparable.

The ASP generator enumerates ALL non-derived triples as negatives, giving a
1.85% positive rate vs. the 17.3% in the exp2 frozen test (where Synthology
produces curated proof-based negatives).  A model calibrated at 17.3% performs
near-randomly at 1.85% — hence the ~1.5% PR-AUC observed in the first run.

This script:
  1. Reads data/asp/family_tree_neutral/test/targets.csv
  2. Keeps ALL inf_root and base_fact rows unchanged
  3. Subsamples neg_inf_root rows per sample_id to reach ~17.3% positive rate
  4. Writes the result to data/asp/family_tree_neutral_balanced/test/targets.csv
     (facts.csv is copied verbatim — it is unaffected by negative sampling)

Run from the repo root:
  uv run python scripts/subsample_asp_negatives.py
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

RANDOM_SEED   = 42
TARGET_POS_RATE = 0.173   # match exp2 frozen-test positive rate

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "data" / "asp" / "family_tree_neutral" / "test"
OUT_DIR   = REPO_ROOT / "data" / "asp" / "family_tree_neutral_balanced" / "test"


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    src_targets = SRC_DIR / "targets.csv"
    src_facts   = SRC_DIR / "facts.csv"

    if not src_targets.exists():
        print(f"ERROR: {src_targets} not found.  Run adapt_asp_for_exp2 first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: count positives per sample_id ─────────────────────────────
    pos_count: dict[str, int] = defaultdict(int)
    neg_rows:  dict[str, list] = defaultdict(list)
    other_rows: list = []

    fieldnames = None
    total_pos = total_neg = total_other = 0

    print("Pass 1: reading targets.csv …")
    with open(src_targets, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            t = row["type"]
            if t == "inf_root":
                pos_count[row["sample_id"]] += 1
                other_rows.append(row)
                total_pos += 1
            elif t == "neg_inf_root":
                neg_rows[row["sample_id"]].append(row)
                total_neg += 1
            else:
                other_rows.append(row)
                total_other += 1

    print(f"  inf_root:     {total_pos:,}")
    print(f"  neg_inf_root: {total_neg:,}")
    print(f"  other:        {total_other:,}")
    print(f"  positive rate (before): {total_pos/(total_pos+total_neg)*100:.2f}%")

    # ── Per-sample subsample ───────────────────────────────────────────────
    # pos / (pos + neg_keep) = TARGET_POS_RATE
    # => neg_keep = pos * (1 - TARGET_POS_RATE) / TARGET_POS_RATE
    kept_neg = []
    for sid, negs in neg_rows.items():
        n_pos = pos_count.get(sid, 0)
        n_keep = int(round(n_pos * (1 - TARGET_POS_RATE) / TARGET_POS_RATE))
        n_keep = min(n_keep, len(negs))
        kept_neg.extend(rng.sample(negs, n_keep))

    kept_neg_count = len(kept_neg)
    effective_rate = total_pos / (total_pos + kept_neg_count) * 100
    print(f"\n  Subsampled neg_inf_root: {kept_neg_count:,} (target rate ≈ {TARGET_POS_RATE*100:.1f}%)")
    print(f"  Effective positive rate: {effective_rate:.2f}%")

    # ── Write output ───────────────────────────────────────────────────────
    print(f"\nWriting to {OUT_DIR / 'targets.csv'} …")
    all_out = other_rows + kept_neg
    # Sort by sample_id then original row order
    all_out.sort(key=lambda r: (r["sample_id"], r.get("subject", ""), r["predicate"]))

    with open(OUT_DIR / "targets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_out)

    total_out = len(all_out)
    print(f"  Written {total_out:,} rows")

    # Copy facts.csv verbatim
    shutil.copy2(src_facts, OUT_DIR / "facts.csv")
    print(f"  Copied facts.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Adapt ASP-generated family-tree test data to the exp2 schema.

The ASP generator (Hohenecker) and Synthology both use the same family.ttl
ontology but differ in how classes are named:
  ASP:        female / male
  exp2 / Synthology:  Woman / Man / Person

All predicates that appear in the ASP data but NOT in the exp2 training schema
are dropped.  The resulting split is written to:
  data/asp/family_tree_neutral/test/facts.csv
  data/asp/family_tree_neutral/test/targets.csv

Only the ASP 'test' split is used (20% of generated samples = ~140 KGs),
giving a dataset comparable in size to the exp2 frozen_test (500 KGs).

Run from the repo root:
  uv run python scripts/adapt_asp_for_exp2.py
"""

import csv
import shutil
from pathlib import Path
from typing import Optional

# ── Schema expected by the exp2-trained models ────────────────────────────────
EXP2_RELATIONS = {
    "auntOf", "auntUncleOf", "boyCousinOf", "brotherOf", "childOf",
    "cousinOf", "daughterOf", "fatherOf", "firstCousinOnceRemovedOf",
    "girlCousinOf", "grandchildOf", "grandfatherOf", "grandmotherOf",
    "grandparentOf", "greatAuntOf", "greatAuntUncleOf", "greatGrandchildOf",
    "greatGrandfatherOf", "greatGrandmotherOf", "greatGrandparentOf",
    "greatUncleOf", "motherOf", "nieceNephewOf", "parentOf",
    "secondAuntUncleOf", "secondCousinOf", "siblingOf", "sisterOf",
    "sonOf", "uncleOf",
}
EXP2_CLASSES = {"Man", "Woman", "Person"}

# ASP → exp2 class mapping
CLASS_MAP = {
    "female": "Woman",
    "male":   "Man",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
ASP_TEST_DIR = REPO_ROOT / "data" / "asp" / "family_tree" / "test"
OUT_DIR      = REPO_ROOT / "data" / "asp" / "family_tree_neutral" / "test"


def _map_class(name: str) -> str:
    return CLASS_MAP.get(name, name)


def _adapt_facts_row(row: dict) -> Optional[dict]:
    p = row["predicate"]
    if p == "rdf:type":
        mapped = _map_class(row["object"])
        if mapped not in EXP2_CLASSES:
            return None
        return {**row, "object": mapped}
    if p in EXP2_RELATIONS:
        return row
    return None


def _adapt_targets_row(row: dict) -> Optional[dict]:
    p = row["predicate"]
    if p == "rdf:type":
        mapped = _map_class(row["object"])
        if mapped not in EXP2_CLASSES:
            return None
        return {**row, "object": mapped}
    if p in EXP2_RELATIONS:
        return row
    return None


def _inject_person_type_streaming(facts_path: Path, targets_path: Path) -> None:
    """
    Every individual in the family tree is also a Person (in the exp2 ontology).
    Collect all individuals from facts.csv, then append rdf:type Person targets
    for any individual that doesn't already have one — streaming to avoid OOM.
    """
    # Pass 1: collect all individuals from facts.csv
    all_individuals: set = set()
    with open(facts_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_individuals.add((row["sample_id"], row["subject"]))
            if row["predicate"] != "rdf:type":
                all_individuals.add((row["sample_id"], row["object"]))

    # Pass 2: collect individuals that already have rdf:type Person
    already_person: set = set()
    with open(targets_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["predicate"] == "rdf:type" and row["object"] == "Person":
                already_person.add((row["sample_id"], row["subject"]))

    to_inject = sorted(all_individuals - already_person)
    if not to_inject:
        print("  No rdf:type Person rows to inject.")
        return

    # Pass 3: append new rows to targets.csv
    with open(targets_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for (sid, subj) in to_inject:
            writer.writerow({
                "sample_id":         sid,
                "subject":           subj,
                "predicate":         "rdf:type",
                "object":            "Person",
                "label":             "1",
                "truth_value":       "True",
                "type":              "inf_root",
                "hops":              "1",
                "corruption_method": "",
            })

    print(f"  Injected {len(to_inject)} rdf:type Person rows.")


def main() -> None:
    if not ASP_TEST_DIR.exists():
        print(f"ERROR: ASP test split not found at {ASP_TEST_DIR}")
        print("Run 'invoke gen-ft-asp' first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    facts_fieldnames   = ["sample_id", "subject", "predicate", "object"]
    targets_fieldnames = [
        "sample_id", "subject", "predicate", "object",
        "label", "truth_value", "type", "hops", "corruption_method",
    ]

    print(f"Adapting ASP test split → exp2 schema")
    print(f"  Source: {ASP_TEST_DIR}")
    print(f"  Output: {OUT_DIR}")

    for fname, adapt_fn, fieldnames in [
        ("facts.csv",   _adapt_facts_row,   facts_fieldnames),
        ("targets.csv", _adapt_targets_row, targets_fieldnames),
    ]:
        src = ASP_TEST_DIR / fname
        dst = OUT_DIR / fname
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping.")
            continue

        kept = total = 0
        with open(src, newline="", encoding="utf-8") as fin, \
             open(dst, "w", newline="", encoding="utf-8") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                adapted = adapt_fn(row)
                if adapted is not None:
                    writer.writerow(adapted)
                    kept += 1

        print(f"  {fname}: {kept}/{total} rows kept")

    # Inject rdf:type Person for all individuals
    print("\nInjecting rdf:type Person for all individuals...")
    _inject_person_type_streaming(OUT_DIR / "facts.csv", OUT_DIR / "targets.csv")

    # Schema validation
    predicates: set = set()
    classes: set    = set()
    with open(OUT_DIR / "targets.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["predicate"] == "rdf:type":
                classes.add(row["object"])
            else:
                predicates.add(row["predicate"])

    unknown_rels = predicates - EXP2_RELATIONS
    unknown_cls  = classes - EXP2_CLASSES
    if unknown_rels:
        print(f"  WARNING: Unexpected relations: {unknown_rels}")
    if unknown_cls:
        print(f"  WARNING: Unexpected classes: {unknown_cls}")
    if not unknown_rels and not unknown_cls:
        print("  Schema check OK — all predicates and classes match exp2.")

    print(f"\nDone. Neutral ASP test set written to: {OUT_DIR}")
    print(f"  Relations: {sorted(predicates)}")
    print(f"  Classes:   {sorted(classes)}")


if __name__ == "__main__":
    main()

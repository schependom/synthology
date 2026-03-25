import csv
import random
from pathlib import Path


def write_split(
    split_dir: Path,
    split_samples: list[tuple[int, list[tuple]]],
    inferred_by_sample: dict[int, list[tuple[str, str, str, int]]],
    target_ratio: float,
    mask_base_facts: bool,
    negatives_per_positive: int,
) -> tuple[int, int]:
    facts_rows = []
    targets_rows = []
    inferred_positive_count = 0

    for sid, sample_base_clean in split_samples:
        all_individuals = set()
        all_classes = set()

        for s, p, o in sample_base_clean:
            all_individuals.add(s)
            if p == "rdf:type":
                all_classes.add(o)
            else:
                all_individuals.add(o)

        positive_targets = []

        for s, p, o in sample_base_clean:
            is_target_only = mask_base_facts and (random.random() < target_ratio)
            fact_type = "inferred" if is_target_only else "base_fact"

            fact = {"sample_id": str(sid), "subject": s, "predicate": p, "object": o}
            target = {
                "sample_id": str(sid),
                "subject": s,
                "predicate": p,
                "object": o,
                "label": 1,
                "truth_value": "True",
                "type": fact_type,
                "hops": 0,
                "corruption_method": "",
            }

            if fact_type == "base_fact":
                facts_rows.append(fact)
                targets_rows.append(target)
            else:
                positive_targets.append(target)
                targets_rows.append(target)

        for s, p, o, hops in inferred_by_sample.get(sid, []):
            inferred_target = {
                "sample_id": str(sid),
                "subject": s,
                "predicate": p,
                "object": o,
                "label": 1,
                "truth_value": "True",
                "type": "inferred",
                "hops": hops,
                "corruption_method": "",
            }
            positive_targets.append(inferred_target)
            targets_rows.append(inferred_target)
            inferred_positive_count += 1

            all_individuals.add(s)
            if p == "rdf:type":
                all_classes.add(o)
            else:
                all_individuals.add(o)

        all_individuals = list(all_individuals)
        all_classes = list(all_classes)

        for pos_tgt in positive_targets:
            for _ in range(max(0, negatives_per_positive)):
                corrupt_object = random.choice([True, False])
                if pos_tgt["predicate"] == "rdf:type" and corrupt_object:
                    choices = all_classes
                else:
                    choices = all_individuals

                if len(choices) < 2:
                    continue

                corrupted_entity = random.choice(choices)
                while corrupted_entity == (pos_tgt["object"] if corrupt_object else pos_tgt["subject"]):
                    corrupted_entity = random.choice(choices)

                neg_target = {
                    "sample_id": pos_tgt["sample_id"],
                    "subject": pos_tgt["subject"] if not corrupt_object else corrupted_entity,
                    "predicate": pos_tgt["predicate"],
                    "object": corrupted_entity if corrupt_object else pos_tgt["object"],
                    "label": 0,
                    "truth_value": "False",
                    "type": "inferred",
                    "hops": pos_tgt.get("hops", 0),
                    "corruption_method": "random",
                }
                targets_rows.append(neg_target)

    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "facts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object"])
        writer.writeheader()
        writer.writerows(facts_rows)

    with open(split_dir / "targets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "subject",
                "predicate",
                "object",
                "label",
                "truth_value",
                "type",
                "hops",
                "corruption_method",
            ],
        )
        writer.writeheader()
        writer.writerows(targets_rows)

    return len(facts_rows), inferred_positive_count

#!/usr/bin/env python3
"""
list_multihop_inferred_facts.py

Prints a few multi-hop inferred facts for each hop depth from UDM baseline targets.csv files (exp2 and exp3).

Usage:
    python list_multihop_inferred_facts.py --exp2 path/to/exp2/targets.csv --exp3 path/to/exp3/targets.csv [--per-hop 3]

"""
import csv
import argparse
from collections import defaultdict
import random


def collect_multihop_facts(targets_path, min_hop=2, per_hop=3):
    """Collects a sample of inferred facts for each hop depth >= min_hop."""
    by_hop = defaultdict(list)
    with open(targets_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hops = int(row.get("hops", 0))
            except Exception:
                continue
            row_type = row.get("type", "").lower()
            if hops >= min_hop and row_type.startswith("inf"):
                by_hop[hops].append(row)
    # Sample per hop
    sampled = {}
    for hop, facts in by_hop.items():
        sampled[hop] = random.sample(facts, min(per_hop, len(facts)))
    return sampled


def print_samples(label, samples):
    print(f"\n=== {label} ===")
    for hop in sorted(samples):
        print(f"\n--- Hop depth {hop} ---")
        for row in samples[hop]:
            print(f"({row['subject']}, {row['predicate']}, {row['object']}) | type: {row['type']} | hops: {row['hops']}")


def main():
    parser = argparse.ArgumentParser(description="List multi-hop inferred facts from UDM baselines.")
    parser.add_argument("--exp2", required=True, help="Path to exp2 UDM baseline targets.csv")
    parser.add_argument("--exp3", required=True, help="Path to exp3 UDM baseline targets.csv")
    parser.add_argument("--per-hop", type=int, default=3, help="Number of samples per hop depth")
    args = parser.parse_args()

    exp2_samples = collect_multihop_facts(args.exp2, per_hop=args.per_hop)
    exp3_samples = collect_multihop_facts(args.exp3, per_hop=args.per_hop)

    print_samples("Exp2 UDM Baseline", exp2_samples)
    print_samples("Exp3 UDM Baseline", exp3_samples)

if __name__ == "__main__":
    main()

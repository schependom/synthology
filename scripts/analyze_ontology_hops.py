#!/usr/bin/env python3
"""
analyze_ontology_hops.py

Computes the theoretical hop-depth distribution achievable from an OWL 2 RL
ontology by analysing the rule dependency graph produced by OntologyParser.

For each rule the script determines:
  - min_depth  : minimum inference-chain length needed to fire the rule
                 starting from base ABox facts
  - max_depth  : maximum achievable chain length (equals max_chain_depth for
                 rules that depend on transitive/recursive predicates)

The distribution is then estimated by counting rule-chain configurations at
every depth and fitting a geometric decay for transitive extensions.

Outputs (written to --out directory):
  theory_hop_distribution.csv   -- main table, readable by MATLAB
  rule_depths.csv               -- per-rule min/max depth + rule type
  theory_hop_distribution.json  -- full metadata (for programmatic access)
  theory_hop_distribution.png   -- matplotlib bar chart

Usage (from repo root):
  uv run --package ont_generator python scripts/analyze_ontology_hops.py \\
      --ontology ontologies/UNIV-BENCH-OWL2RL.owl \\
      --max-depth 10 \\
      --out reports/ontology_hop_analysis/owl2bench

  uv run --package ont_generator python scripts/analyze_ontology_hops.py \\
      --ontology ontologies/family.ttl \\
      --max-depth 12 \\
      --out reports/ontology_hop_analysis/family
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

# Make sure the synthology packages are importable when run directly.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "apps" / "ont_generator" / "src"))
sys.path.insert(0, str(_repo_root / "src"))

from rdflib.namespace import RDF
from synthology.data_structures import Atom, Class, ExecutableRule, Relation, Var
from ont_generator.parse import OntologyParser


# ---------------------------------------------------------------------------
# Atom signature helpers
# ---------------------------------------------------------------------------

def atom_sig(atom: Atom) -> Tuple:
    """
    Return a canonical (kind, name) key for an atom, used to match
    rule heads against rule bodies independent of variable names.

    Type atoms  → ('type',     class_name)
    Relation atoms → ('relation', predicate_name)
    """
    if atom.predicate == RDF.type:
        obj = atom.object
        name = obj.name if hasattr(obj, "name") else str(obj)
        return ("type", name)
    else:
        pred = atom.predicate
        name = pred.name if hasattr(pred, "name") else str(pred)
        return ("relation", name)


# ---------------------------------------------------------------------------
# Rule graph construction
# ---------------------------------------------------------------------------

def build_graph(rules: List[ExecutableRule]):
    """
    Returns:
      head_sig_to_rules  : sig  -> list[rule]   rules that produce this sig
      rule_body_sigs     : rule -> frozenset     sigs needed by this rule
      inferrable_sigs    : set of all sigs that any rule can produce
      self_referential   : set of rules whose head sig appears in their body
    """
    head_sig_to_rules: Dict[tuple, List[ExecutableRule]] = defaultdict(list)
    rule_body_sigs: Dict[ExecutableRule, FrozenSet] = {}

    for rule in rules:
        if rule.conclusion is None:
            continue
        hsig = atom_sig(rule.conclusion)
        head_sig_to_rules[hsig].append(rule)
        rule_body_sigs[rule] = frozenset(atom_sig(p) for p in rule.premises)

    inferrable_sigs: Set[tuple] = set(head_sig_to_rules.keys())

    self_referential: Set[ExecutableRule] = {
        r for r in rules
        if r.conclusion is not None
        and atom_sig(r.conclusion) in rule_body_sigs.get(r, frozenset())
    }

    return head_sig_to_rules, rule_body_sigs, inferrable_sigs, self_referential


# ---------------------------------------------------------------------------
# Minimum depth computation  (fixpoint over the rule graph)
# ---------------------------------------------------------------------------

def compute_min_depths(
    rules: List[ExecutableRule],
    head_sig_to_rules: Dict,
    rule_body_sigs: Dict,
    inferrable_sigs: Set,
    self_referential: Set,
    max_depth: int,
) -> Dict[ExecutableRule, int]:
    """
    Return {rule: min_depth} via Bellman-Ford–style fixpoint.

    For self-referential (transitive / symmetric) rules the recursive premise
    is treated as satisfiable from base facts (the base case), so min_depth = 1.
    """
    INF = max_depth + 1
    min_d: Dict[ExecutableRule, int] = {}

    def _min_d_for_rule(rule: ExecutableRule) -> int:
        """Compute the minimum depth to fire `rule` given current min_d."""
        best_premise_depths = []
        for bsig in rule_body_sigs.get(rule, frozenset()):
            if bsig not in inferrable_sigs:
                # Pure base-fact premise — costs 0 inference steps.
                best_premise_depths.append(0)
            else:
                producers = head_sig_to_rules[bsig]
                producer_min = INF
                for producer in producers:
                    if producer is rule:
                        # Self-loop: base case is a direct ABox fact → cost 0.
                        producer_min = 0
                    elif producer in min_d:
                        producer_min = min(producer_min, min_d[producer])
                if producer_min < INF:
                    best_premise_depths.append(producer_min)
                # If no producer has a known depth yet, skip this premise for now;
                # the fixpoint will revisit it.

        if not best_premise_depths:
            return INF
        # Depth = max over all required premises + 1 own step.
        return max(best_premise_depths) + 1

    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.conclusion is None:
                continue
            new_d = _min_d_for_rule(rule)
            if new_d < min_d.get(rule, INF):
                min_d[rule] = new_d
                changed = True

    # Fill any unreachable rules with max_depth (defensive).
    for rule in rules:
        if rule.conclusion is not None and rule not in min_d:
            min_d[rule] = max_depth

    return min_d


# ---------------------------------------------------------------------------
# Maximum depth  (propagate unboundedness from transitive/recursive rules)
# ---------------------------------------------------------------------------

def compute_max_depths(
    rules: List[ExecutableRule],
    head_sig_to_rules: Dict,
    rule_body_sigs: Dict,
    inferrable_sigs: Set,
    self_referential: Set,
    min_depths: Dict,
    max_depth: int,
) -> Dict[ExecutableRule, int]:
    """
    A rule has unbounded depth potential if its head predicate is produced by a
    recursive rule, OR if any of its premises can be produced at unbounded depth.
    Propagate this forward through the rule dependency graph.
    """
    # Seed: sigs produced by self-referential rules are unbounded.
    unbounded_sigs: Set[tuple] = set()
    for rule in self_referential:
        if rule.conclusion is not None:
            unbounded_sigs.add(atom_sig(rule.conclusion))

    # Propagate: a rule whose body contains an unbounded sig produces an unbounded sig.
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.conclusion is None:
                continue
            hsig = atom_sig(rule.conclusion)
            if hsig in unbounded_sigs:
                continue
            if any(bsig in unbounded_sigs for bsig in rule_body_sigs.get(rule, frozenset())):
                unbounded_sigs.add(hsig)
                changed = True

    max_d: Dict[ExecutableRule, int] = {}
    for rule in rules:
        if rule.conclusion is None:
            continue
        hsig = atom_sig(rule.conclusion)
        if hsig in unbounded_sigs or rule in self_referential:
            max_d[rule] = max_depth
        else:
            max_d[rule] = min_depths.get(rule, 1)

    return max_d


# ---------------------------------------------------------------------------
# Distribution estimation
# ---------------------------------------------------------------------------

def estimate_distribution(
    rules: List[ExecutableRule],
    min_depths: Dict,
    max_depths: Dict,
    max_depth: int,
    transitive_decay: float = 0.65,
) -> Dict[int, float]:
    """
    Estimate the relative frequency of inference chains at each depth.

    For a rule with min_depth m and max_depth M:
      - Contributes weight 1.0 at depth m.
      - Each additional depth d > m contributes decay^(d-m) (geometric decay).
        This models the decreasing probability of finding longer transitive chains.

    Rules with max_depth == min_depth (no transitive extension) contribute
    only at their exact depth.

    Returns {depth: float_weight} normalised to sum to 1.
    """
    weights: Dict[int, float] = defaultdict(float)

    for rule in rules:
        if rule.conclusion is None:
            continue
        m = min_depths.get(rule, 1)
        M = max_depths.get(rule, m)
        for d in range(m, min(M, max_depth) + 1):
            w = transitive_decay ** (d - m)
            weights[d] += w

    total = sum(weights.values())
    if total == 0:
        return {}
    return {d: w / total for d, w in sorted(weights.items())}


# ---------------------------------------------------------------------------
# Rule type classification  (for the per-rule CSV)
# ---------------------------------------------------------------------------

def classify_rule(rule: ExecutableRule, self_referential: Set) -> str:
    name = rule.name.lower()
    if rule in self_referential:
        return "transitive/recursive"
    if "chain" in name:
        return "property_chain"
    if "inverseof" in name or "inverse" in name:
        return "inverse"
    if "subclassof" in name or "subclass" in name:
        return "subClassOf"
    if "subpropertyof" in name or "subproperty" in name:
        return "subPropertyOf"
    if "domain" in name:
        return "domain"
    if "range" in name:
        return "range"
    if "guarded" in name:
        return "guarded_derivation"
    if "symmetric" in name:
        return "symmetric"
    return "other"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_theory_csv(dist: Dict[int, float], path: Path) -> None:
    """Write the main distribution CSV for MATLAB ingestion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "relative_frequency"])
        for d in sorted(dist):
            w.writerow([d, f"{dist[d]:.6f}"])
    print(f"Wrote {path}")


def write_rule_depths_csv(
    rules: List[ExecutableRule],
    min_depths: Dict,
    max_depths: Dict,
    self_referential: Set,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule_name", "rule_type", "min_depth", "max_depth"])
        for rule in sorted(rules, key=lambda r: (min_depths.get(r, 99), r.name)):
            if rule.conclusion is None:
                continue
            w.writerow([
                rule.name,
                classify_rule(rule, self_referential),
                min_depths.get(rule, "?"),
                max_depths.get(rule, "?"),
            ])
    print(f"Wrote {path}")


def write_json(
    ontology_path: str,
    max_depth: int,
    dist: Dict[int, float],
    min_depths: Dict,
    max_depths: Dict,
    self_referential: Set,
    rules: List[ExecutableRule],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ontology": ontology_path,
        "max_depth_param": max_depth,
        "n_rules": sum(1 for r in rules if r.conclusion is not None),
        "n_recursive_rules": len(self_referential),
        "theoretical_distribution": {str(d): v for d, v in dist.items()},
        "overall_max_achievable_depth": max(
            (max_depths[r] for r in rules if r.conclusion is not None), default=0
        ),
        "rule_summary": {
            rule.name: {
                "type": classify_rule(rule, self_referential),
                "min_depth": min_depths.get(rule, None),
                "max_depth": max_depths.get(rule, None),
            }
            for rule in rules if rule.conclusion is not None
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {path}")


def write_figure(
    dist: Dict[int, float],
    ontology_name: str,
    max_depth: int,
    path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping PNG figure.")
        return

    depths = sorted(dist.keys())
    freqs = [dist[d] for d in depths]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(depths, freqs, color="#003D73", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Proof depth (hops)", fontsize=13)
    ax.set_ylabel("Relative frequency (normalised)", fontsize=13)
    ax.set_title(
        f"Theoretical hop-depth distribution — {ontology_name}\n"
        f"(rule-graph analysis, max depth = {max_depth})",
        fontsize=13,
    )
    ax.set_xticks(depths)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(freqs) * 1.15)

    for d, f in zip(depths, freqs):
        ax.text(d, f + 0.002, f"{f:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Theoretical hop-depth distribution for an OWL 2 RL ontology."
    )
    parser.add_argument(
        "--ontology", required=True,
        help="Path to the ontology file (.ttl, .owl, …)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=15,
        help="Cap for transitive chain depth (default: 15). Match the "
             "global_max_depth used in the generator config.",
    )
    parser.add_argument(
        "--transitive-decay", type=float, default=0.65,
        help="Geometric decay factor for each additional transitive step "
             "(default: 0.65 → each extra hop is 35%% less likely).",
    )
    parser.add_argument(
        "--out", default="reports/ontology_hop_analysis",
        help="Output directory (default: reports/ontology_hop_analysis).",
    )
    args = parser.parse_args()

    ont_path = Path(args.ontology)
    if not ont_path.is_absolute():
        ont_path = _repo_root / ont_path
    if not ont_path.exists():
        print(f"[ERROR] Ontology not found: {ont_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ont_name = ont_path.stem

    print(f"Parsing ontology: {ont_path}")
    ont = OntologyParser(str(ont_path))
    rules = ont.rules
    print(f"  {len(rules)} executable rules")

    print("Building rule dependency graph …")
    head_sig_to_rules, rule_body_sigs, inferrable_sigs, self_referential = build_graph(rules)
    print(f"  {len(inferrable_sigs)} inferrable predicates")
    print(f"  {len(self_referential)} recursive/transitive rules")

    print("Computing minimum derivation depths …")
    min_depths = compute_min_depths(
        rules, head_sig_to_rules, rule_body_sigs,
        inferrable_sigs, self_referential, args.max_depth,
    )

    print("Computing maximum derivation depths …")
    max_depths = compute_max_depths(
        rules, head_sig_to_rules, rule_body_sigs,
        inferrable_sigs, self_referential, min_depths, args.max_depth,
    )

    print("Estimating theoretical distribution …")
    dist = estimate_distribution(
        rules, min_depths, max_depths, args.max_depth, args.transitive_decay,
    )

    # Print summary table to stdout
    print(f"\n{'Depth':>6}  {'Rel. freq':>10}  {'Cumulative':>10}")
    print("─" * 32)
    cum = 0.0
    for d in sorted(dist):
        cum += dist[d]
        print(f"  d={d:>2}   {dist[d]:>9.3f}   {cum:>9.3f}")
    print()

    overall_max = max(
        (max_depths[r] for r in rules if r.conclusion is not None), default=0
    )
    print(f"Maximum achievable proof depth: {overall_max}")

    # Write outputs
    write_theory_csv(dist, out_dir / "theory_hop_distribution.csv")
    write_rule_depths_csv(rules, min_depths, max_depths, self_referential,
                          out_dir / "rule_depths.csv")
    write_json(str(ont_path), args.max_depth, dist, min_depths, max_depths,
               self_referential, rules, out_dir / "theory_hop_distribution.json")
    write_figure(dist, ont_name, args.max_depth, out_dir / "theory_hop_distribution.png")

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()

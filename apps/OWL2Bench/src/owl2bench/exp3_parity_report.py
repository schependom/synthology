"""Report utilities for Exp 3 parity attempts against a Synthology reference dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

RDF_TYPE = "rdf:type"


def _to_int(value: str | int | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_target_stats(targets_csv: Path, min_deep_hops: int) -> dict[str, Any]:
    hop_hist: Counter[int] = Counter()
    deep_count = 0
    positives = 0
    negatives = 0
    inferred_positives = 0

    with open(targets_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row.get("label", "1")).strip()
            if label != "1":
                negatives += 1
                continue

            positives += 1
            hops = _to_int(row.get("hops"), default=0)
            hop_hist[hops] += 1

            row_type = str(row.get("type", "")).strip().lower()
            is_inferred = row_type.startswith("inf") or row_type == "inf_root"
            if is_inferred and hops >= min_deep_hops:
                deep_count += 1
            if is_inferred:
                inferred_positives += 1

    pos_neg_ratio = (positives / negatives) if negatives > 0 else float(positives > 0)
    inferred_positive_share = (inferred_positives / positives) if positives > 0 else 0.0

    return {
        "positives": positives,
        "negatives": negatives,
        "deep_count": deep_count,
        "pos_neg_ratio": pos_neg_ratio,
        "inferred_positive_share": inferred_positive_share,
        "hop_histogram": {str(k): v for k, v in sorted(hop_hist.items())},
    }


def extract_facts_stats(facts_csv: Path) -> dict[str, Any]:
    nodes: set[str] = set()
    relation_edges = 0
    all_facts = 0

    with open(facts_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            all_facts += 1
            subject = str(row.get("subject", "")).strip()
            predicate = str(row.get("predicate", "")).strip()
            obj = str(row.get("object", "")).strip()

            if subject:
                nodes.add(subject)
            if predicate != RDF_TYPE and obj:
                nodes.add(obj)
                relation_edges += 1

    node_count = len(nodes)
    max_directed_edges = node_count * (node_count - 1)
    edge_density = (relation_edges / max_directed_edges) if max_directed_edges > 0 else 0.0

    return {
        "node_count": node_count,
        "relation_edge_count": relation_edges,
        "fact_count": all_facts,
        "edge_density": edge_density,
    }


def extract_dataset_stats(targets_csv: Path, facts_csv: Path, min_deep_hops: int) -> dict[str, Any]:
    return {
        **extract_target_stats(targets_csv=targets_csv, min_deep_hops=min_deep_hops),
        **extract_facts_stats(facts_csv=facts_csv),
        "targets_csv": str(targets_csv),
        "facts_csv": str(facts_csv),
        "min_deep_hops": min_deep_hops,
    }


def load_synth_runtime_seconds(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    runtime = payload.get("runtime_seconds", {})
    total = runtime.get("total")
    try:
        return float(total)
    except (TypeError, ValueError):
        return None


def _discover_attempt_targets(attempts_root: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for attempt_dir in sorted(attempts_root.glob("attempt_*")):
        candidates = sorted(attempt_dir.glob("**/train/targets.csv"))
        if candidates:
            result.append((attempt_dir.name, candidates[0]))
    return result


def _attempt_facts_from_targets(targets_csv: Path) -> Path:
    return targets_csv.parent / "facts.csv"


def build_parity_report(
    synth_targets: Path,
    synth_facts: Path,
    attempts_root: Path,
    min_deep_hops: int,
) -> dict[str, Any]:
    synth_stats = extract_dataset_stats(synth_targets, synth_facts, min_deep_hops=min_deep_hops)
    attempts = []

    for attempt_name, targets_csv in _discover_attempt_targets(attempts_root):
        facts_csv = _attempt_facts_from_targets(targets_csv)
        if not facts_csv.exists():
            logger.warning("Skipping {} because facts.csv is missing", attempt_name)
            continue
        stats = extract_dataset_stats(targets_csv, facts_csv, min_deep_hops=min_deep_hops)
        stats["attempt"] = attempt_name
        attempts.append(stats)

    return {
        "synthology": {
            "targets_csv": str(synth_targets),
            "facts_csv": str(synth_facts),
            "k_deep": synth_stats["deep_count"],
            "hop_histogram": synth_stats["hop_histogram"],
            "positives": synth_stats["positives"],
            "negatives": synth_stats["negatives"],
            "node_count": synth_stats["node_count"],
            "relation_edge_count": synth_stats["relation_edge_count"],
            "fact_count": synth_stats["fact_count"],
            "edge_density": synth_stats["edge_density"],
            "pos_neg_ratio": synth_stats["pos_neg_ratio"],
            "inferred_positive_share": synth_stats["inferred_positive_share"],
            "min_deep_hops": min_deep_hops,
        },
        "attempts": attempts,
    }


def _write_attempts_csv(report: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "attempt",
                "targets_csv",
                "facts_csv",
                "positives",
                "negatives",
                "deep_count",
                "node_count",
                "relation_edge_count",
                "edge_density",
                "pos_neg_ratio",
                "inferred_positive_share",
                "hop_histogram_json",
            ],
        )
        writer.writeheader()
        for item in report.get("attempts", []):
            writer.writerow(
                {
                    "attempt": item.get("attempt"),
                    "targets_csv": item.get("targets_csv"),
                    "facts_csv": item.get("facts_csv"),
                    "positives": item.get("positives"),
                    "negatives": item.get("negatives"),
                    "deep_count": item.get("deep_count"),
                    "node_count": item.get("node_count"),
                    "relation_edge_count": item.get("relation_edge_count"),
                    "edge_density": item.get("edge_density"),
                    "pos_neg_ratio": item.get("pos_neg_ratio"),
                    "inferred_positive_share": item.get("inferred_positive_share"),
                    "hop_histogram_json": json.dumps(item.get("hop_histogram", {}), sort_keys=True),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Exp 3 parity attempts against Synthology deep/structural stats."
    )
    parser.add_argument("--synth-targets", required=True, help="Path to Synthology train targets.csv")
    parser.add_argument("--synth-facts", required=True, help="Path to Synthology train facts.csv")
    parser.add_argument(
        "--attempts-root",
        default="data/exp3/baseline/parity_runs",
        help="Directory containing attempt_* outputs",
    )
    parser.add_argument("--min-deep-hops", type=int, default=3)
    parser.add_argument("--summary-json", default="", help="Optional exp3 parity loop summary JSON to enrich report")
    parser.add_argument("--out-json", default="data/exp3/baseline/parity_runs/parity_report.json")
    parser.add_argument("--out-csv", default="data/exp3/baseline/parity_runs/parity_attempts.csv")
    args = parser.parse_args()

    synth_targets = Path(args.synth_targets)
    synth_facts = Path(args.synth_facts)
    attempts_root = Path(args.attempts_root)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    if not synth_targets.exists():
        raise FileNotFoundError(f"Synthology targets not found: {synth_targets}")
    if not synth_facts.exists():
        raise FileNotFoundError(f"Synthology facts not found: {synth_facts}")
    if not attempts_root.exists():
        raise FileNotFoundError(f"Attempts root not found: {attempts_root}")

    report = build_parity_report(
        synth_targets=synth_targets,
        synth_facts=synth_facts,
        attempts_root=attempts_root,
        min_deep_hops=args.min_deep_hops,
    )

    if args.summary_json:
        summary_path = Path(args.summary_json)
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as handle:
                summary_payload = json.load(handle)
            report["time_to_parity"] = {
                "synth_runtime_seconds": summary_payload.get("synth_runtime_seconds"),
                "baseline_time_to_parity_seconds": summary_payload.get("baseline_time_to_parity_seconds"),
                "baseline_vs_synth_time_ratio": summary_payload.get("baseline_vs_synth_time_ratio"),
                "matched_attempt": summary_payload.get("matched_attempt"),
            }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    _write_attempts_csv(report, out_csv)

    synth = report["synthology"]
    logger.info(
        "Exp3 parity report | K_deep={} | synth_hops={} | attempts={} | json={} | csv={}",
        synth["k_deep"],
        synth["hop_histogram"],
        len(report.get("attempts", [])),
        out_json,
        out_csv,
    )


if __name__ == "__main__":
    main()

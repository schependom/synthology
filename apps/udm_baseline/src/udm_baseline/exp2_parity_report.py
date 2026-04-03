"""Report utilities for Exp 2 UDM parity attempts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

RDF_TYPE = "rdf:type"


def _bucketize_hops(hop_histogram: dict[str, int], min_deep_hops: int) -> dict[str, int]:
    hop0 = 0
    hop1 = 0
    hop2_to_3 = 0
    deep = 0

    for hop_key, count in hop_histogram.items():
        hop = _to_int(hop_key, default=0)
        c = _to_int(count, default=0)
        if hop == 0:
            hop0 += c
        elif hop == 1:
            hop1 += c
        elif 2 <= hop <= 3:
            hop2_to_3 += c
        if hop >= min_deep_hops:
            deep += c

    return {
        "hop_0": hop0,
        "hop_1": hop1,
        "hop_2_to_3": hop2_to_3,
        "hop_ge_min_deep": deep,
    }


def _bucket_shares(bucket_counts: dict[str, int], positives: int) -> dict[str, float]:
    denom = float(positives) if positives > 0 else 0.0
    if denom == 0.0:
        return {k: 0.0 for k in bucket_counts.keys()}
    return {k: (float(v) / denom) for k, v in bucket_counts.items()}


def _load_parity_loop_summary(attempts_root: Path) -> dict[str, Any] | None:
    summary_path = attempts_root / "parity_loop_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, encoding="utf-8") as handle:
        return json.load(handle)


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
    # Edge density is measured on directed non-type relation edges over discovered individuals.
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
    target_stats = extract_target_stats(targets_csv=targets_csv, min_deep_hops=min_deep_hops)
    fact_stats = extract_facts_stats(facts_csv=facts_csv)
    return {
        **target_stats,
        **fact_stats,
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
        targets_csv = attempt_dir / "train" / "targets.csv"
        if targets_csv.exists():
            result.append((attempt_dir.name, targets_csv))
    return result


def _attempt_facts_from_targets(targets_csv: Path) -> Path:
    return targets_csv.parent / "facts.csv"


def build_parity_report(
    synth_targets: Path,
    synth_facts: Path,
    attempts_root: Path,
    min_deep_hops: int,
    synth_generation_metrics: Path | None = None,
) -> dict[str, Any]:
    synth_stats = extract_dataset_stats(synth_targets, synth_facts, min_deep_hops=min_deep_hops)
    synth_hop_buckets = _bucketize_hops(synth_stats["hop_histogram"], min_deep_hops=min_deep_hops)
    synth_hop_bucket_shares = _bucket_shares(synth_hop_buckets, positives=int(synth_stats["positives"]))

    loop_summary = _load_parity_loop_summary(attempts_root)
    attempts_timing_by_name: dict[str, dict[str, Any]] = {}
    if loop_summary is not None:
        for attempt in loop_summary.get("attempts", []):
            name = str(attempt.get("attempt", "")).strip()
            if name:
                attempts_timing_by_name[name] = {
                    "attempt_wall_seconds": attempt.get("attempt_wall_seconds"),
                    "cumulative_attempt_seconds": attempt.get("cumulative_attempt_seconds"),
                    "success": attempt.get("success"),
                }

    synth_runtime_seconds = None
    if synth_generation_metrics is not None:
        synth_runtime_seconds = load_synth_runtime_seconds(synth_generation_metrics)
    if synth_runtime_seconds is None and loop_summary is not None:
        synth_runtime_seconds = loop_summary.get("synth_runtime_seconds")

    attempts = []

    for attempt_name, targets_csv in _discover_attempt_targets(attempts_root):
        facts_csv = _attempt_facts_from_targets(targets_csv)
        if not facts_csv.exists():
            logger.warning("Skipping {} because facts.csv is missing", attempt_name)
            continue
        stats = extract_dataset_stats(targets_csv, facts_csv, min_deep_hops=min_deep_hops)
        hop_buckets = _bucketize_hops(stats["hop_histogram"], min_deep_hops=min_deep_hops)
        hop_bucket_shares = _bucket_shares(hop_buckets, positives=int(stats["positives"]))
        stats["attempt"] = attempt_name
        stats["hop_buckets"] = hop_buckets
        stats["hop_bucket_shares"] = hop_bucket_shares
        stats.update(attempts_timing_by_name.get(attempt_name, {}))
        attempts.append(stats)

    matched_attempt = loop_summary.get("matched_attempt") if loop_summary is not None else None
    baseline_time_to_parity_seconds = None
    baseline_vs_synth_time_ratio = None

    if loop_summary is not None:
        baseline_time_to_parity_seconds = loop_summary.get("baseline_time_to_parity_seconds")
        baseline_vs_synth_time_ratio = loop_summary.get("baseline_vs_synth_time_ratio")
    elif matched_attempt is not None:
        baseline_time_to_parity_seconds = matched_attempt.get("cumulative_attempt_seconds")

    if baseline_vs_synth_time_ratio is None and baseline_time_to_parity_seconds and synth_runtime_seconds:
        if float(synth_runtime_seconds) > 0:
            baseline_vs_synth_time_ratio = float(baseline_time_to_parity_seconds) / float(synth_runtime_seconds)

    timing_summary = {
        "synth_runtime_seconds": synth_runtime_seconds,
        "baseline_time_to_parity_seconds": baseline_time_to_parity_seconds,
        "baseline_vs_synth_time_ratio": baseline_vs_synth_time_ratio,
        "attempt_count": len(attempts),
        "matched_attempt": matched_attempt,
    }

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
            "hop_buckets": synth_hop_buckets,
            "hop_bucket_shares": synth_hop_bucket_shares,
            "min_deep_hops": min_deep_hops,
        },
        "timing": timing_summary,
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
                "attempt_wall_seconds",
                "cumulative_attempt_seconds",
                "success",
                "hop_0",
                "hop_1",
                "hop_2_to_3",
                "hop_ge_min_deep",
                "hop_0_share",
                "hop_1_share",
                "hop_2_to_3_share",
                "hop_ge_min_deep_share",
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
                    "attempt_wall_seconds": item.get("attempt_wall_seconds"),
                    "cumulative_attempt_seconds": item.get("cumulative_attempt_seconds"),
                    "success": item.get("success"),
                    "hop_0": item.get("hop_buckets", {}).get("hop_0", 0),
                    "hop_1": item.get("hop_buckets", {}).get("hop_1", 0),
                    "hop_2_to_3": item.get("hop_buckets", {}).get("hop_2_to_3", 0),
                    "hop_ge_min_deep": item.get("hop_buckets", {}).get("hop_ge_min_deep", 0),
                    "hop_0_share": item.get("hop_bucket_shares", {}).get("hop_0", 0.0),
                    "hop_1_share": item.get("hop_bucket_shares", {}).get("hop_1", 0.0),
                    "hop_2_to_3_share": item.get("hop_bucket_shares", {}).get("hop_2_to_3", 0.0),
                    "hop_ge_min_deep_share": item.get("hop_bucket_shares", {}).get("hop_ge_min_deep", 0.0),
                    "hop_histogram_json": json.dumps(item.get("hop_histogram", {}), sort_keys=True),
                }
            )


def _write_markdown_summary(report: dict[str, Any], out_path: Path) -> None:
    synth = report.get("synthology", {})
    timing = report.get("timing", {})
    attempts = report.get("attempts", [])
    matched = timing.get("matched_attempt") or {}

    lines = [
        "# Exp2 Parity Report",
        "",
        "## Timing",
        "",
        f"- synth_runtime_seconds: {timing.get('synth_runtime_seconds')}",
        f"- baseline_time_to_parity_seconds: {timing.get('baseline_time_to_parity_seconds')}",
        f"- baseline_vs_synth_time_ratio: {timing.get('baseline_vs_synth_time_ratio')}",
        f"- attempts_evaluated: {timing.get('attempt_count')}",
        f"- matched_attempt: {matched.get('attempt')}",
        "",
        "## Synthology Reference",
        "",
        f"- k_deep (hop >= {synth.get('min_deep_hops')}): {synth.get('k_deep')}",
        f"- positives: {synth.get('positives')}",
        f"- negatives: {synth.get('negatives')}",
        "",
        "### Synthology Hop Buckets (Positive Targets)",
        "",
        "| bucket | count | share |",
        "| --- | ---: | ---: |",
    ]

    synth_bucket_counts = synth.get("hop_buckets", {})
    synth_bucket_shares = synth.get("hop_bucket_shares", {})
    for key, label in (
        ("hop_0", "hop=0"),
        ("hop_1", "hop=1"),
        ("hop_2_to_3", "2<=hop<=3"),
        ("hop_ge_min_deep", f"hop>={synth.get('min_deep_hops')}"),
    ):
        lines.append(
            f"| {label} | {synth_bucket_counts.get(key, 0)} | {float(synth_bucket_shares.get(key, 0.0)):.2%} |"
        )

    lines.extend(
        [
            "",
            "## Attempts",
            "",
            "| attempt | deep_count | hop>=d share | attempt_s | cumulative_s | success |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for item in attempts:
        bucket_shares = item.get("hop_bucket_shares", {})
        lines.append(
            "| {} | {} | {:.2%} | {} | {} | {} |".format(
                item.get("attempt"),
                item.get("deep_count", 0),
                float(bucket_shares.get("hop_ge_min_deep", 0.0)),
                item.get("attempt_wall_seconds"),
                item.get("cumulative_attempt_seconds"),
                item.get("success"),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Exp 2 UDM parity attempts against Synthology K_deep.")
    parser.add_argument(
        "--synth-targets",
        default="data/exp2/synthology/family_tree/train/targets.csv",
        help="Path to Synthology train targets.csv used to derive K_deep",
    )
    parser.add_argument(
        "--synth-facts",
        default="data/exp2/synthology/family_tree/train/facts.csv",
        help="Path to Synthology train facts.csv used for structural parity stats",
    )
    parser.add_argument(
        "--attempts-root",
        default="data/exp2/baseline/parity_runs",
        help="Directory containing attempt_*/train/targets.csv",
    )
    parser.add_argument(
        "--synth-generation-metrics",
        default="data/exp2/synthology/family_tree/generation_metrics.json",
        help="Synthology generation metrics JSON used to report runtime ratio",
    )
    parser.add_argument("--min-deep-hops", type=int, default=3, help="Minimum hop threshold for deep facts")
    parser.add_argument(
        "--out-json",
        default="data/exp2/baseline/parity_runs/parity_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--out-csv",
        default="data/exp2/baseline/parity_runs/parity_attempts.csv",
        help="Output CSV summary path",
    )
    parser.add_argument(
        "--out-md",
        default="data/exp2/baseline/parity_runs/parity_report.md",
        help="Output Markdown summary path",
    )
    args = parser.parse_args()

    synth_targets = Path(args.synth_targets)
    synth_facts = Path(args.synth_facts)
    attempts_root = Path(args.attempts_root)
    synth_generation_metrics = Path(args.synth_generation_metrics)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

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
        synth_generation_metrics=synth_generation_metrics,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    _write_attempts_csv(report, out_csv)
    _write_markdown_summary(report, out_md)

    synth = report["synthology"]
    logger.info(
        "Exp2 parity report | K_deep={} | synth_hops={} | attempts={} | json={} | csv={} | md={}",
        synth["k_deep"],
        synth["hop_histogram"],
        len(report.get("attempts", [])),
        out_json,
        out_csv,
        out_md,
    )


if __name__ == "__main__":
    main()

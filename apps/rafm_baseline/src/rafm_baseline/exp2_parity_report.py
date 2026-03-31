"""Report utilities for Exp 2 RAFM parity attempts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger


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

    with open(targets_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row.get("label", "1")).strip()
            if label != "1":
                continue

            positives += 1
            hops = _to_int(row.get("hops"), default=0)
            hop_hist[hops] += 1

            row_type = str(row.get("type", "")).strip().lower()
            is_inferred = row_type.startswith("inf") or row_type == "inf_root"
            if is_inferred and hops >= min_deep_hops:
                deep_count += 1

    return {
        "positives": positives,
        "deep_count": deep_count,
        "hop_histogram": {str(k): v for k, v in sorted(hop_hist.items())},
    }


def _discover_attempt_targets(attempts_root: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for attempt_dir in sorted(attempts_root.glob("attempt_*")):
        targets_csv = attempt_dir / "train" / "targets.csv"
        if targets_csv.exists():
            result.append((attempt_dir.name, targets_csv))
    return result


def build_parity_report(
    synth_targets: Path,
    attempts_root: Path,
    min_deep_hops: int,
) -> dict[str, Any]:
    synth_stats = extract_target_stats(synth_targets, min_deep_hops=min_deep_hops)
    attempts = []

    for attempt_name, targets_csv in _discover_attempt_targets(attempts_root):
        stats = extract_target_stats(targets_csv, min_deep_hops=min_deep_hops)
        stats["attempt"] = attempt_name
        stats["targets_csv"] = str(targets_csv)
        attempts.append(stats)

    return {
        "synthology": {
            "targets_csv": str(synth_targets),
            "k_deep": synth_stats["deep_count"],
            "hop_histogram": synth_stats["hop_histogram"],
            "positives": synth_stats["positives"],
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
                "positives",
                "deep_count",
                "hop_histogram_json",
            ],
        )
        writer.writeheader()
        for item in report.get("attempts", []):
            writer.writerow(
                {
                    "attempt": item.get("attempt"),
                    "targets_csv": item.get("targets_csv"),
                    "positives": item.get("positives"),
                    "deep_count": item.get("deep_count"),
                    "hop_histogram_json": json.dumps(item.get("hop_histogram", {}), sort_keys=True),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Exp 2 RAFM parity attempts against Synthology K_deep.")
    parser.add_argument(
        "--synth-targets",
        default="data/exp2/synthology/family_tree/train/targets.csv",
        help="Path to Synthology train targets.csv used to derive K_deep",
    )
    parser.add_argument(
        "--attempts-root",
        default="data/exp2/baseline/parity_runs",
        help="Directory containing attempt_*/train/targets.csv",
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
    args = parser.parse_args()

    synth_targets = Path(args.synth_targets)
    attempts_root = Path(args.attempts_root)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    if not synth_targets.exists():
        raise FileNotFoundError(f"Synthology targets not found: {synth_targets}")
    if not attempts_root.exists():
        raise FileNotFoundError(f"Attempts root not found: {attempts_root}")

    report = build_parity_report(
        synth_targets=synth_targets,
        attempts_root=attempts_root,
        min_deep_hops=args.min_deep_hops,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    _write_attempts_csv(report, out_csv)

    synth = report["synthology"]
    logger.info(
        "Exp2 parity report | K_deep={} | synth_hops={} | attempts={} | json={} | csv={}",
        synth["k_deep"],
        synth["hop_histogram"],
        len(report.get("attempts", [])),
        out_json,
        out_csv,
    )


if __name__ == "__main__":
    main()

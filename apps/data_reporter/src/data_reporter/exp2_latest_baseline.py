from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _configure_plot_style() -> None:
    latex_available = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "text.usetex": latex_available,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _discover_latest_baseline_run(repo_root: Path) -> Path:
    root = repo_root / "reports" / "experiment_runs"
    candidates = [
        path for path in root.glob("*/exp2/generate_baseline/*") if path.is_dir() and (path / "manifest.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No archived Exp2 baseline runs found under reports/experiment_runs.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_run_id_from_log(run_log_path: Path) -> str | None:
    if not run_log_path.exists():
        return None
    for line in run_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Timing recorder enabled" not in line or "run_id=" not in line:
            continue
        fragment = line.split("run_id=", maxsplit=1)[1]
        return fragment.split("|", maxsplit=1)[0].strip()
    return None


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    sorted_values = sorted(values)
    n = len(sorted_values)
    p95_idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    if n % 2 == 1:
        median = sorted_values[n // 2]
    else:
        median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0
    return {
        "count": n,
        "mean": sum(sorted_values) / n,
        "median": median,
        "p95": sorted_values[p95_idx],
        "max": sorted_values[-1],
    }


def _analyze_targets(split_dir: Path) -> dict[str, Any]:
    targets_rows = _read_csv(split_dir / "targets.csv")
    facts_rows = _read_csv(split_dir / "facts.csv")

    per_sample_targets: Counter[str] = Counter()
    per_sample_facts: Counter[str] = Counter()
    hop_counts: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()
    type_counts: Counter[str] = Counter()

    for row in targets_rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            per_sample_targets[sid] += 1
        hop_counts[_safe_int(row.get("hops", 0), 0)] += 1
        label_counts[_safe_int(row.get("label", 1), 1)] += 1
        type_counts[str(row.get("type", "unknown"))] += 1

    for row in facts_rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            per_sample_facts[sid] += 1

    inferred_like = (
        type_counts.get("inf_root", 0) + type_counts.get("inf_intermediate", 0) + type_counts.get("inferred", 0)
    )

    return {
        "rows": {
            "facts": len(facts_rows),
            "targets": len(targets_rows),
            "positive_targets": label_counts.get(1, 0),
            "negative_targets": label_counts.get(0, 0),
        },
        "samples": {
            "count": len(set(per_sample_facts.keys()) | set(per_sample_targets.keys())),
            "avg_facts_per_sample": (sum(per_sample_facts.values()) / len(per_sample_facts))
            if per_sample_facts
            else 0.0,
            "avg_targets_per_sample": (sum(per_sample_targets.values()) / len(per_sample_targets))
            if per_sample_targets
            else 0.0,
            "max_facts_per_sample": max(per_sample_facts.values()) if per_sample_facts else 0,
            "max_targets_per_sample": max(per_sample_targets.values()) if per_sample_targets else 0,
        },
        "hops": {str(k): int(v) for k, v in sorted(hop_counts.items(), key=lambda item: item[0])},
        "type_counts": dict(type_counts),
        "inferred_like_targets": inferred_like,
    }


def _analyze_timing(timing_csv_path: Path, run_id: str | None) -> dict[str, Any]:
    rows = _read_csv(timing_csv_path)
    all_rows = list(rows)
    run_id_filter_applied = False
    run_id_filter_matched = True
    if run_id is not None and rows and "run_id" in rows[0]:
        run_id_filter_applied = True
        rows = [row for row in rows if str(row.get("run_id", "")).strip() == run_id]
        if not rows:
            run_id_filter_matched = False
            rows = all_rows

    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", "unknown"))].append(row)

    metric_columns = ["serialize_seconds", "java_seconds", "parse_seconds", "total_seconds", "function_total_seconds"]
    count_columns = ["base_triples", "closure_triples", "newly_inferred"]

    split_summary: dict[str, Any] = {}
    for split_name, split_rows in by_split.items():
        split_result: dict[str, Any] = {"events": len(split_rows)}
        for metric in metric_columns:
            values = [_safe_float(row.get(metric, 0.0), 0.0) for row in split_rows]
            split_result[metric] = _summary_stats(values)
        for metric in count_columns:
            values = [_safe_float(row.get(metric, 0.0), 0.0) for row in split_rows]
            split_result[metric] = _summary_stats(values)
        split_summary[split_name] = split_result

    overall: dict[str, Any] = {"events": len(rows)}
    for metric in metric_columns + count_columns:
        values = [_safe_float(row.get(metric, 0.0), 0.0) for row in rows]
        overall[metric] = _summary_stats(values)

    return {
        "timing_csv": str(timing_csv_path),
        "run_id_filter": run_id,
        "run_id_filter_applied": run_id_filter_applied,
        "run_id_filter_matched": run_id_filter_matched,
        "total_events_used": len(rows),
        "overall": overall,
        "by_split": split_summary,
    }


def _plot_train_hops(train_hops: dict[str, int], out_path: Path) -> None:
    _configure_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(train_hops.keys(), key=lambda item: int(item))
    values = [train_hops[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#2E5EAA")
    ax.set_title("Exp2 Baseline Train Hop Distribution")
    ax.set_xlabel("Hops")
    ax.set_ylabel("Target Count")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    run = summary["run"]
    dataset = summary["dataset"]
    timing = summary["timing"]

    lines = [
        "# Exp2 Latest Baseline Analysis",
        "",
        "## Run",
        "",
        f"- archive_dir: {run['archive_dir']}",
        f"- run_id: {run.get('timing_run_id')}",
        f"- train_fact_cap: {run.get('train_fact_cap')}",
        "",
        "## Train Split Overview",
        "",
    ]

    train = dataset.get("train", {})
    rows = train.get("rows", {})
    samples = train.get("samples", {})
    lines.extend(
        [
            f"- facts rows: {rows.get('facts', 0)}",
            f"- targets rows: {rows.get('targets', 0)}",
            f"- positive targets: {rows.get('positive_targets', 0)}",
            f"- negative targets: {rows.get('negative_targets', 0)}",
            f"- avg facts/sample: {samples.get('avg_facts_per_sample', 0.0):.2f}",
            f"- avg targets/sample: {samples.get('avg_targets_per_sample', 0.0):.2f}",
            f"- max facts/sample: {samples.get('max_facts_per_sample', 0)}",
            f"- max targets/sample: {samples.get('max_targets_per_sample', 0)}",
            "",
            "## Timing Overview",
            "",
        ]
    )

    overall = timing.get("overall", {})
    lines.append(f"- events used: {timing.get('total_events_used', 0)}")
    for metric in ["serialize_seconds", "java_seconds", "parse_seconds", "total_seconds"]:
        stats = overall.get(metric, {})
        mean = stats.get("mean")
        p95 = stats.get("p95")
        max_value = stats.get("max")
        if mean is None:
            continue
        lines.append(f"- {metric}: mean={mean:.6f}s, p95={p95:.6f}s, max={max_value:.6f}s")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latest archived Exp2 baseline output.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--archive-dir", default=None, help="Optional explicit baseline archive run dir")
    parser.add_argument("--baseline-dir", default=None, help="Optional explicit baseline dataset root")
    parser.add_argument("--timing-csv", default=None, help="Optional explicit timing CSV path")
    parser.add_argument("--out-dir", required=True, help="Directory where summary and plots are written")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = Path(args.archive_dir).resolve() if args.archive_dir else _discover_latest_baseline_run(repo_root)

    manifest_path = archive_dir / "manifest.json"
    run_log_path = archive_dir / "run.log"
    config_path = archive_dir / "configs" / "exp2_baseline.yaml"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    timing_run_id = _extract_run_id_from_log(run_log_path)

    train_fact_cap: str | None = None
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("train_fact_cap:"):
                train_fact_cap = line.split(":", maxsplit=1)[1].strip()
                break

    archived_baseline_dir = archive_dir / "artifacts" / "data" / "exp2" / "baseline" / "family_tree"
    live_baseline_dir = repo_root / "data" / "exp2" / "baseline" / "family_tree"
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir).resolve()
    elif archived_baseline_dir.exists():
        baseline_dir = archived_baseline_dir
    else:
        baseline_dir = live_baseline_dir

    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline dataset directory not found: {baseline_dir}")

    if args.timing_csv:
        timing_csv = Path(args.timing_csv).resolve()
    else:
        timing_csv = repo_root / "data" / "exp2" / "timings" / "exp2_baseline_jena_events.csv"

    if not timing_csv.exists():
        raise FileNotFoundError(f"Timing CSV not found: {timing_csv}")

    dataset_summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_dir = baseline_dir / split
        if split_dir.exists():
            dataset_summary[split] = _analyze_targets(split_dir)

    timing_summary = _analyze_timing(timing_csv, timing_run_id)
    train_hops = dataset_summary.get("train", {}).get("hops", {})
    _plot_train_hops(train_hops, out_dir / "train_hop_distribution")

    summary = {
        "run": {
            "archive_dir": str(archive_dir),
            "manifest_path": str(manifest_path),
            "run_log_path": str(run_log_path),
            "task": manifest.get("task"),
            "command": manifest.get("command"),
            "timing_run_id": timing_run_id,
            "train_fact_cap": train_fact_cap,
        },
        "inputs": {
            "baseline_dir": str(baseline_dir),
            "timing_csv": str(timing_csv),
        },
        "dataset": dataset_summary,
        "timing": timing_summary,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / "summary.md")

    print(f"Wrote analysis to: {out_dir}")
    print(f"Latest baseline archive: {archive_dir}")
    print(f"Detected train_fact_cap: {train_fact_cap}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
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


def _discover_latest_exp3_run(repo_root: Path) -> Path:
    root = repo_root / "reports" / "experiment_runs"
    candidates = [
        path for path in root.glob("*/exp3/generate_baseline/*") if path.is_dir() and (path / "manifest.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No archived Exp3 baseline runs found under reports/experiment_runs.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _discover_latest_diagnostics_summary(repo_root: Path, universities: int) -> Path | None:
    diagnostics_root = repo_root / "data" / "owl2bench" / "output" / "diagnostics"
    pattern = f"owl2bench_{universities}_*/summary.json"
    candidates = [path for path in diagnostics_root.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_diagnostics_summary_for_archive(repo_root: Path, archive_dir: Path, universities: int) -> tuple[dict[str, Any], str | None]:
    run_log = archive_dir / "abox_generation" / "run.log"
    if run_log.exists():
        text = run_log.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"Diagnostics exported \| dir=([^|\n]+)", text)
        if matches:
            raw_dir = matches[-1].strip()
            diag_dir = Path(raw_dir)
            if not diag_dir.is_absolute():
                diag_dir = (repo_root / diag_dir).resolve()
            summary_path = diag_dir / "summary.json"
            if summary_path.exists():
                return json.loads(summary_path.read_text(encoding="utf-8")), str(summary_path)

    fallback = _discover_latest_diagnostics_summary(repo_root, universities)
    if fallback and fallback.exists():
        return json.loads(fallback.read_text(encoding="utf-8")), str(fallback)
    return {}, None


def _analyze_split(split_dir: Path) -> dict[str, Any]:
    facts_rows = _read_csv(split_dir / "facts.csv")
    targets_rows = _read_csv(split_dir / "targets.csv")

    labels = Counter()
    types = Counter()
    hops_all = Counter()
    hops_positive = Counter()
    predicate_positive = Counter()
    sample_ids = set()

    for row in facts_rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            sample_ids.add(sid)

    for row in targets_rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            sample_ids.add(sid)
        label = _safe_int(row.get("label", 1), 1)
        labels[str(label)] += 1
        row_type = str(row.get("type", "unknown"))
        types[row_type] += 1
        hop = _safe_int(row.get("hops", 0), 0)
        hops_all[str(hop)] += 1
        if label == 1:
            hops_positive[str(hop)] += 1
            predicate_positive[str(row.get("predicate", ""))] += 1

    positives = int(labels.get("1", 0))
    negatives = int(labels.get("0", 0))

    return {
        "rows": {
            "facts": len(facts_rows),
            "targets": len(targets_rows),
            "positive": positives,
            "negative": negatives,
            "positive_negative_ratio": (positives / negatives) if negatives else None,
        },
        "samples": {
            "count": len(sample_ids),
        },
        "type_counts": dict(types),
        "hops_all": {k: int(v) for k, v in sorted(hops_all.items(), key=lambda item: int(item[0]))},
        "hops_positive": {k: int(v) for k, v in sorted(hops_positive.items(), key=lambda item: int(item[0]))},
        "top_positive_predicates": dict(predicate_positive.most_common(15)),
    }


def _check_target_integrity(split_dir: Path) -> dict[str, Any]:
    targets_rows = _read_csv(split_dir / "targets.csv")
    positives = set()
    negatives = set()

    for row in targets_rows:
        key = (
            str(row.get("sample_id", "")),
            str(row.get("subject", "")),
            str(row.get("predicate", "")),
            str(row.get("object", "")),
        )
        label = _safe_int(row.get("label", 1), 1)
        if label == 1:
            positives.add(key)
        elif label == 0:
            negatives.add(key)

    overlap = positives.intersection(negatives)
    return {
        "positive_unique": len(positives),
        "negative_unique": len(negatives),
        "positive_negative_overlap": len(overlap),
    }


def _analyze_materialization_timing(timing_csv: Path) -> dict[str, Any]:
    rows = _read_csv(timing_csv)
    if not rows:
        return {"events": 0}

    latest = rows[-1]
    return {
        "events": len(rows),
        "latest_event": {
            "run_id": latest.get("run_id"),
            "base_triples": _safe_int(latest.get("base_triples", 0), 0),
            "closure_triples": _safe_int(latest.get("closure_triples", 0), 0),
            "inferred_triples": _safe_int(latest.get("inferred_triples", 0), 0),
            "reasoning_seconds": _safe_float(latest.get("reasoning_seconds", 0.0), 0.0),
            "run_total_seconds": _safe_float(latest.get("run_total_seconds", 0.0), 0.0),
        },
    }


def _plot_train_hops(train_hops_positive: dict[str, int], out_path: Path) -> None:
    _configure_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(train_hops_positive.keys(), key=lambda item: int(item))
    values = [train_hops_positive[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#1E4F8A")
    ax.set_title("Exp3 Baseline Train Hop Distribution (Positive Targets)")
    ax.set_xlabel("Hops")
    ax.set_ylabel("Target Count")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_train_labels(train_rows: dict[str, Any], out_path: Path) -> None:
    _configure_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["positive", "negative"]
    values = [int(train_rows.get("positive", 0)), int(train_rows.get("negative", 0))]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(labels, values, color=["#2E8B57", "#B22222"])
    ax.set_title("Exp3 Baseline Train Label Counts")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    run = summary["run"]
    train = summary.get("dataset", {}).get("train", {})
    train_rows = train.get("rows", {})
    train_types = train.get("type_counts", {})
    integrity = summary.get("integrity", {})
    timing_latest = summary.get("timing", {}).get("latest_event", {})
    diagnostics = summary.get("diagnostics", {})
    final_export = diagnostics.get("final_export", {})
    split_export = final_export.get("split_export", {}) if isinstance(final_export, dict) else {}
    train_export = split_export.get("train", {}) if isinstance(split_export, dict) else {}

    positives = int(train_rows.get("positive", 0) or 0)
    negatives = int(train_rows.get("negative", 0) or 0)
    base_fact_pos = int(train_types.get("base_fact", 0) or 0)
    inferred_pos = positives - base_fact_pos

    lines = [
        "# Exp3 Latest Baseline Analysis",
        "",
        "## Run",
        "",
        f"- archive_dir: {run['archive_dir']}",
        f"- universities: {run.get('universities')}",
        f"- final_jena_profile: {run.get('final_jena_profile')}",
        f"- final_reasoning_input_triple_cap: {run.get('final_reasoning_input_triple_cap')}",
        "",
        "## Train Split",
        "",
        f"- facts rows: {train_rows.get('facts', 0)}",
        f"- targets rows: {train_rows.get('targets', 0)}",
        f"- positive rows: {positives}",
        f"- negative rows: {negatives}",
        f"- positive/negative ratio: {train_rows.get('positive_negative_ratio')}",
        "",
        "## Ratio Accounting",
        "",
        f"- diagnostics summary source: {summary.get('diagnostics_summary_path')}",
        f"- overall eligible positive targets for negatives: {final_export.get('eligible_positive_targets_for_negatives')}",
        f"- overall expected negatives: {final_export.get('expected_negative_targets')}",
        f"- overall generated negatives: {final_export.get('generated_negative_targets')}",
        f"- overall skipped negative slots: {final_export.get('skipped_negative_slots')}",
        f"- overall rejected negative collisions with positives: {final_export.get('rejected_negative_collisions_with_positive')}",
        f"- overall rejected duplicate negatives: {final_export.get('rejected_duplicate_negatives')}",
        f"- train eligible positives for negatives: {train_export.get('eligible_positive_targets')}",
        f"- train expected negatives: {train_export.get('expected_negative_targets')}",
        f"- train generated negatives: {train_export.get('generated_negative_targets')}",
        f"- train skipped negative slots: {train_export.get('skipped_negative_slots')}",
        "",
        "## Label Skew Diagnosis",
        "",
        f"- base_fact positives (always label=1): {base_fact_pos}",
        f"- inferred positives: {inferred_pos}",
        "- negatives are generated only for inferred/base targets selected as positive_targets, not one-for-one against every base fact row.",
        "- with negatives_per_positive=1 and many retained base facts, positive-heavy targets are expected.",
        "",
        "## Integrity Checks",
        "",
        f"- unique positive triples: {integrity.get('positive_unique', 0)}",
        f"- unique negative triples: {integrity.get('negative_unique', 0)}",
        f"- positive-negative triple overlap: {integrity.get('positive_negative_overlap', 0)}",
        "",
        "## Materialization Timing (Latest Event)",
        "",
        f"- base_triples: {timing_latest.get('base_triples')}",
        f"- closure_triples: {timing_latest.get('closure_triples')}",
        f"- inferred_triples: {timing_latest.get('inferred_triples')}",
        f"- reasoning_seconds: {timing_latest.get('reasoning_seconds')}",
        f"- run_total_seconds: {timing_latest.get('run_total_seconds')}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latest archived Exp3 baseline output.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--archive-dir", default=None, help="Optional explicit Exp3 baseline archive run dir")
    parser.add_argument("--dataset-dir", default=None, help="Optional explicit OWL2Bench dataset root")
    parser.add_argument("--timing-csv", default=None, help="Optional explicit timing CSV path")
    parser.add_argument("--out-dir", required=True, help="Directory where summary and plots are written")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = Path(args.archive_dir).resolve() if args.archive_dir else _discover_latest_exp3_run(repo_root)
    manifest_path = archive_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    universities = int(manifest.get("universities", 1) or 1)
    live_dataset_dir = repo_root / "data" / "owl2bench" / "output" / f"owl2bench_{universities}"
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else live_dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    timing_csv = (
        Path(args.timing_csv).resolve()
        if args.timing_csv
        else archive_dir / "materialization" / "timings" / "exp3_materialize_abox_jena_events.csv"
    )
    if not timing_csv.exists():
        raise FileNotFoundError(f"Timing CSV not found: {timing_csv}")

    dataset_summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if split_dir.exists():
            dataset_summary[split] = _analyze_split(split_dir)

    train_integrity = _check_target_integrity(dataset_dir / "train") if (dataset_dir / "train").exists() else {}
    timing_summary = _analyze_materialization_timing(timing_csv)

    diagnostics_summary, diagnostics_summary_path = _load_diagnostics_summary_for_archive(repo_root, archive_dir, universities)

    train_hops_positive = dataset_summary.get("train", {}).get("hops_positive", {})
    _plot_train_hops(train_hops_positive, out_dir / "train_hop_distribution")
    _plot_train_labels(dataset_summary.get("train", {}).get("rows", {}), out_dir / "train_label_distribution")

    summary = {
        "run": {
            "archive_dir": str(archive_dir),
            "manifest_path": str(manifest_path),
            "command": manifest.get("command"),
            "universities": universities,
            "reasoning_input_triple_cap": manifest.get("reasoning_input_triple_cap"),
            "final_reasoning_input_triple_cap": manifest.get("final_reasoning_input_triple_cap"),
            "final_jena_profile": manifest.get("final_jena_profile"),
        },
        "inputs": {
            "dataset_dir": str(dataset_dir),
            "timing_csv": str(timing_csv),
        },
        "dataset": dataset_summary,
        "integrity": train_integrity,
        "timing": timing_summary,
        "diagnostics": diagnostics_summary,
        "diagnostics_summary_path": diagnostics_summary_path,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / "summary.md")

    print(f"Wrote analysis to: {out_dir}")
    print(f"Latest exp3 baseline archive: {archive_dir}")


if __name__ == "__main__":
    main()

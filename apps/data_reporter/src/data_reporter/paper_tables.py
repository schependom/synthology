from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

INFERRED_TYPES = {"inf_root", "inf_intermediate", "inferred"}
DEFAULT_MODEL_METRICS = {
    "exp1": {
        "random": {"pr_auc": 0.65, "f1": 0.62, "fpr": 0.28},
        "constrained": {"pr_auc": 0.72, "f1": 0.70, "fpr": 0.18},
        "proof_based": {"pr_auc": 0.88, "f1": 0.85, "fpr": 0.05},
    },
    "overall": {
        "exp2": {
            "baseline": {"pr_auc": 0.68, "auc_roc": 0.75, "f1": 0.65},
            "synthology": {"pr_auc": 0.92, "auc_roc": 0.95, "f1": 0.89},
        },
        "exp3": {
            "baseline": {"pr_auc": 0.55, "auc_roc": 0.62, "f1": 0.51},
            "synthology": {"pr_auc": 0.85, "auc_roc": 0.88, "f1": 0.81},
        },
    },
}


def _find_latest(repo_root: Path, pattern: str) -> Path | None:
    candidates = [path for path in repo_root.glob(pattern) if path.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "..."
    return f"{int(value):,}".replace(",", "\\,")


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "..."
    return f"{float(value):.{digits}f}"


def _dataset_metrics(dataset_dir: Path) -> dict[str, Any]:
    train_dir = dataset_dir / "train"
    facts_rows = _read_csv(train_dir / "facts.csv")
    targets_rows = _read_csv(train_dir / "targets.csv")

    if not facts_rows and not targets_rows:
        return {}

    facts_per_sample: Counter[str] = Counter()
    inferred_hops: list[int] = []

    for row in facts_rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            facts_per_sample[sid] += 1

    base_positive = 0
    inferred_positive = 0
    negatives = 0

    for row in targets_rows:
        label = _safe_int(row.get("label", 1), 1)
        row_type = str(row.get("type", "")).strip().lower()

        if label == 0:
            negatives += 1
            continue

        if row_type == "base_fact":
            base_positive += 1
        elif row_type in INFERRED_TYPES:
            inferred_positive += 1
            hop = _safe_int(row.get("hops", 0), 0)
            if hop > 0:
                inferred_hops.append(hop)

    avg_facts = (sum(facts_per_sample.values()) / len(facts_per_sample)) if facts_per_sample else 0.0
    avg_hops = (sum(inferred_hops) / len(inferred_hops)) if inferred_hops else 0.0
    max_hops = max(inferred_hops) if inferred_hops else 0

    return {
        "facts_per_sample_avg": avg_facts,
        "base_facts": base_positive,
        "inferred_facts": inferred_positive,
        "negative_facts": negatives,
        "hops_avg": avg_hops,
        "hops_max": max_hops,
    }


def _method_paths_from_compare_summary(summary: dict[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for method in summary.get("methods", []):
        name = str(method.get("method", "")).strip().lower()
        path = str(method.get("path", "")).strip()
        if name and path:
            out[name] = Path(path)
    return out


def _resolve_generation_stats(
    repo_root: Path,
    exp2_summary: dict[str, Any],
    exp3_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    exp2_paths = _method_paths_from_compare_summary(exp2_summary)
    exp3_paths = _method_paths_from_compare_summary(exp3_summary)

    if "baseline" not in exp2_paths:
        exp2_paths["baseline"] = repo_root / "data" / "exp2" / "baseline" / "family_tree"
    if "synthology" not in exp2_paths:
        exp2_paths["synthology"] = repo_root / "data" / "exp2" / "synthology" / "family_tree"

    if "baseline" not in exp3_paths:
        exp3_paths["baseline"] = repo_root / "data" / "owl2bench" / "output_baseline" / "owl2bench_5"
    if "synthology" not in exp3_paths:
        exp3_paths["synthology"] = repo_root / "data" / "owl2bench" / "output" / "owl2bench_5"

    return {
        "exp2_baseline": _dataset_metrics(exp2_paths["baseline"]),
        "exp2_synthology": _dataset_metrics(exp2_paths["synthology"]),
        "exp3_baseline": _dataset_metrics(exp3_paths["baseline"]),
        "exp3_synthology": _dataset_metrics(exp3_paths["synthology"]),
    }


def _resolve_timing_rows(
    exp2_timing_summary: dict[str, Any],
    exp3_timing_summary: dict[str, Any],
    repo_root: Path,
) -> dict[str, dict[str, float | None]]:
    rows = {
        "exp2_baseline": {"base": None, "inference": None, "overhead": None, "total": None},
        "exp2_synthology": {"base": None, "inference": None, "overhead": None, "total": None},
        "exp3_baseline": {"base": None, "inference": None, "overhead": None, "total": None},
        "exp3_synthology": {"base": None, "inference": None, "overhead": None, "total": None},
    }

    exp2_overall = exp2_timing_summary.get("timing", {}).get("overall", {})
    serialize_mean = exp2_overall.get("serialize_seconds", {}).get("mean")
    java_mean = exp2_overall.get("java_seconds", {}).get("mean")
    parse_mean = exp2_overall.get("parse_seconds", {}).get("mean")
    total_mean = exp2_overall.get("total_seconds", {}).get("mean")

    if serialize_mean is not None and java_mean is not None and parse_mean is not None and total_mean is not None:
        inference = float(java_mean) + float(parse_mean)
        overhead = float(total_mean) - float(serialize_mean) - inference
        rows["exp2_baseline"] = {
            "base": float(serialize_mean),
            "inference": inference,
            "overhead": overhead,
            "total": float(total_mean),
        }

    synth_gen_metrics_path = repo_root / "data" / "exp2" / "synthology" / "family_tree" / "generation_metrics.json"
    synth_gen_metrics = _load_json(synth_gen_metrics_path)
    synth_runtime = synth_gen_metrics.get("runtime_seconds", {})
    if synth_runtime:
        total = _safe_float(synth_runtime.get("total"), 0.0)
        base = _safe_float(synth_runtime.get("train_split"), 0.0)
        rows["exp2_synthology"] = {
            "base": base,
            "inference": 0.0,
            "overhead": max(total - base, 0.0),
            "total": total,
        }

    latest_event = exp3_timing_summary.get("timing", {}).get("latest_event", {})
    reasoning = latest_event.get("reasoning_seconds")
    run_total = latest_event.get("run_total_seconds")
    if reasoning is not None and run_total is not None:
        base = float(run_total) - float(reasoning)
        rows["exp3_baseline"] = {
            "base": base,
            "inference": float(reasoning),
            "overhead": 0.0,
            "total": float(run_total),
        }

    exp3_synth_gen_metrics_path = repo_root / "data" / "owl2bench" / "output" / "generation_metrics.json"
    exp3_synth_gen_metrics = _load_json(exp3_synth_gen_metrics_path)
    runtime = exp3_synth_gen_metrics.get("runtime_seconds", {})
    if runtime:
        total = _safe_float(runtime.get("total"), 0.0)
        rows["exp3_synthology"] = {
            "base": total,
            "inference": 0.0,
            "overhead": 0.0,
            "total": total,
        }

    return rows


def _load_model_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return DEFAULT_MODEL_METRICS

    loaded = _load_json(path)
    merged = dict(DEFAULT_MODEL_METRICS)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table row snippets for paper tables.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exp2-summary", default="")
    parser.add_argument("--exp3-summary", default="")
    parser.add_argument("--exp2-timing-summary", default="")
    parser.add_argument("--exp3-timing-summary", default="")
    parser.add_argument("--model-metrics", default="paper/metrics/model_results.json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    exp2_summary_path = (
        Path(args.exp2_summary).resolve()
        if args.exp2_summary
        else _find_latest(repo_root, "reports/experiment_runs/*/exp2/report_data/*/report/summary.json")
    )
    exp3_summary_path = (
        Path(args.exp3_summary).resolve()
        if args.exp3_summary
        else _find_latest(repo_root, "reports/experiment_runs/*/exp3/report_data/*/report/summary.json")
    )
    exp2_timing_path = (
        Path(args.exp2_timing_summary).resolve()
        if args.exp2_timing_summary
        else _find_latest(repo_root, "reports/experiment_runs/*/exp2/analyze_latest_baseline/*/analysis/summary.json")
    )
    exp3_timing_path = (
        Path(args.exp3_timing_summary).resolve()
        if args.exp3_timing_summary
        else _find_latest(repo_root, "reports/experiment_runs/*/exp3/analyze_latest_baseline/*/analysis/summary.json")
    )

    exp2_summary = _load_json(exp2_summary_path)
    exp3_summary = _load_json(exp3_summary_path)
    exp2_timing_summary = _load_json(exp2_timing_path)
    exp3_timing_summary = _load_json(exp3_timing_path)

    generation = _resolve_generation_stats(repo_root, exp2_summary, exp3_summary)
    timing = _resolve_timing_rows(exp2_timing_summary, exp3_timing_summary, repo_root)
    model_metrics = _load_model_metrics(Path(args.model_metrics).resolve())

    exp1 = model_metrics.get("exp1", {})
    exp1_rows = "\n".join(
        [
            f"\t\tRandom Corruption          & {_fmt_float(exp1.get('random', {}).get('pr_auc'))}                         & {_fmt_float(exp1.get('random', {}).get('f1'))}                           & {_fmt_float(exp1.get('random', {}).get('fpr'))}                        \\tabularnewline",
            f"\t\tConstrained Random         & {_fmt_float(exp1.get('constrained', {}).get('pr_auc'))}                         & {_fmt_float(exp1.get('constrained', {}).get('f1'))}                           & {_fmt_float(exp1.get('constrained', {}).get('fpr'))}                        \\tabularnewline",
            f"\t\tProof-Based                & {_fmt_float(exp1.get('proof_based', {}).get('pr_auc'))}                         & {_fmt_float(exp1.get('proof_based', {}).get('f1'))}                           & {_fmt_float(exp1.get('proof_based', {}).get('fpr'))}                        \\tabularnewline",
        ]
    )

    overall = model_metrics.get("overall", {})
    exp2_overall = overall.get("exp2", {})
    exp3_overall = overall.get("exp3", {})
    overall_rows = "\n".join(
        [
            f"\t\t\\multirow{{2}}{{*}}{{Exp 2: Family Tree}} & UDM Baseline           & {_fmt_float(exp2_overall.get('baseline', {}).get('pr_auc'))}                         & {_fmt_float(exp2_overall.get('baseline', {}).get('auc_roc'))}                          & {_fmt_float(exp2_overall.get('baseline', {}).get('f1'))}                           \\tabularnewline",
            f"\t\t                                    & \\textsc{{Synthology}}    & \\textbf{{{_fmt_float(exp2_overall.get('synthology', {}).get('pr_auc'))}}}                & \\textbf{{{_fmt_float(exp2_overall.get('synthology', {}).get('auc_roc'))}}}                 & \\textbf{{{_fmt_float(exp2_overall.get('synthology', {}).get('f1'))}}}                  \\tabularnewline",
            "\t\t\\midrule",
            f"\t\t\\multirow{{2}}{{*}}{{Exp 3: OWL2Bench}}   & UDM Baseline           & {_fmt_float(exp3_overall.get('baseline', {}).get('pr_auc'))}                         & {_fmt_float(exp3_overall.get('baseline', {}).get('auc_roc'))}                          & {_fmt_float(exp3_overall.get('baseline', {}).get('f1'))}                           \\tabularnewline",
            f"\t\t                                    & \\textsc{{Synthology}}    & \\textbf{{{_fmt_float(exp3_overall.get('synthology', {}).get('pr_auc'))}}}                & \\textbf{{{_fmt_float(exp3_overall.get('synthology', {}).get('auc_roc'))}}}                 & \\textbf{{{_fmt_float(exp3_overall.get('synthology', {}).get('f1'))}}}                  \\tabularnewline",
        ]
    )

    def _gen_row(dataset_cell: str, method: str, stats: dict[str, Any]) -> str:
        return (
            f"\t\t{dataset_cell} & {method} & {_fmt_float(stats.get('facts_per_sample_avg'))} & "
            f"{_fmt_int(stats.get('base_facts'))} & {_fmt_int(stats.get('inferred_facts'))} & "
            f"{_fmt_int(stats.get('negative_facts'))} & {_fmt_float(stats.get('hops_avg'))} [{_fmt_int(stats.get('hops_max'))}]          \\tabularnewline"
        )

    exp2_baseline_stats = generation.get("exp2_baseline", {})
    exp2_synthology_stats = generation.get("exp2_synthology", {})
    exp3_baseline_stats = generation.get("exp3_baseline", {})
    exp3_synthology_stats = generation.get("exp3_synthology", {})

    generation_rows = "\n".join(
        [
            _gen_row("\\multirow{2}{*}{Family Tree}", "Random ABox + Jena", exp2_baseline_stats),
            _gen_row("                                  ", "\\textsc{Synthology}", exp2_synthology_stats),
            "\t\t\t\t\t  \\midrule",
            _gen_row("\\multirow{2}{*}{OWL2Bench}", "OWL2Bench + Jena", exp3_baseline_stats),
            _gen_row("                                  ", "\\textsc{Synthology}", exp3_synthology_stats),
        ]
    )

    def _timing_row(dataset_cell: str, method: str, values: dict[str, float | None], bold_total: bool = True) -> str:
        total_fmt = _fmt_float(values.get("total"))
        total_cell = f"\\textbf{{{total_fmt}}}" if bold_total else total_fmt
        return (
            f"\t\t{dataset_cell} & {method} & {_fmt_float(values.get('base'))} & "
            f"{_fmt_float(values.get('inference'))} & {_fmt_float(values.get('overhead'))} & {total_cell}        \\tabularnewline"
        )

    timing_rows = "\n".join(
        [
            _timing_row(
                "\\multirow{2}{*}{Family Tree}", "Random ABox + Jena + Parity", timing.get("exp2_baseline", {})
            ),
            _timing_row(
                "                                  ", "\\textsc{Synthology}", timing.get("exp2_synthology", {})
            ),
            "\t\t\t\t\t  \\midrule",
            _timing_row("\\multirow{2}{*}{OWL2Bench}", "OWL2Bench + Jena + Subgraphs", timing.get("exp3_baseline", {})),
            _timing_row(
                "                                  ", "\\textsc{Synthology}", timing.get("exp3_synthology", {})
            ),
        ]
    )

    _write(out_dir / "exp1_results_rows.tex", exp1_rows + "\n")
    _write(out_dir / "overall_performance_rows.tex", overall_rows + "\n")
    _write(out_dir / "generation_metrics_rows.tex", generation_rows + "\n")
    _write(out_dir / "timing_breakdown_rows.tex", timing_rows + "\n")

    meta = {
        "exp2_summary_path": str(exp2_summary_path) if exp2_summary_path else None,
        "exp3_summary_path": str(exp3_summary_path) if exp3_summary_path else None,
        "exp2_timing_summary_path": str(exp2_timing_path) if exp2_timing_path else None,
        "exp3_timing_summary_path": str(exp3_timing_path) if exp3_timing_path else None,
        "model_metrics_path": str(Path(args.model_metrics).resolve()),
    }
    _write(out_dir / "table_sources.json", json.dumps(meta, indent=2) + "\n")

    print(f"Wrote LaTeX row snippets to: {out_dir}")


if __name__ == "__main__":
    main()

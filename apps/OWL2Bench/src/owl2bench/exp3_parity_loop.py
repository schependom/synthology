"""Retry OWL2Bench baseline generation until deep/structural parity is reached."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from loguru import logger

from owl2bench.exp3_parity_report import extract_dataset_stats, load_synth_runtime_seconds


def _run_baseline_attempt(
    attempt_dir: Path,
    universities: int,
    seed: int,
    extra_overrides: list[str],
) -> None:
    cmd = [
        "uv",
        "run",
        "--package",
        "owl2bench",
        "python",
        "-m",
        "owl2bench.pipeline",
        f"dataset.universities=[{universities}]",
        f"dataset.output_dir={attempt_dir}",
        f"generator.seed={seed}",
    ] + extra_overrides

    subprocess.run(cmd, check=True)


def _within_tolerance(reference: int, observed: int, tolerance_pct: float) -> bool:
    if reference <= 0:
        return observed <= 0

    lower = math.floor(reference * (1.0 - tolerance_pct / 100.0))
    upper = math.ceil(reference * (1.0 + tolerance_pct / 100.0))
    return lower <= observed <= upper


def _pct_diff(reference: float, observed: float) -> float:
    if reference == 0:
        return 0.0 if observed == 0 else 100.0
    return abs(observed - reference) / abs(reference) * 100.0


def _evaluate_structural_parity(
    synth_stats: dict[str, Any],
    observed_stats: dict[str, Any],
    *,
    deep_count_mode: str,
    deep_count_tolerance_pct: float,
    node_tolerance_pct: float,
    edge_density_tolerance_pct: float,
    target_ratio_tolerance_pct: float,
    inferred_share_tolerance_pct: float,
) -> dict[str, Any]:
    synth_deep = int(synth_stats.get("deep_count", 0))
    observed_deep = int(observed_stats.get("deep_count", 0))

    if deep_count_mode == "exact":
        deep_pass = observed_deep == synth_deep
    else:
        deep_pass = _within_tolerance(synth_deep, observed_deep, deep_count_tolerance_pct)

    checks = {
        "deep_count": {
            "reference": synth_deep,
            "observed": observed_deep,
            "diff_pct": _pct_diff(float(synth_deep), float(observed_deep)),
            "tolerance_pct": 0.0 if deep_count_mode == "exact" else deep_count_tolerance_pct,
            "pass": deep_pass,
        },
        "node_count": {
            "reference": int(synth_stats.get("node_count", 0)),
            "observed": int(observed_stats.get("node_count", 0)),
            "tolerance_pct": node_tolerance_pct,
        },
        "edge_density": {
            "reference": float(synth_stats.get("edge_density", 0.0)),
            "observed": float(observed_stats.get("edge_density", 0.0)),
            "tolerance_pct": edge_density_tolerance_pct,
        },
        "pos_neg_ratio": {
            "reference": float(synth_stats.get("pos_neg_ratio", 0.0)),
            "observed": float(observed_stats.get("pos_neg_ratio", 0.0)),
            "tolerance_pct": target_ratio_tolerance_pct,
        },
        "inferred_positive_share": {
            "reference": float(synth_stats.get("inferred_positive_share", 0.0)),
            "observed": float(observed_stats.get("inferred_positive_share", 0.0)),
            "tolerance_pct": inferred_share_tolerance_pct,
        },
    }

    for metric_name in ("node_count", "edge_density", "pos_neg_ratio", "inferred_positive_share"):
        metric = checks[metric_name]
        metric["diff_pct"] = _pct_diff(metric["reference"], metric["observed"])
        metric["pass"] = metric["diff_pct"] <= metric["tolerance_pct"]

    all_passed = all(metric["pass"] for metric in checks.values())
    max_diff_pct = max(metric["diff_pct"] for metric in checks.values()) if checks else 0.0
    mean_diff_pct = sum(metric["diff_pct"] for metric in checks.values()) / len(checks) if checks else 0.0

    return {
        "checks": checks,
        "all_passed": all_passed,
        "parity_score": {
            "max_diff_pct": max_diff_pct,
            "mean_diff_pct": mean_diff_pct,
        },
    }


def _resolve_train_csvs(attempt_dir: Path) -> tuple[Path, Path]:
    targets_candidates = sorted(attempt_dir.glob("**/train/targets.csv"))
    if not targets_candidates:
        raise FileNotFoundError(f"Could not find train/targets.csv under {attempt_dir}")

    targets_csv = targets_candidates[0]
    facts_csv = targets_csv.parent / "facts.csv"
    if not facts_csv.exists():
        raise FileNotFoundError(f"Could not find sibling facts.csv for {targets_csv}")
    return targets_csv, facts_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp 3 parity loop using OWL2Bench baseline generation.")
    parser.add_argument("--universities", type=int, default=50)
    parser.add_argument("--synth-targets", required=True, help="Synthology train targets.csv used to derive K_deep")
    parser.add_argument("--synth-facts", required=True, help="Synthology train facts.csv used for structural metrics")
    parser.add_argument(
        "--synth-generation-metrics",
        default="",
        help="Optional Synthology generation metrics JSON to compute time-to-parity ratio",
    )
    parser.add_argument("--attempts-root", default="data/exp3/baseline/parity_runs")
    parser.add_argument("--min-deep-hops", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--deep-count-mode", choices=["exact", "tolerance"], default="exact")
    parser.add_argument("--deep-count-tolerance-pct", type=float, default=10.0)
    parser.add_argument("--node-tolerance-pct", type=float, default=10.0)
    parser.add_argument("--edge-density-tolerance-pct", type=float, default=15.0)
    parser.add_argument("--target-ratio-tolerance-pct", type=float, default=10.0)
    parser.add_argument("--inferred-share-tolerance-pct", type=float, default=10.0)
    parser.add_argument("--seed-start", type=int, default=23)
    parser.add_argument("--keep-failed-attempts", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override passed to owl2bench.pipeline (repeatable)",
    )
    args = parser.parse_args()

    synth_targets = Path(args.synth_targets)
    synth_facts = Path(args.synth_facts)
    attempts_root = Path(args.attempts_root)
    attempts_root.mkdir(parents=True, exist_ok=True)

    if not synth_targets.exists():
        raise FileNotFoundError(f"Synthology targets not found: {synth_targets}")
    if not synth_facts.exists():
        raise FileNotFoundError(f"Synthology facts not found: {synth_facts}")

    synth_stats = extract_dataset_stats(synth_targets, synth_facts, min_deep_hops=args.min_deep_hops)
    k_deep = int(synth_stats["deep_count"])

    synth_runtime_seconds = None
    if args.synth_generation_metrics:
        synth_runtime_seconds = load_synth_runtime_seconds(Path(args.synth_generation_metrics))

    logger.info("Exp3 parity loop started | K_deep={} | synth_hops={}", k_deep, synth_stats["hop_histogram"])

    attempt_summaries: list[dict[str, Any]] = []
    matched_attempt = None
    cumulative_attempt_seconds = 0.0

    for attempt_idx in range(1, args.max_attempts + 1):
        attempt_name = f"attempt_{attempt_idx:04d}"
        attempt_dir = attempts_root / attempt_name
        seed = args.seed_start + attempt_idx - 1

        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running {} with seed={} universities={}", attempt_name, seed, args.universities)
        attempt_start = time.perf_counter()
        _run_baseline_attempt(
            attempt_dir=attempt_dir,
            universities=args.universities,
            seed=seed,
            extra_overrides=args.override,
        )
        attempt_wall_seconds = time.perf_counter() - attempt_start
        cumulative_attempt_seconds += attempt_wall_seconds

        targets_csv, facts_csv = _resolve_train_csvs(attempt_dir)
        stats = extract_dataset_stats(targets_csv, facts_csv, min_deep_hops=args.min_deep_hops)

        structural = _evaluate_structural_parity(
            synth_stats,
            stats,
            deep_count_mode=args.deep_count_mode,
            deep_count_tolerance_pct=args.deep_count_tolerance_pct,
            node_tolerance_pct=args.node_tolerance_pct,
            edge_density_tolerance_pct=args.edge_density_tolerance_pct,
            target_ratio_tolerance_pct=args.target_ratio_tolerance_pct,
            inferred_share_tolerance_pct=args.inferred_share_tolerance_pct,
        )
        success = bool(structural["all_passed"])

        summary = {
            "attempt": attempt_name,
            "seed": seed,
            "targets_csv": str(targets_csv),
            "facts_csv": str(facts_csv),
            "deep_count": int(stats["deep_count"]),
            "hop_histogram": stats["hop_histogram"],
            "node_count": int(stats["node_count"]),
            "edge_density": float(stats["edge_density"]),
            "pos_neg_ratio": float(stats["pos_neg_ratio"]),
            "inferred_positive_share": float(stats["inferred_positive_share"]),
            "attempt_wall_seconds": attempt_wall_seconds,
            "cumulative_attempt_seconds": cumulative_attempt_seconds,
            "parity": structural,
            "success": success,
        }
        attempt_summaries.append(summary)

        logger.info(
            "{} complete | success={} | deep_count={} | node_count={} | edge_density={:.4f} | attempt_s={:.2f}",
            attempt_name,
            success,
            stats["deep_count"],
            stats["node_count"],
            stats["edge_density"],
            attempt_wall_seconds,
        )

        if success:
            matched_attempt = summary
            logger.success("Parity condition reached at {}", attempt_name)
            break

        if not args.keep_failed_attempts:
            shutil.rmtree(attempt_dir, ignore_errors=True)

    baseline_time_to_parity_seconds = (
        float(matched_attempt["cumulative_attempt_seconds"])
        if matched_attempt is not None
        else cumulative_attempt_seconds
    )
    baseline_vs_synth_time_ratio = None
    if synth_runtime_seconds is not None and synth_runtime_seconds > 0:
        baseline_vs_synth_time_ratio = baseline_time_to_parity_seconds / synth_runtime_seconds

    run_summary = {
        "universities": args.universities,
        "k_deep": k_deep,
        "synthology_stats": synth_stats,
        "min_deep_hops": args.min_deep_hops,
        "max_attempts": args.max_attempts,
        "deep_count_mode": args.deep_count_mode,
        "deep_count_tolerance_pct": args.deep_count_tolerance_pct,
        "node_tolerance_pct": args.node_tolerance_pct,
        "edge_density_tolerance_pct": args.edge_density_tolerance_pct,
        "target_ratio_tolerance_pct": args.target_ratio_tolerance_pct,
        "inferred_share_tolerance_pct": args.inferred_share_tolerance_pct,
        "synth_runtime_seconds": synth_runtime_seconds,
        "baseline_time_to_parity_seconds": baseline_time_to_parity_seconds,
        "baseline_vs_synth_time_ratio": baseline_vs_synth_time_ratio,
        "matched_attempt": matched_attempt,
        "attempts": attempt_summaries,
    }

    summary_path = attempts_root / "parity_loop_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)

    if matched_attempt is None:
        logger.warning(
            "Parity loop ended without structural parity after {} attempts | baseline_time_s={:.2f} | summary={}",
            args.max_attempts,
            baseline_time_to_parity_seconds,
            summary_path,
        )
    else:
        logger.success(
            "Parity loop finished | match={} | baseline_time_s={:.2f} | synth_time_s={} | ratio={} | summary={}",
            matched_attempt["attempt"],
            baseline_time_to_parity_seconds,
            synth_runtime_seconds,
            baseline_vs_synth_time_ratio,
            summary_path,
        )


if __name__ == "__main__":
    main()

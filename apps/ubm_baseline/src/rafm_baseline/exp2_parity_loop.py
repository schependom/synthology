"""Dedicated Exp 2 parity loop: retry UDM generation until deep-count parity target is reached."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger
from udm_baseline.exp2_parity_report import build_parity_report, extract_target_stats


def _run_baseline_attempt(
    attempt_dir: Path,
    seed: int,
    extra_overrides: list[str],
) -> None:
    cmd = [
        "uv",
        "run",
        "--package",
        "udm_baseline",
        "python",
        "-m",
        "udm_baseline.create_data",
        "--config-name=exp2_baseline",
        f"dataset.output_dir={attempt_dir}",
        f"dataset.seed={seed}",
    ] + extra_overrides

    subprocess.run(cmd, check=True)


def _within_tolerance(k_deep: int, observed: int, tolerance_pct: float) -> bool:
    if k_deep <= 0:
        return observed <= 0

    lower = math.floor(k_deep * (1.0 - tolerance_pct / 100.0))
    upper = math.ceil(k_deep * (1.0 + tolerance_pct / 100.0))
    return lower <= observed <= upper


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp 2 UDM parity loop with Jena materialization.")
    parser.add_argument(
        "--synth-targets",
        default="data/exp2/synthology/family_tree/train/targets.csv",
        help="Synthology train targets.csv used to derive K_deep",
    )
    parser.add_argument(
        "--attempts-root",
        default="data/exp2/baseline/parity_runs",
        help="Where attempt_* directories and summary are written",
    )
    parser.add_argument("--min-deep-hops", type=int, default=3, help="Minimum hop threshold for deep facts")
    parser.add_argument("--max-attempts", type=int, default=250, help="Maximum retry count")
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=10.0,
        help="Parity tolerance band (plus/minus percentage around K_deep)",
    )
    parser.add_argument("--seed-start", type=int, default=23, help="Initial dataset seed")
    parser.add_argument(
        "--keep-failed-attempts",
        action="store_true",
        help="Keep non-matching attempt directories (default removes them)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override passed to udm_baseline.create_data (repeatable)",
    )
    args = parser.parse_args()

    synth_targets = Path(args.synth_targets)
    attempts_root = Path(args.attempts_root)
    attempts_root.mkdir(parents=True, exist_ok=True)

    if not synth_targets.exists():
        raise FileNotFoundError(f"Synthology targets not found: {synth_targets}")

    synth_stats = extract_target_stats(synth_targets, min_deep_hops=args.min_deep_hops)
    k_deep = int(synth_stats["deep_count"])
    logger.info("Exp2 parity loop started | K_deep={} | synth_hops={}", k_deep, synth_stats["hop_histogram"])

    attempt_summaries: list[dict[str, Any]] = []
    matched_attempt = None

    for attempt_idx in range(1, args.max_attempts + 1):
        attempt_name = f"attempt_{attempt_idx:04d}"
        attempt_dir = attempts_root / attempt_name
        seed = args.seed_start + attempt_idx - 1

        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running {} with seed={}", attempt_name, seed)
        _run_baseline_attempt(attempt_dir=attempt_dir, seed=seed, extra_overrides=args.override)

        targets_csv = attempt_dir / "train" / "targets.csv"
        stats = extract_target_stats(targets_csv, min_deep_hops=args.min_deep_hops)
        deep_count = int(stats["deep_count"])

        hit_ge_k = deep_count >= k_deep
        hit_tolerance = _within_tolerance(k_deep, deep_count, args.tolerance_pct)
        success = hit_ge_k or hit_tolerance

        summary = {
            "attempt": attempt_name,
            "seed": seed,
            "targets_csv": str(targets_csv),
            "deep_count": deep_count,
            "hop_histogram": stats["hop_histogram"],
            "hit_ge_k": hit_ge_k,
            "hit_tolerance": hit_tolerance,
            "success": success,
        }
        attempt_summaries.append(summary)

        logger.info(
            "{} complete | deep_count={} | hit_ge_k={} | hit_tolerance={} | hop_hist={}",
            attempt_name,
            deep_count,
            hit_ge_k,
            hit_tolerance,
            stats["hop_histogram"],
        )

        if success:
            matched_attempt = summary
            logger.success("Parity condition reached at {}", attempt_name)
            break

        if not args.keep_failed_attempts:
            shutil.rmtree(attempt_dir, ignore_errors=True)

    report = build_parity_report(
        synth_targets=synth_targets,
        attempts_root=attempts_root,
        min_deep_hops=args.min_deep_hops,
    )

    run_summary = {
        "k_deep": k_deep,
        "min_deep_hops": args.min_deep_hops,
        "max_attempts": args.max_attempts,
        "tolerance_pct": args.tolerance_pct,
        "matched_attempt": matched_attempt,
        "attempts": attempt_summaries,
        "report": report,
    }

    summary_path = attempts_root / "parity_loop_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)

    if matched_attempt is None:
        logger.warning(
            "Parity loop ended without a match after {} attempts | summary={}",
            args.max_attempts,
            summary_path,
        )
    else:
        logger.success("Parity loop finished | match={} | summary={}", matched_attempt["attempt"], summary_path)


if __name__ == "__main__":
    main()

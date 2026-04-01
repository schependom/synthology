from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from loguru import logger
from rdflib import Graph


def _read_targets_stats(path: Path) -> dict[str, Any]:
    base_pos = 0
    inferred_pos = 0
    hops = Counter()

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("label", "1")).strip() != "1":
                continue
            row_type = str(row.get("type", "")).strip().lower()
            hop = str(int(float(row.get("hops", 0))))
            hops[hop] += 1

            if row_type == "base_fact":
                base_pos += 1
            elif row_type.startswith("inf") or row_type == "inferred":
                inferred_pos += 1

    return {
        "base_positive": base_pos,
        "inferred_positive": inferred_pos,
        "hops": dict(sorted(hops.items(), key=lambda x: int(x[0]))),
    }


def _read_nt_count(path: Path) -> int:
    graph = Graph()
    fmt = "nt" if path.suffix == ".nt" else "xml"
    graph.parse(path, format=fmt)
    return len(graph)


def _plot_base_vs_inferred(synth_stats: dict[str, Any], baseline_stats: dict[str, Any], out_path: Path) -> None:
    labels = ["synthology", "baseline"]
    base_vals = [synth_stats["base_positive"], baseline_stats["base_positive"]]
    inf_vals = [synth_stats["inferred_positive"], baseline_stats["inferred_positive"]]

    x = [0, 1]
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([v - width / 2 for v in x], base_vals, width=width, label="base positives")
    ax.bar([v + width / 2 for v in x], inf_vals, width=width, label="inferred positives")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Train Targets: Base vs Inferred Positives")
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hops(synth_stats: dict[str, Any], baseline_stats: dict[str, Any], out_path: Path) -> None:
    all_hops = sorted(set(synth_stats["hops"].keys()) | set(baseline_stats["hops"].keys()), key=lambda x: int(x))
    x = list(range(len(all_hops)))
    synth_vals = [synth_stats["hops"].get(h, 0) for h in all_hops]
    base_vals = [baseline_stats["hops"].get(h, 0) for h in all_hops]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, synth_vals, marker="o", label="synthology")
    ax.plot(x, base_vals, marker="o", label="baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(all_hops)
    ax.set_xlabel("hops")
    ax.set_ylabel("positive target count")
    ax.set_title("Hop Distribution (Train Positives)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_parity_attempts(parity_summary: dict[str, Any], out_path: Path) -> None:
    attempts = parity_summary.get("attempts", [])
    if not attempts:
        return

    xs = list(range(1, len(attempts) + 1))
    ys = [int(a.get("deep_count", 0)) for a in attempts]
    k_deep = int(parity_summary.get("k_deep", 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, marker="o", label="baseline deep_count")
    ax.axhline(y=k_deep, color="red", linestyle="--", label=f"K_deep={k_deep}")
    ax.set_xlabel("attempt")
    ax.set_ylabel("deep inferred count")
    ax.set_title("Parity Attempts: Deep Inferred Count")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_exp3_base_vs_inferred(base_count: int, inferred_count: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["base facts", "inferred facts"], [base_count, inferred_count])
    ax.set_title("Exp3 OWL2Bench: Base vs Inferred (Jena)")
    ax.set_ylabel("triple count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_exp3_hops(exp3_stats: dict[str, Any], out_path: Path) -> None:
    hops = exp3_stats.get("hops", {})
    if not hops:
        return
    labels = sorted(hops.keys(), key=lambda x: int(x))
    values = [hops[h] for h in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title("Exp3 Hop Distribution (Train Positives)")
    ax.set_xlabel("hops")
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready plots for Exp2/Exp3 baseline analysis.")
    parser.add_argument("--exp2-synth-targets", default="data/exp2/synthology/family_tree/train/targets.csv")
    parser.add_argument("--exp2-baseline-targets", default="")
    parser.add_argument("--exp2-parity-summary", default="data/exp2/baseline/parity_runs/parity_loop_summary.json")
    parser.add_argument("--exp3-targets", default="")
    parser.add_argument("--exp3-abox", default="")
    parser.add_argument("--exp3-inferred", default="")
    parser.add_argument("--out-dir", default="reports/paper")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_targets = Path(args.exp2_synth_targets)
    if not synth_targets.exists():
        raise FileNotFoundError(f"Missing synth targets: {synth_targets}")

    parity_summary_path = Path(args.exp2_parity_summary)
    if not parity_summary_path.exists():
        raise FileNotFoundError(f"Missing parity summary: {parity_summary_path}")

    with open(parity_summary_path, encoding="utf-8") as handle:
        parity_summary = json.load(handle)

    baseline_targets = Path(args.exp2_baseline_targets) if args.exp2_baseline_targets else None
    if baseline_targets is None:
        matched = parity_summary.get("matched_attempt") or {}
        if matched.get("targets_csv"):
            baseline_targets = Path(matched["targets_csv"])

    if baseline_targets is None or not baseline_targets.exists():
        raise FileNotFoundError(
            "Baseline targets could not be resolved. Provide --exp2-baseline-targets or ensure matched_attempt exists."
        )

    synth_stats = _read_targets_stats(synth_targets)
    baseline_stats = _read_targets_stats(baseline_targets)

    _plot_base_vs_inferred(synth_stats, baseline_stats, out_dir / "exp2_base_vs_inferred.png")
    _plot_hops(synth_stats, baseline_stats, out_dir / "exp2_hops_distribution.png")
    _plot_parity_attempts(parity_summary, out_dir / "exp2_parity_attempts.png")

    exp3_summary = None
    if args.exp3_abox and args.exp3_inferred:
        abox_path = Path(args.exp3_abox)
        inferred_path = Path(args.exp3_inferred)
        if abox_path.exists() and inferred_path.exists():
            base_count = _read_nt_count(abox_path)
            inferred_count = _read_nt_count(inferred_path)
            _plot_exp3_base_vs_inferred(base_count, inferred_count, out_dir / "exp3_base_vs_inferred.png")
            exp3_summary = {
                "abox": str(abox_path),
                "inferred": str(inferred_path),
                "base_count": base_count,
                "inferred_count": inferred_count,
            }

    if args.exp3_targets:
        exp3_targets = Path(args.exp3_targets)
        if exp3_targets.exists():
            exp3_targets_stats = _read_targets_stats(exp3_targets)
            _plot_exp3_hops(exp3_targets_stats, out_dir / "exp3_hops_distribution.png")
            if exp3_summary is None:
                exp3_summary = {}
            exp3_summary["targets_csv"] = str(exp3_targets)
            exp3_summary["targets_stats"] = exp3_targets_stats

    matched_attempt = (parity_summary.get("matched_attempt") or {}).get("attempt")
    tries_before_match = None
    if isinstance(matched_attempt, str):
        match = re.search(r"(\d+)$", matched_attempt)
        if match:
            tries_before_match = int(match.group(1))

    summary = {
        "exp2": {
            "synth_targets": str(synth_targets),
            "baseline_targets": str(baseline_targets),
            "synth_stats": synth_stats,
            "baseline_stats": baseline_stats,
            "matched_attempt": matched_attempt,
            "tries_before_match": tries_before_match,
        },
        "exp3": exp3_summary,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    logger.info("Paper plots generated at {}", out_dir)


if __name__ == "__main__":
    main()

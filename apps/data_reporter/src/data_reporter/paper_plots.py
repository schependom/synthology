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
from loguru import logger
from rdflib import Graph


def _configure_plot_style() -> dict[str, bool]:
    """Configure publication-oriented plotting style.

    If a LaTeX executable is available, matplotlib text rendering switches to
    LaTeX for paper-ready typography. The function returns metadata about the
    selected rendering mode for reporting.
    """

    latex_available = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "text.usetex": latex_available,
        }
    )
    return {"latex_enabled": latex_available}


def _save_figure(fig: Any, out_path: Path) -> None:
    """Save both PNG and PDF variants for report and paper integration."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


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
    _save_figure(fig, out_path)
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
    _save_figure(fig, out_path)
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
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_exp3_base_vs_inferred(base_count: int, inferred_count: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["base facts", "inferred facts"], [base_count, inferred_count])
    ax.set_title("Exp3 OWL2Bench: Base vs Inferred (Jena)")
    ax.set_ylabel("triple count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_exp3_hops(exp3_stats: dict[str, Any], out_path: Path) -> None:
    hops = exp3_stats.get("hops", {})
    if not hops:
        return
    labels = sorted(hops.keys(), key=lambda x: int(x))
    values = [hops[h] for h in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title(
        "Exp3 Hop Distribution (Train Positives)\nNote: Baseline depths are obscured (1-hop) by single-pass Jena. Synthology natively provides deep-hop metadata.",
        fontsize=10,
    )
    ax.set_xlabel("hops")
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _hop_bucket_counts(stats: dict[str, Any]) -> list[int]:
    """Return hop buckets [0, 1, 2, 3+] for compact comparison charts."""

    buckets = [0, 0, 0, 0]
    for hop_str, count in stats.get("hops", {}).items():
        hop = int(hop_str)
        if hop <= 0:
            buckets[0] += int(count)
        elif hop == 1:
            buckets[1] += int(count)
        elif hop == 2:
            buckets[2] += int(count)
        else:
            buckets[3] += int(count)
    return buckets


def _plot_small_density(
    domain: str, synth_stats: dict[str, Any], baseline_stats: dict[str, Any], out_path: Path
) -> None:
    """Compact chart to visualize KG density between methods."""

    labels = ["synthology", "udm_baseline"]
    total_pos = [
        int(synth_stats["base_positive"]) + int(synth_stats["inferred_positive"]),
        int(baseline_stats["base_positive"]) + int(baseline_stats["inferred_positive"]),
    ]
    inferred_ratio = [
        float(synth_stats["inferred_positive"]) / max(1, int(synth_stats["base_positive"])),
        float(baseline_stats["inferred_positive"]) / max(1, int(baseline_stats["base_positive"])),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))

    ax1.bar(labels, total_pos, color=["#1f77b4", "#7f7f7f"])
    ax1.set_title("positive targets")
    ax1.set_ylabel("count")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2.bar(labels, inferred_ratio, color=["#1f77b4", "#7f7f7f"])
    ax2.set_title("inferred/base ratio")
    ax2.set_ylabel("ratio")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(f"{domain}: KG Density (Compact)", fontsize=11)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_small_multihop(
    domain: str, synth_stats: dict[str, Any], baseline_stats: dict[str, Any], out_path: Path
) -> None:
    """Compact stacked bars to compare hop-depth richness."""

    synth_buckets = _hop_bucket_counts(synth_stats)
    baseline_buckets = _hop_bucket_counts(baseline_stats)
    bucket_labels = ["0", "1", "2", "3+"]
    colors = ["#c7c7c7", "#9ecae1", "#6baed6", "#2171b5"]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x_labels = ["synthology", "udm_baseline"]

    bottom = [0, 0]
    for idx, bucket_label in enumerate(bucket_labels):
        values = [synth_buckets[idx], baseline_buckets[idx]]
        ax.bar(x_labels, values, bottom=bottom, label=f"hop={bucket_label}", color=colors[idx])
        bottom = [bottom[0] + values[0], bottom[1] + values[1]]

    ax.set_title(f"{domain}: Multi-hop Richness (Compact)")
    ax.set_ylabel("positive target count")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready plots for Exp2/Exp3 baseline analysis.")
    parser.add_argument("--exp2-synth-targets", default="data/exp2/synthology/family_tree/train/targets.csv")
    parser.add_argument("--exp2-baseline-targets", default="")
    parser.add_argument("--exp2-parity-summary", default="data/exp2/baseline/parity_runs/parity_loop_summary.json")
    parser.add_argument("--exp3-targets", default="")
    parser.add_argument("--exp3-synth-targets", default="")
    parser.add_argument("--exp3-baseline-targets", default="")
    parser.add_argument("--exp3-abox", default="")
    parser.add_argument("--exp3-inferred", default="")
    parser.add_argument("--out-dir", default="reports/paper")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style_info = _configure_plot_style()

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

    # Compact paper graphs for family-tree baseline vs synthology.
    _plot_small_density("Family Tree", synth_stats, baseline_stats, out_dir / "family_tree_density_small.png")
    _plot_small_multihop(
        "Family Tree",
        synth_stats,
        baseline_stats,
        out_dir / "family_tree_multihop_small.png",
    )

    exp3_pair_summary = None
    if args.exp3_synth_targets and args.exp3_baseline_targets:
        exp3_synth_targets = Path(args.exp3_synth_targets)
        exp3_baseline_targets = Path(args.exp3_baseline_targets)
        if exp3_synth_targets.exists() and exp3_baseline_targets.exists():
            exp3_synth_stats = _read_targets_stats(exp3_synth_targets)
            exp3_baseline_stats = _read_targets_stats(exp3_baseline_targets)
            _plot_small_density(
                "OWL2Bench",
                exp3_synth_stats,
                exp3_baseline_stats,
                out_dir / "owl2bench_density_small.png",
            )
            _plot_small_multihop(
                "OWL2Bench",
                exp3_synth_stats,
                exp3_baseline_stats,
                out_dir / "owl2bench_multihop_small.png",
            )
            exp3_pair_summary = {
                "synth_targets": str(exp3_synth_targets),
                "baseline_targets": str(exp3_baseline_targets),
                "synth_stats": exp3_synth_stats,
                "baseline_stats": exp3_baseline_stats,
            }

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
        "plot_style": style_info,
        "exp2": {
            "synth_targets": str(synth_targets),
            "baseline_targets": str(baseline_targets),
            "synth_stats": synth_stats,
            "baseline_stats": baseline_stats,
            "matched_attempt": matched_attempt,
            "tries_before_match": tries_before_match,
            "time_to_parity": {
                "synth_runtime_seconds": parity_summary.get("synth_runtime_seconds"),
                "baseline_time_to_parity_seconds": parity_summary.get("baseline_time_to_parity_seconds"),
                "baseline_vs_synth_time_ratio": parity_summary.get("baseline_vs_synth_time_ratio"),
            },
        },
        "exp3": exp3_summary,
        "exp3_pair": exp3_pair_summary,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    logger.info("Paper plots generated at {}", out_dir)


if __name__ == "__main__":
    main()

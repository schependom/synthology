from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

import hydra
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _configure_plot_style() -> dict[str, bool]:
    """Configure consistent plotting style with optional LaTeX text rendering."""

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


def _save_figure(fig, out_path: Path) -> None:
    """Save a figure as both PNG and PDF."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


@dataclass
class SplitStats:
    split: str
    facts_count: int = 0
    targets_count: int = 0
    positives: int = 0
    negatives: int = 0
    unique_predicates: int = 0
    unique_entities: int = 0
    unique_sample_ids: int = 0

    predicate_counts_facts: Counter = field(default_factory=Counter)
    predicate_counts_targets: Counter = field(default_factory=Counter)
    predicate_counts_by_fact_group: Counter = field(default_factory=Counter)
    type_counts: Counter = field(default_factory=Counter)
    corruption_counts: Counter = field(default_factory=Counter)
    hops_counts: Counter = field(default_factory=Counter)


@dataclass
class MethodStats:
    name: str
    path: str
    split_stats: Dict[str, SplitStats] = field(default_factory=dict)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _positive_fact_group(type_value: str, label: int) -> Optional[str]:
    """Maps positive target row type to a broad fact group for plotting."""
    if label != 1:
        return None

    t = (type_value or "").strip().lower()
    if t == "base_fact":
        return "base"
    if t == "inf_intermediate":
        return "intermediate"
    if t in {"inf_root", "inferred"}:
        return "inferred"
    return None


def _collect_split_stats(method_name: str, method_path: Path, split: str) -> SplitStats:
    split_dir = method_path / split
    facts_path = split_dir / "facts.csv"
    targets_path = split_dir / "targets.csv"

    facts_rows = _read_csv_rows(facts_path)
    targets_rows = _read_csv_rows(targets_path)

    st = SplitStats(split=split)

    st.facts_count = len(facts_rows)
    st.targets_count = len(targets_rows)

    sample_ids = set()
    entities = set()

    for r in facts_rows:
        sample_ids.add(r.get("sample_id", ""))
        s = r.get("subject", "")
        o = r.get("object", "")
        p = r.get("predicate", "")
        if s:
            entities.add(s)
        if p != "rdf:type" and o:
            entities.add(o)
        st.predicate_counts_facts[p] += 1

    for r in targets_rows:
        sample_ids.add(r.get("sample_id", ""))
        s = r.get("subject", "")
        o = r.get("object", "")
        p = r.get("predicate", "")
        if s:
            entities.add(s)
        if p != "rdf:type" and o:
            entities.add(o)

        label = _safe_int(r.get("label", "1"), 1)
        if label == 1:
            st.positives += 1
        else:
            st.negatives += 1

        st.predicate_counts_targets[p] += 1
        row_type = r.get("type", "unknown")
        st.type_counts[row_type] += 1
        st.corruption_counts[r.get("corruption_method", "unknown")] += 1
        st.hops_counts[str(_safe_int(r.get("hops", "0"), 0))] += 1

        fact_group = _positive_fact_group(row_type, label)
        if fact_group is not None:
            st.predicate_counts_by_fact_group[(p, fact_group)] += 1

    st.unique_sample_ids = len(sample_ids)
    st.unique_entities = len(entities)
    st.unique_predicates = len(st.predicate_counts_targets)

    logger.info(
        f"[{method_name}][{split}] facts={st.facts_count}, targets={st.targets_count}, "
        f"pos={st.positives}, neg={st.negatives}, samples={st.unique_sample_ids}"
    )

    return st


def _sum_counters(stats: Iterable[SplitStats], attr: str) -> Counter:
    total: Counter = Counter()
    for st in stats:
        total += getattr(st, attr)
    return total


def _method_summary(method: MethodStats) -> Dict:
    split_values = list(method.split_stats.values())
    facts_total = sum(s.facts_count for s in split_values)
    targets_total = sum(s.targets_count for s in split_values)
    positives_total = sum(s.positives for s in split_values)
    negatives_total = sum(s.negatives for s in split_values)

    type_counts = _sum_counters(split_values, "type_counts")
    corruption_counts = _sum_counters(split_values, "corruption_counts")
    hops_counts = _sum_counters(split_values, "hops_counts")
    predicate_counts = _sum_counters(split_values, "predicate_counts_targets")
    predicate_fact_group_counts = _sum_counters(split_values, "predicate_counts_by_fact_group")

    predicate_fact_group_nested: Dict[str, Dict[str, int]] = {}
    for (predicate, group), count in predicate_fact_group_counts.items():
        predicate_fact_group_nested.setdefault(predicate, {})[group] = int(count)

    pos_neg_ratio = (positives_total / negatives_total) if negatives_total > 0 else None

    return {
        "method": method.name,
        "path": method.path,
        "facts_total": facts_total,
        "targets_total": targets_total,
        "positives_total": positives_total,
        "negatives_total": negatives_total,
        "positive_negative_ratio": pos_neg_ratio,
        "avg_targets_per_sample": mean(
            s.targets_count / s.unique_sample_ids for s in split_values if s.unique_sample_ids
        ),
        "avg_facts_per_sample": mean(s.facts_count / s.unique_sample_ids for s in split_values if s.unique_sample_ids),
        "type_counts": dict(type_counts),
        "corruption_counts": dict(corruption_counts),
        "hops_counts": dict(hops_counts),
        "top_predicates": dict(predicate_counts.most_common(30)),
        "predicate_fact_group_counts": predicate_fact_group_nested,
        "splits": {
            s.split: {
                "facts_count": s.facts_count,
                "targets_count": s.targets_count,
                "positives": s.positives,
                "negatives": s.negatives,
                "unique_entities": s.unique_entities,
                "unique_predicates": s.unique_predicates,
                "unique_sample_ids": s.unique_sample_ids,
            }
            for s in split_values
        },
    }


def _write_csv_summary(summary: List[Dict], out_dir: Path) -> None:
    path = out_dir / "method_summary.csv"
    fieldnames = [
        "method",
        "path",
        "facts_total",
        "targets_total",
        "positives_total",
        "negatives_total",
        "positive_negative_ratio",
        "avg_facts_per_sample",
        "avg_targets_per_sample",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _plot_bar(values: Dict[str, float], title: str, ylabel: str, out_path: Path) -> None:
    labels = list(values.keys())
    data = [values[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, data)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_type_distribution(summary: List[Dict], out_path: Path) -> None:
    all_types = sorted({t for m in summary for t in m["type_counts"].keys()})
    methods = [m["method"] for m in summary]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = [0.0] * len(methods)

    for t in all_types:
        vals = [float(m["type_counts"].get(t, 0)) for m in summary]
        ax.bar(methods, vals, bottom=bottom, label=t)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_title("Target Type Distribution by Method")
    ax.set_ylabel("Count")
    ax.set_xlabel("Method")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_hops_distribution(summary: List[Dict], out_path: Path) -> None:
    hop_values = sorted({h for m in summary for h in m["hops_counts"].keys()}, key=lambda x: int(x))
    methods = [m["method"] for m in summary]

    x = list(range(len(hop_values)))
    fig, ax = plt.subplots(figsize=(12, 6))

    for method in methods:
        item = next(m for m in summary if m["method"] == method)
        y = [item["hops_counts"].get(h, 0) for h in hop_values]
        ax.plot(x, y, marker="o", label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(hop_values)
    ax.set_title("Hop Distribution by Method")
    ax.set_xlabel("Hops")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_top_predicates_per_method(summary: List[Dict], out_dir: Path, top_k: int) -> None:
    for m in summary:
        top = list(m["top_predicates"].items())[:top_k]
        if not top:
            continue
        labels = [k for k, _ in top]
        vals = [v for _, v in top]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(labels[::-1], vals[::-1])
        ax.set_title(f"Top {top_k} Predicates: {m['method']}")
        ax.set_xlabel("Count")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        _save_figure(fig, out_dir / f"top_predicates_{m['method'].lower()}.png")
        plt.close(fig)


def _plot_predicate_distribution_by_fact_group(summary: List[Dict], out_dir: Path) -> None:
    fact_groups = ["base", "inferred", "intermediate"]

    for m in summary:
        nested_counts: Dict[str, Dict[str, int]] = m.get("predicate_fact_group_counts", {})
        predicates = sorted(nested_counts.keys())
        if not predicates:
            continue

        visible_groups = [g for g in fact_groups if any(nested_counts[p].get(g, 0) > 0 for p in predicates)]
        if not visible_groups:
            continue

        x = list(range(len(predicates)))
        width = 0.8 / len(visible_groups)
        fig_width = max(12, 0.35 * len(predicates) + 6)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        for i, group in enumerate(visible_groups):
            offset = (i - (len(visible_groups) - 1) / 2) * width
            y = [nested_counts[p].get(group, 0) for p in predicates]
            ax.bar([xi + offset for xi in x], y, width=width, label=group)

        ax.set_xticks(x)
        ax.set_xticklabels(predicates, rotation=90)
        ax.set_title(f"Predicate Distribution by Fact Group: {m['method']}")
        ax.set_xlabel("Predicate")
        ax.set_ylabel("Count")
        ax.legend(title="Fact Group")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        _save_figure(fig, out_dir / f"predicate_distribution_by_fact_group_{m['method'].lower()}.png")
        plt.close(fig)


def _write_markdown_report(summary: List[Dict], out_dir: Path) -> None:
    md_path = out_dir / "report.md"

    lines = [
        "# Dataset Comparison Report",
        "",
        "## Method Summary",
        "",
        "| Method | Facts Total | Targets Total | Positives | Negatives | Pos/Neg Ratio | Avg Facts/Sample | Avg Targets/Sample |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for m in summary:
        ratio = m["positive_negative_ratio"]
        ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
        lines.append(
            f"| {m['method']} | {m['facts_total']} | {m['targets_total']} | {m['positives_total']} | "
            f"{m['negatives_total']} | {ratio_str} | {m['avg_facts_per_sample']:.2f} | {m['avg_targets_per_sample']:.2f} |"
        )

    lines.append("")
    lines.append("## Split-Level Stats")
    lines.append("")

    for m in summary:
        lines.append(f"### {m['method']}")
        lines.append("")
        lines.append(
            "| Split | Facts | Targets | Positives | Negatives | Unique Entities | Unique Predicates | Samples |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for split_name, split in m["splits"].items():
            lines.append(
                f"| {split_name} | {split['facts_count']} | {split['targets_count']} | {split['positives']} | "
                f"{split['negatives']} | {split['unique_entities']} | {split['unique_predicates']} | {split['unique_sample_ids']} |"
            )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _generate_plots(summary: List[Dict], out_dir: Path, top_k_predicates: int) -> None:
    _plot_bar(
        {m["method"]: m["targets_total"] for m in summary},
        "Total Targets by Method",
        "Targets",
        out_dir / "targets_total_by_method.png",
    )
    _plot_bar(
        {m["method"]: m["facts_total"] for m in summary},
        "Total Base Facts by Method",
        "Facts",
        out_dir / "facts_total_by_method.png",
    )
    _plot_bar(
        {m["method"]: m["avg_targets_per_sample"] for m in summary},
        "Average Targets per Sample",
        "Targets/Sample",
        out_dir / "avg_targets_per_sample.png",
    )
    _plot_type_distribution(summary, out_dir / "type_distribution_stacked.png")
    _plot_hops_distribution(summary, out_dir / "hops_distribution.png")
    _plot_predicate_distribution_by_fact_group(summary, out_dir)
    _plot_top_predicates_per_method(summary, out_dir, top_k_predicates)


def _analyze_method(name: str, path: Path, splits: List[str]) -> MethodStats:
    method = MethodStats(name=name, path=str(path))

    for split in splits:
        method.split_stats[split] = _collect_split_stats(name, path, split)

    return method


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/data_reporter", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=cfg.logging.level, colorize=True)
    logger.info("Running data reporter with config:\n" + OmegaConf.to_yaml(cfg))

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style_info = _configure_plot_style()

    methods: List[MethodStats] = []

    for method_cfg in cfg.methods:
        method_name = str(method_cfg.name)
        method_path = Path(str(method_cfg.path))
        methods.append(_analyze_method(method_name, method_path, list(cfg.splits)))

    summary = [_method_summary(m) for m in methods]

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"methods": summary, "plot_style": style_info}, f, indent=2)

    _write_csv_summary(summary, out_dir)
    _write_markdown_report(summary, out_dir)

    if cfg.output.save_plots:
        _generate_plots(summary, out_dir, int(cfg.output.top_k_predicates))

    logger.success(f"Saved reports to {out_dir}")


if __name__ == "__main__":
    main()

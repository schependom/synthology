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
from matplotlib.patches import Patch
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _configure_plot_style() -> dict[str, bool]:
    """Configure consistent plotting style with optional LaTeX text rendering."""

    disable_latex = os.environ.get("SYNTHOLOGY_DISABLE_LATEX_PLOTS", "").strip().lower() in {"1", "true", "yes"}
    latex_available = (shutil.which("latex") is not None) and not disable_latex
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


def _hop_bucket_counts(hops_counts: Counter) -> Dict[str, int]:
    d1 = 0
    d2 = 0
    d3p = 0
    for hop_key, count in hops_counts.items():
        hop = _safe_int(str(hop_key), 0)
        if hop <= 1:
            d1 += int(count)
        elif hop == 2:
            d2 += int(count)
        else:
            d3p += int(count)
    return {"d1": d1, "d2": d2, "d3p": d3p}


def _method_summary(method: MethodStats) -> Dict:
    split_values = list(method.split_stats.values())
    facts_total = sum(s.facts_count for s in split_values)
    targets_total = sum(s.targets_count for s in split_values)
    positives_total = sum(s.positives for s in split_values)
    negatives_total = sum(s.negatives for s in split_values)

    type_counts = _sum_counters(split_values, "type_counts")
    corruption_counts = _sum_counters(split_values, "corruption_counts")
    hops_counts = _sum_counters(split_values, "hops_counts")
    hop_buckets = _hop_bucket_counts(hops_counts)
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
        "hop_buckets": hop_buckets,
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


def _write_hops_csv(summary: List[Dict], out_dir: Path) -> None:
    path = out_dir / "hops_by_method.csv"
    fieldnames = ["method", "hop", "count", "share"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            hops = {int(k): int(v) for k, v in row.get("hops_counts", {}).items()}
            total = sum(hops.values())
            for hop in sorted(hops.keys()):
                count = hops[hop]
                share = (count / total) if total else 0.0
                writer.writerow(
                    {
                        "method": row["method"],
                        "hop": hop,
                        "count": count,
                        "share": f"{share:.8f}",
                    }
                )


def _write_hop_bucket_csv(summary: List[Dict], out_dir: Path) -> None:
    path = out_dir / "hop_buckets_by_method.csv"
    fieldnames = ["method", "bucket", "count", "share"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            buckets = row.get("hop_buckets", {})
            total = int(sum(int(v) for v in buckets.values()))
            for bucket in ("d1", "d2", "d3p"):
                count = int(buckets.get(bucket, 0))
                share = (count / total) if total else 0.0
                writer.writerow(
                    {
                        "method": row["method"],
                        "bucket": bucket,
                        "count": count,
                        "share": f"{share:.8f}",
                    }
                )


def _write_predicate_fact_group_csv(summary: List[Dict], out_dir: Path) -> None:
    path = out_dir / "predicate_fact_group_counts.csv"
    fieldnames = ["method", "predicate", "fact_group", "count"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            nested_counts: Dict[str, Dict[str, int]] = row.get("predicate_fact_group_counts", {})
            for predicate in sorted(nested_counts.keys()):
                groups = nested_counts[predicate]
                for fact_group in sorted(groups.keys()):
                    writer.writerow(
                        {
                            "method": row["method"],
                            "predicate": predicate,
                            "fact_group": fact_group,
                            "count": int(groups[fact_group]),
                        }
                    )


def _write_inferred_predicates_csv(summary: List[Dict], out_dir: Path, ignore_predicates: Optional[List[str]] = None) -> None:
    path = out_dir / "inferred_predicates_by_method.csv"
    fieldnames = ["predicate", "method", "count"]
    ignored = _normalize_predicate_filter(ignore_predicates)

    methods = [m["method"] for m in summary]
    all_predicates = set()
    inferred_by_method: Dict[str, Dict[str, int]] = {}

    for m in summary:
        nested_counts: Dict[str, Dict[str, int]] = m.get("predicate_fact_group_counts", {})
        inferred_counts: Dict[str, int] = {}
        for predicate, groups in nested_counts.items():
            if str(predicate).strip().lower() in ignored:
                continue
            inferred_count = int(groups.get("inferred", 0))
            if inferred_count > 0:
                inferred_counts[predicate] = inferred_count
                all_predicates.add(predicate)
        inferred_by_method[m["method"]] = inferred_counts

    ordered_predicates = sorted(
        all_predicates,
        key=lambda p: sum(inferred_by_method.get(method, {}).get(p, 0) for method in methods),
        reverse=True,
    )

    rows: List[Dict[str, str]] = []
    for predicate in ordered_predicates:
        for method in methods:
            rows.append(
                {
                    "predicate": predicate,
                    "method": method,
                    "count": str(inferred_by_method.get(method, {}).get(predicate, 0)),
                }
            )

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_derived_predicates_csv(summary: List[Dict], out_dir: Path, ignore_predicates: Optional[List[str]] = None) -> None:
    path = out_dir / "derived_predicates_by_method.csv"
    fieldnames = ["predicate", "method", "count"]
    ignored = _normalize_predicate_filter(ignore_predicates)

    methods = [m["method"] for m in summary]
    all_predicates = set()
    derived_by_method: Dict[str, Dict[str, int]] = {}

    for m in summary:
        nested_counts: Dict[str, Dict[str, int]] = m.get("predicate_fact_group_counts", {})
        derived_counts: Dict[str, int] = {}
        for predicate, groups in nested_counts.items():
            if str(predicate).strip().lower() in ignored:
                continue
            derived_count = int(groups.get("inferred", 0)) + int(groups.get("intermediate", 0))
            if derived_count > 0:
                derived_counts[predicate] = derived_count
                all_predicates.add(predicate)
        derived_by_method[m["method"]] = derived_counts

    ordered_predicates = sorted(
        all_predicates,
        key=lambda p: sum(derived_by_method.get(method, {}).get(p, 0) for method in methods),
        reverse=True,
    )

    rows: List[Dict[str, str]] = []
    for predicate in ordered_predicates:
        for method in methods:
            rows.append(
                {
                    "predicate": predicate,
                    "method": method,
                    "count": str(derived_by_method.get(method, {}).get(predicate, 0)),
                }
            )

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_missing_inferred_predicates_csv(
    summary: List[Dict], out_dir: Path, ignore_predicates: Optional[List[str]] = None
) -> None:
    path = out_dir / "missing_inferred_predicates_by_method.csv"
    fieldnames = ["method", "predicate", "inferred_count", "derived_count"]
    ignored = _normalize_predicate_filter(ignore_predicates)

    rows: List[Dict[str, str]] = []
    for m in summary:
        method_name = str(m.get("method", ""))
        nested_counts: Dict[str, Dict[str, int]] = m.get("predicate_fact_group_counts", {})
        for predicate in sorted(nested_counts.keys()):
            if str(predicate).strip().lower() in ignored:
                continue
            groups = nested_counts[predicate]
            inferred_count = int(groups.get("inferred", 0))
            derived_count = inferred_count + int(groups.get("intermediate", 0))
            if inferred_count == 0:
                rows.append(
                    {
                        "method": method_name,
                        "predicate": str(predicate),
                        "inferred_count": str(inferred_count),
                        "derived_count": str(derived_count),
                    }
                )

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _normalize_predicate_filter(ignore_predicates: Optional[List[str]]) -> set[str]:
    return {str(p).strip().lower() for p in (ignore_predicates or []) if str(p).strip()}


def _merge_type_counts_for_plot(type_counts: Dict[str, int]) -> Dict[str, int]:
    merged: Counter = Counter()
    for row_type, count in type_counts.items():
        key = str(row_type)
        if key in {"inf_root", "inf_intermediate", "inferred"}:
            key = "inferred"
        elif key in {"neg_inf_root", "neg_inf_intermediate", "neg_inferred"}:
            key = "neg_inferred"
        merged[key] += int(count)
    return dict(merged)


def _merge_fact_group_counts_for_plot(nested_counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    merged: Dict[str, Dict[str, int]] = {}
    for predicate, groups in nested_counts.items():
        predicate_totals: Counter = Counter()
        for group, count in groups.items():
            key = "inferred" if str(group) == "intermediate" else str(group)
            predicate_totals[key] += int(count)
        merged[predicate] = dict(predicate_totals)
    return merged


def _plot_summary_view(summary: List[Dict], merge_intermediate_into_inferred: bool) -> List[Dict]:
    if not merge_intermediate_into_inferred:
        return summary

    plot_summary: List[Dict] = []
    for row in summary:
        merged_row = dict(row)
        merged_row["type_counts"] = _merge_type_counts_for_plot(row.get("type_counts", {}))
        merged_row["predicate_fact_group_counts"] = _merge_fact_group_counts_for_plot(
            row.get("predicate_fact_group_counts", {})
        )
        plot_summary.append(merged_row)

    return plot_summary


def _plot_top_predicates_per_method(
    summary: List[Dict], out_dir: Path, top_k: int, ignore_predicates: Optional[List[str]] = None
) -> None:
    ignored = _normalize_predicate_filter(ignore_predicates)

    for m in summary:
        filtered = [(k, v) for k, v in m["top_predicates"].items() if str(k).strip().lower() not in ignored]
        top = filtered[:top_k]
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


def _plot_top_predicates_combined(
    summary: List[Dict], out_path: Path, top_k: int, ignore_predicates: Optional[List[str]] = None
) -> None:
    methods = [m["method"] for m in summary]
    if not methods:
        return

    ignored = _normalize_predicate_filter(ignore_predicates)

    discovered_groups = {
        str(group)
        for m in summary
        for groups in m.get("predicate_fact_group_counts", {}).values()
        for group in groups.keys()
    }
    preferred_order = ["base", "inferred", "intermediate"]
    fact_groups = [g for g in preferred_order if g in discovered_groups]
    if not fact_groups:
        return
    group_colors = {
        "base": "#1f77b4",
        "inferred": "#ff7f0e",
        "intermediate": "#2ca02c",
    }
    method_hatches = {
        methods[0]: "",
    }
    if len(methods) > 1:
        method_hatches[methods[1]] = "//"

    combined_counts: Counter = Counter()
    per_method: Dict[str, Dict[str, Dict[str, int]]] = {}

    for m in summary:
        nested_counts: Dict[str, Dict[str, int]] = {
            p: {g: int(c) for g, c in groups.items()} for p, groups in m.get("predicate_fact_group_counts", {}).items()
        }
        nested_counts = {p: groups for p, groups in nested_counts.items() if str(p).strip().lower() not in ignored}
        per_method[m["method"]] = nested_counts
        for predicate, groups in nested_counts.items():
            combined_counts[predicate] += sum(int(groups.get(g, 0)) for g in fact_groups)

    predicates = [p for p, _ in combined_counts.most_common(top_k)]
    if not predicates:
        return

    x = list(range(len(predicates)))
    width = 0.8 / max(len(methods), 1)

    fig_width = max(12, 0.65 * len(predicates) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        nested = per_method.get(method, {})
        bottom = [0.0] * len(predicates)

        for fact_group in fact_groups:
            vals = [float(nested.get(p, {}).get(fact_group, 0)) for p in predicates]
            ax.bar(
                [xi + offset for xi in x],
                vals,
                width=width,
                bottom=bottom,
                color=group_colors[fact_group],
                edgecolor="black",
                linewidth=0.35,
                hatch=method_hatches.get(method, ".."),
            )
            bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(predicates, rotation=45, ha="right")
    ax.set_title(f"Top {top_k} Predicates by Method and Fact Group")
    ax.set_xlabel("Predicate")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    group_handles = [Patch(facecolor=group_colors.get(g, "#7f7f7f"), edgecolor="black", label=g) for g in fact_groups]
    method_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=method_hatches.get(m, ".."), label=m) for m in methods
    ]
    legend_groups = ax.legend(handles=group_handles, title="Fact Group", loc="upper right")
    ax.add_artist(legend_groups)
    ax.legend(handles=method_handles, title="Method", loc="upper left")

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_predicate_distribution_by_fact_group(summary: List[Dict], out_dir: Path, ignore_predicates: Optional[List[str]] = None) -> None:
    fact_groups = ["base", "inferred", "intermediate"]
    ignored = _normalize_predicate_filter(ignore_predicates)

    for m in summary:
        nested_counts: Dict[str, Dict[str, int]] = {
            p: groups
            for p, groups in m.get("predicate_fact_group_counts", {}).items()
            if str(p).strip().lower() not in ignored
        }
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


def _plot_all_inferred_predicates_by_method(
    summary: List[Dict], out_path: Path, ignore_predicates: Optional[List[str]] = None
) -> None:
    methods = [m["method"] for m in summary]
    if not methods:
        return

    ignored = _normalize_predicate_filter(ignore_predicates)
    inferred_by_method: Dict[str, Dict[str, int]] = {}
    all_predicates: set[str] = set()

    for m in summary:
        nested_counts: Dict[str, Dict[str, int]] = m.get("predicate_fact_group_counts", {})
        inferred_counts: Dict[str, int] = {}
        for predicate, groups in nested_counts.items():
            if str(predicate).strip().lower() in ignored:
                continue
            count = int(groups.get("inferred", 0))
            if count > 0:
                inferred_counts[predicate] = count
                all_predicates.add(predicate)
        inferred_by_method[m["method"]] = inferred_counts

    if not all_predicates:
        return

    ordered_predicates = sorted(
        all_predicates,
        key=lambda p: sum(inferred_by_method.get(method, {}).get(p, 0) for method in methods),
        reverse=True,
    )

    x = list(range(len(ordered_predicates)))
    width = 0.8 / max(len(methods), 1)
    fig_width = max(14, 0.45 * len(ordered_predicates) + 6)

    fig, ax = plt.subplots(figsize=(fig_width, 6))
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        y = [inferred_by_method.get(method, {}).get(p, 0) for p in ordered_predicates]
        ax.bar([xi + offset for xi in x], y, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_predicates, rotation=90)
    ax.set_title("All Inferred Predicates by Method (inferred only)")
    ax.set_xlabel("Predicate")
    ax.set_ylabel("Inferred Count")
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_all_derived_predicates_by_method(
    summary: List[Dict], out_path: Path, ignore_predicates: Optional[List[str]] = None
) -> None:
    methods = [m["method"] for m in summary]
    if not methods:
        return

    ignored = _normalize_predicate_filter(ignore_predicates)
    derived_by_method: Dict[str, Dict[str, int]] = {}
    all_predicates: set[str] = set()

    for m in summary:
        nested_counts: Dict[str, Dict[str, int]] = m.get("predicate_fact_group_counts", {})
        derived_counts: Dict[str, int] = {}
        for predicate, groups in nested_counts.items():
            if str(predicate).strip().lower() in ignored:
                continue
            count = int(groups.get("inferred", 0)) + int(groups.get("intermediate", 0))
            if count > 0:
                derived_counts[predicate] = count
                all_predicates.add(predicate)
        derived_by_method[m["method"]] = derived_counts

    if not all_predicates:
        return

    ordered_predicates = sorted(
        all_predicates,
        key=lambda p: sum(derived_by_method.get(method, {}).get(p, 0) for method in methods),
        reverse=True,
    )

    x = list(range(len(ordered_predicates)))
    width = 0.8 / max(len(methods), 1)
    fig_width = max(14, 0.45 * len(ordered_predicates) + 6)

    fig, ax = plt.subplots(figsize=(fig_width, 6))
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        y = [derived_by_method.get(method, {}).get(p, 0) for p in ordered_predicates]
        ax.bar([xi + offset for xi in x], y, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_predicates, rotation=90)
    ax.set_title("All Derived Predicates by Method (inferred + intermediate)")
    ax.set_xlabel("Predicate")
    ax.set_ylabel("Derived Count")
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _write_markdown_report(summary: List[Dict], out_dir: Path, ignore_predicates: Optional[List[str]] = None) -> None:
    md_path = out_dir / "report.md"
    ignored = _normalize_predicate_filter(ignore_predicates)

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

    lines.append("## Budget + Hop Diagnostics")
    lines.append("")
    lines.append(
        "| Method | Facts Total | Positives | Negatives | Hop d=1 | Hop d=2 | Hop d>=3 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in summary:
        hop_buckets = m.get("hop_buckets", {})
        lines.append(
            f"| {m['method']} | {m['facts_total']} | {m['positives_total']} | {m['negatives_total']} | "
            f"{int(hop_buckets.get('d1', 0))} | {int(hop_buckets.get('d2', 0))} | {int(hop_buckets.get('d3p', 0))} |"
        )
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

    lines.append("## Predicates With No Inferred Facts")
    lines.append("")
    lines.append("(Inferred = counts from `inf_root`/`inferred`. Derived adds `inf_intermediate`.)")
    lines.append("")

    for m in summary:
        method_name = str(m.get("method", ""))
        nested_counts: Dict[str, Dict[str, int]] = m.get("predicate_fact_group_counts", {})
        missing_rows = []
        for predicate in sorted(nested_counts.keys()):
            if str(predicate).strip().lower() in ignored:
                continue
            groups = nested_counts[predicate]
            inferred_count = int(groups.get("inferred", 0))
            if inferred_count == 0:
                derived_count = inferred_count + int(groups.get("intermediate", 0))
                missing_rows.append((predicate, derived_count))

        lines.append(f"### {method_name}")
        lines.append("")
        if not missing_rows:
            lines.append("- None")
            lines.append("")
            continue

        lines.append("| Predicate | Inferred Count | Derived Count |")
        lines.append("| --- | ---: | ---: |")
        for predicate, derived_count in missing_rows:
            lines.append(f"| {predicate} | 0 | {derived_count} |")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _generate_plots(
    summary: List[Dict], out_dir: Path, top_k_predicates: int, ignore_predicates: Optional[List[str]] = None
) -> None:
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
    _plot_predicate_distribution_by_fact_group(summary, out_dir, ignore_predicates)
    _plot_top_predicates_per_method(summary, out_dir, top_k_predicates, ignore_predicates)
    _plot_top_predicates_combined(summary, out_dir / "top_predicates_combined.png", top_k_predicates, ignore_predicates)
    _plot_all_inferred_predicates_by_method(summary, out_dir / "all_inferred_predicates_by_method.png", ignore_predicates)
    _plot_all_derived_predicates_by_method(summary, out_dir / "all_derived_predicates_by_method.png", ignore_predicates)


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
    _write_hops_csv(summary, out_dir)
    _write_hop_bucket_csv(summary, out_dir)
    _write_predicate_fact_group_csv(summary, out_dir)
    ignore_predicates = list(cfg.output.get("ignore_predicates", []))
    _write_inferred_predicates_csv(summary, out_dir, ignore_predicates)
    _write_derived_predicates_csv(summary, out_dir, ignore_predicates)
    _write_missing_inferred_predicates_csv(summary, out_dir, ignore_predicates)
    _write_markdown_report(summary, out_dir, ignore_predicates)

    if cfg.output.save_plots:
        merge_intermediate_into_inferred = bool(cfg.output.get("merge_intermediate_into_inferred_for_plots", False))
        plot_summary = _plot_summary_view(summary, merge_intermediate_into_inferred)
        _generate_plots(plot_summary, out_dir, int(cfg.output.top_k_predicates), ignore_predicates)

    logger.success(f"Saved reports to {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[4]


def _normalize_sample_id(sample_id: object) -> str:
    return str(sample_id).strip()


def _safe_node_id(term: str) -> str:
    safe = term.replace('"', "'").replace(" ", "_")
    return f"n_{abs(hash(term))}_{safe[:24]}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    if not rows:
        logger.warning("CSV '{}' has no rows", path)
    return rows


def _group_rows_by_sample(rows: list[dict[str, str]], sample_id: str) -> list[dict[str, str]]:
    selected = [r for r in rows if _normalize_sample_id(r.get("sample_id", "")) == sample_id]
    if not selected:
        raise ValueError(f"No rows found for sample_id={sample_id}")
    return selected


def _resolve_targets_csv(input_csv: Path, configured_targets: str | None, include_sibling_targets: bool) -> Path | None:
    if configured_targets:
        return Path(configured_targets)
    if not include_sibling_targets:
        return None

    sibling = input_csv.parent / "targets.csv"
    if sibling.exists() and sibling != input_csv:
        return sibling
    return None


def _predicate_local_name(predicate: str) -> str:
    value = predicate.strip()
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    if "/" in value:
        return value.rsplit("/", 1)[-1]
    if ":" in value:
        return value.rsplit(":", 1)[-1]
    return value


def _is_skipped_predicate(predicate: str, cfg: DictConfig) -> bool:
    if not bool(cfg.filters.ignore_same_as_different_from):
        return False

    local = _predicate_local_name(predicate)
    return local in {"sameAs", "differentFrom"}


def _classify_edges(
    input_rows: list[dict[str, str]],
    targets_rows: list[dict[str, str]] | None,
    cfg: DictConfig,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    base_edges: set[tuple[str, str, str]] = set()
    inferred_edges: set[tuple[str, str, str]] = set()
    negative_edges: set[tuple[str, str, str]] = set()

    has_label = bool(input_rows and "label" in input_rows[0])

    if has_label:
        all_rows = input_rows
    else:
        for row in input_rows:
            if _is_skipped_predicate(row["predicate"], cfg):
                continue
            base_edges.add((row["subject"], row["predicate"], row["object"]))
        all_rows = targets_rows or []

    for row in all_rows:
        if _is_skipped_predicate(row["predicate"], cfg):
            continue

        triple = (row["subject"], row["predicate"], row["object"])
        label = str(row.get("label", "1")).strip()
        fact_type = str(row.get("type", "")).strip().lower()

        if label == "0":
            negative_edges.add(triple)
            continue

        if fact_type == "base_fact":
            base_edges.add(triple)
        else:
            inferred_edges.add(triple)

    inferred_edges.difference_update(base_edges)
    negative_edges.difference_update(base_edges)
    negative_edges.difference_update(inferred_edges)
    return base_edges, inferred_edges, negative_edges


def _collect_nodes(
    base_edges: set[tuple[str, str, str]],
    inferred_edges: set[tuple[str, str, str]],
    negative_edges: set[tuple[str, str, str]],
    class_nodes: bool,
) -> set[str]:
    nodes: set[str] = set()
    for s, p, o in base_edges | inferred_edges | negative_edges:
        nodes.add(s)
        if p != "rdf:type":
            nodes.add(o)
        elif class_nodes:
            nodes.add(o)
    return nodes


def _collect_class_memberships(
    edges: set[tuple[str, str, str]],
) -> dict[str, set[str]]:
    memberships: dict[str, set[str]] = {}
    for s, p, o in edges:
        if p != "rdf:type":
            continue
        memberships.setdefault(s, set()).add(o)
    return memberships


def _build_dot(
    cfg: DictConfig,
    sample_id: str,
    base_edges: set[tuple[str, str, str]],
    inferred_edges: set[tuple[str, str, str]],
    negative_edges: set[tuple[str, str, str]],
):
    try:
        import graphviz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Python graphviz package is not installed") from exc

    dot = graphviz.Digraph(comment=f"kg_sample_{sample_id}", engine=str(cfg.render.engine))

    dot.attr(rankdir=str(cfg.render.rankdir))
    dot.attr(splines=str(cfg.render.splines))
    dot.attr(overlap=str(cfg.render.overlap))
    dot.attr(sep=str(cfg.render.sep))

    if cfg.render.title:
        dot.attr(
            label=str(cfg.render.title).format(sample_id=sample_id),
            labelloc="t",
            fontsize=str(cfg.render.title_fontsize),
        )

    class_nodes_enabled = bool(cfg.render.class_nodes)
    membership_edges = base_edges | inferred_edges
    class_memberships = _collect_class_memberships(membership_edges)

    node_ids: dict[str, str] = {}
    for node in sorted(_collect_nodes(base_edges, inferred_edges, negative_edges, class_nodes=class_nodes_enabled)):
        node_id = _safe_node_id(node)
        node_ids[node] = node_id

        is_class = any(p == "rdf:type" and o == node for _, p, o in (base_edges | inferred_edges))
        shape = str(cfg.style.class_node_shape if is_class else cfg.style.entity_node_shape)
        fill = str(cfg.style.class_node_fill if is_class else cfg.style.entity_node_fill)
        if class_nodes_enabled:
            node_label = node
        else:
            classes = sorted(class_memberships.get(node, set()))
            if classes:
                preview = ", ".join(classes[: int(cfg.render.class_annotation_max_items)])
                if len(classes) > int(cfg.render.class_annotation_max_items):
                    preview += ", ..."
                node_label = f"{node}\\n[{preview}]"
            else:
                node_label = node

        dot.node(
            node_id,
            label=node_label,
            shape=shape,
            style="filled,rounded",
            fillcolor=fill,
            fontname=str(cfg.style.font_name),
            fontsize=str(cfg.style.node_fontsize),
        )

    def add_edge(s: str, p: str, o: str, color: str, style: str, penwidth: str, prefix: str = "") -> None:
        label = f"{prefix}{p}" if cfg.render.show_edge_labels else ""
        dot.edge(
            node_ids[s],
            node_ids[o],
            label=label,
            color=color,
            style=style,
            penwidth=penwidth,
            fontname=str(cfg.style.font_name),
            fontsize=str(cfg.style.edge_fontsize),
            fontcolor=color,
        )

    for s, p, o in sorted(base_edges):
        if p == "rdf:type" and not class_nodes_enabled:
            continue
        add_edge(
            s,
            p,
            o,
            color=str(cfg.style.base_edge_color),
            style=str(cfg.style.base_edge_style),
            penwidth=str(cfg.style.base_edge_width),
        )

    for s, p, o in sorted(inferred_edges):
        if p == "rdf:type" and not class_nodes_enabled:
            continue
        add_edge(
            s,
            p,
            o,
            color=str(cfg.style.inferred_edge_color),
            style=str(cfg.style.inferred_edge_style),
            penwidth=str(cfg.style.inferred_edge_width),
        )

    if cfg.filters.include_negatives:
        for s, p, o in sorted(negative_edges):
            add_edge(
                s,
                p,
                o,
                color=str(cfg.style.negative_edge_color),
                style=str(cfg.style.negative_edge_style),
                penwidth=str(cfg.style.negative_edge_width),
                prefix=str(cfg.style.negative_label_prefix),
            )

    return dot


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs" / "kgvisualiser"), config_name="config")
def main(cfg: DictConfig) -> None:
    input_csv = Path(str(cfg.io.input_csv))
    sample_id = _normalize_sample_id(cfg.io.sample_id)

    output_dir = Path(str(cfg.output.dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = str(cfg.output.name_template).format(sample_id=sample_id)
    output_format = str(cfg.output.format)

    input_rows = _group_rows_by_sample(_read_rows(input_csv), sample_id)

    targets_csv = _resolve_targets_csv(
        input_csv=input_csv,
        configured_targets=cfg.io.targets_csv,
        include_sibling_targets=bool(cfg.io.include_sibling_targets),
    )

    targets_rows = None
    if targets_csv is not None:
        targets_rows = _group_rows_by_sample(_read_rows(targets_csv), sample_id)

    base_edges, inferred_edges, negative_edges = _classify_edges(input_rows, targets_rows, cfg)

    if cfg.filters.max_edges > 0:
        max_edges = int(cfg.filters.max_edges)
        base_edges = set(sorted(base_edges)[:max_edges])
        inferred_edges = set(sorted(inferred_edges)[:max_edges])
        negative_edges = set(sorted(negative_edges)[:max_edges])

    logger.info(
        "Sample {} | base={} inferred={} negative={} | input={} targets={}",
        sample_id,
        len(base_edges),
        len(inferred_edges),
        len(negative_edges),
        input_csv,
        targets_csv,
    )

    dot = _build_dot(
        cfg=cfg,
        sample_id=sample_id,
        base_edges=base_edges,
        inferred_edges=inferred_edges,
        negative_edges=negative_edges,
    )

    output_path = output_dir / output_name
    rendered = dot.render(filename=output_path.name, directory=str(output_dir), format=output_format, cleanup=True)
    logger.info("Graph saved to {}", rendered)


if __name__ == "__main__":
    main()

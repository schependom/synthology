import csv
import json
from pathlib import Path

import rdflib
from loguru import logger


def node_to_str(node: rdflib.term.Node) -> str:
    return node.n3() if hasattr(node, "n3") else str(node)


def write_triples_tsv(
    path: Path,
    rows: list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
    max_rows: int,
) -> int:
    if max_rows > 0:
        rows_to_write = rows[:max_rows]
    else:
        rows_to_write = rows

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["reason", "subject", "predicate", "object"])
        for reason, s, p, o in rows_to_write:
            writer.writerow([reason, node_to_str(s), node_to_str(p), node_to_str(o)])
    return len(rows_to_write)


def write_run_diagnostics(
    diagnostics_cfg: dict,
    diagnostics_id: str,
    generated_owl: Path,
    reasoning_input_graph: rdflib.Graph,
    base_graph: rdflib.Graph,
    raw_inferred_rows: list,
    rejected_rows: list,
    accepted_rows: list,
    inference_stats: dict,
    total_facts: int,
    total_inferred: int,
) -> None:
    if not diagnostics_cfg.get("enabled", True):
        return

    diagnostics_root = Path(str(diagnostics_cfg.get("output_dir", "data/owl2bench/diagnostics")))
    max_rows = int(diagnostics_cfg.get("max_rows_per_file", 0))
    run_dir = diagnostics_root / diagnostics_id
    run_dir.mkdir(parents=True, exist_ok=True)

    base_rows = [("base_fact", s, p, o) for (s, p, o) in base_graph]
    base_written = write_triples_tsv(run_dir / "base_facts.tsv", base_rows, max_rows)
    raw_written = write_triples_tsv(run_dir / "reasoner_raw_inferred.tsv", raw_inferred_rows, max_rows)
    rejected_written = write_triples_tsv(run_dir / "rejected_inferred.tsv", rejected_rows, max_rows)
    accepted_written = write_triples_tsv(run_dir / "accepted_inferred.tsv", accepted_rows, max_rows)

    summary = {
        "diagnostics_id": diagnostics_id,
        "generated_owl": str(generated_owl),
        "reasoning_input_triples": len(reasoning_input_graph),
        "base_exportable_facts": len(base_graph),
        "base_tsv_rows_written": base_written,
        "raw_inferred_tsv_rows_written": raw_written,
        "rejected_tsv_rows_written": rejected_written,
        "accepted_tsv_rows_written": accepted_written,
        "inference": inference_stats,
        "final_export": {
            "facts_csv_rows": total_facts,
            "inferred_targets_csv_rows": total_inferred,
        },
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Diagnostics exported | dir={} | base={} | reasoner_raw={} | rejected={} | accepted={}",
        run_dir,
        base_written,
        raw_written,
        rejected_written,
        accepted_written,
    )

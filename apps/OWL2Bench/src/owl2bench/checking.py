import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import rdflib
from loguru import logger

from owl2bench.parsing import RDF_TYPE_URI


def compute_inferred_triples(
    base_graph: rdflib.Graph,
    tbox_graph: rdflib.Graph,
    known_individuals: set[rdflib.term.Node],
    nemo_cfg: dict[str, Any],
) -> tuple[
    list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]],
    dict[str, int],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
]:
    nmo_executable = str(nemo_cfg.get("executable", "nmo"))
    command_template = str(nemo_cfg.get("command_template", "")).strip()
    program = str(nemo_cfg.get("program", "")).strip()

    if not command_template:
        raise RuntimeError("No `dataset.reasoning.nemo.command_template` configured for OWL2Bench pipeline.")

    with tempfile.TemporaryDirectory(prefix="owl2bench_nemo_") as td:
        tmp_dir = Path(td)
        abox_path = tmp_dir / "abox.ttl"
        tbox_path = tmp_dir / "tbox.ttl"
        output_path = tmp_dir / "nemo_materialized.ttl"

        base_graph.serialize(destination=str(abox_path), format="turtle")
        tbox_graph.serialize(destination=str(tbox_path), format="turtle")

        command = command_template.format(
            nmo_executable=shlex.quote(nmo_executable),
            program=shlex.quote(program),
            input_abox=shlex.quote(str(abox_path)),
            input_tbox=shlex.quote(str(tbox_path)),
            output_ttl=shlex.quote(str(output_path)),
            output_name=shlex.quote(output_path.name),
            output_dir=shlex.quote(str(tmp_dir)),
            tmp_dir=shlex.quote(str(tmp_dir)),
            param_input_tbox=shlex.quote(f'input_tbox="{tbox_path}"'),
            param_input_abox=shlex.quote(f'input_abox="{abox_path}"'),
            param_output_ttl=shlex.quote(f'output_ttl="{output_path.name}"'),
        )

        proc = subprocess.run(command, shell=True, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "NeMo materialization command failed\n"
                f"Command: {command}\n"
                f"Exit code: {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        if not output_path.exists():
            raise RuntimeError(
                "NeMo command completed but produced no materialized Turtle output\n"
                f"Expected output: {output_path}\n"
                f"Command: {command}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        if proc.stdout.strip():
            logger.info("NeMo materializer STDOUT:\n{}", proc.stdout.strip())

        closure_graph = rdflib.Graph()
        closure_graph.parse(output_path, format="turtle")

    inferred = []
    raw_inferred_rows: list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]] = []
    rejected_rows: list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]] = []
    accepted_rows: list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]] = []

    novel_before_filter = 0
    dropped_invalid_abox = 0
    dropped_unknown_subject = 0
    dropped_unknown_object = 0
    dropped_subject_bnode = 0
    dropped_object_bnode = 0
    dropped_object_literal = 0

    reasoner_raw_triples = len(closure_graph)

    for s, p, o in closure_graph:
        raw_inferred_rows.append(("reasoner_output", s, p, o))
        if (s, p, o) in base_graph:
            continue
        if (s, p, o) in tbox_graph:
            continue
        novel_before_filter += 1
        if isinstance(s, rdflib.BNode):
            dropped_invalid_abox += 1
            dropped_subject_bnode += 1
            rejected_rows.append(("subject_bnode", s, p, o))
            continue
        if isinstance(o, rdflib.BNode):
            dropped_invalid_abox += 1
            dropped_object_bnode += 1
            rejected_rows.append(("object_bnode", s, p, o))
            continue
        if isinstance(o, rdflib.Literal):
            dropped_invalid_abox += 1
            dropped_object_literal += 1
            rejected_rows.append(("object_literal", s, p, o))
            continue
        if s not in known_individuals:
            dropped_unknown_subject += 1
            rejected_rows.append(("unknown_subject", s, p, o))
            continue
        if str(p) != RDF_TYPE_URI and o not in known_individuals:
            dropped_unknown_object += 1
            rejected_rows.append(("unknown_object", s, p, o))
            continue
        inferred.append((s, p, o, 1))
        accepted_rows.append(("accepted", s, p, o))

    inferred.sort(key=lambda t: (t[3], str(t[0]), str(t[1]), str(t[2])))
    logger.info(
        "Inference filter diagnostics | novel_before_filter={} | accepted={} | dropped_invalid_abox={} | dropped_unknown_subject={} | dropped_unknown_object={}",
        novel_before_filter,
        len(inferred),
        dropped_invalid_abox,
        dropped_unknown_subject,
        dropped_unknown_object,
    )
    logger.info(
        "Invalid ABox breakdown | subject_bnode={} | object_bnode={} | object_literal={}",
        dropped_subject_bnode,
        dropped_object_bnode,
        dropped_object_literal,
    )

    stats = {
        "reasoner_raw_triples": reasoner_raw_triples,
        "novel_before_filter": novel_before_filter,
        "accepted": len(inferred),
        "dropped_invalid_abox": dropped_invalid_abox,
        "dropped_unknown_subject": dropped_unknown_subject,
        "dropped_unknown_object": dropped_unknown_object,
        "dropped_subject_bnode": dropped_subject_bnode,
        "dropped_object_bnode": dropped_object_bnode,
        "dropped_object_literal": dropped_object_literal,
    }
    return inferred, stats, raw_inferred_rows, rejected_rows, accepted_rows

"""Reusable Apache Jena materialization CLI for UDM baseline workflows."""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Tuple

from loguru import logger
from rdflib import Graph, URIRef
from udm_baseline.create_data import JenaMaterializer

Triple = Tuple[URIRef, URIRef, URIRef]


def _uri_triples(graph: Graph) -> Set[Triple]:
    return {(s, p, o) for s, p, o in graph if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize TBox+ABox with Apache Jena and export inferred triples.")
    parser.add_argument("--tbox", required=True, help="Path to ontology/TBox file")
    parser.add_argument("--abox", required=True, help="Path to ABox/base facts file")
    parser.add_argument("--closure-out", required=True, help="Output path for full closure (N-Triples)")
    parser.add_argument("--inferred-out", required=True, help="Output path for inferred-only triples (N-Triples)")
    parser.add_argument("--timing-dir", default="", help="Directory for scientific timing logs (JSONL + CSV)")
    parser.add_argument("--timing-tag", default="exp3_materialize_abox", help="Tag prefix for timing files")
    parser.add_argument(
        "--jena-profile",
        default="owl_mini",
        choices=["owl_micro", "owl_mini", "owl_full"],
        help="Apache Jena reasoner profile",
    )
    args = parser.parse_args()

    run_started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    tbox_path = Path(args.tbox).resolve()
    abox_path = Path(args.abox).resolve()
    closure_out = Path(args.closure_out).resolve()
    inferred_out = Path(args.inferred_out).resolve()

    tbox_parse_start = time.perf_counter()
    ontology_graph = Graph()
    ontology_graph.parse(tbox_path)
    schema_uri = _uri_triples(ontology_graph)
    tbox_parse_seconds = time.perf_counter() - tbox_parse_start

    abox_parse_start = time.perf_counter()
    base_graph = Graph()
    base_graph.parse(abox_path)
    base_uri = _uri_triples(base_graph)
    abox_parse_seconds = time.perf_counter() - abox_parse_start

    materializer = JenaMaterializer()
    reasoning_start = time.perf_counter()
    closure_uri = materializer.materialize(str(tbox_path), base_uri, jena_profile=args.jena_profile)
    reasoning_seconds = time.perf_counter() - reasoning_start
    inferred_uri = closure_uri - base_uri - schema_uri

    closure_write_start = time.perf_counter()
    closure_graph = Graph()
    for triple in closure_uri:
        closure_graph.add(triple)
    closure_out.parent.mkdir(parents=True, exist_ok=True)
    closure_graph.serialize(closure_out, format="nt", encoding="utf-8")
    closure_write_seconds = time.perf_counter() - closure_write_start

    inferred_write_start = time.perf_counter()
    inferred_graph = Graph()
    for triple in inferred_uri:
        inferred_graph.add(triple)
    inferred_out.parent.mkdir(parents=True, exist_ok=True)
    inferred_graph.serialize(inferred_out, format="nt", encoding="utf-8")
    inferred_write_seconds = time.perf_counter() - inferred_write_start

    run_total_seconds = time.perf_counter() - run_started

    timing_event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_tag": args.timing_tag,
        "event_type": "exp3_materialize_abox",
        "jena_profile": args.jena_profile,
        "tbox_path": str(tbox_path),
        "abox_path": str(abox_path),
        "closure_out": str(closure_out),
        "inferred_out": str(inferred_out),
        "schema_triples": len(schema_uri),
        "base_triples": len(base_uri),
        "closure_triples": len(closure_uri),
        "inferred_triples": len(inferred_uri),
        "tbox_parse_seconds": tbox_parse_seconds,
        "abox_parse_seconds": abox_parse_seconds,
        "reasoning_seconds": reasoning_seconds,
        "closure_write_seconds": closure_write_seconds,
        "inferred_write_seconds": inferred_write_seconds,
        "run_total_seconds": run_total_seconds,
        "jena_last_run_timing": materializer.last_run_timing,
    }

    if args.timing_dir:
        timing_dir = Path(args.timing_dir).resolve()
        timing_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = timing_dir / f"{args.timing_tag}_jena_events.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(timing_event, sort_keys=True) + "\n")

        csv_path = timing_dir / f"{args.timing_tag}_jena_events.csv"
        csv_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "timestamp_utc",
                "run_id",
                "run_tag",
                "event_type",
                "schema_triples",
                "base_triples",
                "closure_triples",
                "inferred_triples",
                "tbox_parse_seconds",
                "abox_parse_seconds",
                "reasoning_seconds",
                "closure_write_seconds",
                "inferred_write_seconds",
                "run_total_seconds",
                "details_json",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            row = {key: timing_event.get(key, "") for key in fieldnames}
            row["details_json"] = json.dumps(timing_event, sort_keys=True)
            writer.writerow(row)

        logger.info(
            "Timing recorded | run_id={} | tag={} | metrics=[schema_triples,base_triples,closure_triples,inferred_triples,"
            "tbox_parse_seconds,abox_parse_seconds,reasoning_seconds,closure_write_seconds,inferred_write_seconds,run_total_seconds,"
            "jena.serialize_seconds,jena.java_seconds,jena.parse_seconds,jena.total_seconds] | jsonl={} | csv={}",
            run_id,
            args.timing_tag,
            jsonl_path,
            csv_path,
        )

    logger.info(
        "Jena materialization complete | base={} | closure={} | inferred={} | closure_out={} | inferred_out={}",
        len(base_uri),
        len(closure_uri),
        len(inferred_uri),
        closure_out,
        inferred_out,
    )
    logger.info(
        "Timing summary | run_total_s={} | reasoning_s={} | tbox_parse_s={} | abox_parse_s={} "
        "| closure_write_s={} | inferred_write_s={} | jena_serialize_s={} | jena_java_s={} | jena_parse_s={} | jena_total_s={}",
        round(run_total_seconds, 6),
        round(reasoning_seconds, 6),
        round(tbox_parse_seconds, 6),
        round(abox_parse_seconds, 6),
        round(closure_write_seconds, 6),
        round(inferred_write_seconds, 6),
        materializer.last_run_timing.get("serialize_seconds", ""),
        materializer.last_run_timing.get("java_seconds", ""),
        materializer.last_run_timing.get("parse_seconds", ""),
        materializer.last_run_timing.get("total_seconds", ""),
    )


if __name__ == "__main__":
    main()

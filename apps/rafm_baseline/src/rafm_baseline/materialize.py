"""Reusable Apache Jena materialization CLI for RAFM baseline workflows."""

import argparse
from pathlib import Path
from typing import Set, Tuple

from loguru import logger
from rdflib import Graph, URIRef

from rafm_baseline.create_data import JenaMaterializer

Triple = Tuple[URIRef, URIRef, URIRef]


def _uri_triples(graph: Graph) -> Set[Triple]:
    return {(s, p, o) for s, p, o in graph if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize TBox+ABox with Apache Jena and export inferred triples.")
    parser.add_argument("--tbox", required=True, help="Path to ontology/TBox file")
    parser.add_argument("--abox", required=True, help="Path to ABox/base facts file")
    parser.add_argument("--closure-out", required=True, help="Output path for full closure (N-Triples)")
    parser.add_argument("--inferred-out", required=True, help="Output path for inferred-only triples (N-Triples)")
    args = parser.parse_args()

    tbox_path = Path(args.tbox).resolve()
    abox_path = Path(args.abox).resolve()
    closure_out = Path(args.closure_out).resolve()
    inferred_out = Path(args.inferred_out).resolve()

    ontology_graph = Graph()
    ontology_graph.parse(tbox_path)
    schema_uri = _uri_triples(ontology_graph)

    base_graph = Graph()
    base_graph.parse(abox_path)
    base_uri = _uri_triples(base_graph)

    materializer = JenaMaterializer()
    closure_uri = materializer.materialize(str(tbox_path), base_uri)
    inferred_uri = closure_uri - base_uri - schema_uri

    closure_graph = Graph()
    for triple in closure_uri:
        closure_graph.add(triple)
    closure_out.parent.mkdir(parents=True, exist_ok=True)
    closure_graph.serialize(closure_out, format="nt")

    inferred_graph = Graph()
    for triple in inferred_uri:
        inferred_graph.add(triple)
    inferred_out.parent.mkdir(parents=True, exist_ok=True)
    inferred_graph.serialize(inferred_out, format="nt")

    logger.info(
        "Jena materialization complete | base={} | closure={} | inferred={} | closure_out={} | inferred_out={}",
        len(base_uri),
        len(closure_uri),
        len(inferred_uri),
        closure_out,
        inferred_out,
    )


if __name__ == "__main__":
    main()

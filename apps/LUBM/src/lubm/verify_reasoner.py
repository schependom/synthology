import csv
import os
import shutil
from pathlib import Path
from typing import Union

import rdflib
from loguru import logger

from lubm.parse_to_csv import RDF_TYPE_URI, compute_inferred_triples, parse_lubm_directory, parse_uri
from synthology.data_structures import (
    Atom,
    Class,
    ExecutableRule,
    Individual,
    KnowledgeGraph,
    Membership,
    Proof,
    Relation,
    Triple,
)


def _repo_root() -> Path:
    env_root = os.environ.get("SYNTHOLOGY_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[4]


def _toy_verify_dir() -> Path:
    return _repo_root() / "data" / "lubm" / "toy_verify"


def _to_readable_name(term) -> str:
    """Convert URI-like terms to compact labels safe for Graphviz node IDs."""
    value = parse_uri(term)

    # Fallback for toy/example URIs not handled by parse_uri prefixes.
    if "://" in value:
        value = value.rsplit("#", 1)[-1]
        value = value.rsplit("/", 1)[-1]

    return value


def _make_toy_graphs() -> tuple[rdflib.Graph, rdflib.Graph]:
    ex = rdflib.Namespace("http://example.org/")

    # Minimal TBox with a 2-hop subclass chain + domain/range typing rules.
    tbox = rdflib.Graph()
    tbox.add((ex.GraduateStudent, rdflib.RDFS.subClassOf, ex.Student))
    tbox.add((ex.Student, rdflib.RDFS.subClassOf, ex.Person))
    tbox.add((ex.teaches, rdflib.RDFS.domain, ex.Faculty))
    tbox.add((ex.takesCourse, rdflib.RDFS.range, ex.Course))

    # Small ABox with very few base facts.
    base = rdflib.Graph()
    base.add((ex.Alice, rdflib.RDF.type, ex.GraduateStudent))
    base.add((ex.Bob, ex.teaches, ex.CS101))
    base.add((ex.Carol, ex.takesCourse, ex.CS101))

    return base, tbox


def _verify_inferred_triples(
    base: rdflib.Graph, tbox: rdflib.Graph
) -> list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]]:
    known = set()
    for s, p, o in base:
        known.add(s)
        if str(p) != RDF_TYPE_URI:
            known.add(o)

    inferred = compute_inferred_triples(base, tbox, known)
    inferred_map = {(str(s), str(p), str(o)): hops for s, p, o, hops in inferred}

    ex = "http://example.org/"
    alice_student = (f"{ex}Alice", RDF_TYPE_URI, f"{ex}Student")
    alice_person = (f"{ex}Alice", RDF_TYPE_URI, f"{ex}Person")
    bob_faculty = (f"{ex}Bob", RDF_TYPE_URI, f"{ex}Faculty")
    cs_course = (f"{ex}CS101", RDF_TYPE_URI, f"{ex}Course")

    missing = [triple for triple in [alice_student, alice_person, bob_faculty, cs_course] if triple not in inferred_map]
    if missing:
        raise AssertionError(f"Reasoner failed toy verification; missing inferred triples: {missing}")

    if inferred_map[alice_person] < inferred_map[alice_student]:
        raise AssertionError(
            "Expected Person(Alice) to require at least as many hops as Student(Alice) in subclass chain."
        )

    logger.success("Toy verification passed for direct inference output.")
    for triple, hops in sorted(inferred_map.items(), key=lambda x: (x[1], x[0])):
        logger.info(f"  inferred[hops={hops}] {triple}")

    return inferred


def _dummy_inferred_proof(subject: Individual, predicate: Relation, object_: Union[Individual, Class]) -> Proof:
    """Create a minimal non-base proof so visualizer marks facts as inferred."""
    goal = Atom(subject=subject, predicate=predicate, object=object_)
    dummy_rule = ExecutableRule(name="TOY_REASONER_INFERRED", conclusion=goal, premises=[])
    return Proof(goal=goal, rule=dummy_rule, sub_proofs=tuple())


def _build_toy_kg_for_visualization(
    base: rdflib.Graph,
    inferred: list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]],
) -> KnowledgeGraph:
    """Build a KnowledgeGraph containing both base and inferred toy facts."""
    individuals: dict[str, Individual] = {}
    classes: dict[str, Class] = {}
    relations: dict[str, Relation] = {"rdf:type": Relation(index=0, name="rdf:type")}

    triples: list[Triple] = []
    memberships: list[Membership] = []
    membership_keys: set[tuple[str, str]] = set()
    triple_keys: set[tuple[str, str, str]] = set()

    def get_individual(name: str) -> Individual:
        if name not in individuals:
            individuals[name] = Individual(index=len(individuals), name=name)
        return individuals[name]

    def get_class(name: str) -> Class:
        if name not in classes:
            classes[name] = Class(index=len(classes), name=name)
        return classes[name]

    def get_relation(name: str) -> Relation:
        if name not in relations:
            relations[name] = Relation(index=len(relations), name=name)
        return relations[name]

    def add_fact(s_raw, p_raw, o_raw, is_inferred: bool, hops: int) -> None:
        s_name = _to_readable_name(s_raw)
        p_name = _to_readable_name(p_raw)
        o_name = _to_readable_name(o_raw)
        if p_name == "type":
            p_name = "rdf:type"

        s_ind = get_individual(s_name)

        if p_name == "rdf:type":
            c = get_class(o_name)
            key = (s_name, o_name)
            if key in membership_keys:
                return
            membership_keys.add(key)

            proofs = []
            if is_inferred:
                rdf_type = get_relation("rdf:type")
                proofs = [_dummy_inferred_proof(s_ind, rdf_type, c)]

            m = Membership(individual=s_ind, cls=c, is_member=True, proofs=proofs)
            m.metadata = {"hops": hops, "type": "inferred" if is_inferred else "base_fact"}
            s_ind.classes.append(m)
            memberships.append(m)
            return

        o_ind = get_individual(o_name)
        rel = get_relation(p_name)
        key = (s_name, p_name, o_name)
        if key in triple_keys:
            return
        triple_keys.add(key)

        proofs = []
        if is_inferred:
            proofs = [_dummy_inferred_proof(s_ind, rel, o_ind)]

        t = Triple(subject=s_ind, predicate=rel, object=o_ind, positive=True, proofs=proofs)
        t.metadata = {"hops": hops, "type": "inferred" if is_inferred else "base_fact"}
        triples.append(t)

    for s, p, o in base:
        add_fact(s, p, o, is_inferred=False, hops=0)

    for s, p, o, hops in inferred:
        add_fact(s, p, o, is_inferred=True, hops=hops)

    return KnowledgeGraph(
        attributes=[],
        classes=list(classes.values()),
        relations=list(relations.values()),
        individuals=list(individuals.values()),
        triples=triples,
        memberships=memberships,
        attribute_triples=[],
    )


def _export_toy_graph(
    base: rdflib.Graph, inferred: list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]]
) -> None:
    toy_root = _toy_verify_dir()
    graph_dir = toy_root / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    kg = _build_toy_kg_for_visualization(base, inferred)
    kg.save_visualization(
        output_path=str(graph_dir),
        output_name="toy_reasoning_graph",
        format="pdf",
        title="Toy LUBM Reasoning Graph (Base + Inferred)",
        display_negatives=False,
    )
    logger.info(f"Toy graph exported to: {graph_dir}")


def _verify_csv_export(base: rdflib.Graph, tbox: rdflib.Graph) -> None:
    toy_root = _toy_verify_dir()
    raw_root = toy_root / "raw"
    out_root = toy_root / "out"

    if raw_root.exists():
        shutil.rmtree(raw_root)
    if out_root.exists():
        shutil.rmtree(out_root)

    raw_dir = raw_root / "lubm_0"
    out_dir = out_root / "lubm_0"
    raw_dir.mkdir(parents=True, exist_ok=True)

    toy_ttl_path = raw_dir / "toy.ttl"
    base.serialize(destination=str(toy_ttl_path), format="turtle")

    parse_lubm_directory(
        raw_dir=raw_dir,
        output_dir=out_dir,
        split_ratios={"train": 1.0, "val": 0.0, "test": 0.0},
        target_ratio=0.0,
        num_samples=1,
        enable_reasoning=True,
        tbox_graph=tbox,
    )

    targets_csv = out_dir / "train" / "targets.csv"
    if not targets_csv.exists():
        raise AssertionError("Toy CSV verification failed: targets.csv not generated.")

    inferred_positive = 0
    max_hops = 0
    with open(targets_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "1" and row["type"] == "inferred":
                inferred_positive += 1
                max_hops = max(max_hops, int(row["hops"]))

    if inferred_positive == 0:
        raise AssertionError("Toy CSV verification failed: no positive inferred targets found.")

    logger.success(
        f"Toy CSV verification passed: inferred_positive={inferred_positive}, max_hops={max_hops}, path={targets_csv}"
    )
    logger.info(f"Toy verification artifacts kept at: {toy_root}")


def main() -> None:
    base, tbox = _make_toy_graphs()
    inferred = _verify_inferred_triples(base, tbox)
    _export_toy_graph(base, inferred)
    _verify_csv_export(base, tbox)
    logger.success("All toy reasoner verification checks passed.")


if __name__ == "__main__":
    main()

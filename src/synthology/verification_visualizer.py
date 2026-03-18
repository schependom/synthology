from pathlib import Path
from typing import Union

import rdflib
from loguru import logger

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


def _readable_name(term) -> str:
    """Convert URI-like terms to compact labels safe for Graphviz node IDs."""
    if isinstance(term, rdflib.URIRef):
        value = str(term)
    else:
        value = str(term)

    if "#" in value:
        value = value.rsplit("#", 1)[-1]
    elif "/" in value:
        value = value.rsplit("/", 1)[-1]

    if value == "type":
        return "rdf:type"

    return value


def _dummy_inferred_proof(subject: Individual, predicate: Relation, object_: Union[Individual, Class]) -> Proof:
    """Create a minimal non-base proof so visualizer marks facts as inferred."""
    goal = Atom(subject=subject, predicate=predicate, object=object_)
    dummy_rule = ExecutableRule(name="VERIFICATION_INFERRED", conclusion=goal, premises=[])
    return Proof(goal=goal, rule=dummy_rule, sub_proofs=tuple())


def build_visualization_kg(
    base_graph: rdflib.Graph,
    inferred: list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]],
) -> KnowledgeGraph:
    """Build a KnowledgeGraph that includes both base and inferred facts."""
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
        s_name = _readable_name(s_raw)
        p_name = _readable_name(p_raw)
        o_name = _readable_name(o_raw)

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

    for s, p, o in base_graph:
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


def export_base_inferred_graph(
    base_graph: rdflib.Graph,
    inferred: list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]],
    output_dir: Path,
    output_name: str,
    title: str,
) -> Path:
    """Export graph visualization with base and inferred facts and return output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    kg = build_visualization_kg(base_graph, inferred)
    kg.save_visualization(
        output_path=str(output_dir),
        output_name=output_name,
        format="pdf",
        title=title,
        display_negatives=False,
    )

    graph_path = output_dir / f"{output_name}.pdf"
    if graph_path.exists():
        logger.info(f"Verification graph exported to: {graph_path}")
    else:
        logger.warning(
            f"Verification graph was not generated at {graph_path} (likely skipped due size cap or missing graphviz backend)"
        )
    return graph_path

import urllib.parse

import rdflib

RDF_TYPE_URI = str(rdflib.RDF.type)


def parse_uri(uri: str) -> str:
    if isinstance(uri, rdflib.URIRef):
        uri = str(uri)

    prefixes_to_strip = [
        "https://kracr.iiitd.edu.in/OWL2Bench#",
        "http://swat.cse.lehigh.edu/onto/univ-bench.owl#",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.",
    ]

    for prefix in prefixes_to_strip:
        if uri.startswith(prefix):
            uri = uri[len(prefix) :]
            break

    return urllib.parse.unquote(uri)


def is_valid_abox_triple(subject: rdflib.term.Node, obj: rdflib.term.Node) -> bool:
    if isinstance(subject, rdflib.BNode) or isinstance(obj, rdflib.BNode):
        return False
    if isinstance(obj, rdflib.Literal):
        return False
    return True


def _remap_uri(node: rdflib.term.Node, namespace_map: dict[str, str]) -> rdflib.term.Node:
    if not isinstance(node, rdflib.URIRef):
        return node

    value = str(node)
    for old_ns, new_ns in namespace_map.items():
        if value.startswith(old_ns):
            return rdflib.URIRef(new_ns + value[len(old_ns) :])
    return node


def apply_namespace_map(graph: rdflib.Graph, namespace_map: dict[str, str]) -> tuple[rdflib.Graph, int]:
    if not namespace_map:
        return graph, 0

    mapped = rdflib.Graph()
    changed_terms = 0

    for s, p, o in graph:
        ns = _remap_uri(s, namespace_map)
        np = _remap_uri(p, namespace_map)
        no = _remap_uri(o, namespace_map)

        if ns != s:
            changed_terms += 1
        if np != p:
            changed_terms += 1
        if no != o:
            changed_terms += 1

        mapped.add((ns, np, no))

    return mapped, changed_terms


def known_individuals_from_base(base_graph: rdflib.Graph) -> set[rdflib.term.Node]:
    known = set()
    for s, p, o in base_graph:
        known.add(s)
        if str(p) != RDF_TYPE_URI:
            known.add(o)
    return known

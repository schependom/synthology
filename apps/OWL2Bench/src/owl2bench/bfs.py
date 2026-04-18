import random
from typing import Optional


def build_bfs_subgraph_samples(
    base_clean: list[tuple[str, str, str]],
    inferred_clean: list[tuple[str, str, str, int]],
    base_sample_id: int,
    sample_count: int,
    max_individuals_per_sample: int,
    use_closure_context: bool = True,
) -> tuple[
    list[tuple[int, list[tuple[str, str, str]]]],
    dict[int, list[tuple[str, str, str, int]]],
]:
    """
    Build BFS subgraph samples from the OWL2Bench materialized graph.

    When use_closure_context=True (default), BFS adjacency is built from the full
    closure (base + inferred) so that each subgraph contains a rich neighbourhood of
    interconnected individuals.  ALL closure facts for the visited individuals are
    written to facts.csv — not just the sparse base ABox triples.  This makes context
    density comparable to what Synthology generates (~500 facts/sample vs 1-9 without).

    When use_closure_context=False, only base_clean triples are used for adjacency and
    context (legacy behaviour).
    """
    if sample_count <= 0:
        return [], {}

    # Build a unified triple list for adjacency and context.
    # Include (s, p, o) from both base and inferred when using closure context.
    if use_closure_context:
        closure_triples: list[tuple[str, str, str]] = list(base_clean) + [
            (s, p, o) for s, p, o, _ in inferred_clean
        ]
    else:
        closure_triples = list(base_clean)

    adjacency: dict[str, set[str]] = {}
    individuals: set[str] = set()

    for s, p, o in closure_triples:
        individuals.add(s)
        adjacency.setdefault(s, set())
        if p != "rdf:type":
            individuals.add(o)
            adjacency.setdefault(o, set())
            adjacency[s].add(o)
            adjacency[o].add(s)

    if not individuals:
        return [], {}

    individual_list = list(individuals)
    random.shuffle(individual_list)

    samples: list[tuple[int, list[tuple[str, str, str]]]] = []
    sample_nodes_by_sid: dict[int, set[str]] = {}

    for i in range(sample_count):
        sid = base_sample_id + i
        seed = individual_list[i % len(individual_list)]

        visited: set[str] = {seed}
        queue = [seed]
        head = 0

        while head < len(queue) and len(visited) < max_individuals_per_sample:
            current = queue[head]
            head += 1
            for nb in adjacency.get(current, set()):
                if nb in visited:
                    continue
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= max_individuals_per_sample:
                    break

        # Collect ALL closure facts whose subject (and object for non-type) are visited.
        sample_triples: list[tuple[str, str, str]] = []
        for s, p, o in closure_triples:
            if s not in visited:
                continue
            if p == "rdf:type":
                sample_triples.append((s, p, o))
            elif o in visited:
                sample_triples.append((s, p, o))

        # Deduplicate while preserving order.
        seen: set[tuple[str, str, str]] = set()
        deduped: list[tuple[str, str, str]] = []
        for t in sample_triples:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        samples.append((sid, deduped))
        sample_nodes_by_sid[sid] = visited

    inferred_by_sample: dict[int, list[tuple[str, str, str, int]]] = {sid: [] for sid, _ in samples}
    sid_order = [sid for sid, _ in samples]

    for s, p, o, hops in inferred_clean:
        assigned_sid: Optional[int] = None
        for sid in sid_order:
            nodes = sample_nodes_by_sid[sid]
            if s not in nodes:
                continue
            if p != "rdf:type" and o not in nodes:
                continue
            assigned_sid = sid
            break

        if assigned_sid is None and sid_order:
            assigned_sid = sid_order[0]

        if assigned_sid is not None:
            inferred_by_sample[assigned_sid].append((s, p, o, hops))

    return samples, inferred_by_sample

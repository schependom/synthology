import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import hydra
import rdflib
from loguru import logger
from omegaconf import DictConfig
from rafm_baseline.create_data import JenaMaterializer

RDF_TYPE_URI = str(rdflib.RDF.type)
REPO_ROOT = Path(__file__).resolve().parents[4]
UNIVERSITY_ID_RES = [
    re.compile(r"\bU(\d+)"),
    re.compile(r"\bUniversity(\d+)\b", re.IGNORECASE),
    re.compile(r"\.University(\d+)", re.IGNORECASE),
]


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


def _node_to_str(node: rdflib.term.Node) -> str:
    return node.n3() if hasattr(node, "n3") else str(node)


def _write_triples_tsv(
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
            writer.writerow([reason, _node_to_str(s), _node_to_str(p), _node_to_str(o)])
    return len(rows_to_write)


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


def run_owl2bench_generator(
    vendor_dir: Path, profile: str, universities: int, seed: int, maven_executable: str
) -> Path:
    cmd = [
        maven_executable,
        "-q",
        "-DskipTests",
        "compile",
        "exec:java",
        "-Dexec.mainClass=ABoxGen.InstanceGenerator.Generator",
        f"-Dexec.args={universities} {profile} {seed}",
    ]

    logger.info("Running OWL2Bench Java generator: {}", " ".join(cmd))
    env = dict(os.environ)
    extra_open = "--add-opens=java.base/java.lang=ALL-UNNAMED"
    current_opts = env.get("MAVEN_OPTS", "").strip()
    env["MAVEN_OPTS"] = f"{current_opts} {extra_open}".strip()

    proc = subprocess.run(cmd, cwd=vendor_dir, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"OWL2Bench generator failed\nExit code: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    generated_name = f"OWL2{profile}-{universities}.owl"
    generated_path = vendor_dir / generated_name
    if not generated_path.exists():
        raise FileNotFoundError(f"Expected generated ontology not found at {generated_path}")

    logger.info("OWL2Bench generator output: {}", generated_path)
    return generated_path


def _uri_triples(graph: rdflib.Graph) -> set[tuple[rdflib.URIRef, rdflib.URIRef, rdflib.URIRef]]:
    return {
        (s, p, o)
        for s, p, o in graph
        if isinstance(s, rdflib.URIRef) and isinstance(p, rdflib.URIRef) and isinstance(o, rdflib.URIRef)
    }


def compute_inferred_triples(
    base_graph: rdflib.Graph,
    tbox_graph: rdflib.Graph,
    tbox_path: Path,
    known_individuals: set[rdflib.term.Node],
    materialization_cfg: dict[str, Any],
) -> tuple[
    list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]],
    dict[str, int],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
]:
    reasoner = str(materialization_cfg.get("reasoner", "jena")).lower()
    iterative = bool(materialization_cfg.get("iterative", True))
    max_iterations = int(materialization_cfg.get("max_iterations", 10))

    if reasoner != "jena":
        raise ValueError(f"Unsupported reasoner '{reasoner}' for OWL2Bench pipeline. Only 'jena' is supported.")

    jena = JenaMaterializer()
    base_uri_triples = _uri_triples(base_graph)
    tbox_uri_triples = _uri_triples(tbox_graph)

    current_working_set = set(base_uri_triples)
    all_inferred: set[tuple[rdflib.URIRef, rdflib.URIRef, rdflib.URIRef]] = set()
    hop_depths: dict[tuple[rdflib.URIRef, rdflib.URIRef, rdflib.URIRef], int] = {}
    closure_final: set[tuple[rdflib.URIRef, rdflib.URIRef, rdflib.URIRef]] = set(base_uri_triples)

    iterations = 1 if not iterative else max_iterations
    for iteration in range(1, iterations + 1):
        closure = jena.materialize(str(tbox_path), current_working_set)
        closure_final = closure
        newly_inferred = closure - current_working_set - tbox_uri_triples

        if not newly_inferred:
            break

        for triple in newly_inferred:
            if triple not in hop_depths:
                hop_depths[triple] = iteration
                all_inferred.add(triple)

        current_working_set |= newly_inferred

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

    reasoner_raw_triples = len(closure_final)

    for s, p, o in closure_final:
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
        inferred.append((s, p, o, hop_depths.get((s, p, o), 1)))
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


def _known_individuals_from_base(base_graph: rdflib.Graph) -> set[rdflib.term.Node]:
    known = set()
    for s, p, o in base_graph:
        known.add(s)
        if str(p) != RDF_TYPE_URI:
            known.add(o)
    return known


def _split_samples(all_triples: list[tuple], num_samples: int, base_sample_id: int) -> list[tuple[int, list[tuple]]]:
    chunk_size = math.ceil(len(all_triples) / max(1, num_samples))
    samples = []
    for i in range(0, len(all_triples), chunk_size):
        chunk = all_triples[i : i + chunk_size]
        samples.append((base_sample_id + len(samples), chunk))
    random.shuffle(samples)
    return samples


def _extract_university_id(value: str) -> Optional[int]:
    for regex in UNIVERSITY_ID_RES:
        match = regex.search(value)
        if match:
            return int(match.group(1))
    return None


def _partition_triples_by_university(
    triples: list[tuple[str, str, str]],
) -> tuple[dict[int, list[tuple[str, str, str]]], list[tuple[str, str, str]]]:
    grouped: dict[int, list[tuple[str, str, str]]] = {}
    unassigned: list[tuple[str, str, str]] = []

    for s, p, o in triples:
        uid = _extract_university_id(s)
        if uid is None:
            uid = _extract_university_id(o)

        if uid is None:
            unassigned.append((s, p, o))
            continue

        grouped.setdefault(uid, []).append((s, p, o))

    return grouped, unassigned


def _build_samples_by_university(
    base_clean: list[tuple[str, str, str]],
    inferred_clean: list[tuple[str, str, str, int]],
    base_sample_id: int,
) -> tuple[
    list[tuple[int, list[tuple[str, str, str]]]],
    dict[int, list[tuple[str, str, str, int]]],
]:
    inferred_triples_only = [(s, p, o) for (s, p, o, _hops) in inferred_clean]

    base_by_uid, base_unassigned = _partition_triples_by_university(base_clean)
    inferred_by_uid_raw, inferred_unassigned = _partition_triples_by_university(inferred_triples_only)

    uid_set = sorted(set(base_by_uid.keys()) | set(inferred_by_uid_raw.keys()))
    samples: list[tuple[int, list[tuple[str, str, str]]]] = []
    inferred_by_sample: dict[int, list[tuple[str, str, str, int]]] = {}

    inferred_hops_index: dict[tuple[str, str, str], list[int]] = {}
    for s, p, o, hops in inferred_clean:
        inferred_hops_index.setdefault((s, p, o), []).append(hops)

    for uid in uid_set:
        sid = base_sample_id + uid
        sample_base = base_by_uid.get(uid, [])
        samples.append((sid, sample_base))

        sample_inferred: list[tuple[str, str, str, int]] = []
        for s, p, o in inferred_by_uid_raw.get(uid, []):
            hops_list = inferred_hops_index.get((s, p, o), [1])
            hops = hops_list.pop(0)
            sample_inferred.append((s, p, o, hops))
        inferred_by_sample[sid] = sample_inferred

    if samples and base_unassigned:
        logger.warning(
            "University partitioning: {} base triples had no university prefix and were assigned to first sample",
            len(base_unassigned),
        )
        first_sid = samples[0][0]
        first_triples = samples[0][1]
        first_triples.extend(base_unassigned)
        samples[0] = (first_sid, first_triples)

    if samples and inferred_unassigned:
        logger.warning(
            "University partitioning: {} inferred triples had no university prefix and were assigned to first sample",
            len(inferred_unassigned),
        )
        first_sid = samples[0][0]
        for s, p, o in inferred_unassigned:
            inferred_by_sample[first_sid].append((s, p, o, 1))

    random.shuffle(samples)
    return samples, inferred_by_sample


def _build_bfs_subgraph_samples(
    base_clean: list[tuple[str, str, str]],
    inferred_clean: list[tuple[str, str, str, int]],
    base_sample_id: int,
    sample_count: int,
    max_individuals_per_sample: int,
) -> tuple[
    list[tuple[int, list[tuple[str, str, str]]]],
    dict[int, list[tuple[str, str, str, int]]],
]:
    if sample_count <= 0:
        return [], {}

    adjacency: dict[str, set[str]] = {}
    individuals: set[str] = set()

    for s, p, o in base_clean:
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

        sample_triples: list[tuple[str, str, str]] = []
        for s, p, o in base_clean:
            if s not in visited:
                continue
            if p == "rdf:type":
                sample_triples.append((s, p, o))
            elif o in visited:
                sample_triples.append((s, p, o))

        samples.append((sid, sample_triples))
        sample_nodes_by_sid[sid] = visited

    inferred_by_sample: dict[int, list[tuple[str, str, str, int]]] = {sid: [] for sid, _ in samples}
    sid_order = [sid for sid, _ in samples]

    for s, p, o, hops in inferred_clean:
        assigned_sid = None
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


def _compute_split_counts(
    total_samples: int,
    split_ratios: dict[str, float],
    require_multiple_graphs_per_csv: bool,
) -> tuple[int, int, int]:
    train_ratio = split_ratios.get("train", 0.8)
    val_ratio = split_ratios.get("val", 0.1)
    test_ratio = split_ratios.get("test", 0.1)
    total_ratio = train_ratio + val_ratio + test_ratio

    if total_samples <= 0 or total_ratio <= 0:
        return 0, 0, 0

    train_count = int(total_samples * (train_ratio / total_ratio))
    val_count = int(total_samples * (val_ratio / total_ratio))
    test_count = total_samples - train_count - val_count

    if require_multiple_graphs_per_csv and total_samples >= 6:
        minimum = 2
        counts = {"train": train_count, "val": val_count, "test": test_count}

        for name in ["train", "val", "test"]:
            while counts[name] < minimum:
                donor = max(counts, key=lambda k: counts[k])
                if counts[donor] <= minimum:
                    break
                counts[donor] -= 1
                counts[name] += 1

        train_count = counts["train"]
        val_count = counts["val"]
        test_count = counts["test"]

    return train_count, val_count, test_count


def _write_split(
    split_dir: Path,
    split_samples: list[tuple[int, list[tuple]]],
    inferred_by_sample: dict[int, list[tuple[str, str, str, int]]],
    target_ratio: float,
    mask_base_facts: bool,
    negatives_per_positive: int,
) -> tuple[int, int]:
    facts_rows = []
    targets_rows = []
    inferred_positive_count = 0

    for sid, sample_base_clean in split_samples:
        all_individuals = set()
        all_classes = set()

        for s, p, o in sample_base_clean:
            all_individuals.add(s)
            if p == "rdf:type":
                all_classes.add(o)
            else:
                all_individuals.add(o)

        positive_targets = []

        for s, p, o in sample_base_clean:
            is_target_only = mask_base_facts and (random.random() < target_ratio)
            fact_type = "inferred" if is_target_only else "base_fact"

            fact = {"sample_id": str(sid), "subject": s, "predicate": p, "object": o}
            target = {
                "sample_id": str(sid),
                "subject": s,
                "predicate": p,
                "object": o,
                "label": 1,
                "truth_value": "True",
                "type": fact_type,
                "hops": 0,
                "corruption_method": "",
            }

            if fact_type == "base_fact":
                facts_rows.append(fact)
                targets_rows.append(target)
            else:
                positive_targets.append(target)
                targets_rows.append(target)

        for s, p, o, hops in inferred_by_sample.get(sid, []):
            inferred_target = {
                "sample_id": str(sid),
                "subject": s,
                "predicate": p,
                "object": o,
                "label": 1,
                "truth_value": "True",
                "type": "inferred",
                "hops": hops,
                "corruption_method": "",
            }
            positive_targets.append(inferred_target)
            targets_rows.append(inferred_target)
            inferred_positive_count += 1

            all_individuals.add(s)
            if p == "rdf:type":
                all_classes.add(o)
            else:
                all_individuals.add(o)

        all_individuals = list(all_individuals)
        all_classes = list(all_classes)

        for pos_tgt in positive_targets:
            for _ in range(max(0, negatives_per_positive)):
                corrupt_object = random.choice([True, False])
                if pos_tgt["predicate"] == "rdf:type" and corrupt_object:
                    choices = all_classes
                else:
                    choices = all_individuals

                if len(choices) < 2:
                    continue

                corrupted_entity = random.choice(choices)
                while corrupted_entity == (pos_tgt["object"] if corrupt_object else pos_tgt["subject"]):
                    corrupted_entity = random.choice(choices)

                neg_target = {
                    "sample_id": pos_tgt["sample_id"],
                    "subject": pos_tgt["subject"] if not corrupt_object else corrupted_entity,
                    "predicate": pos_tgt["predicate"],
                    "object": corrupted_entity if corrupt_object else pos_tgt["object"],
                    "label": 0,
                    "truth_value": "False",
                    "type": "inferred",
                    "hops": pos_tgt.get("hops", 0),
                    "corruption_method": "random",
                }
                targets_rows.append(neg_target)

    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "facts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object"])
        writer.writeheader()
        writer.writerows(facts_rows)

    with open(split_dir / "targets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "subject",
                "predicate",
                "object",
                "label",
                "truth_value",
                "type",
                "hops",
                "corruption_method",
            ],
        )
        writer.writeheader()
        writer.writerows(targets_rows)

    return len(facts_rows), inferred_positive_count


def _parse_generated_to_csv(
    generated_owl: Path,
    tbox_graph: rdflib.Graph,
    tbox_path: Path,
    namespace_map: dict[str, str],
    output_dir: Path,
    split_ratios: dict[str, float],
    target_ratio: float,
    mask_base_facts: bool,
    materialization_cfg: dict[str, Any],
    partition_mode: str,
    num_samples: int,
    bfs_sample_count: int,
    bfs_max_individuals_per_sample: int,
    inferred_target_limit: int,
    negatives_per_positive: int,
    require_multiple_graphs_per_csv: bool,
    base_sample_id: int,
    diagnostics_cfg: dict[str, Any],
    diagnostics_id: str,
) -> None:
    generated_graph = rdflib.Graph()
    generated_graph.parse(generated_owl, format="xml")

    if namespace_map:
        generated_graph, changed_terms = apply_namespace_map(generated_graph, namespace_map)
        logger.info("Applied namespace mapping to generated graph | mapped_terms={}", changed_terms)

    # Keep literals/bnodes for reasoning input; filter only for CSV exportable ABox facts.
    reasoning_input_graph = rdflib.Graph()
    for s, p, o in generated_graph:
        if (s, p, o) in tbox_graph:
            continue
        reasoning_input_graph.add((s, p, o))

    base_graph = rdflib.Graph()
    for s, p, o in reasoning_input_graph:
        if not is_valid_abox_triple(s, o):
            continue
        base_graph.add((s, p, o))

    known_individuals = _known_individuals_from_base(base_graph)
    inferred_raw, inference_stats, raw_inferred_rows, rejected_rows, accepted_rows = compute_inferred_triples(
        reasoning_input_graph,
        tbox_graph,
        tbox_path,
        known_individuals,
        materialization_cfg,
    )

    base_clean = []
    for s, p, o in base_graph:
        s_clean = parse_uri(s)
        p_clean = parse_uri(p)
        o_clean = parse_uri(o)
        if p_clean == "type":
            p_clean = "rdf:type"
        base_clean.append((s_clean, p_clean, o_clean))

    inferred_clean = []
    for s, p, o, hops in inferred_raw:
        s_clean = parse_uri(s)
        p_clean = parse_uri(p)
        o_clean = parse_uri(o)
        if p_clean == "type":
            p_clean = "rdf:type"
        inferred_clean.append((s_clean, p_clean, o_clean, hops))

    if inferred_target_limit > 0 and len(inferred_clean) > inferred_target_limit:
        inferred_clean = random.sample(inferred_clean, inferred_target_limit)
        logger.info(
            "Trimmed inferred positives to configured limit | limit={} | kept={}",
            inferred_target_limit,
            len(inferred_clean),
        )

    if partition_mode == "university_prefix":
        samples, inferred_by_sample = _build_samples_by_university(base_clean, inferred_clean, base_sample_id)
        logger.info("Partitioned graph into {} university-based samples", len(samples))
    elif partition_mode == "bfs_subgraphs":
        samples, inferred_by_sample = _build_bfs_subgraph_samples(
            base_clean,
            inferred_clean,
            base_sample_id,
            bfs_sample_count,
            bfs_max_individuals_per_sample,
        )
        logger.info(
            "Partitioned graph into {} BFS subgraph samples (target_count={}, max_individuals={})",
            len(samples),
            bfs_sample_count,
            bfs_max_individuals_per_sample,
        )
    else:
        samples = _split_samples(base_clean, num_samples, base_sample_id)
        inferred_by_sample = {sid: [] for sid, _ in samples}
        if samples:
            inferred_by_sample[samples[0][0]] = inferred_clean
        logger.warning(
            "Using fallback chunk partitioning mode='{}'; this may cut graph chains.",
            partition_mode,
        )

    total_samples = len(samples)
    train_count, val_count, test_count = _compute_split_counts(
        total_samples,
        split_ratios,
        require_multiple_graphs_per_csv,
    )
    train_idx = train_count
    val_idx = train_count + val_count

    splits = {
        "train": samples[:train_idx],
        "val": samples[train_idx:val_idx],
        "test": samples[val_idx : val_idx + test_count],
    }

    split_sample_counts = {name: len(s) for name, s in splits.items()}
    logger.info(
        "Sample distribution by split | total_samples={} | train={} | val={} | test={}",
        len(samples),
        split_sample_counts["train"],
        split_sample_counts["val"],
        split_sample_counts["test"],
    )

    if require_multiple_graphs_per_csv:
        for split_name, split_samples in splits.items():
            if len(split_samples) == 1:
                raise ValueError(
                    "Split '{}' contains only one graph sample. Increase dataset.universities or adjust split ratios "
                    "to keep multiple sample_ids in each CSV.".format(split_name)
                )

    total_facts = 0
    total_inferred = 0
    for split_name, split_samples in splits.items():
        if not split_samples:
            continue
        facts_count, inferred_count = _write_split(
            output_dir / split_name,
            split_samples,
            inferred_by_sample,
            target_ratio,
            mask_base_facts,
            negatives_per_positive,
        )
        total_facts += facts_count
        total_inferred += inferred_count

    logger.success(
        "Generated OWL2Bench CSV | base_facts={} | inferred_targets={} | output={}",
        total_facts,
        total_inferred,
        output_dir,
    )

    if diagnostics_cfg.get("enabled", True):
        diagnostics_root = Path(str(diagnostics_cfg.get("output_dir", "data/owl2bench/diagnostics")))
        max_rows = int(diagnostics_cfg.get("max_rows_per_file", 0))
        run_dir = diagnostics_root / diagnostics_id
        run_dir.mkdir(parents=True, exist_ok=True)

        base_rows = [("base_fact", s, p, o) for (s, p, o) in base_graph]
        base_written = _write_triples_tsv(run_dir / "base_facts.tsv", base_rows, max_rows)
        raw_written = _write_triples_tsv(run_dir / "reasoner_raw_inferred.tsv", raw_inferred_rows, max_rows)
        rejected_written = _write_triples_tsv(run_dir / "rejected_inferred.tsv", rejected_rows, max_rows)
        accepted_written = _write_triples_tsv(run_dir / "accepted_inferred.tsv", accepted_rows, max_rows)

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


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs" / "owl2bench"), config_name="config")
def main(cfg: DictConfig) -> None:
    random.seed(42)
    t_start = time.perf_counter()

    cwd = Path(hydra.utils.get_original_cwd())

    vendor_dir = cwd / str(cfg.generator.vendor_dir)
    profile = str(cfg.generator.profile)
    seed = int(cfg.generator.seed)
    maven_executable = str(cfg.generator.maven_executable)

    if profile.upper() != "RL":
        raise ValueError("This pipeline is restricted to OWL2Bench RL profile.")

    tbox_path = cwd / str(cfg.dataset.reasoning.tbox_path)
    if not tbox_path.exists():
        raise FileNotFoundError(f"OWL2Bench RL ontology not found: {tbox_path}")

    tbox_graph = rdflib.Graph()
    tbox_graph.parse(tbox_path, format="xml")

    base_output_dir = cwd / str(cfg.dataset.output_dir)
    raw_base_dir = base_output_dir / "raw"

    split_ratios = dict(cfg.dataset.get("split", {"train": 0.8, "val": 0.1, "test": 0.1}))
    target_ratio = float(cfg.dataset.get("target_ratio", 0.0))
    num_samples = int(cfg.dataset.get("num_samples", 1))
    mask_base_facts = bool(cfg.dataset.get("mask_base_facts", False))
    partition_mode = str(cfg.dataset.get("partition_mode", "university_prefix"))
    bfs_cfg = dict(cfg.dataset.get("bfs", {}))
    bfs_sample_count = int(bfs_cfg.get("sample_count", 5000))
    bfs_max_individuals_per_sample = int(bfs_cfg.get("max_individuals_per_sample", 200))
    inferred_target_limit = int(cfg.dataset.get("inferred_target_limit", 0))
    negatives_per_positive = int(cfg.dataset.get("negatives_per_positive", 1))
    require_multiple_graphs_per_csv = bool(cfg.dataset.get("require_multiple_graphs_per_csv", False))
    namespace_map = dict(cfg.dataset.reasoning.get("namespace_map", {}))
    materialization_cfg = dict(cfg.dataset.reasoning.get("materialization", {}))
    diagnostics_cfg = dict(cfg.dataset.get("diagnostics", {}))

    for universities in cfg.dataset.universities:
        universities = int(universities)
        size_label = f"owl2bench_{universities}"

        generated_owl = run_owl2bench_generator(
            vendor_dir=vendor_dir,
            profile=profile,
            universities=universities,
            seed=seed,
            maven_executable=maven_executable,
        )

        raw_dir = raw_base_dir / size_label
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_copy_path = raw_dir / generated_owl.name
        shutil.copy2(generated_owl, raw_copy_path)

        csv_output_dir = base_output_dir / size_label
        diagnostics_id = f"{size_label}_{int(time.time())}"
        _parse_generated_to_csv(
            generated_owl=raw_copy_path,
            tbox_graph=tbox_graph,
            tbox_path=tbox_path,
            namespace_map=namespace_map,
            output_dir=csv_output_dir,
            split_ratios=split_ratios,
            target_ratio=target_ratio,
            mask_base_facts=mask_base_facts,
            materialization_cfg=materialization_cfg,
            partition_mode=partition_mode,
            num_samples=num_samples,
            bfs_sample_count=bfs_sample_count,
            bfs_max_individuals_per_sample=bfs_max_individuals_per_sample,
            inferred_target_limit=inferred_target_limit,
            negatives_per_positive=negatives_per_positive,
            require_multiple_graphs_per_csv=require_multiple_graphs_per_csv,
            base_sample_id=700000 + universities * 10000,
            diagnostics_cfg=diagnostics_cfg,
            diagnostics_id=diagnostics_id,
        )

    logger.info("OWL2Bench pipeline completed in {:.2f}s", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()

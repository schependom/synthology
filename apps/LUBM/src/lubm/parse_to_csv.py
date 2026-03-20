import csv
import os
import random
import shlex
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import hydra
import rdflib
from loguru import logger
from omegaconf import DictConfig

RDF_TYPE_URI = str(rdflib.RDF.type)


def _namespace_of(uri: rdflib.term.Node) -> str:
    value = str(uri)
    if "#" in value:
        return value.rsplit("#", 1)[0] + "#"
    if "/" in value:
        return value.rsplit("/", 1)[0] + "/"
    return value


def _log_namespace_diagnostics(base_graph: rdflib.Graph, tbox_graph: Optional[rdflib.Graph]) -> None:
    if tbox_graph is None:
        return

    base_ns = set()
    for s, p, o in base_graph:
        base_ns.add(_namespace_of(s))
        base_ns.add(_namespace_of(p))
        if not isinstance(o, rdflib.Literal):
            base_ns.add(_namespace_of(o))

    tbox_ns = set()
    for s, p, o in tbox_graph:
        tbox_ns.add(_namespace_of(s))
        tbox_ns.add(_namespace_of(p))
        if not isinstance(o, rdflib.Literal):
            tbox_ns.add(_namespace_of(o))

    overlap = base_ns & tbox_ns
    logger.info(
        "Namespace diagnostics | base_namespaces={} | tbox_namespaces={} | overlap={}",
        len(base_ns),
        len(tbox_ns),
        len(overlap),
    )

    if len(overlap) == 0:
        logger.warning(
            "No overlapping namespaces between ABox and TBox. This usually indicates a namespace mismatch, "
            "which can cause zero inferences."
        )

    logger.info("ABox namespace sample: {}", sorted(base_ns)[:5])
    logger.info("TBox namespace sample: {}", sorted(tbox_ns)[:5])


def _remap_uri(node: rdflib.term.Node, namespace_map: dict[str, str]) -> rdflib.term.Node:
    if not isinstance(node, rdflib.URIRef):
        return node

    value = str(node)
    for old_ns, new_ns in namespace_map.items():
        if value.startswith(old_ns):
            return rdflib.URIRef(new_ns + value[len(old_ns) :])
    return node


def apply_namespace_map(graph: rdflib.Graph, namespace_map: dict[str, str]) -> tuple[rdflib.Graph, int]:
    """Return a mapped graph with URI namespaces rewritten according to namespace_map."""
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


def parse_uri(uri: str) -> str:
    """Simplify URIs into readable strings by stripping the ontology prefix."""
    # Handle rdflib URIRefs directly
    if isinstance(uri, rdflib.URIRef):
        uri = str(uri)

    # Strip typical LUBM ontology prefixes
    prefixes_to_strip = [
        "http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.",
    ]

    for prefix in prefixes_to_strip:
        if uri.startswith(prefix):
            uri = uri[len(prefix) :]
            break

    # Clean up trailing extensions if any
    if uri.endswith(".edu"):
        uri = uri[:-4]

    # Unquote any URL encodings
    uri = urllib.parse.unquote(uri)
    return uri


def is_valid_abox_triple(subject, predicate, obj) -> bool:
    """Filter out triples that are not suitable as ABox training assertions."""
    if isinstance(subject, rdflib.BNode) or isinstance(obj, rdflib.BNode):
        return False
    if isinstance(obj, rdflib.Literal):
        return False
    return True


def _invalid_abox_reason(subject: rdflib.term.Node, obj: rdflib.term.Node) -> Optional[str]:
    """Return the specific reason a triple fails ABox-shape validation, if any."""
    if isinstance(subject, rdflib.BNode):
        return "subject_bnode"
    if isinstance(obj, rdflib.BNode):
        return "object_bnode"
    if isinstance(obj, rdflib.Literal):
        return "object_literal"
    return None


def _write_dropped_debug_report(
    dropped_records: list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
    reason_counts: dict[str, int],
    output_dir: Path,
) -> Path:
    """Persist a TSV + summary text file with sampled dropped inferred triples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_stem = f"dropped_inferred_{int(time.time() * 1000)}"
    tsv_path = output_dir / f"{report_stem}.tsv"
    summary_path = output_dir / f"{report_stem}_summary.txt"

    with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["reason", "subject", "predicate", "object"])
        for reason, s, p, o in dropped_records:
            writer.writerow([reason, s.n3(), p.n3(), o.n3()])

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Dropped inferred triple reason counts\n")
        handle.write("===================================\n")
        for reason in sorted(reason_counts.keys()):
            handle.write(f"{reason}\t{reason_counts[reason]}\n")

    logger.info("Wrote dropped-triple debug report | tsv={} | summary={}", tsv_path, summary_path)
    return tsv_path


def compute_inferred_triples(
    base_graph: rdflib.Graph,
    tbox_graph: Optional[rdflib.Graph],
    known_individuals: set[rdflib.term.Node],
    jena_cfg: dict[str, Any],
) -> list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]]:
    """Run inference via Apache Jena CLI and return inferred ABox triples.

    Requires `dataset.reasoning.jena.command_template` in config.
    The template can use placeholders:
      {jena_executable}, {input_abox}, {input_tbox}, {output_ttl}
    """
    jena_executable = str(jena_cfg.get("executable", "jena"))
    command_template = str(jena_cfg.get("command_template", "")).strip()

    if not command_template:
        raise RuntimeError(
            "Jena backend selected, but `dataset.reasoning.jena.command_template` is empty. "
            "Please configure a Jena materialization command template using placeholders "
            "{jena_executable}, {input_abox}, {input_tbox}, {output_ttl}."
        )

    with tempfile.TemporaryDirectory(prefix="lubm_jena_") as td:
        tmp_dir = Path(td)
        abox_path = tmp_dir / "abox.ttl"
        tbox_path = tmp_dir / "tbox.ttl"
        output_path = tmp_dir / "materialized.ttl"

        base_graph.serialize(destination=str(abox_path), format="turtle")
        if tbox_graph is not None:
            tbox_graph.serialize(destination=str(tbox_path), format="turtle")
        else:
            tbox_path.write_text("", encoding="utf-8")

        command = command_template.format(
            jena_executable=shlex.quote(jena_executable),
            input_abox=shlex.quote(str(abox_path)),
            input_tbox=shlex.quote(str(tbox_path)),
            output_ttl=shlex.quote(str(output_path)),
        )

        proc = subprocess.run(command, shell=True, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Apache Jena materialization command failed\n"
                f"Command: {command}\n"
                f"Exit code: {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        if proc.stdout.strip():
            logger.info("Jena materializer STDOUT:\n{}", proc.stdout.strip())

        if not output_path.exists():
            raise RuntimeError(
                "Apache Jena command completed but no output file was produced at "
                f"{output_path}. Check jena.command_template export/output command."
            )

        closure_graph = rdflib.Graph()
        closure_graph.parse(output_path, format="turtle")

    inferred = []
    novel_before_filter = 0
    dropped_invalid_abox = 0
    dropped_unknown_subject = 0
    dropped_unknown_object = 0
    dropped_reason_counts: dict[str, int] = {
        "subject_bnode": 0,
        "object_bnode": 0,
        "object_literal": 0,
        "unknown_subject": 0,
        "unknown_object": 0,
    }

    debug_dropped_triples = bool(jena_cfg.get("debug_dropped_triples", False))
    debug_dropped_limit = int(jena_cfg.get("debug_dropped_limit", 100))
    dropped_records: list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]] = []

    def _record_drop(reason: str, s: rdflib.term.Node, p: rdflib.term.Node, o: rdflib.term.Node) -> None:
        dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1
        if debug_dropped_triples and len(dropped_records) < debug_dropped_limit:
            dropped_records.append((reason, s, p, o))

    for s, p, o in closure_graph:
        if (s, p, o) in base_graph:
            continue
        if tbox_graph is not None and (s, p, o) in tbox_graph:
            continue
        novel_before_filter += 1
        invalid_reason = _invalid_abox_reason(s, o)
        if invalid_reason is not None:
            dropped_invalid_abox += 1
            _record_drop(invalid_reason, s, p, o)
            continue
        if s not in known_individuals:
            dropped_unknown_subject += 1
            _record_drop("unknown_subject", s, p, o)
            continue
        if str(p) != RDF_TYPE_URI and o not in known_individuals:
            dropped_unknown_object += 1
            _record_drop("unknown_object", s, p, o)
            continue

        # Jena materialization output does not expose derivation-cycle depth in this pipeline.
        inferred.append((s, p, o, 1))

    inferred.sort(key=lambda t: (t[3], str(t[0]), str(t[1]), str(t[2])))
    logger.info(
        "Filter diagnostics | novel_before_filter={} | accepted={} | dropped_invalid_abox={} | dropped_unknown_subject={} | dropped_unknown_object={}",
        novel_before_filter,
        len(inferred),
        dropped_invalid_abox,
        dropped_unknown_subject,
        dropped_unknown_object,
    )
    logger.info(
        "Drop reason diagnostics | subject_bnode={} | object_bnode={} | object_literal={} | unknown_subject={} | unknown_object={}",
        dropped_reason_counts.get("subject_bnode", 0),
        dropped_reason_counts.get("object_bnode", 0),
        dropped_reason_counts.get("object_literal", 0),
        dropped_reason_counts.get("unknown_subject", 0),
        dropped_reason_counts.get("unknown_object", 0),
    )

    if debug_dropped_triples and dropped_records:
        debug_output_dir = Path(str(jena_cfg.get("debug_output_dir", "data/lubm/reasoning_debug")))
        _write_dropped_debug_report(dropped_records, dropped_reason_counts, debug_output_dir)

    return inferred


def load_optional_tbox(tbox_path: Path) -> Optional[rdflib.Graph]:
    """Load TBox ontology if available, otherwise continue without it."""
    if not tbox_path.exists():
        logger.warning(f"TBox file not found at {tbox_path}; reasoning will run without TBox axioms.")
        return None

    g = rdflib.Graph()
    try:
        g.parse(tbox_path)
    except Exception as exc:
        logger.warning(f"Failed to parse TBox at {tbox_path}: {exc}. Reasoning will continue without TBox.")
        return None

    logger.info(f"Loaded TBox ontology from {tbox_path} with {len(g)} triples.")
    return g


def parse_lubm_directory(
    raw_dir: Path,
    output_dir: Path,
    split_ratios: dict,
    target_ratio: float = 0.0,
    num_samples: Optional[int] = None,
    mask_base_facts: bool = True,
    enable_reasoning: bool = True,
    jena_cfg: Optional[dict[str, Any]] = None,
    tbox_graph: Optional[rdflib.Graph] = None,
):
    """
    Parses all .ttl files in a given LUBM raw directory and exports
    them as facts.csv and targets.csv in standard format, split into
    train, val, and test subdirectories.
    """
    if not raw_dir.exists():
        logger.error(f"Cannot find raw directory {raw_dir}")
        return None

    import math

    # RRN expects arbitrary sample IDs for independent KGs.
    # Use 500000+ to avoid collision with ASP (100000+) and ONT generators.
    # Fallback gracefully when raw_dir does not end with a numeric suffix.
    try:
        size_token = int(raw_dir.name.split("_")[-1])
    except ValueError:
        size_token = abs(hash(raw_dir.name)) % 10000
    base_sample_id = 500000 + size_token * 10000

    ttl_files = list(raw_dir.glob("*.ttl"))
    logger.info(f"Parsing {len(ttl_files)} TTL files in {raw_dir} sequentially to preserve structural local density...")

    t_parse_start = time.perf_counter()
    ttl_parse_seconds = 0.0
    reasoning_seconds = 0.0
    reasoning_calls = 0
    inferred_positive_count = 0
    max_observed_hops = 0

    random.seed(42)  # Replicable splits

    all_triples = []

    for i, ttl in enumerate(ttl_files):
        local_graph = rdflib.Graph()
        t_file_start = time.perf_counter()
        try:
            local_graph.parse(ttl, format="turtle")
        except Exception as e:
            logger.error(f"Error parsing {ttl.name}: {e}")
            continue
        ttl_parse_seconds += time.perf_counter() - t_file_start

        for subject, predicate, obj in local_graph:
            if not is_valid_abox_triple(subject, predicate, obj):
                continue

            all_triples.append((subject, predicate, obj))

    if num_samples is None:
        num_samples = len(ttl_files)

    chunk_size = math.ceil(len(all_triples) / max(1, num_samples))
    logger.info(
        f"Parsed {len(all_triples)} individual triples. Slicing sequentially into {num_samples} samples of roughly {chunk_size} triples each."
    )

    samples = []
    for i in range(0, len(all_triples), chunk_size):
        chunk = all_triples[i : i + chunk_size]
        current_sample_id = base_sample_id + len(samples)
        samples.append((current_sample_id, chunk))

    # Shuffle the samples globally to mix subgraphs across train/val/test splits
    random.shuffle(samples)

    total_samples = len(samples)
    train_ratio = split_ratios.get("train", 0.8)
    val_ratio = split_ratios.get("val", 0.1)
    test_ratio = split_ratios.get("test", 0.1)

    total_ratio = train_ratio + val_ratio + test_ratio
    train_idx = int(total_samples * (train_ratio / total_ratio))
    val_idx = train_idx + int(total_samples * (val_ratio / total_ratio))

    splits = {"train": samples[:train_idx], "val": samples[train_idx:val_idx], "test": samples[val_idx:]}

    for split_name, split_samples in splits.items():
        if not split_samples:
            continue

        facts_rows = []
        targets_rows = []

        # --- Positive Targets ---
        for sid, sample_triples in split_samples:
            sample_base_graph = rdflib.Graph()
            sample_base_clean = []

            # Pre-compute all unique entities and classes in this specific dataset
            all_individuals = set()
            all_classes = set()
            for s, p, o in sample_triples:
                sample_base_graph.add((s, p, o))

                s_clean = parse_uri(s)
                p_clean = parse_uri(p)
                o_clean = parse_uri(o)
                if p_clean == "type":
                    p_clean = "rdf:type"

                sample_base_clean.append((s_clean, p_clean, o_clean))

                all_individuals.add(s_clean)
                if p_clean == "rdf:type":
                    all_classes.add(o_clean)
                else:
                    all_individuals.add(o_clean)

            all_individuals = list(all_individuals)
            all_classes = list(all_classes)

            inferred_clean = []
            if enable_reasoning and len(sample_base_graph) > 0:
                _log_namespace_diagnostics(sample_base_graph, tbox_graph)
                known_individuals = set()
                for s, p, o in sample_base_graph:
                    if str(p) == RDF_TYPE_URI:
                        known_individuals.add(s)
                    else:
                        known_individuals.add(s)
                        known_individuals.add(o)

                t_reason_start = time.perf_counter()
                inferred_raw = compute_inferred_triples(
                    base_graph=sample_base_graph,
                    tbox_graph=tbox_graph,
                    known_individuals=known_individuals,
                    jena_cfg=jena_cfg or {},
                )
                reasoning_seconds += time.perf_counter() - t_reason_start
                reasoning_calls += 1

                for s, p, o, hops in inferred_raw:
                    s_clean = parse_uri(s)
                    p_clean = parse_uri(p)
                    o_clean = parse_uri(o)
                    if p_clean == "type":
                        p_clean = "rdf:type"
                    inferred_clean.append((s_clean, p_clean, o_clean, hops))

                    # Expand corruption candidates with inferred nodes/classes.
                    all_individuals.append(s_clean)
                    if p_clean == "rdf:type":
                        all_classes.append(o_clean)
                    else:
                        all_individuals.append(o_clean)

                all_individuals = list(set(all_individuals))
                all_classes = list(set(all_classes))

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

            # Reasoner-derived positives are always target facts (not context facts).
            for s, p, o, hops in inferred_clean:
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
                if hops > max_observed_hops:
                    max_observed_hops = hops

            # --- Negative Targets (Link Prediction Sparsity) ---
            # Generate 1 negative sample per positive target to prevent class imbalance
            # and to allow the model to actually learn from false links.
            for pos_tgt in positive_targets:
                # Randomly corrupt subject or object (basic negative sampling)
                corrupt_object = random.choice([True, False])

                if pos_tgt["predicate"] == "rdf:type" and corrupt_object:
                    choices = all_classes
                else:
                    choices = all_individuals

                if len(choices) < 2:
                    # Rare fallback if there's only 1 valid option, don't accidentally create infinite loop
                    corrupted_entity = "Dummy_Fallback"
                else:
                    corrupted_entity = random.choice(choices)
                    while corrupted_entity == (pos_tgt["object"] if corrupt_object else pos_tgt["subject"]):
                        corrupted_entity = random.choice(choices)

                neg_target = {
                    "sample_id": pos_tgt["sample_id"],
                    "subject": pos_tgt["subject"] if not corrupt_object else corrupted_entity,
                    "predicate": pos_tgt["predicate"],
                    "object": corrupted_entity if corrupt_object else pos_tgt["object"],
                    "label": 0,  # Negative!
                    "truth_value": "False",
                    "type": "inferred",
                    "hops": pos_tgt.get("hops", 0),
                    "corruption_method": "random",
                }
                targets_rows.append(neg_target)

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        facts_path = split_dir / "facts.csv"
        with open(facts_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object"])
            writer.writeheader()
            writer.writerows(facts_rows)

        targets_path = split_dir / "targets.csv"
        with open(targets_path, "w", newline="", encoding="utf-8") as f:
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

        logger.success(f"Generated {len(facts_rows)} base facts and {len(targets_rows)} targets in {split_dir}")

    total_seconds = time.perf_counter() - t_parse_start
    stats = {
        "raw_dir": str(raw_dir),
        "n_ttl_files": len(ttl_files),
        "n_base_triples": len(all_triples),
        "n_samples": len(samples),
        "ttl_parse_seconds": ttl_parse_seconds,
        "reasoning_seconds": reasoning_seconds,
        "reasoning_calls": reasoning_calls,
        "inferred_positive_count": inferred_positive_count,
        "max_observed_hops": max_observed_hops,
        "total_seconds": total_seconds,
    }

    logger.info(
        "LUBM parse stats | ttl_files={} | base_triples={} | samples={} | ttl_parse={:.2f}s | reasoning={:.2f}s ({} calls) | inferred_pos={} | max_hops={} | total={:.2f}s",
        stats["n_ttl_files"],
        stats["n_base_triples"],
        stats["n_samples"],
        stats["ttl_parse_seconds"],
        stats["reasoning_seconds"],
        stats["reasoning_calls"],
        stats["inferred_positive_count"],
        stats["max_observed_hops"],
        stats["total_seconds"],
    )

    return stats


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/lubm", config_name="config")
def main(cfg: DictConfig):
    t_main_start = time.perf_counter()
    original_cwd = Path(hydra.utils.get_original_cwd())
    output_dir_cfg = cfg.dataset.output_dir
    base_dir = Path(output_dir_cfg) if Path(output_dir_cfg).is_absolute() else original_cwd / output_dir_cfg

    raw_dir_base = base_dir / "raw"

    if not raw_dir_base.exists():
        logger.error(f"Base data directory not found: {raw_dir_base}")
        return

    dataset_configs = cfg.dataset.sizes
    split_ratios = cfg.dataset.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
    mask_base_facts = cfg.dataset.get("mask_base_facts", True)
    reasoning_cfg = cfg.dataset.get("reasoning", {})
    enable_reasoning = reasoning_cfg.get("enabled", True)
    jena_cfg = reasoning_cfg.get("jena", {})
    namespace_map = dict(reasoning_cfg.get("namespace_map", {}))

    tbox_graph = None
    if enable_reasoning:
        tbox_path_cfg = reasoning_cfg.get("tbox_path", "data/ont/input/lubm.ttl")
        tbox_path = Path(tbox_path_cfg) if Path(tbox_path_cfg).is_absolute() else original_cwd / tbox_path_cfg
        tbox_graph = load_optional_tbox(tbox_path)
        if tbox_graph is not None and namespace_map:
            tbox_graph, changed_terms = apply_namespace_map(tbox_graph, namespace_map)
            logger.info("Applied namespace mapping to TBox | mapped_terms={}", changed_terms)

    for univ_count in dataset_configs:
        size_label = f"lubm_{univ_count}"
        raw_ds = raw_dir_base / size_label

        if not raw_ds.exists():
            logger.warning(f"Raw directory for '{size_label}' does not exist, skipping CSV parsing.")
            continue

        output_dir = base_dir / size_label  # data/lubm/lubm_1
        target_ratio = cfg.dataset.get("target_ratio", 0.0)
        num_samples = cfg.dataset.get("num_samples", None)
        logger.info(f"Processing CSV output for {size_label} into train, val and test files...")
        stats = parse_lubm_directory(
            raw_ds,
            output_dir,
            split_ratios,
            target_ratio,
            num_samples,
            mask_base_facts=mask_base_facts,
            enable_reasoning=enable_reasoning,
            jena_cfg=jena_cfg,
            tbox_graph=tbox_graph,
        )
        if stats:
            logger.info(
                "Dataset {} summary | parse_total={:.2f}s | reasoning={:.2f}s | max_hops={}",
                size_label,
                stats["total_seconds"],
                stats["reasoning_seconds"],
                stats["max_observed_hops"],
            )

    logger.info("Total parse_to_csv runtime: {:.2f}s", time.perf_counter() - t_main_start)


if __name__ == "__main__":
    main()

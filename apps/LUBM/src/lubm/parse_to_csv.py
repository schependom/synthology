import csv
import os
import random
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import hydra
import rdflib
from loguru import logger
from omegaconf import DictConfig
from owlrl import OWLRL_Semantics

RDF_TYPE_URI = str(rdflib.RDF.type)


class HopTrackingOWLRL(OWLRL_Semantics):
    """OWL RL semantics that records the first cycle that derives each triple."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_cycle = 0
        self.first_cycle_by_triple: dict[tuple, int] = {}

    def rules(self, t, cycle_num):
        self.current_cycle = cycle_num
        return super().rules(t, cycle_num)

    def store_triple(self, t):
        """Store inferred triples and record the earliest cycle in which they appear."""
        s, p, o = t
        if not isinstance(p, rdflib.Literal) and not (t in self.destination or t in self.graph):
            self.added_triples.add(t)
            self.first_cycle_by_triple.setdefault(t, max(1, self.current_cycle))


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


def compute_inferred_triples(
    base_graph: rdflib.Graph,
    tbox_graph: Optional[rdflib.Graph],
    known_individuals: set[rdflib.term.Node],
) -> list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]]:
    """Run OWL RL closure and return inferred ABox triples only."""
    closure_graph = rdflib.Graph()
    for triple in base_graph:
        closure_graph.add(triple)

    if tbox_graph is not None:
        for triple in tbox_graph:
            closure_graph.add(triple)

    reasoner = HopTrackingOWLRL(closure_graph, False, False)
    reasoner.closure()

    inferred = []
    for s, p, o in closure_graph:
        if (s, p, o) in base_graph:
            continue
        if tbox_graph is not None and (s, p, o) in tbox_graph:
            continue
        if not is_valid_abox_triple(s, p, o):
            continue

        # Keep only assertions grounded in sample individuals to avoid
        # exporting TBox-only closure artifacts.
        if s not in known_individuals:
            continue
        if str(p) != RDF_TYPE_URI and o not in known_individuals:
            continue

        hop_count = reasoner.first_cycle_by_triple.get((s, p, o), 1)
        inferred.append((s, p, o, hop_count))

    inferred.sort(key=lambda t: (t[3], str(t[0]), str(t[1]), str(t[2])))
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
    enable_reasoning: bool = True,
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
    base_sample_id = 500000 + int(raw_dir.name.split("_")[-1]) * 10000

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
                is_target_only = random.random() < target_ratio
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
    reasoning_cfg = cfg.dataset.get("reasoning", {})
    enable_reasoning = reasoning_cfg.get("enabled", True)

    tbox_graph = None
    if enable_reasoning:
        tbox_path_cfg = reasoning_cfg.get("tbox_path", "data/ont/input/lubm.ttl")
        tbox_path = Path(tbox_path_cfg) if Path(tbox_path_cfg).is_absolute() else original_cwd / tbox_path_cfg
        tbox_graph = load_optional_tbox(tbox_path)

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
            enable_reasoning=enable_reasoning,
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

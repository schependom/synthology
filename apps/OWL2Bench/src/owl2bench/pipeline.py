import csv
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any

import hydra
import rdflib
from loguru import logger
from omegaconf import DictConfig

RDF_TYPE_URI = str(rdflib.RDF.type)
REPO_ROOT = Path(__file__).resolve().parents[4]


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


def compute_inferred_triples(
    base_graph: rdflib.Graph,
    tbox_graph: rdflib.Graph,
    known_individuals: set[rdflib.term.Node],
    jena_cfg: dict[str, Any],
) -> tuple[
    list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]],
    dict[str, int],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
    list[tuple[str, rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]],
]:
    jena_executable = str(jena_cfg.get("executable", "java"))
    command_template = str(jena_cfg.get("command_template", "")).strip()

    if not command_template:
        raise RuntimeError("No `dataset.reasoning.jena.command_template` configured for OWL2Bench pipeline.")

    with tempfile.TemporaryDirectory(prefix="owl2bench_jena_") as td:
        tmp_dir = Path(td)
        abox_path = tmp_dir / "abox.ttl"
        tbox_path = tmp_dir / "tbox.ttl"
        output_path = tmp_dir / "materialized.ttl"

        base_graph.serialize(destination=str(abox_path), format="turtle")
        tbox_graph.serialize(destination=str(tbox_path), format="turtle")

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

    jena_raw_triples = len(closure_graph)

    for s, p, o in closure_graph:
        raw_inferred_rows.append(("jena_output", s, p, o))
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
        "jena_raw_triples": jena_raw_triples,
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


def _write_split(
    split_dir: Path,
    split_samples: list[tuple[int, list[tuple]]],
    inferred_by_sample: dict[int, list[tuple[str, str, str, int]]],
    target_ratio: float,
    mask_base_facts: bool,
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
    namespace_map: dict[str, str],
    output_dir: Path,
    split_ratios: dict[str, float],
    target_ratio: float,
    num_samples: int,
    mask_base_facts: bool,
    jena_cfg: dict[str, Any],
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
        known_individuals,
        jena_cfg,
    )

    base_clean = []
    for s, p, o in base_graph:
        s_clean = parse_uri(s)
        p_clean = parse_uri(p)
        o_clean = parse_uri(o)
        if p_clean == "type":
            p_clean = "rdf:type"
        base_clean.append((s_clean, p_clean, o_clean))

    samples = _split_samples(base_clean, num_samples, base_sample_id)

    inferred_clean = []
    for s, p, o, hops in inferred_raw:
        s_clean = parse_uri(s)
        p_clean = parse_uri(p)
        o_clean = parse_uri(o)
        if p_clean == "type":
            p_clean = "rdf:type"
        inferred_clean.append((s_clean, p_clean, o_clean, hops))

    inferred_by_sample = {sid: [] for sid, _ in samples}
    if samples:
        inferred_by_sample[samples[0][0]] = inferred_clean

    train_ratio = split_ratios.get("train", 0.8)
    val_ratio = split_ratios.get("val", 0.1)
    test_ratio = split_ratios.get("test", 0.1)
    total_ratio = train_ratio + val_ratio + test_ratio

    total_samples = len(samples)
    train_idx = int(total_samples * (train_ratio / total_ratio))
    val_idx = train_idx + int(total_samples * (val_ratio / total_ratio))

    splits = {
        "train": samples[:train_idx],
        "val": samples[train_idx:val_idx],
        "test": samples[val_idx:],
    }

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
        raw_written = _write_triples_tsv(run_dir / "jena_raw_inferred.tsv", raw_inferred_rows, max_rows)
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
            "Diagnostics exported | dir={} | base={} | jena_raw={} | rejected={} | accepted={}",
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
    namespace_map = dict(cfg.dataset.reasoning.get("namespace_map", {}))
    jena_cfg = dict(cfg.dataset.reasoning.get("jena", {}))
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
            namespace_map=namespace_map,
            output_dir=csv_output_dir,
            split_ratios=split_ratios,
            target_ratio=target_ratio,
            num_samples=num_samples,
            mask_base_facts=mask_base_facts,
            jena_cfg=jena_cfg,
            base_sample_id=700000 + universities * 10000,
            diagnostics_cfg=diagnostics_cfg,
            diagnostics_id=diagnostics_id,
        )

    logger.info("OWL2Bench pipeline completed in {:.2f}s", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()

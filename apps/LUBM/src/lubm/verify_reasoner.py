import csv
import os
import random
import shutil
from pathlib import Path

import hydra
import rdflib
from loguru import logger
from omegaconf import DictConfig

from lubm.parse_to_csv import RDF_TYPE_URI, compute_inferred_triples, is_valid_abox_triple, parse_lubm_directory
from synthology.verification_visualizer import export_base_inferred_graph

REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


def _resolve_path(path_str: str, original_cwd: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return original_cwd / path


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


def _load_tbox_graph(tbox_path: Path) -> rdflib.Graph:
    tbox = rdflib.Graph()
    tbox.parse(tbox_path)
    logger.info(f"Loaded TBox from {tbox_path} with {len(tbox)} triples")
    return tbox


def _build_subset_graph(raw_dir: Path, ttl_glob: str, max_base_facts: int, seed: int) -> rdflib.Graph:
    ttl_files = sorted(raw_dir.glob(ttl_glob))
    if not ttl_files:
        raise FileNotFoundError(f"No TTL files found in {raw_dir} with pattern '{ttl_glob}'")

    all_triples = []
    for ttl_file in ttl_files:
        local_graph = rdflib.Graph()
        local_graph.parse(ttl_file, format="turtle")
        for s, p, o in local_graph:
            if is_valid_abox_triple(s, p, o):
                all_triples.append((s, p, o))

    if not all_triples:
        raise ValueError(f"No valid ABox triples found in {raw_dir}")

    rng = random.Random(seed)
    rng.shuffle(all_triples)

    target_count = min(max_base_facts, len(all_triples))
    subset_graph = rdflib.Graph()
    for triple in all_triples[:target_count]:
        subset_graph.add(triple)

    logger.info(
        f"Subset graph built from {raw_dir}: selected {target_count}/{len(all_triples)} base triples (seed={seed})"
    )
    return subset_graph


def _known_individuals_from_base(base: rdflib.Graph) -> set[rdflib.term.Node]:
    known = set()
    for s, p, o in base:
        known.add(s)
        if str(p) != RDF_TYPE_URI:
            known.add(o)
    return known


def _sample_for_graph_export(
    base: rdflib.Graph,
    inferred: list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]],
    max_base_facts: int,
) -> tuple[rdflib.Graph, list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]]]:
    """Bound graph size to keep visualization readable and below hard caps."""
    if max_base_facts <= 0 or len(base) <= max_base_facts:
        return base, inferred

    sampled_base = rdflib.Graph()
    ordered_base = sorted(base, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))
    for triple in ordered_base[:max_base_facts]:
        sampled_base.add(triple)

    known = _known_individuals_from_base(sampled_base)
    sampled_inferred = []
    for s, p, o, hops in inferred:
        if s not in known:
            continue
        if str(p) != RDF_TYPE_URI and o not in known:
            continue
        sampled_inferred.append((s, p, o, hops))

    return sampled_base, sampled_inferred


def _collect_targets_stats(targets_paths: list[Path]) -> tuple[int, int]:
    inferred_positive = 0
    max_hops = 0

    for targets_path in targets_paths:
        with open(targets_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["label"] == "1" and row["type"] == "inferred":
                    inferred_positive += 1
                    max_hops = max(max_hops, int(row["hops"]))

    return inferred_positive, max_hops


def _assert_minimum_quality(inferred_positive: int, max_hops: int, cfg: DictConfig) -> None:
    min_inferred = int(cfg.checks.get("min_inferred_positive", 1))
    min_hops = int(cfg.checks.get("min_max_hops", 1))

    if inferred_positive < min_inferred:
        raise AssertionError(
            f"Verification failed: inferred_positive={inferred_positive} is below min_inferred_positive={min_inferred}"
        )
    if max_hops < min_hops:
        raise AssertionError(f"Verification failed: max_hops={max_hops} is below min_max_hops={min_hops}")


def _prepare_output_dirs(
    output_root: Path,
    clean_before_run: bool,
    clean_graph: bool = True,
) -> tuple[Path, Path, Path]:
    raw_root = output_root / "raw"
    out_root = output_root / "out"
    graph_root = output_root / "graph"

    if clean_before_run:
        if raw_root.exists():
            shutil.rmtree(raw_root)
        if out_root.exists():
            shutil.rmtree(out_root)
        if clean_graph and graph_root.exists():
            shutil.rmtree(graph_root)

    raw_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    graph_root.mkdir(parents=True, exist_ok=True)
    return raw_root, out_root, graph_root


def _run_csv_export_verification(base: rdflib.Graph, tbox: rdflib.Graph, cfg: DictConfig, output_root: Path) -> None:
    clean_before_run = bool(cfg.output.get("clean_before_run", True))
    dataset_label = str(cfg.output.get("dataset_label", "verify_0"))
    source_ttl_name = str(cfg.output.get("source_ttl_name", "verify_input.ttl"))

    raw_root, out_root, _graph_root = _prepare_output_dirs(output_root, clean_before_run, clean_graph=False)

    raw_dir = raw_root / dataset_label
    out_dir = out_root / dataset_label
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_subset_ttl = raw_dir / source_ttl_name
    base.serialize(destination=str(raw_subset_ttl), format="turtle")

    split_cfg = cfg.output.get("split", {"train": 1.0, "val": 0.0, "test": 0.0})
    target_ratio = float(cfg.output.get("target_ratio", 0.0))
    num_samples_cfg = cfg.output.get("num_samples", 1)
    num_samples = None if num_samples_cfg is None else int(num_samples_cfg)

    parse_lubm_directory(
        raw_dir=raw_dir,
        output_dir=out_dir,
        split_ratios=dict(split_cfg),
        target_ratio=target_ratio,
        num_samples=num_samples,
        enable_reasoning=True,
        tbox_graph=tbox,
    )

    targets_paths = sorted(out_dir.glob("*/targets.csv"))
    if not targets_paths:
        raise AssertionError(f"Verification failed: no targets.csv found under {out_dir}")

    inferred_positive, max_hops = _collect_targets_stats(targets_paths)
    _assert_minimum_quality(inferred_positive, max_hops, cfg)

    logger.success(
        "CSV verification passed: inferred_positive={}, max_hops={}, targets_files={}",
        inferred_positive,
        max_hops,
        len(targets_paths),
    )


def _export_graph_if_enabled(
    base: rdflib.Graph,
    inferred: list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]],
    cfg: DictConfig,
    output_root: Path,
) -> None:
    graph_cfg = cfg.output.get("graph", {})
    enabled = bool(graph_cfg.get("enabled", True))
    if not enabled:
        logger.info("Graph export disabled by configuration")
        return

    output_name = str(graph_cfg.get("name", "verification_reasoning_graph"))
    title = str(graph_cfg.get("title", "Verification Graph (Base + Inferred)"))
    max_base_facts = int(graph_cfg.get("max_base_facts", 100))
    graph_dir = output_root / "graph"

    current_limit = max_base_facts
    for _attempt in range(6):
        graph_base, graph_inferred = _sample_for_graph_export(base, inferred, current_limit)
        logger.info(
            f"Graph export subset: base={len(graph_base)} triples, inferred={len(graph_inferred)} triples (max_base_facts={current_limit})"
        )

        graph_path = export_base_inferred_graph(
            base_graph=graph_base,
            inferred=graph_inferred,
            output_dir=graph_dir,
            output_name=output_name,
            title=title,
        )
        if graph_path.exists():
            return

        if current_limit <= 10:
            break
        current_limit = max(10, current_limit // 2)

    logger.warning("Unable to export verification graph after adaptive downsampling retries")


def _run_toy_expected_checks(inferred: list[tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node, int]]) -> None:
    inferred_map = {(str(s), str(p), str(o)): hops for s, p, o, hops in inferred}

    ex = "http://example.org/"
    alice_student = (f"{ex}Alice", RDF_TYPE_URI, f"{ex}Student")
    alice_person = (f"{ex}Alice", RDF_TYPE_URI, f"{ex}Person")
    bob_faculty = (f"{ex}Bob", RDF_TYPE_URI, f"{ex}Faculty")
    cs_course = (f"{ex}CS101", RDF_TYPE_URI, f"{ex}Course")

    expected = [alice_student, alice_person, bob_faculty, cs_course]
    missing = [triple for triple in expected if triple not in inferred_map]
    if missing:
        raise AssertionError(f"Toy verification failed; missing inferred triples: {missing}")

    if inferred_map[alice_person] < inferred_map[alice_student]:
        raise AssertionError("Toy verification failed: Person(Alice) should be at least as deep as Student(Alice)")

    logger.success("Toy expected-triple checks passed")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/lubm", config_name="verify_reasoner")
def main(cfg: DictConfig) -> None:
    original_cwd = Path(hydra.utils.get_original_cwd())
    output_root = _resolve_path(str(cfg.output.root_dir), original_cwd)

    mode = str(cfg.verification.get("mode", "subset")).lower()

    if mode == "toy":
        base_graph, tbox_graph = _make_toy_graphs()
        logger.info("Running toy verification mode")
    elif mode == "subset":
        source_raw_dir = _resolve_path(str(cfg.source.raw_dir), original_cwd)
        tbox_path = _resolve_path(str(cfg.source.tbox_path), original_cwd)
        ttl_glob = str(cfg.source.get("ttl_glob", "*.ttl"))
        max_base_facts = int(cfg.subset.get("max_base_facts", 250))
        seed = int(cfg.subset.get("seed", 42))

        base_graph = _build_subset_graph(
            raw_dir=source_raw_dir,
            ttl_glob=ttl_glob,
            max_base_facts=max_base_facts,
            seed=seed,
        )
        tbox_graph = _load_tbox_graph(tbox_path)
        logger.info("Running subset verification mode")
    else:
        raise ValueError(f"Unsupported verification.mode '{mode}'. Use 'subset' or 'toy'.")

    known = _known_individuals_from_base(base_graph)
    inferred = compute_inferred_triples(base_graph=base_graph, tbox_graph=tbox_graph, known_individuals=known)

    if mode == "toy":
        _run_toy_expected_checks(inferred)

    _run_csv_export_verification(base=base_graph, tbox=tbox_graph, cfg=cfg, output_root=output_root)
    _export_graph_if_enabled(base=base_graph, inferred=inferred, cfg=cfg, output_root=output_root)

    logger.success(f"Verification completed successfully. Artifacts kept at: {output_root}")


if __name__ == "__main__":
    main()

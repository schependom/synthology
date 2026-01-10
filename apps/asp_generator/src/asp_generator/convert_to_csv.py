import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from synthology.data_structures import (
    Class,
    ExecutableRule,
    Individual,
    KnowledgeGraph,
    Membership,
    Proof,
    Relation,
    Triple,
)

# Add project root to sys.path to allow importing synthology
# Assuming script is in apps/asp_generator/
project_root = Path(__file__).resolve().parents[2]
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))


# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_reldata_kg(rd_kg) -> KnowledgeGraph:
    """
    Converts a reldata KnowledgeGraph to Synthology KnowledgeGraph.
    """
    # Mappings to maintain unique objects
    classes_map: Dict[str, Class] = {}
    relations_map: Dict[str, Relation] = {}
    individuals_map: Dict[str, Individual] = {}

    s_classes = []
    s_relations = []
    s_attributes = []
    s_individuals = []
    s_triples = []
    s_memberships = []
    s_attr_triples = []

    # Helper to mark fact as inferred if needed
    def check_inference(fact, origin_obj):
        # Check standard property names
        is_inf = getattr(origin_obj, "inferred", getattr(origin_obj, "is_inferred", False))
        if is_inf:
            goal = fact.to_atom()
            # Create a dummy rule that signifies origin from ASP generator
            # We must create a new rule instance per fact because 'conclusion' differs
            rule = ExecutableRule("asp_inferred", goal, [])
            proof = Proof(goal, rule=rule)
            fact.proofs.append(proof)

    # Helper to get or create Class
    def get_class(rd_cls) -> Class:
        name = str(rd_cls)
        # Try to get name from attribute if available
        if hasattr(rd_cls, "name"):
            name = rd_cls.name
        # Some reldata classes might be primitives?

        if name not in classes_map:
            idx = len(s_classes)
            s_cls = Class(idx, name)
            classes_map[name] = s_cls
            s_classes.append(s_cls)
        return classes_map[name]

    # Helper to get or create Relation
    def get_relation(rd_rel) -> Relation:
        name = str(rd_rel)
        if hasattr(rd_rel, "name"):
            name = rd_rel.name

        if name not in relations_map:
            idx = len(s_relations)
            s_rel = Relation(idx, name)
            relations_map[name] = s_rel
            s_relations.append(s_rel)
        return relations_map[name]

    # Helper to get or create Individual
    def get_individual(rd_ind) -> Individual:
        name = str(rd_ind)
        if hasattr(rd_ind, "name"):
            name = rd_ind.name  # e.g. "0", "1"
        elif hasattr(rd_ind, "index"):
            name = f"Ind_{rd_ind.index}"

        if name not in individuals_map:
            idx = len(s_individuals)
            s_ind = Individual(idx, name)
            individuals_map[name] = s_ind
            s_individuals.append(s_ind)

            # Reconstruct class memberships if stored on individual
            # Check for 'classes' attribute on reldata individual
            if hasattr(rd_ind, "classes"):
                for mem in rd_ind.classes:
                    # mem is ClassMembership(cls, is_member)
                    # Handle attribute access divergence
                    cls_obj = getattr(mem, "class_type", getattr(mem, "cls", None))
                    if cls_obj:
                        s_cls = get_class(cls_obj)
                        is_member = getattr(mem, "is_member", True)

                        s_mem = Membership(s_ind, s_cls, is_member)
                        check_inference(s_mem, mem)
                        s_ind.classes.append(s_mem)
                        s_memberships.append(s_mem)

        return individuals_map[name]

    # 1. Iterate over triples
    # reldata KG typically exposes 'triples' as a set/list
    if hasattr(rd_kg, "triples"):
        for i, rd_triple in enumerate(rd_kg.triples):
            # Subject
            subj_raw = getattr(rd_triple, "subject", None)
            if not subj_raw:
                continue

            # Relation
            pred_raw = getattr(rd_triple, "predicate", getattr(rd_triple, "relation", None))
            if not pred_raw:
                continue

            # Object
            obj_raw = getattr(rd_triple, "object", None)
            if not obj_raw:
                continue

            # Positive
            pos = getattr(rd_triple, "positive", True)

            s_subj = get_individual(subj_raw)
            s_pred = get_relation(pred_raw)
            s_obj = get_individual(obj_raw)

            s_triple = Triple(s_subj, s_pred, s_obj, pos)
            check_inference(s_triple, rd_triple)
            s_triples.append(s_triple)

    # 2. Iterate over individuals if possible to catch those without triples
    # Some KGs might lists all individuals
    if hasattr(rd_kg, "individuals"):
        # If it's a dict or list
        inds = rd_kg.individuals
        iterable = inds.values() if isinstance(inds, dict) else inds
        for rd_ind in iterable:
            get_individual(rd_ind)

    return KnowledgeGraph(
        attributes=s_attributes,
        classes=s_classes,
        relations=s_relations,
        individuals=s_individuals,
        triples=s_triples,
        memberships=s_memberships,
        attribute_triples=s_attr_triples,
    )


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/asp_generator", config_name="config")
def main(cfg: DictConfig):
    # args are now handled by hydra
    # We still need input_dir, but we can get it from cfg or infer from standard locations?
    # The prompt implies we should base locations on config.

    # Original args: input_dir, output_dir.
    # New logic:
    #   input_dir = cfg.dataset.output_dir (where reldata was saved)
    #   output_dir structure = data/asp/{dataset.name}/{train/test}

    input_dir = cfg.dataset.output_dir
    # Resolve relative path if needed? cfg.dataset.output_dir is usually relative to running location
    # But hydra changes working directory.
    # We should use `hydra.utils.get_original_cwd()` to resolve paths if they are relative to invocation.

    import hydra.utils

    original_cwd = Path(hydra.utils.get_original_cwd())

    # If input_dir starts with ./, it's relative to original_cwd
    input_path = Path(input_dir)
    if not input_path.is_absolute():
        input_path = original_cwd / input_path

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    dataset_name = cfg.dataset.name

    base_output_dir = original_cwd / "data" / "asp" / dataset_name
    # /data/asp/{dataset.name}/train_val
    # /data/asp/{dataset.name}/test
    train_val_dir = base_output_dir / "train_val"
    test_dir = base_output_dir / "test"
    # Final structure:
    #   {base}/train_val/
    #   {base}/test/

    if train_val_dir.exists():
        shutil.rmtree(train_val_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    train_val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        from reldata.io import kg_reader
    except ImportError:
        logger.error("Could not import 'reldata'. verify it is installed.")
        return

    # Identify samples by scanning for unique basenames (e.g. "0", "1")
    unique_basenames = set()
    for f in input_path.iterdir():
        if f.is_file() and not f.name.startswith("."):
            parts = f.name.split(".")
            if parts and parts[0].isdigit():
                unique_basenames.add(parts[0])

    samples: List[KnowledgeGraph] = []

    # Sort to ensure reproducibility if reading order matters (it shouldn't for list, but nice for logs)
    sorted_basenames = sorted(unique_basenames, key=lambda x: int(x))

    for basename in sorted_basenames:
        try:
            rd_kg = kg_reader.KgReader.read(str(input_path), basename)
            if rd_kg:
                # logger.info(f"Converting sample '{basename}'...")
                s_kg = convert_reldata_kg(rd_kg)
                samples.append(s_kg)
        except Exception as e:
            logger.warning(f"Skipping sample '{basename}': {e}")

    if not samples:
        logger.warning("No valid samples found or converted.")
        return

    logger.info(f"Total samples converted: {len(samples)}")

    # Shuffle and split
    # Use cfg.dataset.seed if available, else cfg.seed?
    seed = getattr(cfg.dataset, "seed", 42)
    random.seed(seed)
    random.shuffle(samples)

    train_val_pct = cfg.train_val_pct
    split_idx = int(len(samples) * train_val_pct)

    train_val_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    logger.info(f"Splitting data (train_val_pct={train_val_pct}):")
    logger.info(f"  Train/Val: {len(train_val_samples)} samples -> {train_val_dir}")
    logger.info(f"  Test:      {len(test_samples)} samples -> {test_dir}")

    # Save Train/Val
    if train_val_samples:
        save_samples(train_val_samples, train_val_dir)

    # Save Test
    if test_samples:
        save_samples(test_samples, test_dir)


def save_samples(samples: List[KnowledgeGraph], output_path: Path):
    try:
        KnowledgeGraph.to_csv_batch(samples, str(output_path))
    except TypeError as e:
        logger.error(f"Error calling to_csv_batch: {e}")
        for idx, kg in enumerate(samples):
            fname = output_path / f"sample_{idx:05d}.csv"
            kg.to_csv(str(fname))


if __name__ == "__main__":
    main()

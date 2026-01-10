import logging
import multiprocessing
import os
import random
import shutil
import sys
from concurrent import futures
from pathlib import Path
from typing import Dict, Tuple

import hydra
from omegaconf import DictConfig

# Now safe to import
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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

project_root = Path(__file__).resolve().parents[4]  # apps/asp_generator/src/asp_generator/
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _process_and_save_task(args: Tuple[str, str, str]) -> bool:
    """
    Worker task: Read -> Convert -> Write to Disk immediately.
    Args:
        args: Tuple of (input_dir_str, output_filepath_str, basename)
    """
    input_dir, output_filepath, basename = args
    try:
        # Import inside worker to avoid pickling complex reader objects
        from reldata.io import kg_reader

        # 1. Read
        rd_kg = kg_reader.KgReader.read(input_dir, basename)

        # 2. Convert
        if rd_kg:
            kg = convert_reldata_kg(rd_kg)

            # 3. Write immediately
            # Ensure parent dir exists (redundancy for safety)
            Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)

            kg.to_csv(output_filepath)
            return True

    except Exception as e:
        # Use simple print for worker errors to avoid Logging lock contention
        sys.stderr.write(f"Worker Error [{basename}]: {e}\n")
        return False

    return False


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
        is_inf = getattr(origin_obj, "inferred", getattr(origin_obj, "is_inferred", False))
        if is_inf:
            goal = fact.to_atom()
            rule = ExecutableRule("asp_inferred", goal, [])
            proof = Proof(goal, rule=rule)
            fact.proofs.append(proof)

    # Helper to get or create Class
    def get_class(rd_cls) -> Class:
        name = str(rd_cls)
        if hasattr(rd_cls, "name"):
            name = rd_cls.name
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
            name = rd_ind.name
        elif hasattr(rd_ind, "index"):
            name = f"Ind_{rd_ind.index}"

        if name not in individuals_map:
            idx = len(s_individuals)
            s_ind = Individual(idx, name)
            individuals_map[name] = s_ind
            s_individuals.append(s_ind)

            if hasattr(rd_ind, "classes"):
                for mem in rd_ind.classes:
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
    if hasattr(rd_kg, "triples"):
        for i, rd_triple in enumerate(rd_kg.triples):
            subj_raw = getattr(rd_triple, "subject", None)
            if not subj_raw:
                continue
            pred_raw = getattr(rd_triple, "predicate", getattr(rd_triple, "relation", None))
            if not pred_raw:
                continue
            obj_raw = getattr(rd_triple, "object", None)
            if not obj_raw:
                continue

            pos = getattr(rd_triple, "positive", True)

            s_subj = get_individual(subj_raw)
            s_pred = get_relation(pred_raw)
            s_obj = get_individual(obj_raw)

            s_triple = Triple(s_subj, s_pred, s_obj, pos)
            check_inference(s_triple, rd_triple)
            s_triples.append(s_triple)

    # 2. Iterate over individuals
    if hasattr(rd_kg, "individuals"):
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
    import hydra.utils

    original_cwd = Path(hydra.utils.get_original_cwd())

    input_dir = cfg.dataset.output_dir
    input_path = Path(input_dir)
    if not input_path.is_absolute():
        input_path = original_cwd / input_path

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    dataset_name = cfg.dataset.name
    base_output_dir = original_cwd / "data" / "asp" / dataset_name

    train_dir = base_output_dir / "train"
    val_dir = base_output_dir / "val"
    test_dir = base_output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        logger.info(f"Preparing output directory: {d}")
        if d.exists():
            logger.info(f"Cleaning existing directory: {d}")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    try:
        from reldata import io as reldata_io
    except ImportError:
        logger.error("Could not import 'reldata'. verify it is installed.")
        return

    try:
        # Lightweight discovery (filenames only)
        all_basenames = reldata_io.find_knowledge_graphs(str(input_path))
        logger.info(f"Found {len(all_basenames)} knowledge graphs to process.")
        if not all_basenames:
            logger.warning("No samples found.")
            return
    except Exception as e:
        logger.error(f"Error finding knowledge graphs: {e}")
        return

    # Shuffle & Split Strings (FAST)
    seed = getattr(cfg.dataset, "seed", 23)
    random.seed(seed)

    all_basenames = list(all_basenames)
    random.shuffle(all_basenames)

    train_val_pct = cfg.train_val_pct
    val_pct = getattr(cfg, "val_pct", 0.1)

    split_idx_test = int(len(all_basenames) * train_val_pct)
    train_val_names = all_basenames[:split_idx_test]
    test_names = all_basenames[split_idx_test:]

    val_count = int(len(train_val_names) * val_pct)
    if len(train_val_names) > 0 and val_count == 0 and val_pct > 0:
        val_count = 1

    train_names = train_val_names[val_count:]
    val_names = train_val_names[:val_count]

    logger.info(f"Splitting data plan: Train={len(train_names)}, Val={len(val_names)}, Test={len(test_names)}")

    # Create Tasks
    tasks = []

    def add_tasks(names_list, dest_dir):
        for idx, bn in enumerate(names_list):
            out_file = dest_dir / f"sample_{idx:05d}.csv"
            # Store strings to keep task pickle size tiny
            tasks.append((str(input_path), str(out_file), bn))

    add_tasks(train_names, train_dir)
    add_tasks(val_names, val_dir)
    add_tasks(test_names, test_dir)

    # Process in Parallel
    num_workers = max(1, multiprocessing.cpu_count() - 2)  # Leave 2 cores for OS/UI
    logger.info(f"Processing {len(tasks)} samples with {num_workers} workers...")

    success_count = 0

    # This prevents the UI from looking "hung" while waiting for the first chunk
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(_process_and_save_task, task): task[2] for task in tasks}

        for future in futures.as_completed(future_to_file):
            basename = future_to_file[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                if success_count % 100 == 0:
                    logger.info(f"Progress: {success_count}/{len(tasks)} converted.")
            except Exception as exc:
                logger.error(f"Sample {basename} generated an exception: {exc}")

    logger.info(f"Complete. Successfully converted: {success_count}/{len(tasks)}")


if __name__ == "__main__":
    main()

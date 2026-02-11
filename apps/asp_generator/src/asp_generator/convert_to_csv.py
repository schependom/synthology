
import csv
import glob
import multiprocessing
import os
import random
import shutil
import sys
from concurrent import futures
from pathlib import Path
from typing import Dict, Tuple, List

import hydra
from loguru import logger
from omegaconf import DictConfig
from reldata import io as reldata_io
from reldata.io import kg_reader

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


def _process_and_save_task(args: Tuple[str, str, str, str]) -> bool:
    """
    Worker task: Read -> Convert -> Write temp CSVs.
    Args:
        args: Tuple of (input_dir_str, temp_output_dir_str, basename, sample_id)
    """
    input_dir, temp_output_dir, basename, sample_id = args
    try:
        # Read
        rd_kg = kg_reader.KgReader.read(input_dir, basename)

        # Convert
        if rd_kg:
            kg = convert_reldata_kg(rd_kg)
            
            # Get standard rows
            rows = kg.to_standard_rows(sample_id)
            
            facts_rows = []
            targets_rows = []
            
            for row in rows:
                # FACTS: Only Positive Base Facts usually, but let's follow ont_generator logic
                if row["type"] == "base_fact" and row["label"] == 1:
                     # Minimal columns for facts.csv
                    min_row = {
                        "sample_id": row["sample_id"],
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"]
                    }
                    facts_rows.append(min_row)
                    # Base facts are ALSO targets
                    targets_rows.append(row)
                else:
                    targets_rows.append(row)

            # Write temp files
            temp_path = Path(temp_output_dir)
            temp_path.mkdir(parents=True, exist_ok=True)
            
            facts_file = temp_path / f"{sample_id}_facts.csv"
            targets_file = temp_path / f"{sample_id}_targets.csv"
            
            if facts_rows:
                with open(facts_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object"])
                    writer.writeheader()
                    writer.writerows(facts_rows)
            
            if targets_rows:
                with open(targets_file, "w", newline="", encoding="utf-8") as f:
                    keys = list(targets_rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(targets_rows)
                    
            return True

    except Exception as e:
        sys.stderr.write(f"Worker Error [{basename}]: {e}\n")
        import traceback
        traceback.print_exc()
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

    # Iterate over triples
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

    # Iterate over individuals
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

def merge_csvs(temp_dir: Path, output_dir: Path):
    """Merges temp CSVs into single facts.csv and targets.csv"""
    logger.info(f"Merging temp CSVs from {temp_dir} to {output_dir}...")
    
    # Merge facts
    all_facts_files = sorted(temp_dir.glob("*_facts.csv"))
    if all_facts_files:
        facts_out = output_dir / "facts.csv"
        with open(facts_out, "w", newline="", encoding="utf-8") as outfile:
            # Write header from first file
            with open(all_facts_files[0], "r", encoding="utf-8") as f:
                header = f.readline()
                outfile.write(header)
            
            # Copy data
            for fname in all_facts_files:
                with open(fname, "r", encoding="utf-8") as infile:
                    next(infile) # skip header
                    shutil.copyfileobj(infile, outfile)
                    
    # Merge targets
    all_targets_files = sorted(temp_dir.glob("*_targets.csv"))
    if all_targets_files:
        targets_out = output_dir / "targets.csv"
        with open(targets_out, "w", newline="", encoding="utf-8") as outfile:
             # Write header from first file
            with open(all_targets_files[0], "r", encoding="utf-8") as f:
                header = f.readline()
                outfile.write(header)
                
            # Copy data
            for fname in all_targets_files:
                with open(fname, "r", encoding="utf-8") as infile:
                    next(infile)
                    shutil.copyfileobj(infile, outfile)

    logger.success(f"Merged {len(all_facts_files)} facts files and {len(all_targets_files)} targets files.")


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/asp_generator", config_name="config")
def main(cfg: DictConfig):
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
    
    # IDs offset for each split to ensure uniqueness if needed, 
    # though sample_id is string so "train_0" vs "test_0" would contain uniqueness info if we used prefix.
    # But RRN expects integer-like sample_ids usually?
    # standard format: sample_id column.
    # We will use 100000 series for ASP to avoid collision with others?
    # Or just simple incrementing.
    
    logger.info(f"Splitting data plan: Train={len(train_names)}, Val={len(val_names)}, Test={len(test_names)}")

    # Process each split
    
    def process_split(split_names, split_dir, start_id):
        if not split_names:
            return
            
        temp_dir = split_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        tasks = []
        for idx, bn in enumerate(split_names):
            sid = str(start_id + idx)
            tasks.append((str(input_path), str(temp_dir), bn, sid))
            
        # Process in Parallel
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        logger.info(f"Processing {len(tasks)} samples for {split_dir.name} with {num_workers} workers...")
        
        success_count = 0
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {executor.submit(_process_and_save_task, task): task for task in tasks}
            
            for future in futures.as_completed(future_to_task):
                try:
                    if future.result():
                        success_count += 1
                    if success_count % 100 == 0:
                        logger.info(f"Progress: {success_count}/{len(tasks)} converted.")
                except Exception as exc:
                    logger.error(f"Task generated exception: {exc}")

        # Merge
        merge_csvs(temp_dir, split_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir)

    process_split(train_names, train_dir, start_id=100000)
    process_split(val_names, val_dir, start_id=200000)
    process_split(test_names, test_dir, start_id=300000)

    logger.success(f"CSV conversion complete.")


if __name__ == "__main__":
    main()

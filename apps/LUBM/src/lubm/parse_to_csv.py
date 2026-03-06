import os
import csv
import urllib.parse
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
import rdflib


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
        "http://www."
    ]
    
    for prefix in prefixes_to_strip:
        if uri.startswith(prefix):
            uri = uri[len(prefix):]
            break

    # Clean up trailing extensions if any
    if uri.endswith(".edu"):
        uri = uri[:-4]
        
    # Unquote any URL encodings
    uri = urllib.parse.unquote(uri)
    return uri


def parse_lubm_directory(raw_dir: Path, output_dir: Path):
    """
    Parses all .ttl files in a given LUBM raw directory and exports 
    them as facts.csv and targets.csv in standard format.
    """
    if not raw_dir.exists():
        logger.error(f"Cannot find raw directory {raw_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # RRN expects arbitrary sample IDs for independent KGs. 
    # Use 500000+ to avoid collision with ASP (100000+) and ONT generators.
    sample_id = 500000 + int(raw_dir.name.split("_")[-1])

    facts_rows = []
    targets_rows = []
    
    ttl_files = list(raw_dir.glob("*.ttl"))
    logger.info(f"Parsing {len(ttl_files)} TTL files in {raw_dir}...")

    # We use a single Graph across all TTL files in a particular size configuration 
    # (e.g. lubm_1, lubm_5) to consolidate knowledge. LUBM produces consolidated graphs.
    combined_graph = rdflib.Graph()
    for ttl in ttl_files:
        try:
            combined_graph.parse(ttl, format="turtle")
        except Exception as e:
            logger.error(f"Error parsing {ttl.name}: {e}")

    logger.info(f"Parsed {len(combined_graph)} triples. Generating CSV rows...")

    for subject, predicate, obj in combined_graph:
        s_clean = parse_uri(subject)
        p_clean = parse_uri(predicate)
        o_clean = parse_uri(obj)
        
        # We only care about object properties or type assignments
        if isinstance(obj, rdflib.Literal):
            # Skip literal properties like 'name' to keep KG strictly relational
            continue
            
        # Re-map standard rdf:type since it's commonly used by RRN as rdf:type
        if p_clean == "type":
            p_clean = "rdf:type"

        fact = {
            "sample_id": str(sample_id),
            "subject": s_clean,
            "predicate": p_clean,
            "object": o_clean
        }

        target = {
            "sample_id": str(sample_id),
            "subject": s_clean,
            "predicate": p_clean,
            "object": o_clean,
            "label": 1,
            "truth_value": "True",
            "type": "base_fact",
            "hops": 0,
            "corruption_method": ""  # Blank for ground truth
        }

        facts_rows.append(fact)
        targets_rows.append(target)

    # Write facts.csv
    facts_path = output_dir / "facts.csv"
    if facts_rows:
        with open(facts_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object"])
            writer.writeheader()
            writer.writerows(facts_rows)

    # Write targets.csv
    targets_path = output_dir / "targets.csv"
    if targets_rows:
        with open(targets_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object", "label", "truth_value", "type", "hops", "corruption_method"])
            writer.writeheader()
            writer.writerows(targets_rows)

    logger.success(f"Generated {len(facts_rows)} facts in {output_dir}")


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")

@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/lubm", config_name="config")
def main(cfg: DictConfig):
    original_cwd = Path(hydra.utils.get_original_cwd())
    output_dir_cfg = cfg.dataset.output_dir
    base_dir = Path(output_dir_cfg) if Path(output_dir_cfg).is_absolute() else original_cwd / output_dir_cfg
    
    raw_dir_base = base_dir / "raw"
    
    if not raw_dir_base.exists():
        logger.error(f"Base data directory not found: {raw_dir_base}")
        return

    dataset_configs = cfg.dataset.sizes
    
    for univ_count in dataset_configs:
        size_label = f"lubm_{univ_count}"
        raw_ds = raw_dir_base / size_label
        
        if not raw_ds.exists():
            logger.warning(f"Raw directory for '{size_label}' does not exist, skipping CSV parsing.")
            continue
            
        output_dir = base_dir / size_label # data/lubm/lubm_1
        logger.info(f"Processing CSV output for {size_label}...")
        parse_lubm_directory(raw_ds, output_dir)


if __name__ == "__main__":
    main()

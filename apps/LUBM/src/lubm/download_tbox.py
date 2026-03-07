import os
import ssl
from pathlib import Path
import urllib.request
import rdflib
from loguru import logger

def download_lubm_tbox():
    """
    Downloads the LUBM (Lehigh University Benchmark) TBox ontology 
    and saves it as both XML (.owl) and Turtle (.ttl) formats.
    """
    
    # Official LUBM Ontology URL
    # Sometimes encounters SSL cert issues depending on local Python env
    url = "https://swat.cse.lehigh.edu/onto/univ-bench.owl"
    
    # Bypass SSL verification if needed
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Determine absolute path pointing to data/ont/input
    repo_root = Path(os.environ.get("SYNTHOLOGY_ROOT", "../../../..")).resolve()
    output_dir = repo_root / "data" / "ont" / "input"
    
    if not output_dir.exists():
        logger.warning(f"Output directory {output_dir} does not exist. Creating it.")
        output_dir.mkdir(parents=True, exist_ok=True)
        
    logger.info(f"Downloading LUBM TBox from {url}...")
    
    # Use rdflib to fetch and parse the XML format
    g = rdflib.Graph()
    try:
        g.parse(url, format="xml")
        logger.info(f"Successfully parsed XML ontology with {len(g)} statements.")
        
        # Save as TTL for the custom ontology generator
        ttl_path = output_dir / "lubm.ttl"
        g.serialize(destination=str(ttl_path), format="turtle")
        logger.success(f"Saved LUBM TBox as Turtle: {ttl_path}")
        
        # Save as OWL for completeness
        owl_path = output_dir / "lubm.owl"
        g.serialize(destination=str(owl_path), format="xml")
        logger.success(f"Saved LUBM TBox as XML: {owl_path}")
        
    except Exception as e:
        logger.error(f"Failed to download or parse the LUBM ontology: {e}")

if __name__ == "__main__":
    download_lubm_tbox()

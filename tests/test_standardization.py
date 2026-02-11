
import pytest
import shutil
import sys
import subprocess
from pathlib import Path
import pandas as pd
from hydra import compose, initialize
from omegaconf import OmegaConf

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "apps" / "ont_generator" / "src"))
sys.path.append(str(project_root / "apps" / "RRN" / "src"))

from ont_generator.create_data import KGEDatasetGenerator
from rrn.dataloading.dataset import RRNDataset

# ---------------------------------------------------------------------------- #
#                                 TEST FIXTURES                                #
# ---------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ont_output_dir():
    """Generates a small ONT dataset for testing."""
    output_dir = Path("tests/tmp_ont_test_data")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    with initialize(version_base=None, config_path="../configs/ont_generator"):
        cfg = compose(config_name="config", overrides=[
            "dataset.n_train=10", 
            "dataset.n_val=5", 
            "dataset.n_test=5",
            f"dataset.output_dir={output_dir}",
            "logging.level=WARNING",
            "dataset.min_individuals=5",
            "dataset.max_individuals=10",
        ])
        
    generator = KGEDatasetGenerator(cfg=cfg, verbose=False)
    generator.generate_dataset(
        n_train=cfg.dataset.n_train,
        n_val=cfg.dataset.n_val,
        n_test=cfg.dataset.n_test,
        min_individuals=cfg.dataset.min_individuals,
        max_individuals=cfg.dataset.max_individuals,
        min_rules=cfg.dataset.min_rules,
        max_rules=cfg.dataset.max_rules,
        min_proofs_per_rule=cfg.dataset.min_proofs_per_rule,
    )
    
    yield output_dir
    # Cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)

@pytest.fixture(scope="module")
def asp_output_dir():
    """Runs ASP adapter to generate standard data."""
    # Assuming ASP data exists in data/asp/family_tree/ (checked in previous steps)
    # We will run the adapter on a small subset or just check existing output if present
    # To be safe and independent, we should probably run the adapter command
    
    # We'll just verify the existing outcome from the previous manual run in data/asp/family_tree
    # If not present, this test might skip or fail. 
    # Better: Ensure it runs.
    
    cmd = [
        "uv", "run", "--package", "asp_generator", 
        "python", "apps/asp_generator/src/asp_generator/convert_to_csv.py",
        "dataset=family_tree",
        "train_val_pct=0.8",
        "val_pct=0.1"
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    return Path("data/asp/family_tree")

# ---------------------------------------------------------------------------- #
#                                 HELPER CHECKS                                #
# ---------------------------------------------------------------------------- #

def check_standard_format(split_dir: Path):
    facts_path = split_dir / "facts.csv"
    targets_path = split_dir / "targets.csv"
    
    assert facts_path.exists(), f"facts.csv missing in {split_dir}"
    assert targets_path.exists(), f"targets.csv missing in {split_dir}"
    
    facts_df = pd.read_csv(facts_path)
    targets_df = pd.read_csv(targets_path)
    
    required_facts = {"sample_id", "subject", "predicate", "object"}
    assert required_facts.issubset(facts_df.columns)
    
    required_targets = {"sample_id", "subject", "predicate", "object", "label", "truth_value", "type", "hops", "corruption_method"}
    assert required_targets.issubset(targets_df.columns)
    
    # Check content logic
    if "hops" in targets_df.columns:
        inferred = targets_df[targets_df["type"].isin(["inf_root", "inf_intermediate"])]
        if not inferred.empty:
            assert inferred["hops"].max() > 0, "Inferred facts should have > 0 hops"

# ---------------------------------------------------------------------------- #
#                                     TESTS                                    #
# ---------------------------------------------------------------------------- #

def test_ont_generator_format(ont_output_dir):
    """Verify ONT generator output format."""
    check_standard_format(ont_output_dir / "train")
    check_standard_format(ont_output_dir / "val")
    check_standard_format(ont_output_dir / "test")

def test_asp_generator_format(asp_output_dir):
    """Verify ASP generator adapter output format."""
    check_standard_format(asp_output_dir / "train")
    check_standard_format(asp_output_dir / "val")
    check_standard_format(asp_output_dir / "test")

def test_rrn_loader_ont(ont_output_dir):
    """Verify RRN loader with ONT data."""
    class MockSchema:
        classes = []
        relations = []
        def get_class_index(self, name): return 0
        def get_relation_index(self, name): return 0
    
    dataset = RRNDataset(str(ont_output_dir / "train"), MockSchema())
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, dict) # Preprocessed sample
    
def test_rrn_loader_asp(asp_output_dir):
    """Verify RRN loader with ASP data."""
    class MockSchema:
        classes = []
        relations = []
        def get_class_index(self, name): return 0
        def get_relation_index(self, name): return 0
        
    dataset = RRNDataset(str(asp_output_dir / "train"), MockSchema())
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, dict)

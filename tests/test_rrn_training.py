"""
Unit tests for RRN training pipeline.
Tests data loading, preprocessing, and basic model instantiation.
"""

import pytest
import shutil
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "apps" / "ont_generator" / "src"))
sys.path.insert(0, str(project_root / "apps" / "RRN" / "src"))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from ont_generator.create_data import KGEDatasetGenerator
from rrn.dataloading.datamodule import RRNDataModule
from rrn.dataloading.schema import scan_schema
from rrn.models.rrn_module import RRNSystem
from synthology.data_structures import Class, Relation


@pytest.fixture(scope="module")
def tiny_dataset():
    """Generates a tiny ONT dataset for testing training."""
    output_dir = Path("tests/tmp_rrn_train_data")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Clear any existing Hydra state
    GlobalHydra.instance().clear()
    
    config_dir = str(project_root / "configs" / "ont_generator")
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[
            "dataset.n_train=5", 
            "dataset.n_val=2", 
            "dataset.n_test=2",
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
    
    # Clear Hydra state after fixture setup
    GlobalHydra.instance().clear()
    
    yield output_dir
    
    # Cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_rrn_datamodule_loading(tiny_dataset):
    """Test that RRNDataModule can load the generated data."""
    # Clear Hydra state
    GlobalHydra.instance().clear()
    
    config_dir = str(project_root / "configs" / "rrn")
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[
            f"data.dataset.train_path={tiny_dataset}/train",
            f"data.dataset.val_path={tiny_dataset}/val",
            f"data.dataset.test_path={tiny_dataset}/test",
            "max_epochs=1",
        ])
        
        # Create DataModule
        dm = RRNDataModule(cfg)
        dm.setup("fit")
        
        # Verify datasets are loaded
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
        
        # Verify schema is populated
        assert dm.schema is not None
        assert len(dm.schema.classes) > 0
        
        # Verify we can iterate through a batch
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        assert "base_triples" in batch
        assert "base_memberships" in batch
        assert "all_triples" in batch
        assert "all_memberships" in batch
        
        print(f"✅ DataModule loaded {len(dm.train_dataset)} train samples")
        print(f"✅ Schema has {len(dm.schema.classes)} classes, {len(dm.schema.relations)} relations")
    
    GlobalHydra.instance().clear()


def test_rrn_model_instantiation(tiny_dataset):
    """Test that RRNSystem can be instantiated with generated data schema."""
    GlobalHydra.instance().clear()
    
    config_dir = str(project_root / "configs" / "rrn")
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[
            f"data.dataset.train_path={tiny_dataset}/train",
            f"data.dataset.val_path={tiny_dataset}/val",
            f"data.dataset.test_path={tiny_dataset}/test",
        ])
        
        # Create DataModule to get schema
        dm = RRNDataModule(cfg)
        
        # Create model using schema
        classes = [Class(i, name) for i, name in enumerate(dm.schema.class_names)]
        relations = [Relation(i, name) for i, name in enumerate(dm.schema.relation_names)]
        
        model = RRNSystem(cfg, classes=classes, relations=relations)
        
        assert model is not None
        assert len(model.mlps) == 1 + len(relations)  # 1 class MLP + 1 per relation
        
        print(f"✅ RRNSystem instantiated with {len(classes)} classes, {len(relations)} relations")
    
    GlobalHydra.instance().clear()

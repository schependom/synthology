import sys
import os
import pytest
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from synthology.data_structures import KnowledgeGraph, Triple, Individual, Relation, Class, Membership

# Add ont_generator src to path (relative to this file)
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path.resolve()))

from generate import KGenerator
from negative_sampler import NegativeSampler
from create_data import KGEDatasetGenerator

# Mock paths - assume tests run from root of repo
ONTOLOGY_PATH = "data/ont/family.ttl"

@pytest.fixture
def mock_cfg():
    """Create a mock Hydra configuration."""
    cfg = OmegaConf.create({
        "ontology": {
            "path": ONTOLOGY_PATH
        },
        "dataset": {
            "n_train": 2,
            "n_test": 1,
            "min_individuals": 2,
            "max_individuals": 10,
            "min_rules": 1,
            "max_rules": 2,
            "min_proofs_per_rule": 1,
            "output_dir": "data/output/test",
            "seed": 42,
            "export_proofs": False,
            "export_graphs": False
        },
        "generator": {
            "max_recursion": 2,
            "global_max_depth": 5,
            "max_proofs_per_atom": 5,
            "individual_pool_size": 20,
            "individual_reuse_prob": 0.9,
            "use_signature_sampling": True
        },
        "negative_sampling": {
            "strategy": "random",
            "ratio": 1.0,
            "corrupt_base_facts": False
        },
        "logging": {
            "level": "INFO"
        }
    })
    return cfg

@pytest.fixture
def generator(mock_cfg):
    """Fixture for KGenerator instance."""
    if not Path(ONTOLOGY_PATH).exists():
        pytest.skip(f"Ontology file not found at {ONTOLOGY_PATH}")
        
    return KGenerator(
        cfg=mock_cfg,
        verbose=False
    )

def test_kgenerator_initialization(generator):
    """Test that KGenerator initializes and parses ontology correctly."""
    assert len(generator.parser.rules) > 0
    assert len(generator.parser.classes) > 0
    assert len(generator.parser.relations) > 0

def test_generate_proofs_for_rule(generator):
    """Test generating proofs for a specific rule."""
    if not generator.parser.rules:
        pytest.skip("No rules found in ontology")
        
    # Find a rule name to test
    rule_name = next(iter(generator.parser.rules)).name
    proofs = generator.generate_proofs_for_rule(rule_name, n_instances=1)
    
    # We might not find proofs if the rule is hard to satisfy with empty pool, 
    # but the call should not crash.
    assert isinstance(proofs, list)
    if proofs:
        assert proofs[0].goal.is_ground()

def test_negative_sampler_strategies(generator, mock_cfg):
    """Test that NegativeSampler can add negatives using different strategies."""
    # Create a small dummy KG
    ind1 = Individual(0, "Ind_0")
    ind2 = Individual(1, "Ind_1")
    rel = Relation(0, "hasParent")
    triple = Triple(ind1, rel, ind2, positive=True, proofs=[])
    
    kg = KnowledgeGraph(
        attributes=[],
        classes=list(generator.schema_classes.values()),
        relations=list(generator.schema_relations.values()),
        individuals=[ind1, ind2],
        triples=[triple],
        memberships=[],
        attribute_triples=[]
    )
    
    sampler = NegativeSampler(
        schema_classes=generator.schema_classes,
        schema_relations=generator.schema_relations,
        cfg=mock_cfg,
        domains=generator.parser.domains,
        ranges=generator.parser.ranges,
        verbose=False
    )
    
    # Test Random Strategy
    kg_random = sampler.add_negative_samples(kg, strategy="random", ratio=1.0)
    assert len(kg_random.triples) > 1
    assert any(not t.positive for t in kg_random.triples)

def test_full_pipeline_integration(mock_cfg):
    """Integration test for KGEDatasetGenerator."""
    if not Path(ONTOLOGY_PATH).exists():
        pytest.skip(f"Ontology file not found at {ONTOLOGY_PATH}")

    ds_gen = KGEDatasetGenerator(
        cfg=mock_cfg,
        verbose=False
    )
    
    # Generate tiny dataset
    train, test = ds_gen.generate_dataset(
        n_train=mock_cfg.dataset.n_train,
        n_test=mock_cfg.dataset.n_test,
        min_individuals=mock_cfg.dataset.min_individuals,
        max_individuals=mock_cfg.dataset.max_individuals,
        min_rules=mock_cfg.dataset.min_rules,
        max_rules=mock_cfg.dataset.max_rules,
        min_proofs_per_rule=mock_cfg.dataset.min_proofs_per_rule
    )
    
    assert len(train) == mock_cfg.dataset.n_train
    assert len(test) == mock_cfg.dataset.n_test
    
    # Check inductive split
    train_prefixes = {ind.name.split('_')[0] for ind in train[0].individuals}
    test_prefixes = {ind.name.split('_')[0] for ind in test[0].individuals}
    
    # Note: Depending on pool usage, we might get 'train' or 'test' prefix
    # But specifically, test set generation should produce test prefixes
    if len(test) > 0:
        test_prefixes = {ind.name.split('_')[0] for ind in test[0].individuals}
        assert "test" in test_prefixes

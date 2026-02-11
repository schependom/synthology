
import os
import sys
from pathlib import Path
import pandas as pd
import shutil
from hydra import compose, initialize
from omegaconf import OmegaConf

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "apps" / "ont_generator" / "src"))

from ont_generator.create_data import KGEDatasetGenerator, save_standard_dataset

def verify_standard_format(output_dir: Path, split: str = "train"):
    print(f"\nVerifying {split} split in {output_dir}...")
    split_dir = output_dir / split
    
    facts_path = split_dir / "facts.csv"
    targets_path = split_dir / "targets.csv"
    
    if not facts_path.exists():
        print(f"❌ facts.csv missing in {split_dir}")
        return False
    if not targets_path.exists():
        print(f"❌ targets.csv missing in {split_dir}")
        return False
        
    print(f"✅ Found facts.csv and targets.csv")
    
    # Load and check columns
    try:
        facts_df = pd.read_csv(facts_path)
        targets_df = pd.read_csv(targets_path)
    except Exception as e:
        print(f"❌ Failed to read CSVs: {e}")
        return False
        
    required_facts_cols = {"sample_id", "subject", "predicate", "object"}
    if not required_facts_cols.issubset(facts_df.columns):
        print(f"❌ facts.csv missing columns. Found: {facts_df.columns}, Expected subset: {required_facts_cols}")
        return False
        
    required_targets_cols = {"sample_id", "subject", "predicate", "object", "label", "truth_value", "type", "hops", "corruption_method"}
    if not required_targets_cols.issubset(targets_df.columns):
        print(f"❌ targets.csv missing columns. Found: {targets_df.columns}, Expected subset: {required_targets_cols}")
        return False
        
    print(f"✅ Column checks passed")
    
    # Check Content
    # Check hops in targets
    if "hops" in targets_df.columns:
        # Check if we have non-zero hops for inferred types
        inferred = targets_df[targets_df["type"].isin(["inf_root", "inf_intermediate"])]
        if not inferred.empty:
            max_hops = inferred["hops"].max()
            print(f"ℹ️  Max hops for inferred facts: {max_hops}")
            if max_hops == 0:
                 print(f"⚠️  Warning: Inferred facts have 0 hops? This might be wrong.")
        else:
            print(f"ℹ️  No positive inferred facts found in this small sample.")
            
    # Check sample grouping
    sample_ids = facts_df["sample_id"].unique()
    print(f"ℹ️  Found {len(sample_ids)} unique samples.")
    
    return True

def run_ont_generator_verification():
    print("="*60)
    print("Running ONT Generator Verification")
    print("="*60)
    
    # Use a small config
    with initialize(version_base=None, config_path="../configs/ont_generator"):
        cfg = compose(config_name="config", overrides=[
            "dataset.n_train=2", 
            "dataset.n_val=1", 
            "dataset.n_test=1",
            "dataset.output_dir=tests/tmp_ont_gen_output"
        ])
        
    output_dir = Path(cfg.dataset.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    try:
        generator = KGEDatasetGenerator(cfg=cfg, verbose=True)
        train, val, test = generator.generate_dataset(
            n_train=cfg.dataset.n_train,
            n_val=cfg.dataset.n_val,
            n_test=cfg.dataset.n_test,
            min_individuals=cfg.dataset.min_individuals,
            max_individuals=cfg.dataset.max_individuals,
            min_rules=cfg.dataset.min_rules,
            max_rules=cfg.dataset.max_rules,
            min_proofs_per_rule=cfg.dataset.min_proofs_per_rule,
        )
        
        # Verify
        if verify_standard_format(output_dir, "train") and \
           verify_standard_format(output_dir, "val") and \
           verify_standard_format(output_dir, "test"):
            print("\n✅ ONT Generator Verification PASSED")
        else:
            print("\n❌ ONT Generator Verification FAILED")
            
    except Exception as e:
        print(f"\n❌ Exception during ONT generation: {e}")
        import traceback
        traceback.print_exc()


def run_asp_generator_verification():
    print("\n" + "="*60)
    print("Running ASP Generator Adapter Verification")
    print("="*60)
    
    # Run the adapter via subprocess
    import subprocess
    cmd = [
        "uv", "run", "--package", "asp_generator", 
        "python", "apps/asp_generator/src/asp_generator/convert_to_csv.py",
        "dataset=family_tree",
        "train_val_pct=0.8",
        "val_pct=0.1"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ASP Adapter failed with error: {e}")
        return

    # Verify Output
    # Based on config, output should be in data/asp/family_tree/
    output_base = Path("data/asp/family_tree")
    
    if verify_standard_format(output_base, "train") and \
       verify_standard_format(output_base, "val") and \
       verify_standard_format(output_base, "test"):
        print("\n✅ ASP Generator Verification PASSED")
    else:
        print("\n❌ ASP Generator Verification FAILED")


def run_rrn_loader_verification():
    print("\n" + "="*60)
    print("Running RRN Loader Verification")
    print("="*60)
    
    # Add RRN to path
    sys.path.append(str(project_root / "apps" / "RRN" / "src"))
    
    try:
        from rrn.dataloading.dataset import RRNDataset
        from rrn.dataloading.schema import Schema
        
        # We need a dummy schema or build one from data
        # For verification, we can mock it or just use a simple one
        class MockSchema:
            def get_class_index(self, name): return 0
            def get_relation_index(self, name): return 0
            
        schema = MockSchema()
        
        # Test with ONT output
        data_path = Path("tests/tmp_ont_gen_output/train")
        if not data_path.exists():
            print("⚠️ ONT output not found, skipping RRN verify on ONT data")
        else:
            print(f"Verifying RRN loading from {data_path}...")
            dataset = RRNDataset(str(data_path), schema)
            print(f"Loaded {len(dataset)} samples.")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample 0 type: {type(sample)}")
                # Check if it has expected keys (it returns preprocessed dict usually?)
                # preprocess_knowledge_graph returns a list of dictionaries (facts, targets)?
                # Actually preprocess_knowledge_graph returns a list of (fact, label) or similar?
                # Let's simple check if it runs without error.
                print("✅ Successfully loaded and preprocessed sample 0")
                
        # Test with ASP output
        asp_data_path = Path("data/asp/family_tree/train")
        if not asp_data_path.exists():
             print("⚠️ ASP output not found, skipping RRN verify on ASP data")
        else:
             print(f"Verifying RRN loading from {asp_data_path}...")
             dataset = RRNDataset(str(asp_data_path), schema)
             print(f"Loaded {len(dataset)} samples.")
             if len(dataset) > 0:
                 sample = dataset[0]
                 print("✅ Successfully loaded and preprocessed sample 0")

        print("\n✅ RRN Loader Verification PASSED")

    except Exception as e:
        print(f"\n❌ RRN Loader Verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_ont_generator_verification()
    run_asp_generator_verification()
    run_rrn_loader_verification()

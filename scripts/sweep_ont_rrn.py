
import argparse
import os
import shutil
import subprocess
import sys
import uuid
import time
from pathlib import Path

def run_command(command, env=None):
    """Run a shell command and stream output."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    # Stream output in real-time
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Run WandB sweep with ont_generator and RRN")
    
    # Generator arguments (prefix: gen.)
    # We use parse_known_args to allow flexible arguments from WandB
    args, unknown = parser.parse_known_args()
    
    # Parsing logic:
    # WandB passes args like --gen.dataset.n_train=100 --rrn.hyperparams.lr=0.01
    # We collect them into separate lists
    
    gen_args = []
    rrn_args = []
    
    # Process sys.argv directly to handle all flags
    # Skip the script name
    for arg in sys.argv[1:]:
        if arg.startswith("--gen."):
            # Strip prefix and add to generator args
            # e.g. --gen.dataset.n_train=100 -> dataset.n_train=100
            key_val = arg[6:] 
            gen_args.append(key_val)
        elif arg.startswith("--rrn."):
            # Strip prefix and add to RRN args
            # e.g. --rrn.hyperparams.lr=0.01 -> hyperparams.lr=0.01
            key_val = arg[6:]
            rrn_args.append(key_val)
        elif arg.startswith("gen."): # handle non-dashed args if any
             gen_args.append(arg[4:])
        elif arg.startswith("rrn."):
             rrn_args.append(arg[4:])
        else:
             # Unknown args, maybe wandb specific flags? 
             # We generally ignore raw flags unless we know what they are. 
             # If they are simple flags like --verbose, we might need to route them manually.
             # For now, print warning.
             print(f"Warning: Unrouted argument: {arg}")

    # Generate a unique directory for this run's data
    run_id = str(uuid.uuid4())[:8]
    base_data_dir = os.path.abspath(f"data/sweep_runs/{run_id}")
    output_dir = os.path.join(base_data_dir, "output")
    
    print(f"--- Starting Sweep Run {run_id} ---")
    print(f"CWD: {os.getcwd()}")
    print(f"Data directory: {base_data_dir}")
    
    # 1. Run Generator
    print("\n[1/2] Running Ontology Generator...")
    gen_cmd_args = " ".join(gen_args)
    # Ensure output_dir is set correctly
    abs_ont_path = os.path.abspath("data/ont/input/family.ttl")
    if not os.path.exists(abs_ont_path):
        abs_ont_path = os.path.abspath("data/ont/output/family.ttl")
        if not os.path.exists(abs_ont_path):
             print(f"Error: Ontology file not found at {abs_ont_path} or data/ont/input/family.ttl")
             sys.exit(1)
             
    gen_cmd_args += f" dataset.output_dir={output_dir} hydra.job.chdir=False ontology.path={abs_ont_path}"
    
    # Using uv run python -m ... directly to avoid invoked overhead/parsing issues
    gen_cmd = f"uv run --package ont_generator python -m ont_generator.create_data {gen_cmd_args}"
    
    # Pass current env (important for wandb if it sets vars)
    env = os.environ.copy()
    env["SYNTHOLOGY_ROOT"] = os.getcwd()
    
    ret_code = run_command(gen_cmd, env=env)
    
    if ret_code != 0:
        print("Generator failed. Aborting run.")
        # Cleanup
        if os.path.exists(base_data_dir):
            shutil.rmtree(base_data_dir)
        sys.exit(ret_code)
        
    # 2. Run RRN Training
    print("\n[2/2] Running RRN Training...")
    
    # RRN expects specific paths for data.
    # The generator outputs to output_dir/train, output_dir/val, output_dir/test
    train_path = os.path.join(output_dir, "train")
    val_path = os.path.join(output_dir, "val")
    test_path = os.path.join(output_dir, "test")
    
    # Construct RRN arguments
    rrn_cmd_args = " ".join(rrn_args)
    
    # Override data paths
    rrn_cmd_args += f" data.train_path={train_path} data.val_path={val_path} data.test_path={test_path}"

    # Also make sure logging directory is unique or handled by WANDB
    # We rely on RRN's internal wandb logger to pick up the sweep configuration from env
    
    rrn_cmd = f"uv run --package rrn python -m rrn.train {rrn_cmd_args}"
    
    ret_code = run_command(rrn_cmd, env=env)
    
    # Cleanup data to save space?
    # Maybe we want to keep it if the run was interesting.
    # For now, let's keep it, or make it configurable. 
    # WandB sweeps can generate a lot of data.
    # Let's delete it if successful to avoid filling disk.
    
    if ret_code == 0:
        print("Training finished successfully.")
        print(f"Cleaning up data directory: {base_data_dir}")
        shutil.rmtree(base_data_dir)
    else:
        print("Training failed.")
        # Keep data for debugging
        print(f"Data preserved at: {base_data_dir}")
        sys.exit(ret_code)

if __name__ == "__main__":
    main()

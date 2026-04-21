import argparse
import sys
import os
import pandas as pd
import wandb
import datetime
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_shared_metrics(api: wandb.Api, project: str, run_ids: list, section: str):
    """Finds all metrics starting with 'section/' that exist in all provided runs."""
    metrics_sets = []
    for run_id in run_ids:
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            runs = api.runs(project, {"display_name": run_id}, order="-created_at")
            if len(runs) == 0:
                print(f"Error: Could not find run '{run_id}' for shared metric calculation.")
                sys.exit(1)
            run = runs[0]
            
        keys = set([k.split("/", 1)[1] for k in run.summary.keys() if k.startswith(f"{section}/")])
        metrics_sets.append(keys)
        
    if not metrics_sets:
        return []
        
    shared = set.intersection(*metrics_sets)
    return sorted(list(shared))

def main():
    parser = argparse.ArgumentParser(description="Export W&B metric traces and configs to LLM-readable CSV/JSON.")
    parser.add_argument("--project", type=str, 
                        help="W&B project path (e.g. 'username/synthology'). Will default to WANDB_ENTITY/WANDB_PROJECT from .env if not provided.")
    parser.add_argument("--runs", nargs="+", required=True, 
                        help="List of WandB run IDs or Display Names to export.")
    parser.add_argument("--labels", nargs="*", 
                        help="List of column labels corresponding to the runs. Must match --runs length.")
    parser.add_argument("--section", type=str, required=True,
                        help="Section logged in wandb (e.g., 'train', 'val', 'test').")
    parser.add_argument("--metric", type=str, required=True,
                        help="Metric name (e.g., 'triple_pr_auc'). Use 'all' to bulk export all shared metrics.")

    args = parser.parse_args()

    project = args.project
    if not project:
        entity = os.environ.get("WANDB_ENTITY")
        proj = os.environ.get("WANDB_PROJECT")
        if entity and proj:
            project = f"{entity}/{proj}"
        elif proj:
            project = proj
        else:
            print("Error: --project argument not provided, and WANDB_PROJECT/WANDB_ENTITY not found in .env")
            sys.exit(1)

    if args.labels and len(args.labels) != len(args.runs):
        print(f"Error: Number of labels ({len(args.labels)}) does not match number of runs ({len(args.runs)}).")
        sys.exit(1)

    api = wandb.Api()
    
    if args.metric.lower() == "all":
        shared_metrics = get_shared_metrics(api, project, args.runs, args.section)
        if not shared_metrics:
            print(f"No shared metrics found in section '{args.section}' for these runs.")
            sys.exit(1)
        metrics_to_export = shared_metrics
    else:
        metrics_to_export = [args.metric]

    expected_keys = [f"{args.section}/{m}" for m in metrics_to_export]
    run_data_list = []
    run_configs = {}
    
    print("Pre-fetching data for all requested runs...")
    for i, run_id in enumerate(args.runs):
        label = args.labels[i] if args.labels else run_id
        
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            runs = api.runs(project, {"display_name": run_id}, order="-created_at")
            if len(runs) == 0:
                print(f"Error: Could not find run '{run_id}'")
                continue
            run = runs[0]
            
        print(f"  -> Bulk fetching '{run.name}' (ID: {run.id})")
        
        # Save config for LLM context
        run_configs[label] = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "config": run.config
        }
        
        # Fetch scan_history
        keys_to_fetch = ["_step"] + expected_keys
        rows = []
        for row in run.scan_history(keys=keys_to_fetch):
            rows.append(row)
            
        df = pd.DataFrame(rows)
        if not df.empty and "_step" in df.columns:
            df = df.sort_values("_step")
            
        run_data_list.append({
            "label": label,
            "df": df,
            "run_id": run_id
        })

    # Prepare output directory
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    out_base_dir = os.path.join("wandb", "csv_exports", date_str, args.section)
    os.makedirs(out_base_dir, exist_ok=True)
    
    # Export Configs
    config_path = os.path.join(out_base_dir, "run_configs.json")
    with open(config_path, "w") as f:
        json.dump(run_configs, f, indent=4)
    print(f"\nExported run configurations to {config_path}")

    print("Generating CSVs...")
    for metric in metrics_to_export:
        metric_key = f"{args.section}/{metric}"
        
        # Merge dataframes on step
        merged_df = None
        
        for run_info in run_data_list:
            df = run_info["df"]
            if df is None or df.empty or metric_key not in df.columns:
                continue
                
            # Filter and rename for clarity
            metric_df = df[["_step", metric_key]].dropna().rename(columns={metric_key: run_info["label"]})
            
            if merged_df is None:
                merged_df = metric_df
            else:
                # Use outer merge so that misaligned steps are still retained safely
                merged_df = pd.merge(merged_df, metric_df, on="_step", how="outer")

        if merged_df is None or merged_df.empty:
            print(f"No data was found for metric {metric_key}. Skipping.")
            continue

        # Sort combined df by step
        merged_df = merged_df.sort_values("_step").set_index("_step")
        
        curr_output = os.path.join(out_base_dir, f"{metric}.csv")
        merged_df.to_csv(curr_output)
            
        if args.metric.lower() != "all" or len(metrics_to_export) <= 5:
            print(f"Successfully saved CSV to: {curr_output}")
            
    if args.metric.lower() == "all":
        print(f"Finished generating all {len(metrics_to_export)} CSVs in '{out_base_dir}'!")

if __name__ == "__main__":
    main()

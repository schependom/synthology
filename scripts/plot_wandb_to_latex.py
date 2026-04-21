import argparse
import sys
import os
import pandas as pd
import wandb
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure matplotlib for LaTeX quality output
import matplotlib as mpl
# We use 'pgf' backend for generating native LaTeX files (.pgf) and PDF files with LaTeX fonts
mpl.use("pgf") 
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "axes.labelsize": 11,
    "font.size": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": [5.5, 3.5], # Fits nicely in a LaTeX standard single-column
    "axes.grid": True,
    "grid.alpha": 0.5,
})
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

KUL_COLORS = {
    "KULijsblauw":  "#DCE7F0",
    "KULwit":       "#FFFFFF",
    "KULcorporate": "#00407A",
    "KULpetrol":    "#147FA1",
    "KULzwart":     "#001D41",
    "KULrood":      "#F0776E",
    "KULoranje":    "#FBB037",
    "KULgeel":      "#E4DA3E",
    "KULgroen":     "#87C0BD",
    "KULblauw":     "#52BDEC",
    "KULpaars":     "#C793AE",
    "blue":         "#00407A",
    "lightBlue":    "#52BDEC",
    "darkBlue":     "#001D41",
    "cyan":         "#147FA1",
    "green":        "#87C0BD",
    "orange":       "#FBB037",
    "red":          "#F0776E",
    "gray":         "#828282",
    "lightGray":    "#DCE7F0",
    "black":        "#001D41",
    "white":        "#FFFFFF",
}

OWN_COLORS = {
    "geeloranje": "#f7aa25",
    "appelblauwzeegroen": "#0cc795"
}

DEFAULT_COLORS = [
    KUL_COLORS["KULcorporate"],
    KUL_COLORS["KULblauw"],
    OWN_COLORS["geeloranje"],   
    KUL_COLORS["KULrood"],
    KUL_COLORS["KULpaars"],
    KUL_COLORS["KULoranje"],
    KUL_COLORS["KULgroen"],
    KUL_COLORS["KULpetrol"],
    KUL_COLORS["KULzwart"],
    OWN_COLORS["appelblauwzeegroen"],
]

def print_available_metrics(run):
    """Print available metrics cleanly formatted per section."""
    sections = {}
    for key in run.summary.keys():
        if "/" in key:
            section, metric = key.split("/", 1)
        else:
            section, metric = "General", key
            
        if section not in sections:
            sections[section] = []
        sections[section].append(metric)
        
    print(f"\n--- Available Metrics for Run '{run.name}' ---")
    for section in sorted(sections.keys()):
        print(f"  {section}:")
        for metric in sorted(sections[section]):
            print(f"    - {metric}")
    print("-" * 45 + "\n")


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


def smooth_series(series: pd.Series, alpha: float) -> pd.Series:
    """Apply Exponential Moving Average smoothing."""
    if alpha <= 0.0:
        return series
    # In pandas ewm, alpha is the standard smoothing parameter
    return series.ewm(alpha=1-alpha, adjust=False).mean()

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX-quality graphs from W&B metrics.")
    parser.add_argument("--project", type=str, 
                        help="W&B project path (e.g. 'username/synthology'). Will default to WANDB_ENTITY/WANDB_PROJECT from .env if not provided.")
    parser.add_argument("--runs", nargs="+", required=True, 
                        help="List of WandB run IDs or Display Names to plot.")
    parser.add_argument("--labels", nargs="*", 
                        help="List of legend labels corresponding to the runs. Must match --runs length.")
    parser.add_argument("--colors", nargs="*", 
                        help="List of color keys from the KUL matlab scheme (e.g. 'KULcorporate' 'KULrood'). Must match --runs length.")
    parser.add_argument("--section", type=str, 
                        help="Section logged in wandb (e.g., 'train', 'val', 'test'). Required unless --list-metrics is used.")
    parser.add_argument("--metric", type=str, 
                        help="Metric name (e.g., 'triple_pr_auc'). Use 'all' to bulk export all shared metrics.")
    parser.add_argument("--list-metrics", action="store_true",
                        help="Only list all available metrics grouped by section for the first provided run, then exit.")
    parser.add_argument("--output", type=str, default="plot.pdf", 
                        help="Output path. Ignored if '--metric all' is used.")
    parser.add_argument("--smooth", type=float, default=0.0, 
                        help="Smoothing alpha (0.0=no smoothing, 0.9=heavy smoothing). Default: 0.0")
    parser.add_argument("--title", type=str, default="", 
                        help="Optional title for the plot.")
    parser.add_argument("--ylabel", type=str, 
                        help="Override Y-axis label. If not provided, uses section/metric.")
    parser.add_argument("--xlabel", type=str, default="Steps", 
                        help="Override X-axis label.")

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
        
    if args.colors and len(args.colors) != len(args.runs):
        print(f"Error: Number of colors ({len(args.colors)}) does not match number of runs ({len(args.runs)}).")
        sys.exit(1)

    api = wandb.Api()
    
    # Handle list metrics request first
    if args.list_metrics:
        # Just grab the first run to list metrics
        try:
            runs = api.runs(project, {"display_name": args.runs[0]}, order="-created_at")
            if len(runs) > 0:
                print_available_metrics(runs[0])
            else:
                run = api.run(f"{project}/{args.runs[0]}")
                print_available_metrics(run)
        except Exception as e:
            print(f"Error fetching run '{args.runs[0]}' to list metrics: {e}")
        return

    if not args.section or not args.metric:
        print("Error: --section and --metric are required when not using --list-metrics")
        sys.exit(1)

    if args.metric.lower() == "all":
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_base_dir = os.path.join("wandb", "graphs", date_str, args.section)
        
        shared_metrics = get_shared_metrics(api, project, args.runs, args.section)
        if not shared_metrics:
            print(f"No shared metrics found in section '{args.section}' for these runs.")
            sys.exit(1)
            
        print(f"Found {len(shared_metrics)} shared metrics in '{args.section}'. Generating bulk plots...")
        metrics_to_plot = shared_metrics
    else:
        out_base_dir = None
        metrics_to_plot = [args.metric]

    expected_keys = [f"{args.section}/{m}" for m in metrics_to_plot]
    run_data_list = []
    
    print("Pre-fetching data for all requested runs...")
    for i, run_id in enumerate(args.runs):
        label = args.labels[i] if args.labels else run_id
        
        if args.colors:
            color_name = args.colors[i]
            color_hex = KUL_COLORS.get(color_name)
            if not color_hex:
                print(f"Warning: Color '{color_name}' not found in KUL palette. Falling back.")
                color_hex = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        else:
            color_hex = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            runs = api.runs(project, {"display_name": run_id}, order="-created_at")
            if len(runs) == 0:
                print(f"Error: Could not find run '{run_id}'")
                continue
            run = runs[0]
            
        print(f"  -> Bulk fetching '{run.name}' (ID: {run.id})")
        
        # We fetch all the required keys at once
        keys_to_fetch = ["_step"] + expected_keys
        rows = []
        for row in run.scan_history(keys=keys_to_fetch):
            rows.append(row)
            
        df = pd.DataFrame(rows)
        if not df.empty and "_step" in df.columns:
            df = df.sort_values("_step")
            
        run_data_list.append({
            "label": label,
            "color": color_hex,
            "df": df,
            "run_id": run_id
        })

    print("\nGenerating plots...")
    for metric in metrics_to_plot:
        metric_key = f"{args.section}/{metric}"
        
        fig, ax = plt.subplots()
        data_found = False
        run_max_steps = []
        
        for run_info in run_data_list:
            df = run_info["df"]
            if df is None or df.empty or metric_key not in df.columns:
                continue
                
            # Filter and drop NA for this specific metric
            metric_df = df[["_step", metric_key]].dropna()
            if metric_df.empty:
                continue
                
            data_found = True
            
            step_series = metric_df["_step"]
            val_series = metric_df[metric_key]
            
            run_max_steps.append(step_series.max())
            
            plot_kwargs = {"label": run_info["label"], "linewidth": 1.5, "color": run_info["color"]}
            
            if args.smooth > 0.0:
                smoothed = smooth_series(val_series, args.smooth)
                p = ax.plot(step_series, smoothed, **plot_kwargs)
                ax.plot(step_series, val_series, color=p[0].get_color(), alpha=0.3, linewidth=0.5)
            else:
                ax.plot(step_series, val_series, **plot_kwargs)

        if not data_found:
            print(f"No data was found for metric {metric_key}. Skipping.")
            plt.close(fig)
            continue

        ax.set_xlabel(args.xlabel)
        if run_max_steps:
            ax.set_xlim(right=min(run_max_steps),left=0)
            
        if args.ylabel:
            ax.set_ylabel(args.ylabel)
        else:
            ax.set_ylabel(f"{metric}".replace("_", " ").title())
            
        if args.title:
            ax.set_title(args.title)
            
        ax.legend(loc="best")
        
        if args.metric.lower() == "all":
            curr_output = os.path.join(out_base_dir, f"{metric}.pdf")
        else:
            curr_output = args.output
            
        out_dir = os.path.dirname(curr_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        plt.tight_layout()
        try:
            plt.savefig(curr_output, dpi=300, bbox_inches="tight")
            if args.metric.lower() != "all" or len(metrics_to_plot) <= 5:
                # To avoid spamming terminal too much
                print(f"Successfully saved plot to: {curr_output}")
        except Exception as e:
            print(f"Error saving {curr_output}: {e}")
            
        plt.close(fig)
        
    if args.metric.lower() == "all":
        print(f"Finished generating all {len(metrics_to_plot)} graphs in '{out_base_dir}'!")

if __name__ == "__main__":
    main()

import argparse
import sys
import os
import pandas as pd
import wandb
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
    "font.size": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": [5.5, 3.5], # Fits nicely in a LaTeX standard single-column
    "axes.grid": True,
    "grid.alpha": 0.5,
})
import matplotlib.pyplot as plt


def smooth_series(series: pd.Series, alpha: float) -> pd.Series:
    """Apply Exponential Moving Average smoothing."""
    if alpha <= 0.0:
        return series
    # In pandas ewm, alpha is the standard smoothing parameter
    return series.ewm(alpha=1-alpha, adjust=False).mean()


def get_run_history(api: wandb.Api, project: str, run_id: str, metric_key: str):
    """Fetch history from WandB. Tries finding by ID first, then by Display Name."""
    try:
        run = api.run(f"{project}/{run_id}")
    except wandb.errors.CommError:
        # Try finding by display name
        runs = api.runs(project, {"display_name": run_id}, order="-created_at")
        if len(runs) == 0:
            print(f"Error: Could not find run with ID or name '{run_id}' in project '{project}'")
            return None
        if len(runs) > 1:
            print(f"Note: Found {len(runs)} runs named '{run_id}'. Selecting the most recent one.")
        run = runs[0]
        
    print(f"Fetching data for run: {run.name} (ID: {run.id})")
    print(f"  -> Created At: {getattr(run, 'created_at', 'Unknown')}")
    print(f"  -> State: {run.state}")
    
    # We use scan_history to get all points (history() is aggressively sampled)
    rows = []
    for row in run.scan_history(keys=["_step", metric_key]):
        # Depending on how the user logged, the step might be stored differently,
        # but "_step" is the global WandB step.
        if metric_key in row and row[metric_key] is not None:
            rows.append((row["_step"], row[metric_key]))
            
    if not rows:
        print(f"Warning: Metric '{metric_key}' not found in run '{run_id}'.")
        print("  -> Troubleshooting: Is the run stale?")
        print("  -> Similar keys found in the summary:")
        similar_keys = [k for k in run.summary.keys() if metric_key.split('/')[-1] in k or (metric_key.split('/')[0] in k)]
        if similar_keys:
            print(f"     {similar_keys}")
        else:
            print(f"     No keys resembling '{metric_key}' found.")
        print("Skipping to next run...")
        return None
        
    df = pd.DataFrame(rows, columns=["step", "value"]).sort_values("step").dropna()
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX-quality graphs from W&B metrics.")
    parser.add_argument("--project", type=str, 
                        help="W&B project path (e.g. 'username/synthology'). Will default to WANDB_ENTITY/WANDB_PROJECT from .env if not provided.")
    parser.add_argument("--runs", nargs="+", required=True, 
                        help="List of WandB run IDs or Display Names to plot.")
    parser.add_argument("--labels", nargs="*", 
                        help="List of legend labels corresponding to the runs. Must match --runs length.")
    parser.add_argument("--section", type=str, required=True, 
                        help="Section logged in wandb (e.g., 'train', 'val', 'test').")
    parser.add_argument("--metric", type=str, required=True, 
                        help="Metric name logged in wandb (e.g., 'triple_pr_auc').")
    parser.add_argument("--output", type=str, default="plot.pdf", 
                        help="Output path. Supports .pdf, .pgf, .png (e.g., 'results/figure1.pgf')")
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

    # W&B metric key is commonly '{section}/{metric}' (e.g. 'val/triple_pr_auc')
    metric_key = f"{args.section}/{args.metric}"
    
    api = wandb.Api()
    
    fig, ax = plt.subplots()
    
    data_found = False
    
    for i, run_id in enumerate(args.runs):
        label = args.labels[i] if args.labels else run_id
        
        df = get_run_history(api, project, run_id, metric_key)
        if df is None or df.empty:
            continue
            
        data_found = True
        
        step_series = df["step"]
        val_series = df["value"]
        
        # Plot original (transparent) if smoothing is aggressively applied
        if args.smooth > 0.0:
            smoothed = smooth_series(val_series, args.smooth)
            p = ax.plot(step_series, smoothed, label=label, linewidth=1.5)
            # Add a fainter line for the noisy original data
            ax.plot(step_series, val_series, color=p[0].get_color(), alpha=0.3, linewidth=0.5)
        else:
            ax.plot(step_series, val_series, label=label, linewidth=1.5)

    if not data_found:
        print("No data was found for any of the supplied runs. Exiting without saving.")
        sys.exit(1)

    ax.set_xlabel(args.xlabel)
    
    # Format Y label nicely if not provided manually
    if args.ylabel:
        ax.set_ylabel(args.ylabel)
    else:
        # e.g., 'val/triple_pr_auc' -> 'Val Triple PR AUC'
        default_ylabel = f"{args.section} {args.metric}".replace("_", " ").title()
        ax.set_ylabel(default_ylabel)
        
    if args.title:
        ax.set_title(args.title)
        
    ax.legend(loc="best")
    
    # Save the figure
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    plt.tight_layout()
    try:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Successfully saved plot to: {args.output}")
        if args.output.endswith(".pgf"):
            print("Note: You can include this in LaTeX via '\\input{...}'. Make sure to import the 'pgf' package.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()

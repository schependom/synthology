import os

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "synthology"
PYTHON_VERSION = "3.12"
# Ensure SYNTHOLOGY_ROOT is set for subprocesses to locate configs
os.environ["SYNTHOLOGY_ROOT"] = os.path.dirname(os.path.abspath(__file__))


@task
def gen_ft_asp(ctx: Context):
    """Generates family tree datasets with ASP solver using default configurations in configs/asp_generator/"""

    # If ./data/asp/out-reldata has content,
    # ask user to confirm cleanup
    if os.path.exists("./data/asp/out-reldata") and os.listdir("./data/asp/out-reldata"):
        response = input(
            "Previous ASP generator outputs detected in ./data/asp/out-reldata.\n"
            "You can convert them to CSV using 'invoke convert-reldata' if needed.\n\n"
            "Do you want to delete them before generating new data? (y/n): "
        )
        if response.lower() != "y":
            print("Aborting dataset generation.")
            return

    # Clean up ./data/asp/out-reldata first
    logger.info("Cleaning up previous ASP generator outputs.")
    ctx.run("rm -rf ./data/asp/out-reldata")
    logger.success("Cleanup done.")

    # Run ASP generator using
    # config from configs/asp_generator/config.yaml
    print("-------------------------------------------------------")
    print("Running family tree ASP generator by Patrick Hohenecker")
    print("-------------------------------------------------------\n")
    ctx.run("export PYTHONUNBUFFERED=1")
    ctx.run("export LOGURU_COLORIZE=1 && uv run --package asp_generator asp_generator")

    # Convert reldata outputs to CSV
    # using config from configs/asp_generator/config.yaml
    print("\n-------------------------------------------")
    print("Converting generated ASP data to CSV format")
    print("--------------------------------------------\n")
    ctx.run("export PYTHONUNBUFFERED=1")
    ctx.run("export LOGURU_COLORIZE=1 && uv run --package asp_generator python -u -m asp_generator.convert_to_csv")

    logger.success("Family tree dataset generation with ASP completed.")


@task
def convert_reldata(ctx: Context):
    """Converts family tree datasets in data/asp/out-reldata to CSV format."""

    print("\n-------------------------------------------")
    print("Converting generated ASP data to CSV format")
    print("--------------------------------------------\n")
    ctx.run("export PYTHONUNBUFFERED=1")
    ctx.run("export LOGURU_COLORIZE=1 && uv run --package asp_generator python -u -m asp_generator.convert_to_csv")

    logger.success("Conversion of family tree dataset from reldata to CSV completed.")


@task
def gen_ft_ont(ctx: Context, args=""):
    """
    Generates family tree datasets with Ontology-based Generator
    using default configurations in configs/ont_generator/config.yaml
    """

    print("\nRunning family tree Ontology-based generator.")
    cmd = "export LOGURU_COLORIZE=1 && "  # Ensure logs are colored
    cmd += "uv run --package ont_generator python -m ont_generator.create_data"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def gen_ft_ont_single(ctx: Context, args=""):
    """
    Generates one single, big family tree dataset with Ontology-based
    Generator using default configurations in configs/ont_generator/config_single_graph.yaml.
    Handy for debugging purposes (visual inspection of proof trees and KG's).
    """

    print("\nRunning family tree Ontology-based generator.")
    cmd = "export LOGURU_COLORIZE=1 && "  # Ensure logs are colored
    cmd += "uv run --package ont_generator python -m ont_generator.generate"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def visualize_proofs(ctx: Context, args=""):
    """
    Generates a small dataset with all negative sampling strategies
    and exports proof tree visualizations for manual inspection.
    Output goes to visualizations/proofs/ and visualizations/graphs/.
    """

    print("\nGenerating proof visualizations (small dataset, mixed strategy).")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package ont_generator python -m ont_generator.create_data --config-name=config_visualize"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def visualize_lubm_proofs(ctx: Context, args=""):
    """
    Generates a few medium-sized datasets based on LUBM with all negative sampling strategies
    and exports proof tree visualizations for manual inspection.
    Output goes to data/ont/output/lubm-visualize/proofs/ and data/ont/output/lubm-visualize/graphs/.
    """

    print("\nGenerating LUBM proof visualizations (medium dataset, mixed strategy).")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package ont_generator python -m ont_generator.create_data --config-name=config_lubm_visualize"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def train_rrn_ont(ctx: Context, args=""):
    """Trains RRN based on default configurations in configs/rrn/"""

    print("\nRunning RRN training with Ontology-based dataset.")

    cmd = "export PYTHONUNBUFFERED=1 && "  # Ensure logs are unbuffered
    cmd += "export LOGURU_COLORIZE=1 && "  # Ensure logs are colored
    cmd += "uv run --package rrn python -m rrn.train data/dataset=ont"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def train_rrn_asp(ctx: Context, args=""):
    """Trains RRN based on default configurations in configs/rrn/"""

    print("\nRunning RRN training with ASP dataset.")

    cmd = "export PYTHONUNBUFFERED=1 && "  # Ensure logs are unbuffered
    cmd += "export LOGURU_COLORIZE=1 && "  # Ensure logs are colored
    cmd += "uv run --package rrn python -m rrn.train data/dataset=asp"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def train_rrn_lubm(ctx: Context, args=""):
    """Trains RRN based on default configurations in configs/rrn/ using LUBM data"""

    print("\nRunning RRN training with LUBM dataset.")

    cmd = "export PYTHONUNBUFFERED=1 && "  # Ensure logs are unbuffered
    cmd += "export LOGURU_COLORIZE=1 && "  # Ensure logs are colored
    # Add LUBM tag to wandb correctly using Hydra dict syntax
    cmd += "uv run --package rrn python -m rrn.train data/dataset=lubm +logger.tags=[LUBM]"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def gen_lubm(ctx: Context, args=""):
    """
    Generates LUBM datasets and parses them into CSV (facts/targets).
    Parsing stage applies OWL RL reasoning to compute inferred targets.
    """
    print("\nRunning LUBM generator orchestrator.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package lubm python -m lubm.orchestrator"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)

    print("\nParsing generated LUBM data into facts.csv and targets.csv (with reasoning).")
    parse_cmd = "export LOGURU_COLORIZE=1 && "
    parse_cmd += "uv run --package lubm python -m lubm.parse_to_csv"
    if args:
        parse_cmd += f" {args}"
    ctx.run(parse_cmd)


@task
def gen_lubm_tbox(ctx: Context, args=""):
    """
    Downloads the LUBM TBox ontology and saves it to data/ont/input/lubm.ttl
    """
    print("\nDownloading LUBM TBox ontology.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package lubm python -m lubm.download_tbox"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def parse_lubm(ctx: Context, args=""):
    """
    Parses LUBM generated TTL data into RRN standard CSV format.
    """
    print("\nRunning LUBM CSV parser.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package lubm python -m lubm.parse_to_csv"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def verify_lubm_reasoner(ctx: Context):
    """
    Runs a tiny deterministic toy verification of LUBM reasoning + CSV export.
    Useful as a static sanity check before expensive full parsing.
    """
    print("\nRunning LUBM toy reasoner verification.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package lubm python -m lubm.verify_reasoner"
    ctx.run(cmd)


@task
def pipeline_lubm(ctx: Context, args=""):
    """
    Runs the full LUBM pipeline sequentially: generation then parsing.
    """
    print("\nStarting full LUBM generation & parsing pipeline.")
    gen_lubm(ctx, args)
    parse_lubm(ctx, args)
    print("\nLUBM pipeline completed successfully.")


# Data version control with DVC+SSH
#
#   1. uv add dvc[ssh] --dev
#
#   2. `dvc init`
#
#   3. add SSH information to ~/.ssh/config file
#
#   4. dvc remote modify --local <host-name> auth ssh
#
#         e.g. `dvc remote modify --local gbar1 auth ssh`
#
#   5. Add the folder you want to track with DVC
#      to .gitignore
#
#   6. Remove folder you want to track with DVC from git tracking:
#
#         git rm -r --cached <folder>
#
#   7. Commit changes to git
#
#   8. Now, `uv run invoke dvc` to add, commit and push data changes
#      as being done below:
#
@task
def dvc(ctx, folder="data", message="Add new data"):
    """Adds, commits and pushes data changes to DVC remote storage."""

    ctx.run(f"dvc add {folder}")
    ctx.run(f"git add {folder}.dvc .gitignore")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")
    ctx.run("dvc push")

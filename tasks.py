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
    ctx.run("uv run --package asp_generator asp_generator")

    # Convert reldata outputs to CSV
    # using config from configs/asp_generator/config.yaml
    print("\n-------------------------------------------")
    print("Converting generated ASP data to CSV format")
    print("--------------------------------------------\n")
    ctx.run("export PYTHONUNBUFFERED=1")
    ctx.run("uv run --package asp_generator python -u -m asp_generator.convert_to_csv")

    logger.success("Family tree dataset generation with ASP completed.")


@task
def convert_reldata(ctx: Context):
    """Converts family tree datasets in data/asp/out-reldata to CSV format."""

    print("\n-------------------------------------------")
    print("Converting generated ASP data to CSV format")
    print("--------------------------------------------\n")
    ctx.run("uv run --package asp_generator python -u -m asp_generator.convert_to_csv")

    logger.success("Conversion of family tree dataset from reldata to CSV completed.")


@task
def gen_ft_ont(ctx: Context, args=""):
    """Generates family tree datasets with Ontology-based Generator using default configurations in configs/ont_generator/"""

    print("\nRunning family tree Ontology-based generator.")
    cmd = "uv run --package ont_generator python -m ont_generator.create_data"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def train_rrn(ctx: Context, args=""):
    """Trains RRN based on default configurations in configs/rrn/"""

    print("\nTraining RRN on Ontology-based generated family tree datasets.")
    cmd = "uv run --package rrn python -m rrn.train"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


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

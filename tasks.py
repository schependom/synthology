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
def ont_visual_inspection(ctx: Context, args=""):
    """
    Generates a few decently sized knowledge graphs that contain both 
    positive, negative, base and inferred samples and visualizes them.
    Uses configs/ont_generator/config_visual_inspection.yaml.
    """

    print("\nRunning Visual Inspection Generator.")
    cmd = (
        f"export LOGURU_COLORIZE=1 && "
        f"uv run --package ont_generator python -m ont_generator.create_data "
        f"--config-name=config_visual_inspection"
    )
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
def train_rrn_owl2bench(ctx: Context, args=""):
    """Trains RRN using OWL2Bench OWL 2 RL dataset."""

    print("\nRunning RRN training with OWL2Bench dataset.")

    cmd = "export PYTHONUNBUFFERED=1 && "
    cmd += "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package rrn python -m rrn.train data/dataset=owl2bench +logger.tags=[OWL2Bench]"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def gen_owl2bench(ctx: Context, args=""):
    """
    Runs the OWL2Bench OWL 2 RL pipeline:
    ABox generation -> NeMo materialization -> CSV export.
    """
    print("\nRunning OWL2Bench OWL 2 RL generation pipeline.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package owl2bench python -m owl2bench.pipeline"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def gen_owl2bench_toy(ctx: Context, args=""):
    """
    Runs a tiny OWL2Bench pipeline config for quick end-to-end verification:
    base -> NeMo materialization -> inferred targets -> negatives.
    """
    print("\nRunning OWL2Bench TOY generation pipeline.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package owl2bench python -m owl2bench.pipeline --config-name=config_toy"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)

    print("\nAuto-visualizing toy sample 710021.")
    viz_cmd = "export LOGURU_COLORIZE=1 && "
    viz_cmd += (
        "uv run --package kgvisualiser python -m kgvisualiser.visualize "
        "io.input_csv=data/owl2bench/output_toy/owl2bench_1/val/targets.csv "
        "io.sample_id=710021"
    )
    ctx.run(viz_cmd)


@task
def visualize_kg_sample(ctx: Context, args=""):
    """
    Visualizes one KG sample with base, inferred and negative facts.
    Uses configs/kgvisualiser/config.yaml by default.
    """

    print("\nRunning KG sample visualization.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package kgvisualiser python -m kgvisualiser.visualize"
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


@task
def exp1_generate_trainval_sets(ctx: Context):
    """Generates train/val sets for Exp 1 for all negative sampling strategies."""
    for strategy in ("random", "constrained", "proof_based"):
        exp1_generate_trainval(ctx, strategy=strategy)

@task
def exp1_generate_trainval(ctx: Context, strategy="proof_based", args=""):
    """Generates train/val sets for Exp 1 with a specific negative sampling strategy."""
    print(f"\nGenerating Exp 1 data using strategy: {strategy}")
    cmd = (
        f"export LOGURU_COLORIZE=1 && "
        f"uv run --package ont_generator python -m ont_generator.create_data "
        f"--config-name=exp1_{strategy}"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)

@task
def exp1_generate_test_set(ctx: Context, args=""):
    """Generates the frozen 'near-miss' hard negative test set for Exp 1."""
    print("\nGenerating Exp 1 frozen test set (near-miss hard negatives)")
    cmd = (
        f"export LOGURU_COLORIZE=1 && "
        f"uv run --package ont_generator python -m ont_generator.create_data "
        f"--config-name=exp1_test"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)

@task
def exp1_train_rrn(ctx: Context, strategy="random", args=""):
    """Trains RRN for Exp 1. Provide a strategy to match datasets/logs."""
    print(f"\nTraining Exp 1 RRN. Strategy: {strategy}")
    cmd = (
        f"export PYTHONUNBUFFERED=1 && export LOGURU_COLORIZE=1 && "
        f"uv run --package rrn python -m rrn.train "
        f"data/dataset=exp1_{strategy} "
        f"+logger.name=exp1_{strategy} "
        f"+logger.group=exp1_negative_sampling "
        f"+logger.tags=[exp1,{strategy}]"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)

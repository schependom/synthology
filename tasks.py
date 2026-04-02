import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "synthology"
PYTHON_VERSION = "3.12"
REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ARCHIVE_ROOT = REPO_ROOT / "reports" / "experiment_runs"

# Ensure SYNTHOLOGY_ROOT is set for subprocesses to locate configs
os.environ["SYNTHOLOGY_ROOT"] = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------ #
# Experiment specs
# ------------------------------------------------------------ #


@dataclass(frozen=True)
class ExperimentRunSpec:
    experiment: str
    task_name: str
    label: Optional[str]
    command: str
    config_paths: tuple[str, ...] = ()
    manifest: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: tuple[str, ...] = ()
    cwd: Optional[Path] = None
    hydra_run_dir: bool = True


# ------------------------------------------------------------ #
# Replication of RRN paper results
# ------------------------------------------------------------ #


# Generate the ASP dataset used by P. Hohenecker et al.
@task
def gen_ft_asp(ctx: Context):
    """Generates family tree ASP datasets with ASP solver
    as described in the RRN paper by P. Hohenecker et al. using
    default configurations in configs/asp_generator/"""

    # If ./data/asp/out-reldata has content,
    # ask user to confirm cleanup
    run_dir = _make_run_archive("asp_generator", "gen_ft_asp", label="family_tree")
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "asp_generator",
            "task": "gen_ft_asp",
            "output_dir": "./data/asp/out-reldata",
        },
    )
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
    _run_logged_command("rm -rf ./data/asp/out-reldata", run_dir / "cleanup.log")
    logger.success("Cleanup done.")

    # Run ASP generator using
    # config from configs/asp_generator/config.yaml
    print("-------------------------------------------------------")
    print("Running family tree ASP generator by Patrick Hohenecker")
    print("-------------------------------------------------------\n")
    generator_cmd = _build_uv_command(
        "asp_generator",
        "asp_generator",
        python_module=False,
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _run_logged_command(generator_cmd, run_dir / "generator.log")

    # Convert reldata outputs to CSV
    # using config from configs/asp_generator/config.yaml
    print("\n-------------------------------------------")
    print("Converting generated ASP data to CSV format")
    print("--------------------------------------------\n")
    convert_cmd = _build_uv_command(
        "asp_generator",
        "asp_generator.convert_to_csv",
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _run_logged_command(convert_cmd, run_dir / "convert.log")

    logger.success("Family tree dataset generation with ASP completed.")


# Convert proprietary reldata format by P. Hohenecker's generator
# to CSV for RRN training/evaluation
@task
def convert_reldata(ctx: Context):
    """Converts family tree datasets in reldata format to CSV format."""

    print("\n-------------------------------------------")
    print("Converting generated ASP data to CSV format")
    print("--------------------------------------------\n")
    run_dir = _make_run_archive("asp_generator", "convert_reldata", label="family_tree")
    cmd = _build_uv_command(
        "asp_generator",
        "asp_generator.convert_to_csv",
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _write_json(run_dir / "manifest.json", {"experiment": "asp_generator", "task": "convert_reldata", "command": cmd})
    _run_logged_command(cmd, run_dir / "run.log")

    logger.success("Conversion of family tree dataset from reldata to CSV completed.")


# Train the RRN on the ASP dataset
@task
def train_rrn_asp(ctx: Context, args=""):
    """Trains RRN on ASP-generated Family Tree dataset."""

    print("\nRunning RRN training with ASP dataset.")
    run_dir = _make_run_archive("rrn", "train_asp", label="asp")
    cmd = _build_uv_command(
        "rrn",
        "rrn.train",
        overrides=("data/dataset=asp",),
        args=args,
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "rrn",
            "task": "train_asp",
            "args": args,
            "command": cmd,
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


# ------------------------------------------------------------ #
# Generate/Train on Family Tree data with Synthology/RRN
# ------------------------------------------------------------ #


@task
def gen_ft_ont(ctx: Context, args=""):
    """
    Generates family tree datasets with Ontology-based 'Synthology' Generator
    using default configurations in configs/ont_generator/config.yaml
    """

    print("\nRunning family tree Ontology-based generator.")
    run_dir = _make_run_archive("ont_generator", "gen_ft_ont", label="family_tree")
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        env={"LOGURU_COLORIZE": "1"},
        args=args,
    )
    _write_json(
        run_dir / "manifest.json", {"experiment": "ont_generator", "task": "gen_ft_ont", "args": args, "command": cmd}
    )
    _run_logged_command(cmd, run_dir / "run.log")


@task
def train_rrn_ont(ctx: Context, args=""):
    """Trains RRN based on default configurations in configs/rrn/"""

    print("\nRunning RRN training with Ontology-based dataset.")
    run_dir = _make_run_archive("rrn", "train_ont", label="ont")
    cmd = _build_uv_command(
        "rrn",
        "rrn.train",
        overrides=("data/dataset=ont",),
        args=args,
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "rrn",
            "task": "train_ont",
            "args": args,
            "command": cmd,
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


@task
def gen_ft_fc(ctx: Context, args=""):
    """
    Generates family tree datasets with random base facts + owlrl
    forward-chaining materialization baseline.
    Uses configs/udm_baseline/config.yaml by default.
    """

    print("\nRunning family tree FC baseline generator.")
    run_dir = _make_run_archive("udm_baseline", "gen_ft_fc", label="family_tree")
    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.create_data",
        env={"LOGURU_COLORIZE": "1"},
        args=args,
    )
    _write_json(
        run_dir / "manifest.json", {"experiment": "udm_baseline", "task": "gen_ft_fc", "args": args, "command": cmd}
    )
    _run_logged_command(cmd, run_dir / "run.log")


@task
def train_rrn_fc(ctx: Context, args=""):
    """Trains RRN on FC baseline dataset."""

    print("\nRunning RRN training with FC baseline dataset.")
    run_dir = _make_run_archive("rrn", "train_fc", label="fc")
    cmd = _build_uv_command(
        "rrn",
        "rrn.train",
        overrides=("data/dataset=fc",),
        args=args,
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "rrn",
            "task": "train_fc",
            "args": args,
            "command": cmd,
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


# ------------------------------------------------------------ #
# Helper commands to verify correctness.
# ------------------------------------------------------------ #


@task
def synthology_visual_verification(ctx: Context, args=""):
    """
    Generates a few decently sized knowledge graphs that contain both
    positive, negative, base and inferred samples and visualizes them.
    Uses configs/ont_generator/config_visual_inspection.yaml.
    Output saved to visual-verification/ folder.
    """

    print("\nRunning Visual Inspection Generator.")
    run_dir = _make_run_archive("ont_generator", "visual_verification", label="inspection")
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        config_name="config_visual_inspection",
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _write_json(
        run_dir / "manifest.json",
        {"experiment": "ont_generator", "task": "visual_verification", "args": args, "command": cmd},
    )
    _run_logged_command(cmd, run_dir / "run.log")


@task
def visualize_proofs(ctx: Context, args=""):
    """
    Generates a small dataset with all negative sampling strategies
    and exports proof tree visualizations for manual inspection.
    Output goes to visualizations/proofs/ and visualizations/graphs/.
    """

    print("\nGenerating proof visualizations (small dataset, mixed strategy).")
    run_dir = _make_run_archive("ont_generator", "visualize_proofs", label="proofs")
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        config_name="config_visualize",
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _write_json(
        run_dir / "manifest.json",
        {"experiment": "ont_generator", "task": "visualize_proofs", "args": args, "command": cmd},
    )
    _run_logged_command(cmd, run_dir / "run.log")


# ------------------------------------------------------------ #
# Helper commands for experiments.
# ------------------------------------------------------------ #


@task
def train_rrn_owl2bench(ctx: Context, args=""):
    """Trains RRN using OWL2Bench OWL 2 RL dataset."""

    print("\nRunning RRN training with OWL2Bench dataset.")
    run_dir = _make_run_archive("rrn", "train_owl2bench", label="owl2bench")
    cmd = _build_uv_command(
        "rrn",
        "rrn.train",
        config_name="exp3_owl2bench_hpc",
        args=args,
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "rrn",
            "task": "train_owl2bench",
            "args": args,
            "command": cmd,
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


@task
def gen_owl2bench(ctx: Context, args=""):
    """
    Runs the OWL2Bench OWL 2 RL pipeline:
    ABox generation -> Apache Jena materialization -> CSV export.
    """
    print("\nRunning OWL2Bench OWL 2 RL generation pipeline.")
    run_dir = _make_run_archive("owl2bench", "generate", label="default")
    cmd = _build_uv_command(
        "owl2bench",
        "owl2bench.pipeline",
        env={"LOGURU_COLORIZE": "1"},
        args=args,
    )
    _write_json(run_dir / "manifest.json", {"task": "generate", "args": args, "command": cmd})
    _run_logged_command(cmd, run_dir / "run.log")


@task
def gen_owl2bench_toy(ctx: Context, args=""):
    """
    Runs a tiny OWL2Bench pipeline config for quick end-to-end verification:
    base -> Jena materialization -> inferred targets -> negatives.
    """
    print("\nRunning OWL2Bench TOY generation pipeline.")
    run_dir = _make_run_archive("owl2bench", "generate_toy", label="toy")
    cmd = _build_uv_command(
        "owl2bench",
        "owl2bench.pipeline",
        config_name="config_toy",
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _write_json(run_dir / "manifest.json", {"task": "generate_toy", "args": args, "command": cmd})
    _run_logged_command(cmd, run_dir / "run.log")

    csv_path, sample_id = _find_visualization_target()
    print(f"\nAuto-visualizing toy sample {sample_id} from {csv_path}.", flush=True)
    viz_cmd = _build_uv_command(
        "kgvisualiser",
        "kgvisualiser.visualize",
        overrides=(f"io.input_csv={csv_path}", f"io.sample_id={sample_id}"),
        env={"LOGURU_COLORIZE": "1"},
    )
    _write_text(run_dir / "visualize.log", f"input_csv={csv_path}\nsample_id={sample_id}\ncommand={viz_cmd}\n")
    _run_logged_command(viz_cmd, run_dir / "visualize-run.log")


@task
def visualize_kg_sample(ctx: Context, args=""):
    """
    Visualizes one KG sample with base, inferred and negative facts.
    Uses configs/kgvisualiser/config.yaml by default.
    """

    print("\nRunning KG sample visualization.")
    run_dir = _make_run_archive("kgvisualiser", "visualize_kg_sample", label="sample")
    cmd = _build_uv_command(
        "kgvisualiser",
        "kgvisualiser.visualize",
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _write_json(
        run_dir / "manifest.json",
        {"experiment": "kgvisualiser", "task": "visualize_kg_sample", "args": args, "command": cmd},
    )
    _run_logged_command(cmd, run_dir / "run.log")


@task
def report_data(ctx: Context, args=""):
    """
    Generates method-comparison dataset reports and plots
    (predicate/type/hops/negative distributions, counts, ratios).
    Uses configs/data_reporter/config.yaml by default.
    """

    print("\nRunning dataset comparison report generator.")
    run_dir = _make_run_archive("data_reporter", "report_data", label="report")
    cmd = _build_uv_command(
        "data_reporter",
        "data_reporter.analyze",
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _write_json(
        run_dir / "manifest.json", {"experiment": "data_reporter", "task": "report_data", "args": args, "command": cmd}
    )
    _run_logged_command(cmd, run_dir / "run.log")


@task
def paper_visual_report(
    ctx: Context,
    exp2_synth_targets="data/exp2/synthology/family_tree/train/targets.csv",
    exp2_parity_summary="data/exp2/baseline/parity_runs/parity_loop_summary.json",
    exp2_baseline_targets="",
    exp3_targets="",
    exp3_abox="",
    exp3_inferred="",
    out_dir="reports/paper",
    args="",
):
    """Generates paper-ready plots for base/inferred counts, hops, and parity attempts."""
    print("\nGenerating paper visual report.")
    run_dir = _make_run_archive("paper", "visual_report", label="paper")
    cmd = _build_uv_command(
        "data_reporter",
        "data_reporter.paper_plots",
        overrides=(
            f"--exp2-synth-targets {exp2_synth_targets}",
            f"--exp2-parity-summary {exp2_parity_summary}",
            f"--out-dir {out_dir}",
        ),
        env={"LOGURU_COLORIZE": "1"},
    )
    if exp2_baseline_targets:
        cmd += f" --exp2-baseline-targets {exp2_baseline_targets}"
    if exp3_targets:
        cmd += f" --exp3-targets {exp3_targets}"
    if exp3_abox and exp3_inferred:
        cmd += f" --exp3-abox {exp3_abox} --exp3-inferred {exp3_inferred}"
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "paper",
            "task": "visual_report",
            "exp2_synth_targets": exp2_synth_targets,
            "exp2_parity_summary": exp2_parity_summary,
            "exp2_baseline_targets": exp2_baseline_targets,
            "exp3_targets": exp3_targets,
            "exp3_abox": exp3_abox,
            "exp3_inferred": exp3_inferred,
            "out_dir": out_dir,
            "args": args,
            "command": cmd,
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    _archive_path(Path(out_dir), run_dir / "artifacts")


# ------------------------------------------------------------ #
# EXP 1: Negative Sampling Strategies for RRN Training
# ------------------------------------------------------------ #


@task
def exp1_generate_trainval_sets(ctx: Context):
    """Generates train/val sets for Exp 1 for all negative sampling strategies."""
    run_dir = _make_run_archive("exp1", "generate_trainval_sets", label="all")
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp1",
            "task": "generate_trainval_sets",
            "strategies": ["random", "constrained", "proof_based"],
        },
    )
    summary_lines = ["Exp1 train/val set generation summary"]
    for strategy in ("random", "constrained", "proof_based"):
        exp1_generate_trainval(ctx, strategy=strategy)
        summary_lines.append(f"- generated strategy={strategy}")
    _write_text(run_dir / "run.log", "\n".join(summary_lines) + "\n")


@task
def exp1_generate_trainval(ctx: Context, strategy="proof_based", args=""):
    """Generates train/val sets for Exp 1 with a specific negative sampling strategy."""
    print(f"\nGenerating Exp 1 data using strategy: {strategy}")
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        config_name=f"exp1_{strategy}",
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    run_dir = _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp1",
            task_name="generate_trainval",
            label=strategy,
            command=cmd,
            config_paths=(f"configs/ont_generator/exp1_{strategy}.yaml", "configs/ont_generator/config.yaml"),
            manifest={
                "strategy": strategy,
                "args": args,
                "config_files": [
                    f"configs/ont_generator/exp1_{strategy}.yaml",
                    "configs/ont_generator/config.yaml",
                ],
            },
            artifact_paths=(str(REPO_ROOT / "data" / "exp1" / strategy),),
        )
    )


@task
def exp1_generate_test_set(ctx: Context, args=""):
    """Generates the frozen 'near-miss' hard negative test set for Exp 1."""
    print("\nGenerating Exp 1 frozen test set (near-miss hard negatives)")
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        config_name="exp1_test",
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp1",
            task_name="generate_test_set",
            label="test_set",
            command=cmd,
            config_paths=("configs/ont_generator/exp1_test.yaml", "configs/ont_generator/config.yaml"),
            manifest={
                "args": args,
                "config_files": ["configs/ont_generator/exp1_test.yaml", "configs/ont_generator/config.yaml"],
            },
            artifact_paths=(str(REPO_ROOT / "data" / "exp1" / "test_set"),),
        )
    )


@task
def exp1_train_rrn(ctx: Context, strategy="random", args=""):
    """Trains RRN for Exp 1. Provide a strategy to match datasets/logs."""
    print(f"\nTraining Exp 1 RRN. Strategy: {strategy}")
    run_dir = _make_run_archive("exp1", "train_rrn", label=strategy)
    config_name = f"exp1_{strategy}_hpc"
    cmd = _build_uv_command(
        "rrn",
        "rrn.train",
        config_name=config_name,
        args=args,
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp1",
            task_name="train_rrn",
            label=strategy,
            command=cmd,
            config_paths=("configs/rrn/config.yaml", f"configs/rrn/{config_name}.yaml"),
            manifest={
                "strategy": strategy,
                "args": args,
                "config_files": ["configs/rrn/config.yaml", f"configs/rrn/{config_name}.yaml"],
            },
        )
    )


# ------------------------------------------------------------ #
# EXP 2: Synthology vs UDM Baseline on Family Tree
# ------------------------------------------------------------ #


@task
def exp2_generate_gold_test(ctx: Context, args=""):
    """Generates the frozen shared test set for Exp 2."""
    print("\nGenerating Exp 2 frozen test set.")
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        config_name="exp2_gold_test",
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp2",
            task_name="generate_gold_test",
            label="gold_test",
            command=cmd,
            config_paths=("configs/ont_generator/exp2_gold_test.yaml", "configs/ont_generator/config.yaml"),
            manifest={
                "args": args,
                "config_files": ["configs/ont_generator/exp2_gold_test.yaml", "configs/ont_generator/config.yaml"],
            },
            artifact_paths=(str(REPO_ROOT / "data" / "exp2" / "frozen_test"),),
        )
    )


@task
def exp2_generate_baseline(ctx: Context, fact_cap=None, target_cap=None, base_facts_per_sample=None, args=""):
    """
    Generates Exp 2 forward-chaining baseline data.

    Args:
        fact_cap: Optional train split fact cap for budget-matched runs.
        target_cap: Optional train split target cap for budget-matched runs.
        base_facts_per_sample: Optional fixed base relation count per sample.
    """
    print("\nGenerating Exp 2 baseline (FC) dataset.")
    overrides: List[str] = []
    if fact_cap is not None:
        overrides.append(f"dataset.train_fact_cap={fact_cap}")
    if target_cap is not None:
        overrides.append(f"dataset.train_target_cap={target_cap}")
    if base_facts_per_sample is not None:
        overrides.append(f"generator.base_relations_per_sample={base_facts_per_sample}")
    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.create_data",
        config_name="exp2_baseline",
        overrides=overrides,
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp2",
            task_name="generate_baseline",
            label="baseline",
            command=cmd,
            config_paths=("configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"),
            manifest={
                "fact_cap": fact_cap,
                "target_cap": target_cap,
                "base_facts_per_sample": base_facts_per_sample,
                "args": args,
                "config_files": ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
            },
            artifact_paths=(str(REPO_ROOT / "data" / "exp2" / "baseline" / "family_tree"),),
        )
    )


@task
def exp2_generate_synthology(ctx: Context, fact_cap=None, target_cap=None, proof_roots_per_rule=None, args=""):
    """
    Generates Exp 2 Synthology backward-chaining data.

    Args:
        fact_cap: Optional train split fact cap for budget-matched runs.
        target_cap: Optional train split target cap for budget-matched runs.
        proof_roots_per_rule: Optional fixed proof roots per selected rule.
    """
    print("\nGenerating Exp 2 synthology dataset.")
    overrides: List[str] = []
    if fact_cap is not None:
        overrides.append(f"dataset.train_fact_cap={fact_cap}")
    if target_cap is not None:
        overrides.append(f"dataset.train_target_cap={target_cap}")
    if proof_roots_per_rule is not None:
        overrides.append(f"generator.proof_roots_per_rule={proof_roots_per_rule}")
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        config_name="exp2_synthology",
        overrides=overrides,
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp2",
            task_name="generate_synthology",
            label="synthology",
            command=cmd,
            config_paths=("configs/ont_generator/exp2_synthology.yaml", "configs/ont_generator/config.yaml"),
            manifest={
                "fact_cap": fact_cap,
                "target_cap": target_cap,
                "proof_roots_per_rule": proof_roots_per_rule,
                "args": args,
                "config_files": ["configs/ont_generator/exp2_synthology.yaml", "configs/ont_generator/config.yaml"],
            },
            artifact_paths=(str(REPO_ROOT / "data" / "exp2" / "synthology" / "family_tree"),),
        )
    )


@task
def exp2_report_data(ctx: Context, args=""):
    """Generates parity/distribution reports for Exp 2 methods."""
    print("\nGenerating Exp 2 comparison report.")
    run_dir = _make_run_archive("exp2", "report_data", label="compare")
    report_dir = run_dir / "report"
    cmd = _build_uv_command(
        "data_reporter",
        "data_reporter.analyze",
        config_name="exp2_compare",
        overrides=(f"output.dir={shlex.quote(str(report_dir))}",),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp2",
            task_name="report_data",
            label="compare",
            command=cmd,
            config_paths=("configs/data_reporter/exp2_compare.yaml", "configs/data_reporter/config.yaml"),
            manifest={
                "args": args,
                "config_files": ["configs/data_reporter/exp2_compare.yaml", "configs/data_reporter/config.yaml"],
                "output_dir": str(report_dir),
            },
        )
    )


@task
def exp2_train_rrn(ctx: Context, dataset="baseline", args=""):
    """Trains RRN for Exp 2 on either baseline or synthology dataset."""
    dataset_key = dataset.strip().lower()
    if dataset_key not in {"baseline", "synthology"}:
        raise ValueError("dataset must be either 'baseline' or 'synthology'")

    rrn_dataset = "exp2_baseline" if dataset_key == "baseline" else "exp2_synthology"
    config_name = f"{rrn_dataset}_hpc"

    print(f"\nTraining Exp 2 RRN on: {dataset_key}")
    run_dir = _make_run_archive("exp2", "train_rrn", label=dataset_key)
    cmd = _build_uv_command(
        "rrn",
        "rrn.train",
        config_name=config_name,
        args=args,
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp2",
            task_name="train_rrn",
            label=dataset_key,
            command=cmd,
            config_paths=("configs/rrn/config.yaml", f"configs/rrn/{config_name}.yaml"),
            manifest={
                "dataset": dataset_key,
                "args": args,
                "config_files": ["configs/rrn/config.yaml", f"configs/rrn/{config_name}.yaml"],
            },
        )
    )


@task
def exp2_generate_both(
    ctx: Context,
    fact_cap=None,
    target_cap=None,
    baseline_base_facts=None,
    synthology_proof_roots=None,
    args="",
):
    """Convenience command to generate both Exp 2 methods with a shared cap."""
    run_dir = _make_run_archive("exp2", "generate_both", label="matched")
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "generate_both",
            "fact_cap": fact_cap,
            "target_cap": target_cap,
            "baseline_base_facts": baseline_base_facts,
            "synthology_proof_roots": synthology_proof_roots,
            "args": args,
        },
    )
    exp2_generate_baseline(
        ctx,
        fact_cap=fact_cap,
        target_cap=target_cap,
        base_facts_per_sample=baseline_base_facts,
        args=args,
    )
    exp2_generate_synthology(
        ctx,
        fact_cap=fact_cap,
        target_cap=target_cap,
        proof_roots_per_rule=synthology_proof_roots,
        args=args,
    )
    _write_text(
        run_dir / "run.log",
        "\n".join(
            [
                "Exp2 matched-budget generation summary",
                f"baseline fact_cap={fact_cap}, target_cap={target_cap}, base_facts_per_sample={baseline_base_facts}",
                f"synthology fact_cap={fact_cap}, target_cap={target_cap}, proof_roots_per_rule={synthology_proof_roots}",
            ]
        )
        + "\n",
    )


@task
def exp2_balance_datasets(
    ctx: Context,
    fact_cap,
    target_cap=None,
    baseline_base_facts=None,
    synthology_proof_roots=None,
    args="",
):
    """Generates both Exp 2 datasets using shared train fact/target caps for budget matching."""
    run_dir = _make_run_archive("exp2", "balance_datasets", label="matched")
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "balance_datasets",
            "fact_cap": fact_cap,
            "target_cap": target_cap,
            "baseline_base_facts": baseline_base_facts,
            "synthology_proof_roots": synthology_proof_roots,
            "args": args,
        },
    )
    exp2_generate_both(
        ctx,
        fact_cap=fact_cap,
        target_cap=target_cap,
        baseline_base_facts=baseline_base_facts,
        synthology_proof_roots=synthology_proof_roots,
        args=args,
    )
    _write_text(
        run_dir / "run.log",
        "\n".join(
            [
                "Exp2 balance datasets summary",
                f"fact_cap={fact_cap}",
                f"target_cap={target_cap}",
                f"baseline_base_facts={baseline_base_facts}",
                f"synthology_proof_roots={synthology_proof_roots}",
            ]
        )
        + "\n",
    )


@task
def exp2_smoke_jena_visual(ctx: Context, args=""):
    """Runs a tiny Jena-backed baseline generation and visualizes one sample graph."""
    print("\nRunning Exp 2 Jena smoke generation (visual).")
    run_dir = _make_run_archive("exp2", "smoke_jena_visual", label="visual")
    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.create_data",
        overrides=(
            "dataset.n_train=1",
            "dataset.n_val=0",
            "dataset.n_test=0",
            "dataset.output_dir=data/exp2/baseline/smoke_visual",
            "materialization.reasoner=jena",
            "materialization.iterative=false",
            "materialization.jena_profile=owl_mini",
            "materialization.timing.enabled=true",
            "materialization.timing.output_dir=data/exp2/timings",
            "materialization.timing.run_tag=exp2_smoke",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _write_json(run_dir / "manifest.json", {"task": "smoke_jena_visual", "args": args, "command": cmd})
    _run_logged_command(cmd, run_dir / "run.log")

    print("\nRendering smoke sample graph to visual-verification/exp2_smoke")
    viz_cmd = _build_uv_command(
        "kgvisualiser",
        "kgvisualiser.visualize",
        overrides=(
            "io.input_csv=data/exp2/baseline/smoke_visual/train/facts.csv",
            "io.sample_id=1000",
            "output.dir=visual-verification/exp2_smoke",
            "output.name_template=exp2_jena_smoke_1000",
        ),
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_logged_command(viz_cmd, run_dir / "visualize-run.log")


@task
def exp2_parity_loop(
    ctx: Context,
    max_attempts=250,
    min_deep_hops=3,
    tolerance_pct=10.0,
    synth_targets="data/exp2/synthology/family_tree/train/targets.csv",
    attempts_root="data/exp2/baseline/parity_runs",
    args="",
):
    """Retries UDM baseline generation until Exp 2 deep-count parity target is reached."""
    print("\nRunning Exp 2 UDM parity loop.")
    run_dir = _make_run_archive("exp2", "parity_loop", label="parity")
    attempts_dir = run_dir / "attempts"
    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.exp2_parity_loop",
        python_module=True,
        overrides=(
            f"--max-attempts {max_attempts}",
            f"--min-deep-hops {min_deep_hops}",
            f"--tolerance-pct {tolerance_pct}",
            f"--synth-targets {synth_targets}",
            f"--attempts-root {shlex.quote(str(attempts_dir))}",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp2",
            task_name="parity_loop",
            label="parity",
            command=cmd,
            config_paths=("configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"),
            manifest={
                "max_attempts": max_attempts,
                "min_deep_hops": min_deep_hops,
                "tolerance_pct": tolerance_pct,
                "synth_targets": synth_targets,
                "args": args,
                "config_files": ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
                "attempts_root": str(attempts_dir),
            },
        )
    )


@task
def exp2_parity_report(
    ctx: Context,
    min_deep_hops=3,
    synth_targets="data/exp2/synthology/family_tree/train/targets.csv",
    attempts_root="data/exp2/baseline/parity_runs",
    out_json="data/exp2/baseline/parity_runs/parity_report.json",
    out_csv="data/exp2/baseline/parity_runs/parity_attempts.csv",
    args="",
):
    """Builds Exp 2 parity report: K_deep plus per-attempt depth histograms."""
    print("\nGenerating Exp 2 parity report.")
    run_dir = _make_run_archive("exp2", "parity_report", label="parity")
    report_json = run_dir / "parity_report.json"
    report_csv = run_dir / "parity_attempts.csv"
    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.exp2_parity_report",
        python_module=True,
        overrides=(
            f"--min-deep-hops {min_deep_hops}",
            f"--synth-targets {synth_targets}",
            f"--attempts-root {attempts_root}",
            f"--out-json {shlex.quote(str(report_json))}",
            f"--out-csv {shlex.quote(str(report_csv))}",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp2",
            task_name="parity_report",
            label="parity",
            command=cmd,
            config_paths=("configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"),
            manifest={
                "min_deep_hops": min_deep_hops,
                "synth_targets": synth_targets,
                "attempts_root": attempts_root,
                "legacy_outputs": {"out_json": out_json, "out_csv": out_csv},
                "archive_outputs": {"out_json": str(report_json), "out_csv": str(report_csv)},
                "args": args,
                "config_files": ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
            },
        )
    )


# ------------------------------------------------------------ #
# EXP 3: OWL2Bench
# ------------------------------------------------------------ #


@task
def exp3_generate_owl2bench_abox(ctx: Context, universities=50, args="", archive_dir: Optional[str] = None):
    """Runs the existing OWL2Bench pipeline and stores raw generated OWL (ABox source)."""
    print(f"\nGenerating OWL2Bench data for Exp 3 (universities={universities}).")
    run_dir = (
        Path(archive_dir)
        if archive_dir
        else _make_run_archive("exp3", "generate_owl2bench_abox", label=str(universities))
    )
    timing_dir = run_dir / "timings"
    cmd = _build_uv_command(
        "owl2bench",
        "owl2bench.pipeline",
        overrides=(
            f"dataset.universities=[{universities}]",
            "dataset.output_dir=data/owl2bench/output",
            f"dataset.reasoning.materialization.timing.output_dir={shlex.quote(str(timing_dir))}",
            "dataset.reasoning.materialization.timing.enabled=true",
            f"dataset.reasoning.materialization.timing.run_tag=exp3_owl2bench_abox_{universities}",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp3",
            task_name="generate_owl2bench_abox",
            label=str(universities),
            command=cmd,
            config_paths=("configs/owl2bench/config.yaml", "configs/owl2bench/config_toy.yaml"),
            manifest={
                "universities": universities,
                "args": args,
                "config_files": ["configs/owl2bench/config.yaml", "configs/owl2bench/config_toy.yaml"],
                "timing_dir": str(timing_dir),
            },
            cwd=REPO_ROOT,
            artifact_paths=(str(REPO_ROOT / "vendor" / "OWL2Bench" / f"OWL2RL-{universities}.owl"),),
        )
    )


@task
def exp3_generate_baseline(ctx: Context, universities=50, args=""):
    """Generates Exp 3 baseline by chaining OWL2Bench generation with UDM/Jena materialization."""
    run_dir = _make_run_archive("exp3", "generate_baseline", label=str(universities))
    _snapshot_configs(
        run_dir,
        [
            "configs/owl2bench/config.yaml",
            "configs/owl2bench/config_toy.yaml",
            "configs/udm_baseline/config.yaml",
        ],
    )

    exp3_generate_owl2bench_abox(
        ctx, universities=universities, args=args, archive_dir=str(run_dir / "abox_generation")
    )

    abox_path = f"data/owl2bench/output/raw/owl2bench_{universities}/OWL2RL-{universities}.owl"
    closure_out = f"data/exp3/baseline/owl2bench_{universities}/closure.nt"
    inferred_out = f"data/exp3/baseline/owl2bench_{universities}/inferred.nt"

    exp3_materialize_abox(
        ctx,
        abox=abox_path,
        tbox="ontologies/UNIV-BENCH-OWL2RL.owl",
        closure_out=closure_out,
        inferred_out=inferred_out,
        archive_dir=str(run_dir / "materialization"),
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp3",
            "task": "generate_baseline",
            "universities": universities,
            "args": args,
            "abox_path": abox_path,
            "closure_out": closure_out,
            "inferred_out": inferred_out,
            "sub_runs": {
                "abox_generation": "abox_generation",
                "materialization": "materialization",
            },
        },
    )
    _write_text(
        run_dir / "run.log",
        "\n".join(
            [
                "Exp3 baseline summary",
                f"abox_generation={run_dir / 'abox_generation'}",
                f"materialization={run_dir / 'materialization'}",
                f"abox_path={abox_path}",
                f"closure_out={closure_out}",
                f"inferred_out={inferred_out}",
            ]
        )
        + "\n",
    )
    generated_dir = REPO_ROOT / "data" / "exp3" / "baseline" / f"owl2bench_{universities}"
    if generated_dir.exists():
        _archive_path(generated_dir, run_dir / "artifacts")


@task
def exp3_materialize_abox(
    ctx: Context,
    abox,
    tbox="ontologies/UNIV-BENCH-OWL2RL.owl",
    closure_out="outputs/exp3/closure.nt",
    inferred_out="outputs/exp3/inferred.nt",
    jena_profile="owl_mini",
    args="",
    archive_dir: Optional[str] = None,
):
    """Materializes an OWL2Bench ABox with UDM/Jena and exports closure + inferred triples."""
    print("\nRunning Exp 3 ABox materialization with UDM/Jena.")
    run_dir = Path(archive_dir) if archive_dir else _make_run_archive("exp3", "materialize_abox", label=jena_profile)
    closure_archive = run_dir / "artifacts" / "closure.nt"
    inferred_archive = run_dir / "artifacts" / "inferred.nt"
    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.materialize",
        overrides=(
            f"--tbox {tbox}",
            f"--abox {abox}",
            f"--closure-out {shlex.quote(str(closure_archive))}",
            f"--inferred-out {shlex.quote(str(inferred_archive))}",
            f"--jena-profile {jena_profile}",
            f"--timing-dir {shlex.quote(str(run_dir / 'timings'))}",
            "--timing-tag exp3_materialize_abox",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp3",
            task_name="materialize_abox",
            label=jena_profile,
            command=cmd,
            config_paths=("configs/udm_baseline/config.yaml",),
            manifest={
                "abox": abox,
                "tbox": tbox,
                "jena_profile": jena_profile,
                "args": args,
                "legacy_outputs": {"closure_out": closure_out, "inferred_out": inferred_out},
                "archive_outputs": {"closure_out": str(closure_archive), "inferred_out": str(inferred_archive)},
                "config_files": ["configs/udm_baseline/config.yaml"],
            },
            hydra_run_dir=False,
        )
    )
    closure_target = Path(closure_out)
    inferred_target = Path(inferred_out)
    closure_target.parent.mkdir(parents=True, exist_ok=True)
    inferred_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(closure_archive, closure_target)
    shutil.copy2(inferred_archive, inferred_target)


# ------------------------------------------------------------ #
# Helper methods
# ------------------------------------------------------------ #


def _run_experiment_spec(spec: ExperimentRunSpec) -> Path:
    run_dir = _make_run_archive(spec.experiment, spec.task_name, label=spec.label)
    if spec.config_paths:
        _snapshot_configs(run_dir, list(spec.config_paths))

    command = spec.command
    if spec.hydra_run_dir:
        command = f"{command} hydra.run.dir={shlex.quote(str(run_dir))}"

    manifest = {
        "experiment": spec.experiment,
        "task": spec.task_name,
        "label": spec.label,
        "command": command,
        **spec.manifest,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _run_logged_command(command, run_dir / "run.log", cwd=spec.cwd)

    for artifact_path in spec.artifact_paths:
        _archive_path(Path(artifact_path), run_dir / "artifacts")

    return run_dir


def _run_logged_command(command: str, log_path: Path, cwd: Optional[Path] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("LOGURU_COLORIZE", "1")
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(cwd or REPO_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {command}\n")
        log_handle.write(f"cwd: {str(cwd or REPO_ROOT)}\n\n")
        log_handle.flush()
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()

        if return_code != 0:
            log_handle.write(f"\n[exit code: {return_code}]\n")

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _archive_path(source: Path, destination_root: Path) -> Path:
    target = destination_root / _relative_snapshot_path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    elif source.exists():
        shutil.copy2(source, target)
    return target


def _is_negative_row(row: dict[str, str]) -> bool:
    label = str(row.get("label", "")).strip().lower()
    truth_value = str(row.get("truth_value", "")).strip().lower()
    row_type = str(row.get("type", "")).strip().lower()
    return label in {"0", "false"} or truth_value == "false" or row_type.startswith("neg")


def _find_visualization_target(output_root: str = "data/owl2bench/output_toy") -> tuple[str, str]:
    """Return (targets_csv_path, sample_id) from the most recent toy split output."""

    root = Path(output_root)
    if not root.exists():
        raise RuntimeError(f"Toy output directory does not exist: {root}")

    split_order = ("train", "val", "test")
    candidates: list[tuple[float, Path, str]] = []

    for split in split_order:
        for csv_path in root.glob(f"**/{split}/targets.csv"):
            try:
                mtime = csv_path.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, csv_path, split))

    if not candidates:
        raise RuntimeError(f"No split targets.csv files found under: {root}")

    # Prefer newest output; within same run we prioritize train -> val -> test.
    split_rank = {"train": 0, "val": 1, "test": 2}
    candidates.sort(key=lambda item: (-item[0], split_rank.get(item[2], 99)))

    # Parse per-sample counts for each candidate split file.
    split_sample_stats: List[Tuple[Path, str, List[Dict[str, Any]]]] = []
    for _, csv_path, split in candidates:
        per_sample: Dict[str, Dict[str, Any]] = {}
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = str(row.get("sample_id", "")).strip()
                if not sample_id:
                    continue

                stats = per_sample.setdefault(
                    sample_id,
                    {
                        "sample_id": sample_id,
                        "total": 0,
                        "base": 0,
                        "inferred": 0,
                        "negative": 0,
                    },
                )
                stats["total"] += 1

                row_type = str(row.get("type", "")).strip().lower()
                if row_type == "base_fact":
                    stats["base"] += 1
                elif row_type.startswith("inf") or row_type == "inferred":
                    stats["inferred"] += 1

                if _is_negative_row(row):
                    stats["negative"] += 1

        ranked = sorted(
            per_sample.values(),
            key=lambda s: (
                int(s["base"]),
                int(s["inferred"]),
                int(s["total"]),
                int(s["negative"]),
                int(str(s["sample_id"])),
            ),
            reverse=True,
        )
        split_sample_stats.append((csv_path, split, ranked))

    print("\nToy generation stats by split:", flush=True)
    for split in split_order:
        entry = next((x for x in split_sample_stats if x[1] == split), None)
        if entry is None:
            print(f"  {split}: samples=0, rows=0", flush=True)
            continue

        csv_path, _, ranked = entry
        sample_count = len(ranked)
        row_count = sum(int(s["total"]) for s in ranked)
        if ranked:
            best = ranked[0]
            print(
                f"  {split}: samples={sample_count}, rows={row_count}, "
                f"best_sample={best['sample_id']} "
                f"(rows={best['total']}, base={best['base']}, inferred={best['inferred']}, negatives={best['negative']})",
                flush=True,
            )
        else:
            print(f"  {split}: samples=0, rows=0", flush=True)

    # Prefer informative, non-tiny, and balanced samples for visualization quality.
    min_rows_for_visualization = 12
    min_base_for_visualization = 5
    min_inferred_for_visualization = 5
    split_priority = {"train": 0, "val": 1, "test": 2}
    all_candidates: List[Tuple[Path, str, Dict[str, Any]]] = []
    for csv_path, split, ranked in split_sample_stats:
        for stats in ranked:
            all_candidates.append((csv_path, split, stats))

    all_candidates.sort(
        key=lambda item: (
            min(int(item[2]["base"]), int(item[2]["inferred"])),
            int(item[2]["base"]),
            int(item[2]["inferred"]),
            int(item[2]["total"]),
            int(item[2]["negative"]),
            -split_priority.get(item[1], 99),
        ),
        reverse=True,
    )

    for csv_path, _, stats in all_candidates:
        if (
            int(stats["base"]) >= min_base_for_visualization
            and int(stats["inferred"]) >= min_inferred_for_visualization
            and int(stats["total"]) >= min_rows_for_visualization
        ):
            return str(csv_path), str(stats["sample_id"])

    for csv_path, _, stats in all_candidates:
        if int(stats["total"]) >= min_rows_for_visualization:
            return str(csv_path), str(stats["sample_id"])

    # Fallback: use the best available sample even if all are small.
    if all_candidates:
        csv_path, _, stats = all_candidates[0]
        return str(csv_path), str(stats["sample_id"])

    raise RuntimeError("Found split targets.csv files, but none contained a non-empty sample_id")


def _slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "run"


def _make_run_archive(experiment: str, task_name: str, label: Optional[str] = None) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    suffix = f"_{_slugify(label)}" if label else ""
    run_dir = EXPERIMENT_ARCHIVE_ROOT / today / experiment / task_name / f"{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _relative_snapshot_path(source_path: Path) -> Path:
    try:
        return source_path.relative_to(REPO_ROOT)
    except ValueError:
        return Path(source_path.name)


def _snapshot_configs(run_dir: Path, config_paths: List[str]) -> List[str]:
    snapshots: List[str] = []
    for config_path in config_paths:
        source_path = Path(config_path)
        if not source_path.exists():
            continue
        target_path = run_dir / "configs" / _relative_snapshot_path(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        snapshots.append(str(target_path.relative_to(run_dir)))
    return snapshots


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _compose_shell_command(parts: Iterable[str]) -> str:
    return " && ".join(part for part in parts if part)


def _build_uv_command(
    package: str,
    module_or_entrypoint: str,
    *,
    python_module: bool = True,
    config_name: Optional[str] = None,
    overrides: Iterable[str] = (),
    args: str = "",
    env: Optional[Dict[str, str]] = None,
) -> str:
    if python_module:
        command_tokens: List[str] = [
            "uv",
            "run",
            "--package",
            package,
            "python",
            "-m",
            module_or_entrypoint,
        ]
    else:
        command_tokens = ["uv", "run", "--package", package, module_or_entrypoint]

    if config_name:
        command_tokens.append(f"--config-name={config_name}")

    command_tokens.extend(overrides)

    if args:
        command_tokens.append(args)

    base_command = " ".join(command_tokens)

    env_commands: List[str] = []
    for key, value in (env or {}).items():
        env_commands.append(f"export {key}={shlex.quote(str(value))}")

    return _compose_shell_command([*env_commands, base_command])

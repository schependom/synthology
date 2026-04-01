import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "synthology"
PYTHON_VERSION = "3.12"
REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ARCHIVE_ROOT = REPO_ROOT / "reports" / "experiment_runs"

# Ensure SYNTHOLOGY_ROOT is set for subprocesses to locate configs
os.environ["SYNTHOLOGY_ROOT"] = os.path.dirname(os.path.abspath(__file__))


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


def _run_logged_command(command: str, log_path: Path, cwd: Optional[Path] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
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
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return_code = process.wait()

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


# Convert proprietary reldata format by P. Hohenecker's generator
# to CSV for RRN training/evaluation
@task
def convert_reldata(ctx: Context):
    """Converts family tree datasets in reldata format to CSV format."""

    print("\n-------------------------------------------")
    print("Converting generated ASP data to CSV format")
    print("--------------------------------------------\n")
    ctx.run("export PYTHONUNBUFFERED=1")
    ctx.run("export LOGURU_COLORIZE=1 && uv run --package asp_generator python -u -m asp_generator.convert_to_csv")

    logger.success("Conversion of family tree dataset from reldata to CSV completed.")


# Train the RRN on the ASP dataset
@task
def train_rrn_asp(ctx: Context, args=""):
    """Trains RRN on ASP-generated Family Tree dataset."""

    print("\nRunning RRN training with ASP dataset.")

    cmd = "export PYTHONUNBUFFERED=1 && "  # Ensure logs are unbuffered
    cmd += "export LOGURU_COLORIZE=1 && "  # Ensure logs are colored
    cmd += "uv run --package rrn python -m rrn.train data/dataset=asp"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


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
    cmd = "export LOGURU_COLORIZE=1 && "  # Ensure logs are colored
    cmd += "uv run --package ont_generator python -m ont_generator.create_data"
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
def gen_ft_fc(ctx: Context, args=""):
    """
    Generates family tree datasets with random base facts + owlrl
    forward-chaining materialization baseline.
    Uses configs/udm_baseline/config.yaml by default.
    """

    print("\nRunning family tree FC baseline generator.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package udm_baseline python -m udm_baseline.create_data"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def train_rrn_fc(ctx: Context, args=""):
    """Trains RRN on FC baseline dataset."""

    print("\nRunning RRN training with FC baseline dataset.")

    cmd = "export PYTHONUNBUFFERED=1 && "
    cmd += "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package rrn python -m rrn.train data/dataset=fc"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


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
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package ont_generator python -m ont_generator.create_data "
        "--config-name=config_visual_inspection"
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


# ------------------------------------------------------------ #
# Helper commands for experiments.
# ------------------------------------------------------------ #


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
    ABox generation -> Apache Jena materialization -> CSV export.
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
    base -> Jena materialization -> inferred targets -> negatives.
    """
    print("\nRunning OWL2Bench TOY generation pipeline.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package owl2bench python -m owl2bench.pipeline --config-name=config_toy"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)

    csv_path, sample_id = _find_visualization_target()
    print(f"\nAuto-visualizing toy sample {sample_id} from {csv_path}.", flush=True)
    viz_cmd = "export LOGURU_COLORIZE=1 && "
    viz_cmd += (
        "uv run --package kgvisualiser python -m kgvisualiser.visualize "
        f"io.input_csv={csv_path} "
        f"io.sample_id={sample_id}"
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


@task
def report_data(ctx: Context, args=""):
    """
    Generates method-comparison dataset reports and plots
    (predicate/type/hops/negative distributions, counts, ratios).
    Uses configs/data_reporter/config.yaml by default.
    """

    print("\nRunning dataset comparison report generator.")
    cmd = "export LOGURU_COLORIZE=1 && "
    cmd += "uv run --package data_reporter python -m data_reporter.analyze"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


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
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package data_reporter python -m data_reporter.paper_plots "
        f"--exp2-synth-targets {exp2_synth_targets} "
        f"--exp2-parity-summary {exp2_parity_summary} "
        f"--out-dir {out_dir}"
    )
    if exp2_baseline_targets:
        cmd += f" --exp2-baseline-targets {exp2_baseline_targets}"
    if exp3_targets:
        cmd += f" --exp3-targets {exp3_targets}"
    if exp3_abox and exp3_inferred:
        cmd += f" --exp3-abox {exp3_abox} --exp3-inferred {exp3_inferred}"
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


# ------------------------------------------------------------ #
# EXP 1: Negative Sampling Strategies for RRN Training
# ------------------------------------------------------------ #


@task
def exp1_generate_trainval_sets(ctx: Context):
    """Generates train/val sets for Exp 1 for all negative sampling strategies."""
    for strategy in ("random", "constrained", "proof_based"):
        exp1_generate_trainval(ctx, strategy=strategy)


@task
def exp1_generate_trainval(ctx: Context, strategy="proof_based", args=""):
    """Generates train/val sets for Exp 1 with a specific negative sampling strategy."""
    print(f"\nGenerating Exp 1 data using strategy: {strategy}")
    run_dir = _make_run_archive("exp1", "generate_trainval", label=strategy)
    _snapshot_configs(
        run_dir,
        [
            f"configs/ont_generator/exp1_{strategy}.yaml",
            "configs/ont_generator/config.yaml",
        ],
    )
    cmd = (
        f"export LOGURU_COLORIZE=1 && "
        f"uv run --package ont_generator python -m ont_generator.create_data "
        f"--config-name=exp1_{strategy}"
    )
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp1",
            "task": "generate_trainval",
            "strategy": strategy,
            "args": args,
            "command": cmd,
            "config_files": [
                f"configs/ont_generator/exp1_{strategy}.yaml",
                "configs/ont_generator/config.yaml",
            ],
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    generated_dir = REPO_ROOT / "data" / "exp1" / strategy
    if generated_dir.exists():
        _archive_path(generated_dir, run_dir / "artifacts")


@task
def exp1_generate_test_set(ctx: Context, args=""):
    """Generates the frozen 'near-miss' hard negative test set for Exp 1."""
    print("\nGenerating Exp 1 frozen test set (near-miss hard negatives)")
    run_dir = _make_run_archive("exp1", "generate_test_set", label="test_set")
    _snapshot_configs(
        run_dir,
        ["configs/ont_generator/exp1_test.yaml", "configs/ont_generator/config.yaml"],
    )
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package ont_generator python -m ont_generator.create_data "
        "--config-name=exp1_test"
    )
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp1",
            "task": "generate_test_set",
            "args": args,
            "command": cmd,
            "config_files": ["configs/ont_generator/exp1_test.yaml", "configs/ont_generator/config.yaml"],
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    generated_dir = REPO_ROOT / "data" / "exp1" / "test_set"
    if generated_dir.exists():
        _archive_path(generated_dir, run_dir / "artifacts")


@task
def exp1_train_rrn(ctx: Context, strategy="random", args=""):
    """Trains RRN for Exp 1. Provide a strategy to match datasets/logs."""
    print(f"\nTraining Exp 1 RRN. Strategy: {strategy}")
    run_dir = _make_run_archive("exp1", "train_rrn", label=strategy)
    log_dir = run_dir / "lightning_logs"
    checkpoint_dir = run_dir / "checkpoints"
    cmd = (
        f"export PYTHONUNBUFFERED=1 && export LOGURU_COLORIZE=1 && "
        f"export WANDB_DIR={shlex.quote(str(run_dir / 'wandb'))} && "
        f"uv run --package rrn python -m rrn.train "
        f"log_dir={shlex.quote(str(log_dir))} "
        f"callbacks.model_checkpoint.dirpath={shlex.quote(str(checkpoint_dir))} "
        f"data/dataset=exp1_{strategy} "
        f"+logger.name=exp1_{strategy} "
        f"+logger.group=exp1_negative_sampling "
        f"+logger.tags=[exp1,{strategy}]"
    )
    if args:
        cmd += f" {args}"
    _snapshot_configs(
        run_dir,
        ["configs/rrn/config.yaml", f"configs/rrn/data/dataset/exp1_{strategy}.yaml"],
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp1",
            "task": "train_rrn",
            "strategy": strategy,
            "args": args,
            "command": cmd,
            "config_files": ["configs/rrn/config.yaml", f"configs/rrn/data/dataset/exp1_{strategy}.yaml"],
            "output_dirs": {"log_dir": str(log_dir), "checkpoint_dir": str(checkpoint_dir)},
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


# ------------------------------------------------------------ #
# EXP 2: ...
# ------------------------------------------------------------ #


@task
def exp2_generate_gold_test(ctx: Context, args=""):
    """Generates the frozen shared test set for Exp 2."""
    print("\nGenerating Exp 2 frozen test set.")
    run_dir = _make_run_archive("exp2", "generate_gold_test", label="gold_test")
    _snapshot_configs(
        run_dir,
        ["configs/ont_generator/exp2_gold_test.yaml", "configs/ont_generator/config.yaml"],
    )
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package ont_generator python -m ont_generator.create_data "
        "--config-name=exp2_gold_test"
    )
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "generate_gold_test",
            "args": args,
            "command": cmd,
            "config_files": ["configs/ont_generator/exp2_gold_test.yaml", "configs/ont_generator/config.yaml"],
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    generated_dir = REPO_ROOT / "data" / "exp2" / "frozen_test"
    if generated_dir.exists():
        _archive_path(generated_dir, run_dir / "artifacts")


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
    run_dir = _make_run_archive("exp2", "generate_baseline", label="baseline")
    _snapshot_configs(
        run_dir,
        ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
    )
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package udm_baseline python -m udm_baseline.create_data "
        "--config-name=exp2_baseline"
    )
    if fact_cap is not None:
        cmd += f" dataset.train_fact_cap={fact_cap}"
    if target_cap is not None:
        cmd += f" dataset.train_target_cap={target_cap}"
    if base_facts_per_sample is not None:
        cmd += f" generator.base_relations_per_sample={base_facts_per_sample}"
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "generate_baseline",
            "fact_cap": fact_cap,
            "target_cap": target_cap,
            "base_facts_per_sample": base_facts_per_sample,
            "args": args,
            "command": cmd,
            "config_files": ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    generated_dir = REPO_ROOT / "data" / "exp2" / "baseline" / "family_tree"
    if generated_dir.exists():
        _archive_path(generated_dir, run_dir / "artifacts")


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
    run_dir = _make_run_archive("exp2", "generate_synthology", label="synthology")
    _snapshot_configs(
        run_dir,
        ["configs/ont_generator/exp2_synthology.yaml", "configs/ont_generator/config.yaml"],
    )
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package ont_generator python -m ont_generator.create_data "
        "--config-name=exp2_synthology"
    )
    if fact_cap is not None:
        cmd += f" dataset.train_fact_cap={fact_cap}"
    if target_cap is not None:
        cmd += f" dataset.train_target_cap={target_cap}"
    if proof_roots_per_rule is not None:
        cmd += f" generator.proof_roots_per_rule={proof_roots_per_rule}"
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "generate_synthology",
            "fact_cap": fact_cap,
            "target_cap": target_cap,
            "proof_roots_per_rule": proof_roots_per_rule,
            "args": args,
            "command": cmd,
            "config_files": ["configs/ont_generator/exp2_synthology.yaml", "configs/ont_generator/config.yaml"],
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    generated_dir = REPO_ROOT / "data" / "exp2" / "synthology" / "family_tree"
    if generated_dir.exists():
        _archive_path(generated_dir, run_dir / "artifacts")


@task
def exp2_report_data(ctx: Context, args=""):
    """Generates parity/distribution reports for Exp 2 methods."""
    print("\nGenerating Exp 2 comparison report.")
    run_dir = _make_run_archive("exp2", "report_data", label="compare")
    _snapshot_configs(
        run_dir,
        ["configs/data_reporter/exp2_compare.yaml", "configs/data_reporter/config.yaml"],
    )
    report_dir = run_dir / "report"
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package data_reporter python -m data_reporter.analyze "
        "--config-name=exp2_compare "
        f"output.dir={shlex.quote(str(report_dir))}"
    )
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "report_data",
            "args": args,
            "command": cmd,
            "config_files": ["configs/data_reporter/exp2_compare.yaml", "configs/data_reporter/config.yaml"],
            "output_dir": str(report_dir),
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


@task
def exp2_train_rrn(ctx: Context, dataset="baseline", args=""):
    """Trains RRN for Exp 2 on either baseline or synthology dataset."""
    dataset_key = dataset.strip().lower()
    if dataset_key not in {"baseline", "synthology"}:
        raise ValueError("dataset must be either 'baseline' or 'synthology'")

    rrn_dataset = "exp2_baseline" if dataset_key == "baseline" else "exp2_synthology"

    print(f"\nTraining Exp 2 RRN on: {dataset_key}")
    run_dir = _make_run_archive("exp2", "train_rrn", label=dataset_key)
    log_dir = run_dir / "lightning_logs"
    checkpoint_dir = run_dir / "checkpoints"
    cmd = (
        "export PYTHONUNBUFFERED=1 && export LOGURU_COLORIZE=1 && "
        f"export WANDB_DIR={shlex.quote(str(run_dir / 'wandb'))} && "
        "uv run --package rrn python -m rrn.train "
        f"data/dataset={rrn_dataset} "
        f"log_dir={shlex.quote(str(log_dir))} "
        f"callbacks.model_checkpoint.dirpath={shlex.quote(str(checkpoint_dir))} "
        f"+logger.name=exp2_{dataset_key} "
        "+logger.group=exp2_multihop "
        f"+logger.tags=[exp2,{dataset_key}]"
    )
    if args:
        cmd += f" {args}"
    _snapshot_configs(
        run_dir,
        ["configs/rrn/config.yaml", f"configs/rrn/data/dataset/{rrn_dataset}.yaml"],
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "train_rrn",
            "dataset": dataset_key,
            "args": args,
            "command": cmd,
            "config_files": ["configs/rrn/config.yaml", f"configs/rrn/data/dataset/{rrn_dataset}.yaml"],
            "output_dirs": {"log_dir": str(log_dir), "checkpoint_dir": str(checkpoint_dir)},
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


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
    exp2_generate_both(
        ctx,
        fact_cap=fact_cap,
        target_cap=target_cap,
        baseline_base_facts=baseline_base_facts,
        synthology_proof_roots=synthology_proof_roots,
        args=args,
    )


@task
def exp2_smoke_jena_visual(ctx: Context, args=""):
    """Runs a tiny Jena-backed baseline generation and visualizes one sample graph."""
    print("\nRunning Exp 2 Jena smoke generation (visual).")
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package udm_baseline python -m udm_baseline.create_data "
        "dataset.n_train=1 dataset.n_val=0 dataset.n_test=0 "
        "dataset.output_dir=data/exp2/baseline/smoke_visual "
        "materialization.reasoner=jena materialization.iterative=false materialization.jena_profile=owl_mini "
        "materialization.timing.enabled=true "
        "materialization.timing.output_dir=data/exp2/timings "
        "materialization.timing.run_tag=exp2_smoke"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)

    print("\nRendering smoke sample graph to visual-verification/exp2_smoke")
    viz_cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package kgvisualiser python -m kgvisualiser.visualize "
        "io.input_csv=data/exp2/baseline/smoke_visual/train/facts.csv "
        "io.sample_id=1000 "
        "output.dir=visual-verification/exp2_smoke "
        "output.name_template=exp2_jena_smoke_1000"
    )
    ctx.run(viz_cmd)


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
    _snapshot_configs(
        run_dir,
        ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
    )
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package udm_baseline python -m udm_baseline.exp2_parity_loop "
        f"--max-attempts {max_attempts} "
        f"--min-deep-hops {min_deep_hops} "
        f"--tolerance-pct {tolerance_pct} "
        f"--synth-targets {synth_targets} "
        f"--attempts-root {shlex.quote(str(attempts_dir))}"
    )
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "parity_loop",
            "max_attempts": max_attempts,
            "min_deep_hops": min_deep_hops,
            "tolerance_pct": tolerance_pct,
            "synth_targets": synth_targets,
            "args": args,
            "command": cmd,
            "config_files": ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
            "attempts_root": str(attempts_dir),
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


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
    _snapshot_configs(
        run_dir,
        ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
    )
    report_json = run_dir / "parity_report.json"
    report_csv = run_dir / "parity_attempts.csv"
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package udm_baseline python -m udm_baseline.exp2_parity_report "
        f"--min-deep-hops {min_deep_hops} "
        f"--synth-targets {synth_targets} "
        f"--attempts-root {attempts_root} "
        f"--out-json {shlex.quote(str(report_json))} "
        f"--out-csv {shlex.quote(str(report_csv))}"
    )
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp2",
            "task": "parity_report",
            "min_deep_hops": min_deep_hops,
            "synth_targets": synth_targets,
            "attempts_root": attempts_root,
            "legacy_outputs": {"out_json": out_json, "out_csv": out_csv},
            "archive_outputs": {"out_json": str(report_json), "out_csv": str(report_csv)},
            "args": args,
            "command": cmd,
            "config_files": ["configs/udm_baseline/exp2_baseline.yaml", "configs/udm_baseline/config.yaml"],
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")


# ------------------------------------------------------------ #
# EXP 3: ...
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
    _snapshot_configs(
        run_dir,
        ["configs/owl2bench/config.yaml", "configs/owl2bench/config_toy.yaml"],
    )
    timing_dir = run_dir / "timings"
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package owl2bench python -m owl2bench.pipeline "
        f"dataset.universities=[{universities}] "
        "dataset.output_dir=data/owl2bench/output "
        f"dataset.reasoning.materialization.timing.output_dir={shlex.quote(str(timing_dir))} "
        "dataset.reasoning.materialization.timing.enabled=true "
        f"dataset.reasoning.materialization.timing.run_tag=exp3_owl2bench_abox_{universities}"
    )
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp3",
            "task": "generate_owl2bench_abox",
            "universities": universities,
            "args": args,
            "command": cmd,
            "config_files": ["configs/owl2bench/config.yaml", "configs/owl2bench/config_toy.yaml"],
            "timing_dir": str(timing_dir),
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    generated_owl = REPO_ROOT / "vendor" / "OWL2Bench" / f"OWL2RL-{universities}.owl"
    if generated_owl.exists():
        _archive_path(generated_owl, run_dir / "artifacts")


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
        tbox="data/owl2bench/input/UNIV-BENCH-OWL2RL.owl",
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
    generated_dir = REPO_ROOT / "data" / "exp3" / "baseline" / f"owl2bench_{universities}"
    if generated_dir.exists():
        _archive_path(generated_dir, run_dir / "artifacts")


@task
def exp3_materialize_abox(
    ctx: Context,
    abox,
    tbox="data/OWL2Bench/input/UNIV-BENCH-OWL2RL.owl",
    closure_out="outputs/exp3/closure.nt",
    inferred_out="outputs/exp3/inferred.nt",
    jena_profile="owl_mini",
    args="",
    archive_dir: Optional[str] = None,
):
    """Materializes an OWL2Bench ABox with UDM/Jena and exports closure + inferred triples."""
    print("\nRunning Exp 3 ABox materialization with UDM/Jena.")
    run_dir = Path(archive_dir) if archive_dir else _make_run_archive("exp3", "materialize_abox", label=jena_profile)
    _snapshot_configs(
        run_dir,
        ["configs/udm_baseline/config.yaml"],
    )
    closure_archive = run_dir / "artifacts" / "closure.nt"
    inferred_archive = run_dir / "artifacts" / "inferred.nt"
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package udm_baseline python -m udm_baseline.materialize "
        f"--tbox {tbox} "
        f"--abox {abox} "
        f"--closure-out {shlex.quote(str(closure_archive))} "
        f"--inferred-out {shlex.quote(str(inferred_archive))} "
        f"--jena-profile {jena_profile} "
        f"--timing-dir {shlex.quote(str(run_dir / 'timings'))} "
        "--timing-tag exp3_materialize_abox"
    )
    if args:
        cmd += f" {args}"
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp3",
            "task": "materialize_abox",
            "abox": abox,
            "tbox": tbox,
            "jena_profile": jena_profile,
            "args": args,
            "command": cmd,
            "legacy_outputs": {"closure_out": closure_out, "inferred_out": inferred_out},
            "archive_outputs": {"closure_out": str(closure_archive), "inferred_out": str(inferred_archive)},
            "config_files": ["configs/udm_baseline/config.yaml"],
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    closure_target = Path(closure_out)
    inferred_target = Path(inferred_out)
    closure_target.parent.mkdir(parents=True, exist_ok=True)
    inferred_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(closure_archive, closure_target)
    shutil.copy2(inferred_archive, inferred_target)

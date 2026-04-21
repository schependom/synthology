import csv
import json
import os
import random
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
from omegaconf import OmegaConf

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
# Smoke tests and paper outputs
# ------------------------------------------------------------ #


@task
def gen_owl2bench_toy(ctx: Context, args=""):
    """
    Runs a tiny OWL2Bench pipeline config for quick end-to-end verification:
    base -> Jena materialization -> inferred targets -> negatives.
    """
    print("\nRunning OWL2Bench TOY generation pipeline.")
    run_dir = _make_run_archive("owl2bench", "generate_toy", label="toy")
    owl2bench_env = _resolve_owl2bench_env(run_dir)
    cmd = _build_uv_command(
        "owl2bench",
        "owl2bench.pipeline",
        config_name="config_toy",
        args=args,
        env=owl2bench_env,
    )
    _write_json(run_dir / "manifest.json", {"task": "generate_toy", "args": args, "command": cmd})
    _run_logged_command(cmd, run_dir / "run.log")

    csv_path, sample_id = _find_visualization_target()
    targets_csv = Path(csv_path)
    facts_csv = targets_csv.with_name("facts.csv")
    print(f"\nAuto-visualizing toy sample {sample_id} from {csv_path}.", flush=True)
    viz_cmd = _build_uv_command(
        "kgvisualiser",
        "kgvisualiser.visualize",
        overrides=(
            f"io.input_csv={facts_csv.as_posix()}",
            f"io.targets_csv={targets_csv.as_posix()}",
            f"io.sample_id={sample_id}",
            "filters.include_negatives=true",
        ),
        env={"LOGURU_COLORIZE": "1"},
    )
    _write_text(
        run_dir / "visualize.log",
        "\n".join(
            [
                f"input_csv={facts_csv.as_posix()}",
                f"targets_csv={targets_csv.as_posix()}",
                f"sample_id={sample_id}",
                f"command={viz_cmd}",
            ]
        )
        + "\n",
    )
    _run_logged_command(viz_cmd, run_dir / "visualize-run.log")


@task
def paper_visual_report(
    ctx: Context,
    exp2_synth_targets="data/exp2/synthology/family_tree/train/targets.csv",
    exp2_parity_summary="data/exp2/baseline/parity_runs/parity_loop_summary.json",
    exp2_baseline_targets="",
    exp3_targets="",
    exp3_synth_targets="",
    exp3_baseline_targets="",
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
    if exp3_synth_targets:
        cmd += f" --exp3-synth-targets {exp3_synth_targets}"
    if exp3_baseline_targets:
        cmd += f" --exp3-baseline-targets {exp3_baseline_targets}"
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
            "exp3_synth_targets": exp3_synth_targets,
            "exp3_baseline_targets": exp3_baseline_targets,
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
            "strategies": ["random", "constrained", "proof_based", "mixed"],
        },
    )
    summary_lines = ["Exp1 train/val set generation summary"]
    for strategy in ("random", "constrained", "proof_based", "mixed"):
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


def _log_dataset_generation_time(name: str, root: Path) -> None:
    """Logs the latest modification time of a dataset split to confirm freshness."""
    sentinel = root / "train" / "facts.csv"
    if sentinel.exists():
        mtime = datetime.fromtimestamp(sentinel.stat().st_mtime)
        logger.info(f"  {name:>12s} dataset last generated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        logger.warning(f"  {name:>12s} dataset NOT FOUND at {sentinel}")


@task
def exp2_report_data(ctx: Context, args="", strict="true"):
    """Generates parity/distribution reports for Exp 2 methods."""
    print("\nGenerating Exp 2 comparison report.")
    strict_mode = _to_bool(strict)

    baseline_root = REPO_ROOT / "data" / "exp2" / "baseline" / "family_tree"
    synthology_root = REPO_ROOT / "data" / "exp2" / "synthology" / "family_tree"

    logger.info("Dataset generation timestamps (train/facts.csv mtime):")
    _log_dataset_generation_time("baseline", baseline_root)
    _log_dataset_generation_time("synthology", synthology_root)
    mismatches = _get_split_sample_count_mismatches(baseline_root, synthology_root, splits=("train", "val", "test"))
    label_mismatches = _get_split_target_label_mismatches(baseline_root, synthology_root, splits=("train", "val", "test"))
    if mismatches:
        mismatch_text = "; ".join(
            f"{split}: baseline={base_count}, synthology={syn_count}" for split, base_count, syn_count in mismatches
        )
        hint = (
            "Dataset splits are not aligned for fair Exp2 comparison. "
            f"{mismatch_text}. Run 'bsub < jobscripts/exp2-balance-datasets.sh' and retry."
        )
        if strict_mode:
            raise ValueError(hint)
        logger.warning(hint)
    if label_mismatches:
        mismatch_text = "; ".join(
            f"{split}: baseline(pos={base_pos},neg={base_neg}) vs synthology(pos={syn_pos},neg={syn_neg})"
            for split, base_pos, base_neg, syn_pos, syn_neg in label_mismatches
        )
        hint = (
            "Target label balance is not aligned for fair Exp2 comparison. "
            f"{mismatch_text}. Run 'bsub < jobscripts/exp2-balance-datasets.sh' and retry."
        )
        if strict_mode:
            raise ValueError(hint)
        logger.warning(hint)

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
                "strict": strict_mode,
                "config_files": ["configs/data_reporter/exp2_compare.yaml", "configs/data_reporter/config.yaml"],
                "output_dir": str(report_dir),
            },
        )
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
def exp2_analyze_latest_baseline(ctx: Context, args=""):
    """Analyzes the latest archived Exp2 baseline run (hop distribution + timing summary)."""
    print("\nAnalyzing latest Exp 2 baseline run.")
    run_dir = _make_run_archive("exp2", "analyze_latest_baseline", label="baseline")
    analysis_dir = run_dir / "analysis"
    cmd = _build_uv_command(
        "data_reporter",
        "data_reporter.exp2_latest_baseline",
        overrides=(
            f"--repo-root {shlex.quote(str(REPO_ROOT))}",
            f"--out-dir {shlex.quote(str(analysis_dir))}",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp2",
            task_name="analyze_latest_baseline",
            label="baseline",
            command=cmd,
            manifest={
                "args": args,
                "output_dir": str(analysis_dir),
            },
            hydra_run_dir=False,
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

    if dataset_key == "baseline":
        train_path = REPO_ROOT / "data" / "exp2" / "baseline" / "family_tree" / "train"
        val_path   = REPO_ROOT / "data" / "exp2" / "baseline" / "family_tree" / "val"
        test_path  = REPO_ROOT / "data" / "exp2" / "frozen_test"
    else:
        train_path = REPO_ROOT / "data" / "exp2" / "synthology" / "family_tree" / "train"
        val_path   = REPO_ROOT / "data" / "exp2" / "synthology" / "family_tree" / "val"
        test_path  = REPO_ROOT / "data" / "exp2" / "frozen_test"

    def _mtime(p: Path) -> str:
        facts = p / "facts.csv"
        if facts.exists():
            import datetime
            return datetime.datetime.fromtimestamp(facts.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return "missing"

    print(f"\nTraining Exp 2 RRN on: {dataset_key}")
    print(f"  train : {train_path}  [{_mtime(train_path)}]")
    print(f"  val   : {val_path}  [{_mtime(val_path)}]")
    print(f"  test  : {test_path}  [{_mtime(test_path)}]")

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
                "train_path": str(train_path),
                "val_path": str(val_path),
                "test_path": str(test_path),
                "train_facts_mtime": _mtime(train_path),
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
    baseline_args="",
    synthology_args="",
):
    """Convenience command to generate both Exp 2 methods with a shared cap."""
    run_dir = _make_run_archive("exp2", "generate_both", label="matched")

    baseline_args_combined = " ".join(token for token in [str(args).strip(), str(baseline_args).strip()] if token)
    synthology_args_combined = " ".join(token for token in [str(args).strip(), str(synthology_args).strip()] if token)

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
            "baseline_args": baseline_args,
            "synthology_args": synthology_args,
        },
    )
    exp2_generate_baseline(
        ctx,
        fact_cap=fact_cap,
        target_cap=target_cap,
        base_facts_per_sample=baseline_base_facts,
        args=baseline_args_combined,
    )
    exp2_generate_synthology(
        ctx,
        fact_cap=fact_cap,
        target_cap=target_cap,
        proof_roots_per_rule=synthology_proof_roots,
        args=synthology_args_combined,
    )
    _write_text(
        run_dir / "run.log",
        "\n".join(
            [
                "Exp2 matched-budget generation summary",
                f"baseline fact_cap={fact_cap}, target_cap={target_cap}, base_facts_per_sample={baseline_base_facts}",
                f"synthology fact_cap={fact_cap}, target_cap={target_cap}, proof_roots_per_rule={synthology_proof_roots}",
                f"baseline args={baseline_args_combined}",
                f"synthology args={synthology_args_combined}",
            ]
        )
        + "\n",
    )


@task
def exp2_balance_datasets(ctx: Context, config_path="configs/experiments/exp2_balance_hpc.yaml"):
    """Generates both Exp 2 datasets using only YAML config files (no CLI overrides)."""
    cfg = _load_yaml_config(config_path)
    run_dir = _make_run_archive("exp2", "balance_datasets", label="matched")
    _write_json(run_dir / "manifest.json", cfg)
    fact_cap = int(cfg["fact_cap"]) if cfg.get("fact_cap") is not None else None
    target_cap = int(cfg["target_cap"]) if cfg.get("target_cap") is not None else None
    baseline_base_facts = int(cfg["baseline_base_facts"]) if cfg.get("baseline_base_facts") is not None else None
    synthology_proof_roots = int(cfg["synthology_proof_roots"]) if cfg.get("synthology_proof_roots") is not None else None
    exp2_generate_both(
        ctx,
        fact_cap=fact_cap,
        target_cap=target_cap,
        baseline_base_facts=baseline_base_facts,
        synthology_proof_roots=synthology_proof_roots,
        args=str(cfg.get("args", "")),
        baseline_args=str(cfg.get("baseline_args", "")),
        synthology_args=str(cfg.get("synthology_args", "")),
    )
    baseline_root = REPO_ROOT / "data" / "exp2" / "baseline" / "family_tree"
    synthology_root = REPO_ROOT / "data" / "exp2" / "synthology" / "family_tree"
    sample_align_summary = _align_exp2_split_sample_counts(baseline_root, synthology_root, splits=("train", "val", "test"))
    _write_json(run_dir / "sample_alignment.json", sample_align_summary)
    label_align_summary = _align_exp2_split_target_labels(baseline_root, synthology_root, splits=("train", "val", "test"))
    _write_json(run_dir / "label_alignment.json", label_align_summary)


@task
def exp2_balance_datasets_hpc(ctx: Context, config_path="configs/experiments/exp2_balance_hpc.yaml"):
    """Runs Exp2 matched-budget dataset generation from centralized YAML config only."""
    exp2_balance_datasets(ctx, config_path=config_path)


@task
def exp2_balance_smoke(
    ctx: Context,
    config_path="configs/experiments/exp2_balance_smoke.yaml",
    run_report="true",
):
    """Runs a short Exp2 matched-budget smoke generation for rapid debugging."""
    cfg = _load_yaml_config(config_path)

    fact_cap = int(cfg["fact_cap"])
    target_cap = cfg.get("target_cap")
    baseline_base_facts = cfg.get("baseline_base_facts")
    synthology_proof_roots = cfg.get("synthology_proof_roots")
    args = str(cfg.get("args", ""))
    baseline_args = str(cfg.get("baseline_args", ""))
    synthology_args = str(cfg.get("synthology_args", ""))

    exp2_balance_datasets(
        ctx,
        fact_cap=fact_cap,
        target_cap=int(target_cap) if target_cap is not None else None,
        baseline_base_facts=int(baseline_base_facts) if baseline_base_facts is not None else None,
        synthology_proof_roots=int(synthology_proof_roots) if synthology_proof_roots is not None else None,
        args=args,
        baseline_args=baseline_args,
        synthology_args=synthology_args,
    )

    if _to_bool(run_report):
        exp2_report_data(ctx)


@task
def exp2_sweep_targetcaps_seeds(ctx: Context, config_path="configs/experiments/exp2_sweep_hpc.yaml"):
    """Runs the Exp2 target-cap/seed sweep from centralized YAML config."""
    cfg = _load_yaml_config(config_path)

    fact_cap = int(cfg["fact_cap"])
    target_caps = [int(value) for value in cfg.get("target_caps", [])]
    seeds = [int(value) for value in cfg.get("seeds", [])]
    baseline_base_facts = int(cfg["baseline_base_facts"])
    synthology_proof_roots = int(cfg["synthology_proof_roots"])
    run_report = bool(cfg.get("run_report", True))

    if not target_caps or not seeds:
        raise ValueError("exp2 sweep config must define non-empty target_caps and seeds")

    for target_cap in target_caps:
        exp2_balance_datasets(
            ctx,
            fact_cap=fact_cap,
            target_cap=target_cap,
            baseline_base_facts=baseline_base_facts,
            synthology_proof_roots=synthology_proof_roots,
        )

        if run_report:
            exp2_report_data(ctx)

        for seed in seeds:
            run_name_baseline = f"exp2_baseline_tc{target_cap}_seed{seed}"
            run_name_synthology = f"exp2_synthology_tc{target_cap}_seed{seed}"

            exp2_train_rrn(
                ctx,
                dataset="baseline",
                args=(
                    f"+seed={seed} +logger.name={run_name_baseline} "
                    f"+logger.group=exp2_multihop +logger.tags=[exp2,baseline,target_cap_{target_cap},seed{seed}]"
                ),
            )
            exp2_train_rrn(
                ctx,
                dataset="synthology",
                args=(
                    f"+seed={seed} +logger.name={run_name_synthology} "
                    f"+logger.group=exp2_multihop +logger.tags=[exp2,synthology,target_cap_{target_cap},seed{seed}]"
                ),
            )



# ------------------------------------------------------------ #
# EXP 3: OWL2Bench
# ------------------------------------------------------------ #


@task
def exp3_report_data(ctx: Context, universities=20, baseline_path="", synthology_path="", args=""):
    """Generates parity/distribution reports for Exp 3 baseline vs synthology datasets."""
    print("\nGenerating Exp 3 comparison report.")
    run_dir = _make_run_archive("exp3", "report_data", label="compare")
    report_dir = run_dir / "report"

    balanced_baseline = "data/exp3/balanced_baseline/owl2bench"
    raw_baseline = "data/owl2bench/output/owl2bench"
    resolved_baseline_path = baseline_path or (balanced_baseline if Path(balanced_baseline).exists() else raw_baseline)
    balanced = "data/exp3/balanced/owl2bench"
    unbalanced = "data/exp3/synthology/owl2bench"
    resolved_synthology_path = synthology_path or (balanced if Path(balanced).exists() else unbalanced)

    cmd = _build_uv_command(
        "data_reporter",
        "data_reporter.analyze",
        config_name="exp3_compare",
        overrides=(
            f"methods.0.path={shlex.quote(str(resolved_baseline_path))}",
            f"methods.1.path={shlex.quote(str(resolved_synthology_path))}",
            f"output.dir={shlex.quote(str(report_dir))}",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp3",
            task_name="report_data",
            label="compare",
            command=cmd,
            config_paths=("configs/data_reporter/exp3_compare.yaml", "configs/data_reporter/config.yaml"),
            manifest={
                "universities": universities,
                "baseline_path": resolved_baseline_path,
                "synthology_path": resolved_synthology_path,
                "args": args,
                "config_files": ["configs/data_reporter/exp3_compare.yaml", "configs/data_reporter/config.yaml"],
                "output_dir": str(report_dir),
            },
        )
    )

@task
def exp3_analyze_latest_baseline(ctx: Context, args=""):
    """Analyzes the latest archived Exp3 baseline run (label balance, hops, timing, integrity)."""
    print("\nAnalyzing latest Exp 3 baseline run.")
    run_dir = _make_run_archive("exp3", "analyze_latest_baseline", label="baseline")
    analysis_dir = run_dir / "analysis"
    cmd = _build_uv_command(
        "data_reporter",
        "data_reporter.exp3_latest_baseline",
        overrides=(
            f"--repo-root {shlex.quote(str(REPO_ROOT))}",
            f"--out-dir {shlex.quote(str(analysis_dir))}",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp3",
            task_name="analyze_latest_baseline",
            label="baseline",
            command=cmd,
            manifest={
                "args": args,
                "output_dir": str(analysis_dir),
            },
            hydra_run_dir=False,
        )
    )


@task
def exp3_generate_owl2bench_abox(
    ctx: Context,
    universities=5,
    args="",
    archive_dir: Optional[str] = None,
    reasoning_input_triple_cap=0,
    abox_jena_heap_mb=8192,
    config_name: str = "config",
):
    """Runs the existing OWL2Bench pipeline and stores raw generated OWL (ABox source)."""
    print(f"\nGenerating OWL2Bench data for Exp 3 (universities={universities}, config={config_name}).")
    run_dir = (
        Path(archive_dir)
        if archive_dir
        else _make_run_archive("exp3", "generate_owl2bench_abox", label=str(universities))
    )
    timing_dir = run_dir / "timings"
    overrides = [
        f"dataset.universities=[{universities}]",
        f"dataset.reasoning.materialization.timing.output_dir={shlex.quote(str(timing_dir))}",
        "dataset.reasoning.materialization.timing.enabled=true",
        f"dataset.reasoning.materialization.timing.run_tag=exp3_owl2bench_abox_{universities}",
    ]
    cap = int(reasoning_input_triple_cap)
    if cap > 0:
        overrides.append(f"+dataset.reasoning_input_triple_cap={cap}")

    abox_jena_heap_mb = str(abox_jena_heap_mb)
    config_file = f"configs/owl2bench/{config_name}.yaml"

    cmd = _build_uv_command(
        "owl2bench",
        "owl2bench.pipeline",
        config_name=config_name,
        overrides=tuple(overrides),
        args=args,
        env={
            "LOGURU_COLORIZE": "1",
            "SYNTHOLOGY_UDM_BASELINE_XMX_MB": abox_jena_heap_mb,
            "SYNTHOLOGY_JENA_XMX_MB": abox_jena_heap_mb,
            "SYNTHOLOGY_HEAP_MB": abox_jena_heap_mb,
        },
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp3",
            task_name="generate_owl2bench_abox",
            label=str(universities),
            command=cmd,
            config_paths=(config_file, "configs/owl2bench/config_toy.yaml"),
            manifest={
                "universities": universities,
                "args": args,
                "config_name": config_name,
                "reasoning_input_triple_cap": cap,
                "abox_jena_heap_mb": abox_jena_heap_mb,
                "config_files": [config_file, "configs/owl2bench/config_toy.yaml"],
                "timing_dir": str(timing_dir),
            },
            cwd=REPO_ROOT,
            artifact_paths=(str(REPO_ROOT / "vendor" / "OWL2Bench" / f"OWL2RL-{universities}.owl"),),
        )
    )


@task
def exp3_generate_baseline(
    ctx: Context,
    universities=5,
    args="",
    reasoning_input_triple_cap=0,
    abox_jena_heap_mb=8192,
    owl2bench_config_name: str = "config",
):
    """Generates Exp 3 baseline: OWL2Bench ABox generation + Jena BFS materialisation."""
    run_dir = _make_run_archive("exp3", "generate_baseline", label=str(universities))
    config_file = f"configs/owl2bench/{owl2bench_config_name}.yaml"
    _snapshot_configs(
        run_dir,
        [
            config_file,
            "configs/owl2bench/config_toy.yaml",
        ],
    )

    reasoning_cap = int(reasoning_input_triple_cap)
    exp3_generate_owl2bench_abox(
        ctx,
        universities=universities,
        args=args,
        archive_dir=str(run_dir / "abox_generation"),
        reasoning_input_triple_cap=reasoning_cap,
        abox_jena_heap_mb=int(abox_jena_heap_mb),
        config_name=owl2bench_config_name,
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp3",
            "task": "generate_baseline",
            "universities": universities,
            "args": args,
            "reasoning_input_triple_cap": reasoning_cap,
            "abox_jena_heap_mb": int(abox_jena_heap_mb),
            "owl2bench_config_name": owl2bench_config_name,
        },
    )
    generated_dir = REPO_ROOT / "data" / "owl2bench" / "output" / "owl2bench"
    if generated_dir.exists():
        _archive_path(generated_dir, run_dir / "artifacts")


@task
def exp3_generate_synthology(ctx: Context, universities=5, args="", config_name="exp3_synthology"):
    """Generates Exp 3 Synthology backward-chaining dataset on the OWL2Bench TBox."""
    print(f"\nGenerating Exp 3 synthology dataset (universities={universities}, config={config_name}).")
    run_dir = _make_run_archive("exp3", "generate_synthology", label=str(universities))
    config_file = f"configs/ont_generator/{config_name}.yaml"
    _snapshot_configs(
        run_dir,
        [
            config_file,
            "configs/ont_generator/config.yaml",
        ],
    )

    output_dir = "data/exp3/synthology/owl2bench"
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        config_name=config_name,
        overrides=(f"dataset.output_dir={output_dir}",),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp3",
            task_name="generate_synthology",
            label=str(universities),
            command=cmd,
            config_paths=(config_file, "configs/ont_generator/config.yaml"),
            manifest={
                "universities": universities,
                "args": args,
                "config_name": config_name,
                "config_files": [config_file, "configs/ont_generator/config.yaml"],
                "output_dir": output_dir,
            },
            artifact_paths=(str(REPO_ROOT / "data" / "exp3" / "synthology" / "owl2bench"),),
        )
    )


@task
def exp3_balance_data(
    ctx: Context,
    universities=5,
    baseline_dir="",
    synthology_dir="",
    output_dir="",
    seed=23,
):
    """Balances Exp 3 datasets: keeps ALL Synthology data, downsamples baseline to match.

    Synthology positives (the hard-to-generate inferred graphs) are fully preserved.
    Baseline is downsampled so both sides contribute identical label counts.
    If the baseline lacks enough negatives to cover Synthology's, Synthology negatives
    are also capped — the stderr waste report will flag this so the university count or
    inferred_target_limit can be adjusted for the next run.
    Balanced baseline is written to data/exp3/balanced_baseline/... and balanced
    Synthology to data/exp3/balanced/...
    """
    print(f"\nBalancing Exp 3 datasets (universities={universities}): keeping all Synthology, downsampling baseline.")
    run_dir = _make_run_archive("exp3", "balance_data", label=str(universities))

    baseline_root = (
        Path(baseline_dir)
        if baseline_dir
        else REPO_ROOT / "data" / "owl2bench" / "output" / "owl2bench"
    )
    synthology_root = (
        Path(synthology_dir)
        if synthology_dir
        else REPO_ROOT / "data" / "exp3" / "synthology" / "owl2bench"
    )
    output_root = (
        Path(output_dir) if output_dir else REPO_ROOT / "data" / "exp3" / "balanced" / "owl2bench"
    )
    baseline_output_root = REPO_ROOT / "data" / "exp3" / "balanced_baseline" / "owl2bench"

    if not baseline_root.exists():
        raise FileNotFoundError(f"Baseline dataset directory not found: {baseline_root}")
    if not synthology_root.exists():
        raise FileNotFoundError(f"Synthology dataset directory not found: {synthology_root}")

    import random

    rng = random.Random(int(seed))
    summary: Dict[str, Any] = {
        "baseline_dir": str(baseline_root),
        "synthology_dir": str(synthology_root),
        "output_dir": str(output_root),
        "seed": int(seed),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        baseline_split = baseline_root / split
        synthology_split = synthology_root / split
        out_split = output_root / split

        baseline_targets = baseline_split / "targets.csv"
        synthology_targets = synthology_split / "targets.csv"
        synthology_facts = synthology_split / "facts.csv"

        if not baseline_targets.exists() or not synthology_targets.exists():
            continue

        baseline_rows = _read_csv_rows(baseline_targets)
        synth_rows = _read_csv_rows(synthology_targets)

        baseline_pos = [row for row in baseline_rows if not _is_negative_row(row)]
        baseline_neg = [row for row in baseline_rows if _is_negative_row(row)]
        synth_pos = [row for row in synth_rows if not _is_negative_row(row)]
        synth_neg = [row for row in synth_rows if _is_negative_row(row)]

        def _random_sample(pool: list, target_n: int) -> list:
            if target_n >= len(pool):
                return list(pool)
            return rng.sample(pool, target_n)

        # Keep ALL Synthology positives (the precious backward-chaining graphs).
        # Downsample baseline pos to match Synthology's pos count.
        # Negatives: baseline is the floor; if baseline has fewer negatives than Synthology,
        # Synthology negatives are also capped (unavoidable — raise inferred_target_limit to fix).
        target_pos = min(len(baseline_pos), len(synth_pos))
        target_neg = min(len(baseline_neg), len(synth_neg))

        # Synthology: keep all positives; cap negatives only if baseline can't cover them.
        selected_synth_pos = synth_pos
        selected_synth_neg = _random_sample(synth_neg, target_neg)
        selected_synth_rows = selected_synth_pos + selected_synth_neg
        rng.shuffle(selected_synth_rows)

        out_split.mkdir(parents=True, exist_ok=True)
        if synthology_facts.exists():
            shutil.copy2(synthology_facts, out_split / "facts.csv")
        synth_fieldnames = list(synth_rows[0].keys()) if synth_rows else None
        _write_csv_rows(out_split / "targets.csv", selected_synth_rows, fieldnames=synth_fieldnames)

        # Baseline: downsample to match Synthology counts.
        selected_baseline_pos = _random_sample(baseline_pos, target_pos)
        selected_baseline_neg = _random_sample(baseline_neg, target_neg)
        selected_baseline_rows = selected_baseline_pos + selected_baseline_neg
        rng.shuffle(selected_baseline_rows)

        baseline_out_split = baseline_output_root / split
        baseline_facts = baseline_split / "facts.csv"
        baseline_out_split.mkdir(parents=True, exist_ok=True)
        if baseline_facts.exists():
            shutil.copy2(baseline_facts, baseline_out_split / "facts.csv")
        baseline_fieldnames = list(baseline_rows[0].keys()) if baseline_rows else None
        _write_csv_rows(baseline_out_split / "targets.csv", selected_baseline_rows, fieldnames=baseline_fieldnames)

        synth_pos_discarded = len(synth_pos) - len(selected_synth_pos)
        synth_neg_discarded = len(synth_neg) - len(selected_synth_neg)
        bl_pos_discarded = len(baseline_pos) - len(selected_baseline_pos)
        bl_neg_discarded = len(baseline_neg) - len(selected_baseline_neg)

        def _pct(n: int, total: int) -> str:
            return f"{n / total:.1%}" if total else "n/a"

        import sys
        print(
            f"[balance/{split}] Synthology: kept {len(selected_synth_pos):,} pos "
            f"(discarded {synth_pos_discarded:,} = {_pct(synth_pos_discarded, len(synth_pos))}), "
            f"kept {len(selected_synth_neg):,} neg "
            f"(discarded {synth_neg_discarded:,} = {_pct(synth_neg_discarded, len(synth_neg))})",
            file=sys.stderr,
        )
        print(
            f"[balance/{split}] Baseline:   kept {len(selected_baseline_pos):,} pos "
            f"(discarded {bl_pos_discarded:,} = {_pct(bl_pos_discarded, len(baseline_pos))}), "
            f"kept {len(selected_baseline_neg):,} neg "
            f"(discarded {bl_neg_discarded:,} = {_pct(bl_neg_discarded, len(baseline_neg))})",
            file=sys.stderr,
        )
        if synth_neg_discarded > 0:
            print(
                f"[balance/{split}] WARNING: Synthology negatives were capped because baseline only has "
                f"{len(baseline_neg):,} neg vs Synthology's {len(synth_neg):,}. "
                "Raise inferred_target_limit in configs/owl2bench/config.yaml to fix.",
                file=sys.stderr,
            )

        summary["splits"][split] = {
            "baseline_positive": len(baseline_pos),
            "baseline_negative": len(baseline_neg),
            "synthology_positive": len(synth_pos),
            "synthology_negative": len(synth_neg),
            "selected_positive": len(selected_synth_pos),
            "selected_negative": len(selected_synth_neg),
            "selected_total": len(selected_synth_rows),
            "synth_positive_discarded": synth_pos_discarded,
            "synth_negative_discarded": synth_neg_discarded,
            "balanced_baseline_positive": len(selected_baseline_pos),
            "balanced_baseline_negative": len(selected_baseline_neg),
            "baseline_positive_discarded": bl_pos_discarded,
            "baseline_negative_discarded": bl_neg_discarded,
        }

    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp3",
            "task": "balance_data",
            "universities": universities,
            "summary": summary,
        },
    )
    _write_text(run_dir / "run.log", json.dumps(summary, indent=2) + "\n")
    _archive_path(output_root, run_dir / "artifacts")
    _archive_path(baseline_output_root, run_dir / "artifacts_baseline")


@task
def exp3_generate_gold_test(
    ctx: Context,
    universities=5,
    source_test_dir="",
    output_test_dir="",
):
    """Freezes an Exp 3 test split for paper evaluation reproducibility."""
    print(f"\nFreezing Exp 3 gold test split (universities={universities}).")
    run_dir = _make_run_archive("exp3", "generate_gold_test", label=str(universities))

    source_dir = (
        Path(source_test_dir)
        if source_test_dir
        else REPO_ROOT / "data" / "exp3" / "balanced" / "owl2bench" / "test"
    )
    if not source_dir.exists():
        source_dir = REPO_ROOT / "data" / "exp3" / "synthology" / "owl2bench" / "test"
    if not source_dir.exists():
        source_dir = REPO_ROOT / "data" / "owl2bench" / "output" / "owl2bench" / "test"

    output_dir_path = (
        Path(output_test_dir)
        if output_test_dir
        else REPO_ROOT / "data" / "exp3" / "frozen_test" / "owl2bench" / "test"
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)

    facts_src = source_dir / "facts.csv"
    targets_src = source_dir / "targets.csv"
    if not facts_src.exists() or not targets_src.exists():
        raise FileNotFoundError(f"Expected facts.csv and targets.csv in source test dir: {source_dir}")

    shutil.copy2(facts_src, output_dir_path / "facts.csv")
    shutil.copy2(targets_src, output_dir_path / "targets.csv")

    summary = {
        "source_test_dir": str(source_dir),
        "output_test_dir": str(output_dir_path),
        "facts_rows": _count_csv_rows(output_dir_path / "facts.csv"),
        "targets_rows": _count_csv_rows(output_dir_path / "targets.csv"),
    }

    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "exp3",
            "task": "generate_gold_test",
            "universities": universities,
            "summary": summary,
        },
    )
    _write_text(run_dir / "run.log", json.dumps(summary, indent=2) + "\n")
    _archive_path(output_dir_path.parent, run_dir / "artifacts")


@task
def exp3_train_rrn(ctx: Context, dataset="baseline", universities=5, args=""):
    """Trains RRN for Exp 3 on baseline or synthology-aligned OWL2Bench splits."""
    dataset_key = dataset.strip().lower()
    if dataset_key not in {"baseline", "synthology"}:
        raise ValueError("dataset must be either 'baseline' or 'synthology'")

    baseline_root = REPO_ROOT / "data" / "owl2bench" / "output" / "owl2bench"

    if dataset_key == "baseline":
        dataset_root = baseline_root
    else:
        balanced_root = REPO_ROOT / "data" / "exp3" / "balanced" / "owl2bench"
        synth_root = REPO_ROOT / "data" / "exp3" / "synthology" / "owl2bench"
        if balanced_root.exists():
            dataset_root = balanced_root
        elif synth_root.exists():
            dataset_root = synth_root
        else:
            raise FileNotFoundError(
                "Synthology dataset not found. Run exp3-generate-synthology first or provide balanced data via exp3-balance-data."
            )

    train_path = dataset_root / "train"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train directory for Exp3 RRN training: {train_path}")

    # Synthology only has a train split; fall back to baseline val/test for evaluation.
    val_path = dataset_root / "val"
    test_path = dataset_root / "test"
    if dataset_key == "synthology":
        if not val_path.exists():
            val_path = baseline_root / "val"
            print(f"[exp3-train-rrn] Synthology val/ not found — using baseline val: {val_path}")
        if not test_path.exists():
            test_path = baseline_root / "test"
            print(f"[exp3-train-rrn] Synthology test/ not found — using baseline test: {test_path}")

    for label, p in [("val", val_path), ("test", test_path)]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} directory for Exp3 RRN training: {p}")

    def _mtime3(p: Path) -> str:
        facts = p / "facts.csv"
        if facts.exists():
            import datetime
            return datetime.datetime.fromtimestamp(facts.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return "missing"

    print(f"\nTraining Exp 3 RRN on: {dataset_key} (universities={universities})")
    print(f"  train : {train_path}  [{_mtime3(train_path)}]")
    print(f"  val   : {val_path}  [{_mtime3(val_path)}]")
    print(f"  test  : {test_path}  [{_mtime3(test_path)}]")

    run_dir = _make_run_archive("exp3", "train_rrn", label=dataset_key)

    override_args = " ".join(
        [
            f"data.dataset.train_path={train_path}",
            f"data.dataset.val_path={val_path}",
            f"data.dataset.test_path={test_path}",
            f"logger.name=exp3_{dataset_key}_owl2bench_u{universities}",
            "logger.group=exp3_scaling",
        ]
    )
    combined_args = " ".join(part for part in (override_args, args) if part).strip()

    cmd = _build_uv_command(
        "rrn",
        "rrn.train",
        config_name="exp3_owl2bench_hpc",
        args=combined_args,
        env={"PYTHONUNBUFFERED": "1", "LOGURU_COLORIZE": "1"},
    )
    _run_experiment_spec(
        ExperimentRunSpec(
            experiment="exp3",
            task_name="train_rrn",
            label=dataset_key,
            command=cmd,
            config_paths=("configs/rrn/config.yaml", "configs/rrn/exp3_owl2bench_hpc.yaml"),
            manifest={
                "dataset": dataset_key,
                "universities": universities,
                "dataset_root": str(dataset_root),
                "train_path": str(train_path),
                "val_path": str(val_path),
                "test_path": str(test_path),
                "train_facts_mtime": _mtime3(train_path),
                "args": args,
                "effective_args": combined_args,
                "config_files": ["configs/rrn/config.yaml", "configs/rrn/exp3_owl2bench_hpc.yaml"],
            },
        )
    )


@task
def exp3_generate_baseline_hpc(ctx: Context, config_path="configs/experiments/exp3_hpc.yaml"):
    """Runs Exp3 baseline generation using centralized YAML preset."""
    cfg = _load_yaml_config(config_path)
    baseline = dict(cfg.get("baseline", {}))
    exp3_generate_baseline(
        ctx,
        universities=int(cfg["universities"]),
        args=str(baseline.get("args", "")),
        reasoning_input_triple_cap=int(baseline.get("reasoning_input_triple_cap", 1200)),
        abox_jena_heap_mb=int(baseline.get("abox_jena_heap_mb", 8192)),
        owl2bench_config_name=str(baseline.get("owl2bench_config_name", "config")),
    )


@task
def exp3_generate_synthology_hpc(ctx: Context, config_path="configs/experiments/exp3_hpc.yaml"):
    """Runs Exp3 synthology generation using centralized YAML preset."""
    cfg = _load_yaml_config(config_path)
    synthology = dict(cfg.get("synthology", {}))
    exp3_generate_synthology(
        ctx,
        universities=int(cfg["universities"]),
        args=str(synthology.get("args", "")),
        config_name=str(synthology.get("config_name", "exp3_synthology")),
    )


@task
def exp3_balance_data_hpc(ctx: Context, config_path="configs/experiments/exp3_hpc.yaml"):
    """Runs Exp3 balance-data step using centralized YAML preset."""
    cfg = _load_yaml_config(config_path)
    balance = dict(cfg.get("balance", {}))
    exp3_balance_data(
        ctx,
        universities=int(cfg["universities"]),
        baseline_dir=str(balance.get("baseline_dir", "")),
        synthology_dir=str(balance.get("synthology_dir", "")),
        output_dir=str(balance.get("output_dir", "")),
        seed=int(balance.get("seed", 23)),
    )


@task
def exp3_generate_gold_test_hpc(ctx: Context, config_path="configs/experiments/exp3_hpc.yaml"):
    """Runs Exp3 gold-test freeze step using centralized YAML preset."""
    cfg = _load_yaml_config(config_path)
    gold_test = dict(cfg.get("gold_test", {}))
    exp3_generate_gold_test(
        ctx,
        universities=int(cfg["universities"]),
        source_test_dir=str(gold_test.get("source_test_dir", "")),
        output_test_dir=str(gold_test.get("output_test_dir", "")),
    )


@task
def exp3_train_rrn_hpc(ctx: Context, dataset="baseline", config_path="configs/experiments/exp3_hpc.yaml"):
    """Trains Exp3 RRN for one arm (baseline or synthology) using centralized YAML preset."""
    cfg = _load_yaml_config(config_path)
    training = dict(cfg.get("training", {}))
    exp3_train_rrn(
        ctx,
        dataset=dataset,
        universities=int(cfg["universities"]),
        args=str(training.get("args", "")),
    )


@task
def exp3_report_and_analyze_hpc(ctx: Context, config_path="configs/experiments/exp3_hpc.yaml"):
    """Runs Exp3 comparison report and baseline analysis using centralized YAML preset."""
    cfg = _load_yaml_config(config_path)
    universities = int(cfg["universities"])
    report_cfg = dict(cfg.get("report", {}))

    baseline_path = str(report_cfg.get("baseline_path", "")).strip()
    if not baseline_path:
        _balanced_bl = "data/exp3/balanced_baseline/owl2bench"
        baseline_path = _balanced_bl if Path(_balanced_bl).exists() else "data/owl2bench/output/owl2bench"

    synthology_path = str(report_cfg.get("synthology_path", "")).strip()
    if not synthology_path:
        use_balanced_synthology = bool(report_cfg.get("use_balanced_synthology", True))
        if use_balanced_synthology:
            synthology_path = "data/exp3/balanced/owl2bench"
        else:
            synthology_path = "data/exp3/synthology/owl2bench"

    exp3_report_data(
        ctx,
        universities=universities,
        baseline_path=baseline_path,
        synthology_path=synthology_path,
        args=str(report_cfg.get("args", "")),
    )

    if bool(report_cfg.get("run_baseline_analysis", True)):
        exp3_analyze_latest_baseline(ctx)

@task
def exp3_materialize_abox(
    ctx: Context,
    abox,
    tbox="ontologies/UNIV-BENCH-OWL2RL.owl",
    closure_out="outputs/exp3/closure.nt",
    inferred_out="outputs/exp3/inferred.nt",
    jena_profile="owl_mini",
    reasoning_input_triple_cap=0,
    args="",
    archive_dir: Optional[str] = None,
):
    """Materializes an OWL2Bench ABox with UDM/Jena and exports closure + inferred triples."""
    print("\nRunning Exp 3 ABox materialization with UDM/Jena.")
    run_dir = Path(archive_dir) if archive_dir else _make_run_archive("exp3", "materialize_abox", label=jena_profile)
    closure_archive = run_dir / "artifacts" / "closure.nt"
    inferred_archive = run_dir / "artifacts" / "inferred.nt"
    cap = int(reasoning_input_triple_cap)
    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.materialize",
        overrides=(
            f"--tbox {tbox}",
            f"--abox {abox}",
            f"--closure-out {shlex.quote(str(closure_archive))}",
            f"--inferred-out {shlex.quote(str(inferred_archive))}",
            f"--jena-profile {jena_profile}",
            f"--reasoning-input-triple-cap {cap}",
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
                "reasoning_input_triple_cap": cap,
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
    _apply_runtime_storage_overrides(environment, cwd or REPO_ROOT)
    _apply_global_hpc_heap_overrides(environment)
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


def _apply_runtime_storage_overrides(environment: Dict[str, str], base_dir: Path) -> None:
    runtime_root = Path(base_dir).resolve() / ".cache" / "runtime"
    path_map = {
        "TMPDIR": runtime_root / "tmp",
        "XDG_CACHE_HOME": runtime_root / "xdg-cache",
        "XDG_CONFIG_HOME": runtime_root / "xdg-config",
        "XDG_DATA_HOME": runtime_root / "xdg-data",
        "XDG_STATE_HOME": runtime_root / "xdg-state",
        "WANDB_DIR": runtime_root / "wandb",
        "WANDB_CACHE_DIR": runtime_root / "wandb-cache",
        "WANDB_ARTIFACT_DIR": runtime_root / "wandb-artifacts",
    }

    for path in path_map.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    for key, value in path_map.items():
        environment.setdefault(key, str(value))


def _apply_global_hpc_heap_overrides(environment: Dict[str, str]) -> None:
    """Apply one-shot JVM heap policy across experiment commands on HPC.

    Priority:
    1) SYNTHOLOGY_JAVA_XMX_MB
    2) SYNTHOLOGY_HEAP_MB
    """

    heap_mb = os.environ.get("SYNTHOLOGY_JAVA_XMX_MB") or os.environ.get("SYNTHOLOGY_HEAP_MB")
    if not heap_mb:
        return

    heap_mb = str(heap_mb).strip()
    if not heap_mb.isdigit() or int(heap_mb) <= 0:
        return

    # UDM baseline Jena helper reads these variables directly.
    environment.setdefault("SYNTHOLOGY_UDM_BASELINE_XMX_MB", heap_mb)
    environment.setdefault("SYNTHOLOGY_JENA_XMX_MB", heap_mb)

    # Maven-backed Java runs (e.g. OWL2Bench generator) should use the same heap cap.
    current_maven_opts = environment.get("MAVEN_OPTS", "")
    environment["MAVEN_OPTS"] = _upsert_xmx_option(current_maven_opts, heap_mb)


def _upsert_xmx_option(raw_options: str, heap_mb: str) -> str:
    """Ensure exactly one -Xmx option is present in a JVM options string."""

    text = str(raw_options or "").strip()
    try:
        tokens = shlex.split(text) if text else []
    except ValueError:
        tokens = text.split() if text else []

    filtered = [token for token in tokens if not token.startswith("-Xmx")]
    filtered.append(f"-Xmx{heap_mb}m")
    return " ".join(filtered).strip()


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


def _load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = (REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    loaded = OmegaConf.load(str(config_path))
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at config root: {config_path}")
    return data


def _load_json_defaults(path: str) -> Dict[str, Any]:
    resolved = (REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _resolve_default(value: Any, defaults: Dict[str, Any], key: str, fallback: Any) -> Any:
    if value is not None:
        return value
    if key in defaults:
        return defaults[key]
    return fallback


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _count_unique_sample_ids(path: Path) -> int:
    if not path.exists():
        return -1
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row.get("sample_id", "")
            if sample_id:
                sample_ids.add(sample_id)
    return len(sample_ids)


def _ordered_sample_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    sample_ids: List[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row.get("sample_id", "")
            if sample_id and sample_id not in seen:
                seen.add(sample_id)
                sample_ids.append(sample_id)
    return sample_ids


def _trim_split_to_sample_ids(split_root: Path, keep_sample_ids: set[str]) -> None:
    for filename in ("facts.csv", "targets.csv"):
        path = split_root / filename
        if not path.exists():
            continue
        rows = _read_csv_rows(path)
        fieldnames = list(rows[0].keys()) if rows else None
        filtered_rows = [row for row in rows if row.get("sample_id", "") in keep_sample_ids]
        _write_csv_rows(path, filtered_rows, fieldnames=fieldnames)


def _align_exp2_split_sample_counts(
    baseline_root: Path, synthology_root: Path, splits: Tuple[str, ...]
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split in splits:
        baseline_targets = baseline_root / split / "targets.csv"
        synthology_targets = synthology_root / split / "targets.csv"

        baseline_ids = _ordered_sample_ids(baseline_targets)
        synthology_ids = _ordered_sample_ids(synthology_targets)
        baseline_count = len(baseline_ids)
        synthology_count = len(synthology_ids)

        if baseline_count == 0 or synthology_count == 0:
            summary[split] = {
                "baseline_before": baseline_count,
                "synthology_before": synthology_count,
                "baseline_after": baseline_count,
                "synthology_after": synthology_count,
                "aligned_to": 0,
            }
            continue

        aligned_to = min(baseline_count, synthology_count)
        baseline_keep_ids = set(baseline_ids[:aligned_to])
        synthology_keep_ids = set(synthology_ids[:aligned_to])

        if baseline_count != aligned_to:
            _trim_split_to_sample_ids(baseline_root / split, baseline_keep_ids)
        if synthology_count != aligned_to:
            _trim_split_to_sample_ids(synthology_root / split, synthology_keep_ids)

        summary[split] = {
            "baseline_before": baseline_count,
            "synthology_before": synthology_count,
            "baseline_after": aligned_to,
            "synthology_after": aligned_to,
            "aligned_to": aligned_to,
        }

    return summary


def _get_split_sample_count_mismatches(
    baseline_root: Path, synthology_root: Path, splits: Tuple[str, ...]
) -> List[Tuple[str, int, int]]:
    mismatches: List[Tuple[str, int, int]] = []
    for split in splits:
        baseline_targets = baseline_root / split / "targets.csv"
        synthology_targets = synthology_root / split / "targets.csv"
        baseline_count = _count_unique_sample_ids(baseline_targets)
        synthology_count = _count_unique_sample_ids(synthology_targets)
        if baseline_count != synthology_count:
            mismatches.append((split, baseline_count, synthology_count))
    return mismatches


def _count_targets_by_label(path: Path) -> Tuple[int, int]:
    rows = _read_csv_rows(path)
    positives = sum(1 for row in rows if not _is_negative_row(row))
    negatives = sum(1 for row in rows if _is_negative_row(row))
    return positives, negatives


def _get_split_target_label_mismatches(
    baseline_root: Path, synthology_root: Path, splits: Tuple[str, ...]
) -> List[Tuple[str, int, int, int, int]]:
    mismatches: List[Tuple[str, int, int, int, int]] = []
    for split in splits:
        baseline_targets = baseline_root / split / "targets.csv"
        synthology_targets = synthology_root / split / "targets.csv"
        if not baseline_targets.exists() or not synthology_targets.exists():
            continue
        baseline_pos, baseline_neg = _count_targets_by_label(baseline_targets)
        synthology_pos, synthology_neg = _count_targets_by_label(synthology_targets)
        if baseline_pos != synthology_pos or baseline_neg != synthology_neg:
            mismatches.append((split, baseline_pos, baseline_neg, synthology_pos, synthology_neg))
    return mismatches


def _trim_targets_to_label_budget(path: Path, target_pos: int, target_neg: int) -> Dict[str, int]:
    rows = _read_csv_rows(path)
    fieldnames = list(rows[0].keys()) if rows else None

    def _select_with_coverage(source_rows: List[Dict[str, str]], target_count: int, seed: int) -> List[Dict[str, str]]:
        if target_count <= 0 or not source_rows:
            return []
        if target_count >= len(source_rows):
            return list(source_rows)

        rng = random.Random(seed)

        # 1) Preserve target-type diversity first (e.g., inf_root vs inf_intermediate).
        rows_by_type: Dict[str, List[Dict[str, str]]] = {}
        for row in source_rows:
            type_key = str(row.get("type", ""))
            rows_by_type.setdefault(type_key, []).append(row)

        type_keys = list(rows_by_type.keys())
        rng.shuffle(type_keys)

        selected: List[Dict[str, str]] = []
        selected_ids: set[int] = set()

        for type_key in type_keys:
            if len(selected) >= target_count:
                break
            group = rows_by_type[type_key]
            pick = rng.choice(group)
            selected.append(pick)
            selected_ids.add(id(pick))

        if len(selected) >= target_count:
            return selected[:target_count]

        # 2) Preserve predicate coverage as much as possible.
        rows_by_predicate: Dict[str, List[Dict[str, str]]] = {}
        for row in source_rows:
            predicate = str(row.get("predicate", ""))
            rows_by_predicate.setdefault(predicate, []).append(row)

        predicate_keys = list(rows_by_predicate.keys())
        rng.shuffle(predicate_keys)

        for predicate in predicate_keys:
            if len(selected) >= target_count:
                break
            candidates = [row for row in rows_by_predicate[predicate] if id(row) not in selected_ids]
            if not candidates:
                continue
            pick = rng.choice(candidates)
            selected.append(pick)
            selected_ids.add(id(pick))

        if len(selected) >= target_count:
            return selected[:target_count]

        # 3) Fill remaining quota uniformly from leftovers.
        leftovers = [row for row in source_rows if id(row) not in selected_ids]
        need = target_count - len(selected)
        if need > 0 and leftovers:
            if need >= len(leftovers):
                selected.extend(leftovers)
            else:
                selected.extend(rng.sample(leftovers, need))

        return selected[:target_count]

    positive_rows = [row for row in rows if not _is_negative_row(row)]
    negative_rows = [row for row in rows if _is_negative_row(row)]

    kept_positive = _select_with_coverage(positive_rows, target_pos, seed=23)
    kept_negative = _select_with_coverage(negative_rows, target_neg, seed=31)
    kept_rows = kept_positive + kept_negative

    _write_csv_rows(path, kept_rows, fieldnames=fieldnames)
    return {
        "kept_pos": len(kept_positive),
        "kept_neg": len(kept_negative),
        "kept_samples": len({row.get("sample_id", "") for row in kept_rows if row.get("sample_id", "")}),
    }


def _align_exp2_split_target_labels(
    baseline_root: Path, synthology_root: Path, splits: Tuple[str, ...]
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split in splits:
        baseline_targets = baseline_root / split / "targets.csv"
        synthology_targets = synthology_root / split / "targets.csv"

        if not baseline_targets.exists() or not synthology_targets.exists():
            continue

        baseline_pos_before, baseline_neg_before = _count_targets_by_label(baseline_targets)
        synthology_pos_before, synthology_neg_before = _count_targets_by_label(synthology_targets)

        target_pos = min(baseline_pos_before, synthology_pos_before)
        target_neg = min(baseline_neg_before, synthology_neg_before)

        baseline_trim = _trim_targets_to_label_budget(baseline_targets, target_pos=target_pos, target_neg=target_neg)
        synthology_trim = _trim_targets_to_label_budget(synthology_targets, target_pos=target_pos, target_neg=target_neg)

        summary[split] = {
            "baseline_pos_before": baseline_pos_before,
            "baseline_neg_before": baseline_neg_before,
            "synthology_pos_before": synthology_pos_before,
            "synthology_neg_before": synthology_neg_before,
            "target_pos": target_pos,
            "target_neg": target_neg,
            "baseline_samples_after": baseline_trim["kept_samples"],
            "synthology_samples_after": synthology_trim["kept_samples"],
        }

    return summary


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = list(fieldnames or (list(rows[0].keys()) if rows else []))
    if not resolved_fieldnames:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


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


def _resolve_owl2bench_env(run_dir: Path) -> Dict[str, str]:
    """Return env vars for OWL2Bench tasks, ensuring Maven is available."""
    env: Dict[str, str] = {"LOGURU_COLORIZE": "1"}

    # Keep Maven artifacts out of small home quotas on HPC nodes.
    maven_repo_local = REPO_ROOT / ".cache" / "m2"
    maven_repo_local.mkdir(parents=True, exist_ok=True)
    existing_opts = os.environ.get("MAVEN_OPTS", "").strip()
    if "maven.repo.local=" in existing_opts:
        env["MAVEN_OPTS"] = existing_opts
    else:
        env["MAVEN_OPTS"] = f"{existing_opts} -Dmaven.repo.local={maven_repo_local}".strip()

    if shutil.which("mvn"):
        return env

    local_mvn = REPO_ROOT / "apache-maven-3.9.13" / "bin" / "mvn"

    if not local_mvn.exists():
        setup_cmd = "bash ./install-mvn.sh"
        _run_logged_command(setup_cmd, run_dir / "maven-setup.log")

    if local_mvn.exists():
        env["MAVEN_EXECUTABLE"] = str(local_mvn)
        return env

    raise RuntimeError("Maven executable could not be resolved. Install Maven or load a module so 'mvn' is available.")

@task
def gen_owl2bench(ctx: Context, args=""):
    """
    Runs the OWL2Bench OWL 2 RL pipeline:
    ABox generation -> Apache Jena materialization -> CSV export.
    """
    print("\nRunning OWL2Bench OWL 2 RL generation pipeline.")
    run_dir = _make_run_archive("owl2bench", "generate", label="default")
    owl2bench_env = _resolve_owl2bench_env(run_dir)
    cmd = _build_uv_command(
        "owl2bench",
        "owl2bench.pipeline",
        env=owl2bench_env,
        args=args,
    )
    _write_json(run_dir / "manifest.json", {"task": "generate", "args": args, "command": cmd})
    _run_logged_command(cmd, run_dir / "run.log")


@task
def udm_visual_verification(ctx: Context, n_samples=3, args=""):
    """
    Generates UDM baseline samples and renders comparable PDF graph visuals
    for side-by-side inspection against synthology-visual-verification outputs.

    Output is written to visual-verification/udm_baseline/.
    """

    print("\nRunning UDM visual verification generator.")
    run_dir = _make_run_archive("udm_baseline", "visual_verification", label="inspection")

    output_root = Path("visual-verification") / "udm_baseline"
    dataset_output_dir = output_root
    # Match Synthology visual output location while keeping UDM filenames explicit.
    graphs_output_dir = Path("visual-verification") / "graphs"

    try:
        n_samples_int = max(1, int(n_samples))
    except (TypeError, ValueError):
        raise ValueError(f"n_samples must be an integer >= 1, got: {n_samples}")

    n_train_for_render = max(1, n_samples_int)

    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.create_data",
        overrides=(
            f"dataset.n_train={n_train_for_render}",
            "dataset.n_val=0",
            "dataset.n_test=0",
            f"dataset.output_dir={dataset_output_dir.as_posix()}",
            "generator.min_individuals=8",
            "generator.max_individuals=18",
            "generator.min_base_relations=8",
            "generator.max_base_relations=24",
            "neg_sampling.ratio=0.5",
            "materialization.reasoner=jena",
            "materialization.jena_profile=owl_mini",
            "materialization.iterative=false",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_logged_command(cmd, run_dir / "run.log")

    targets_csv = dataset_output_dir / "train" / "targets.csv"
    facts_csv = dataset_output_dir / "train" / "facts.csv"

    if not targets_csv.exists():
        raise RuntimeError(f"Expected targets CSV not found at {targets_csv}")
    if not facts_csv.exists():
        raise RuntimeError(f"Expected facts CSV not found at {facts_csv}")

    per_sample: Dict[str, Dict[str, Any]] = {}
    with targets_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
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

    def _numeric_sample_key(sample_id: str) -> int:
        try:
            return int(sample_id)
        except ValueError:
            return -1

    ranked_samples = sorted(
        per_sample.values(),
        key=lambda s: (
            min(int(s["base"]), int(s["inferred"])),
            int(s["base"]),
            int(s["inferred"]),
            int(s["total"]),
            int(s["negative"]),
            _numeric_sample_key(str(s["sample_id"])),
        ),
        reverse=True,
    )

    if not ranked_samples:
        raise RuntimeError(f"No sample_id values found in {targets_csv}")

    selected_sample_ids = [str(stats["sample_id"]) for stats in ranked_samples[:n_samples_int]]
    graphs_output_dir.mkdir(parents=True, exist_ok=True)

    visualization_commands = []
    for sample_id in selected_sample_ids:
        output_stem = f"udm_baseline_sample_{sample_id}"
        viz_cmd = _build_uv_command(
            "kgvisualiser",
            "kgvisualiser.visualize",
            overrides=(
                f"io.input_csv={facts_csv.as_posix()}",
                f"io.targets_csv={targets_csv.as_posix()}",
                f"io.sample_id={sample_id}",
                f"output.dir={graphs_output_dir.as_posix()}",
                f"output.name_template={output_stem}",
                "output.format=pdf",
                "filters.include_negatives=false",
                "filters.max_edges=75",
                "render.engine=dot",
                "render.overlap=false",
                "render.splines=curved",
                "render.class_nodes=false",
                "render.show_edge_labels=true",
            ),
            env={"LOGURU_COLORIZE": "1"},
        )
        _run_logged_command(viz_cmd, run_dir / f"visualize_{sample_id}.log")

        expected_pdf = graphs_output_dir / f"{output_stem}.pdf"
        if not expected_pdf.exists():
            source_graph = graphs_output_dir / output_stem
            if source_graph.exists():
                fallback_cmd = f"dot -Tpdf {shlex.quote(str(source_graph))} -o {shlex.quote(str(expected_pdf))}"
                _run_logged_command(fallback_cmd, run_dir / f"render_fallback_{sample_id}.log")

            if not expected_pdf.exists():
                raise RuntimeError(
                    f"Expected PDF not generated for sample {sample_id}: {expected_pdf}. "
                    f"Ensure Graphviz is installed and available on PATH."
                )

        visualization_commands.append(viz_cmd)

    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "udm_baseline",
            "task": "visual_verification",
            "args": args,
            "n_samples": n_samples_int,
            "selected_sample_ids": selected_sample_ids,
            "generation_command": cmd,
            "visualization_commands": visualization_commands,
            "outputs": {
                "dataset_output_dir": dataset_output_dir.as_posix(),
                "graphs_output_dir": graphs_output_dir.as_posix(),
            },
        },
    )

    print(
        f"Rendered {len(selected_sample_ids)} UDM visual verification PDF(s) to {graphs_output_dir.as_posix()}",
        flush=True,
    )


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
def paper_export_tables(
    ctx: Context,
    out_dir="paper/generated",
    exp2_summary="",
    exp3_summary="",
    exp2_timing_summary="",
    exp3_timing_summary="",
    model_metrics="paper/metrics/model_results.json",
    args="",
):
    """Exports LaTeX table row snippets for the paper from latest run artifacts."""
    print("\nExporting paper table rows.")
    run_dir = _make_run_archive("paper", "export_tables", label="tables")
    cmd = _build_uv_command(
        "data_reporter",
        "data_reporter.paper_tables",
        overrides=(
            f"--repo-root {shlex.quote(str(REPO_ROOT))}",
            f"--out-dir {out_dir}",
            f"--model-metrics {model_metrics}",
        ),
        env={"LOGURU_COLORIZE": "1"},
    )
    if exp2_summary:
        cmd += f" --exp2-summary {exp2_summary}"
    if exp3_summary:
        cmd += f" --exp3-summary {exp3_summary}"
    if exp2_timing_summary:
        cmd += f" --exp2-timing-summary {exp2_timing_summary}"
    if exp3_timing_summary:
        cmd += f" --exp3-timing-summary {exp3_timing_summary}"
    if args:
        cmd += f" {args}"

    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "paper",
            "task": "export_tables",
            "out_dir": out_dir,
            "exp2_summary": exp2_summary,
            "exp3_summary": exp3_summary,
            "exp2_timing_summary": exp2_timing_summary,
            "exp3_timing_summary": exp3_timing_summary,
            "model_metrics": model_metrics,
            "args": args,
            "command": cmd,
        },
    )
    _run_logged_command(cmd, run_dir / "run.log")
    _archive_path(Path(out_dir), run_dir / "artifacts")



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
def udm_visual_verification(ctx: Context, n_samples=3, args=""):
    """
    Generates UDM baseline samples and renders comparable PDF graph visuals
    for side-by-side inspection against synthology-visual-verification outputs.

    Output is written to visual-verification/udm_baseline/.
    """

    print("\nRunning UDM visual verification generator.")
    run_dir = _make_run_archive("udm_baseline", "visual_verification", label="inspection")

    output_root = Path("visual-verification") / "udm_baseline"
    dataset_output_dir = output_root
    # Match Synthology visual output location while keeping UDM filenames explicit.
    graphs_output_dir = Path("visual-verification") / "graphs"

    try:
        n_samples_int = max(1, int(n_samples))
    except (TypeError, ValueError):
        raise ValueError(f"n_samples must be an integer >= 1, got: {n_samples}")

    n_train_for_render = max(1, n_samples_int)

    cmd = _build_uv_command(
        "udm_baseline",
        "udm_baseline.create_data",
        overrides=(
            f"dataset.n_train={n_train_for_render}",
            "dataset.n_val=0",
            "dataset.n_test=0",
            f"dataset.output_dir={dataset_output_dir.as_posix()}",
            "generator.min_individuals=8",
            "generator.max_individuals=18",
            "generator.min_base_relations=8",
            "generator.max_base_relations=24",
            "neg_sampling.ratio=0.5",
            "materialization.reasoner=jena",
            "materialization.jena_profile=owl_mini",
            "materialization.iterative=false",
        ),
        args=args,
        env={"LOGURU_COLORIZE": "1"},
    )
    _run_logged_command(cmd, run_dir / "run.log")

    targets_csv = dataset_output_dir / "train" / "targets.csv"
    facts_csv = dataset_output_dir / "train" / "facts.csv"

    if not targets_csv.exists():
        raise RuntimeError(f"Expected targets CSV not found at {targets_csv}")
    if not facts_csv.exists():
        raise RuntimeError(f"Expected facts CSV not found at {facts_csv}")

    per_sample: Dict[str, Dict[str, Any]] = {}
    with targets_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
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

    def _numeric_sample_key(sample_id: str) -> int:
        try:
            return int(sample_id)
        except ValueError:
            return -1

    ranked_samples = sorted(
        per_sample.values(),
        key=lambda s: (
            min(int(s["base"]), int(s["inferred"])),
            int(s["base"]),
            int(s["inferred"]),
            int(s["total"]),
            int(s["negative"]),
            _numeric_sample_key(str(s["sample_id"])),
        ),
        reverse=True,
    )

    if not ranked_samples:
        raise RuntimeError(f"No sample_id values found in {targets_csv}")

    selected_sample_ids = [str(stats["sample_id"]) for stats in ranked_samples[:n_samples_int]]
    graphs_output_dir.mkdir(parents=True, exist_ok=True)

    visualization_commands = []
    for sample_id in selected_sample_ids:
        output_stem = f"udm_baseline_sample_{sample_id}"
        viz_cmd = _build_uv_command(
            "kgvisualiser",
            "kgvisualiser.visualize",
            overrides=(
                f"io.input_csv={facts_csv.as_posix()}",
                f"io.targets_csv={targets_csv.as_posix()}",
                f"io.sample_id={sample_id}",
                f"output.dir={graphs_output_dir.as_posix()}",
                f"output.name_template={output_stem}",
                "output.format=pdf",
                "filters.include_negatives=false",
                "filters.max_edges=75",
                "render.engine=dot",
                "render.overlap=false",
                "render.splines=curved",
                "render.class_nodes=false",
                "render.show_edge_labels=true",
            ),
            env={"LOGURU_COLORIZE": "1"},
        )
        _run_logged_command(viz_cmd, run_dir / f"visualize_{sample_id}.log")

        expected_pdf = graphs_output_dir / f"{output_stem}.pdf"
        if not expected_pdf.exists():
            source_graph = graphs_output_dir / output_stem
            if source_graph.exists():
                fallback_cmd = f"dot -Tpdf {shlex.quote(str(source_graph))} -o {shlex.quote(str(expected_pdf))}"
                _run_logged_command(fallback_cmd, run_dir / f"render_fallback_{sample_id}.log")

            if not expected_pdf.exists():
                raise RuntimeError(
                    f"Expected PDF not generated for sample {sample_id}: {expected_pdf}. "
                    f"Ensure Graphviz is installed and available on PATH."
                )

        visualization_commands.append(viz_cmd)

    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": "udm_baseline",
            "task": "visual_verification",
            "args": args,
            "n_samples": n_samples_int,
            "selected_sample_ids": selected_sample_ids,
            "generation_command": cmd,
            "visualization_commands": visualization_commands,
            "outputs": {
                "dataset_output_dir": dataset_output_dir.as_posix(),
                "graphs_output_dir": graphs_output_dir.as_posix(),
            },
        },
    )

    print(
        f"Rendered {len(selected_sample_ids)} UDM visual verification PDF(s) to {graphs_output_dir.as_posix()}",
        flush=True,
    )
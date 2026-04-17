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
def exp3_report_data(ctx: Context, universities=20, baseline_path="", synthology_path="", args=""):
    """Generates parity/distribution reports for Exp 3 baseline vs synthology datasets."""
    print("\nGenerating Exp 3 comparison report.")
    run_dir = _make_run_archive("exp3", "report_data", label="compare")
    report_dir = run_dir / "report"

    resolved_baseline_path = baseline_path or f"data/owl2bench/output/owl2bench_{universities}"
    balanced = f"data/exp3/balanced/owl2bench_{universities}"
    unbalanced = f"data/exp3/synthology/owl2bench_{universities}"
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
def exp3_generate_owl2bench_abox(
    ctx: Context,
    universities=5,
    args="",
    archive_dir: Optional[str] = None,
    reasoning_input_triple_cap=0,
    abox_jena_heap_mb=8192,
):
    """Runs the existing OWL2Bench pipeline and stores raw generated OWL (ABox source)."""
    print(f"\nGenerating OWL2Bench data for Exp 3 (universities={universities}).")
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
        # Exp3 baseline only needs a stable generated ABox artifact; capping reasoning input avoids Jena OOM.
        overrides.append(f"+dataset.reasoning_input_triple_cap={cap}")

    abox_jena_heap_mb = str(abox_jena_heap_mb)

    cmd = _build_uv_command(
        "owl2bench",
        "owl2bench.pipeline",
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
            config_paths=("configs/owl2bench/config.yaml", "configs/owl2bench/config_toy.yaml"),
            manifest={
                "universities": universities,
                "args": args,
                "reasoning_input_triple_cap": cap,
                "abox_jena_heap_mb": abox_jena_heap_mb,
                "config_files": ["configs/owl2bench/config.yaml", "configs/owl2bench/config_toy.yaml"],
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
    reasoning_input_triple_cap=1200,
    abox_jena_heap_mb=8192,
    final_reasoning_input_triple_cap=15000,
    final_jena_profile="owl_mini",
):
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

    reasoning_cap = int(reasoning_input_triple_cap)
    final_reasoning_cap = int(final_reasoning_input_triple_cap)
    final_jena_profile = str(final_jena_profile)
    exp3_generate_owl2bench_abox(
        ctx,
        universities=universities,
        args=args,
        archive_dir=str(run_dir / "abox_generation"),
        reasoning_input_triple_cap=reasoning_cap,
        abox_jena_heap_mb=int(abox_jena_heap_mb),
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
        jena_profile=final_jena_profile,
        reasoning_input_triple_cap=final_reasoning_cap,
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
            "reasoning_input_triple_cap": reasoning_cap,
            "abox_jena_heap_mb": int(abox_jena_heap_mb),
            "final_reasoning_input_triple_cap": final_reasoning_cap,
            "final_jena_profile": final_jena_profile,
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
def exp3_generate_synthology(ctx: Context, universities=5, args=""):
    """Generates Exp 3 Synthology backward-chaining dataset on the OWL2Bench TBox."""
    print(f"\nGenerating Exp 3 synthology dataset (universities={universities}).")
    run_dir = _make_run_archive("exp3", "generate_synthology", label=str(universities))
    _snapshot_configs(
        run_dir,
        [
            "configs/ont_generator/exp3_synthology.yaml",
            "configs/ont_generator/config.yaml",
        ],
    )

    output_dir = f"data/exp3/synthology/owl2bench_{universities}"
    cmd = _build_uv_command(
        "ont_generator",
        "ont_generator.create_data",
        config_name="exp3_synthology",
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
            config_paths=("configs/ont_generator/exp3_synthology.yaml", "configs/ont_generator/config.yaml"),
            manifest={
                "universities": universities,
                "args": args,
                "config_files": ["configs/ont_generator/exp3_synthology.yaml", "configs/ont_generator/config.yaml"],
                "output_dir": output_dir,
            },
            artifact_paths=(str(REPO_ROOT / "data" / "exp3" / "synthology" / f"owl2bench_{universities}"),),
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
    """Downsamples Synthology targets to match baseline per-split label counts for Exp 3."""
    print(f"\nBalancing Exp 3 Synthology targets to baseline counts (universities={universities}).")
    run_dir = _make_run_archive("exp3", "balance_data", label=str(universities))

    baseline_root = (
        Path(baseline_dir)
        if baseline_dir
        else REPO_ROOT / "data" / "owl2bench" / "output" / f"owl2bench_{universities}"
    )
    synthology_root = (
        Path(synthology_dir)
        if synthology_dir
        else REPO_ROOT / "data" / "exp3" / "synthology" / f"owl2bench_{universities}"
    )
    output_root = (
        Path(output_dir) if output_dir else REPO_ROOT / "data" / "exp3" / "balanced" / f"owl2bench_{universities}"
    )

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

        target_pos = min(len(baseline_pos), len(synth_pos))
        target_neg = min(len(baseline_neg), len(synth_neg))

        # Stratified sampling: match baseline hop-bucket distribution (d=1, d=2, d>=3)
        # so Synthology preserves the same depth profile as the baseline rather than
        # inheriting the raw generator's ~57% hop=0 skew.
        def _hop_bucket(row: dict[str, str]) -> str:
            try:
                h = int(row.get("hops", "0") or "0")
            except ValueError:
                h = 0
            if h <= 1:
                return "d1"
            if h == 2:
                return "d2"
            return "d3p"

        def _stratified_sample(pool: list, target_n: int, reference: list) -> list:
            if target_n >= len(pool):
                return list(pool)
            # Compute reference bucket fractions
            bucket_counts: dict[str, int] = {"d1": 0, "d2": 0, "d3p": 0}
            for r in reference:
                bucket_counts[_hop_bucket(r)] += 1
            total_ref = sum(bucket_counts.values()) or 1
            # Group pool by bucket
            pool_buckets: dict[str, list] = {"d1": [], "d2": [], "d3p": []}
            for r in pool:
                pool_buckets[_hop_bucket(r)].append(r)
            # Allocate target_n proportionally; remainder goes to largest bucket
            alloc: dict[str, int] = {}
            assigned = 0
            for bkt in ("d1", "d2", "d3p"):
                n = int(round(target_n * bucket_counts[bkt] / total_ref))
                n = min(n, len(pool_buckets[bkt]))
                alloc[bkt] = n
                assigned += n
            # Adjust for rounding errors, favouring d3p then d2 then d1
            for bkt in ("d3p", "d2", "d1"):
                while assigned < target_n and alloc[bkt] < len(pool_buckets[bkt]):
                    alloc[bkt] += 1
                    assigned += 1
                while assigned > target_n and alloc[bkt] > 0:
                    alloc[bkt] -= 1
                    assigned -= 1
            selected: list = []
            for bkt in ("d1", "d2", "d3p"):
                n = alloc[bkt]
                candidates = pool_buckets[bkt]
                selected.extend(rng.sample(candidates, n) if n < len(candidates) else list(candidates))
            return selected

        selected_pos = _stratified_sample(synth_pos, target_pos, baseline_pos)
        selected_neg = _stratified_sample(synth_neg, target_neg, baseline_neg)
        selected_rows = selected_pos + selected_neg
        rng.shuffle(selected_rows)

        out_split.mkdir(parents=True, exist_ok=True)
        if synthology_facts.exists():
            shutil.copy2(synthology_facts, out_split / "facts.csv")
        csv_fieldnames = list(synth_rows[0].keys()) if synth_rows else None
        _write_csv_rows(out_split / "targets.csv", selected_rows, fieldnames=csv_fieldnames)

        summary["splits"][split] = {
            "baseline_positive": len(baseline_pos),
            "baseline_negative": len(baseline_neg),
            "synthology_positive": len(synth_pos),
            "synthology_negative": len(synth_neg),
            "selected_positive": len(selected_pos),
            "selected_negative": len(selected_neg),
            "selected_total": len(selected_rows),
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
        else REPO_ROOT / "data" / "exp3" / "balanced" / f"owl2bench_{universities}" / "test"
    )
    if not source_dir.exists():
        source_dir = REPO_ROOT / "data" / "exp3" / "synthology" / f"owl2bench_{universities}" / "test"
    if not source_dir.exists():
        source_dir = REPO_ROOT / "data" / "owl2bench" / "output" / f"owl2bench_{universities}" / "test"

    output_dir_path = (
        Path(output_test_dir)
        if output_test_dir
        else REPO_ROOT / "data" / "exp3" / "frozen_test" / f"owl2bench_{universities}" / "test"
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

    if dataset_key == "baseline":
        dataset_root = REPO_ROOT / "data" / "owl2bench" / "output" / f"owl2bench_{universities}"
    else:
        balanced_root = REPO_ROOT / "data" / "exp3" / "balanced" / f"owl2bench_{universities}"
        synth_root = REPO_ROOT / "data" / "exp3" / "synthology" / f"owl2bench_{universities}"
        if balanced_root.exists():
            dataset_root = balanced_root
        elif synth_root.exists():
            dataset_root = synth_root
        else:
            raise FileNotFoundError(
                "Synthology dataset not found. Run exp3-generate-synthology first or provide balanced data via exp3-balance-data."
            )

    for split in ("train", "val", "test"):
        split_path = dataset_root / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split directory for Exp3 RRN training: {split_path}")

    print(f"\nTraining Exp 3 RRN on: {dataset_key} (universities={universities})")
    run_dir = _make_run_archive("exp3", "train_rrn", label=dataset_key)

    override_args = " ".join(
        [
            f"data.train_path={dataset_root / 'train'}",
            f"data.val_path={dataset_root / 'val'}",
            f"data.test_path={dataset_root / 'test'}",
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
        final_reasoning_input_triple_cap=int(baseline.get("final_reasoning_input_triple_cap", 15000)),
        final_jena_profile=str(baseline.get("final_jena_profile", "owl_mini")),
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
def exp3_paper_visual_report_hpc(ctx: Context, config_path="configs/experiments/exp3_hpc.yaml"):
    """Runs paper visual report with centralized Exp3 YAML preset."""
    cfg = _load_yaml_config(config_path)
    report = dict(cfg.get("paper_visual_report", {}))
    paper_visual_report(
        ctx,
        exp2_synth_targets=str(report["exp2_synth_targets"]),
        exp2_parity_summary=str(report["exp2_parity_summary"]),
        exp3_targets=str(report["exp3_targets"]),
        exp3_abox=str(report["exp3_abox"]),
        exp3_inferred=str(report["exp3_inferred"]),
        out_dir=str(report["out_dir"]),
        args=str(report.get("args", "")),
    )


@task
def exp3_report_and_analyze_hpc(ctx: Context, config_path="configs/experiments/exp3_hpc.yaml"):
    """Runs Exp3 comparison report and baseline analysis using centralized YAML preset."""
    cfg = _load_yaml_config(config_path)
    universities = int(cfg["universities"])
    report_cfg = dict(cfg.get("report", {}))

    baseline_path = str(report_cfg.get("baseline_path", "")).strip()
    if not baseline_path:
        baseline_path = f"data/owl2bench/output/owl2bench_{universities}"

    synthology_path = str(report_cfg.get("synthology_path", "")).strip()
    if not synthology_path:
        use_balanced_synthology = bool(report_cfg.get("use_balanced_synthology", True))
        if use_balanced_synthology:
            synthology_path = f"data/exp3/balanced/owl2bench_{universities}"
        else:
            synthology_path = f"data/exp3/synthology/owl2bench_{universities}"

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

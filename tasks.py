import os

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "synthology"
PYTHON_VERSION = "3.12"

# Ensure SYNTHOLOGY_ROOT is set for subprocesses to locate configs
os.environ["SYNTHOLOGY_ROOT"] = os.path.dirname(os.path.abspath(__file__))


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
        "export LOGURU_COLORIZE=1 && "
        "uv run --package ont_generator python -m ont_generator.create_data "
        "--config-name=exp1_test"
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


# ------------------------------------------------------------ #
# EXP 2: ...
# ------------------------------------------------------------ #


@task
def exp2_generate_gold_test(ctx: Context, args=""):
    """Generates the frozen shared test set for Exp 2."""
    print("\nGenerating Exp 2 frozen test set.")
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package ont_generator python -m ont_generator.create_data "
        "--config-name=exp2_gold_test"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


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
    ctx.run(cmd)


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
    ctx.run(cmd)


@task
def exp2_report_data(ctx: Context, args=""):
    """Generates parity/distribution reports for Exp 2 methods."""
    print("\nGenerating Exp 2 comparison report.")
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package data_reporter python -m data_reporter.analyze "
        "--config-name=exp2_compare"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def exp2_train_rrn(ctx: Context, dataset="baseline", args=""):
    """Trains RRN for Exp 2 on either baseline or synthology dataset."""
    dataset_key = dataset.strip().lower()
    if dataset_key not in {"baseline", "synthology"}:
        raise ValueError("dataset must be either 'baseline' or 'synthology'")

    rrn_dataset = "exp2_baseline" if dataset_key == "baseline" else "exp2_synthology"

    print(f"\nTraining Exp 2 RRN on: {dataset_key}")
    cmd = (
        "export PYTHONUNBUFFERED=1 && export LOGURU_COLORIZE=1 && "
        "uv run --package rrn python -m rrn.train "
        f"data/dataset={rrn_dataset} "
        f"+logger.name=exp2_{dataset_key} "
        "+logger.group=exp2_multihop "
        f"+logger.tags=[exp2,{dataset_key}]"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


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
        "materialization.reasoner=jena materialization.iterative=true "
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
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package udm_baseline python -m udm_baseline.exp2_parity_loop "
        f"--max-attempts {max_attempts} "
        f"--min-deep-hops {min_deep_hops} "
        f"--tolerance-pct {tolerance_pct} "
        f"--synth-targets {synth_targets} "
        f"--attempts-root {attempts_root}"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


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
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package udm_baseline python -m udm_baseline.exp2_parity_report "
        f"--min-deep-hops {min_deep_hops} "
        f"--synth-targets {synth_targets} "
        f"--attempts-root {attempts_root} "
        f"--out-json {out_json} "
        f"--out-csv {out_csv}"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


# ------------------------------------------------------------ #
# EXP 3: ...
# ------------------------------------------------------------ #


@task
def exp3_generate_owl2bench_abox(ctx: Context, universities=50, args=""):
    """Runs the existing OWL2Bench pipeline and stores raw generated OWL (ABox source)."""
    print(f"\nGenerating OWL2Bench data for Exp 3 (universities={universities}).")
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package owl2bench python -m owl2bench.pipeline "
        f"dataset.universities=[{universities}] "
        "dataset.output_dir=data/owl2bench/output "
        "dataset.reasoning.materialization.timing.enabled=true "
        "dataset.reasoning.materialization.timing.output_dir=data/exp3/timings "
        "dataset.reasoning.materialization.timing.run_tag=exp3_owl2bench_abox"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)


@task
def exp3_generate_baseline(ctx: Context, universities=50, args=""):
    """Generates Exp 3 baseline by chaining OWL2Bench generation with UDM/Jena materialization."""
    exp3_generate_owl2bench_abox(ctx, universities=universities, args=args)

    abox_path = f"data/owl2bench/output/raw/owl2bench_{universities}/OWL2RL-{universities}.owl"
    closure_out = f"data/exp3/baseline/owl2bench_{universities}/closure.nt"
    inferred_out = f"data/exp3/baseline/owl2bench_{universities}/inferred.nt"

    exp3_materialize_abox(
        ctx,
        abox=abox_path,
        tbox="data/owl2bench/input/UNIV-BENCH-OWL2RL.owl",
        closure_out=closure_out,
        inferred_out=inferred_out,
    )


@task
def exp3_materialize_abox(
    ctx: Context,
    abox,
    tbox="data/OWL2Bench/input/UNIV-BENCH-OWL2RL.owl",
    closure_out="outputs/exp3/closure.nt",
    inferred_out="outputs/exp3/inferred.nt",
    args="",
):
    """Materializes an OWL2Bench ABox with UDM/Jena and exports closure + inferred triples."""
    print("\nRunning Exp 3 ABox materialization with UDM/Jena.")
    cmd = (
        "export LOGURU_COLORIZE=1 && "
        "uv run --package udm_baseline python -m udm_baseline.materialize "
        f"--tbox {tbox} "
        f"--abox {abox} "
        f"--closure-out {closure_out} "
        f"--inferred-out {inferred_out} "
        "--timing-dir data/exp3/timings "
        "--timing-tag exp3_materialize_abox"
    )
    if args:
        cmd += f" {args}"
    ctx.run(cmd)

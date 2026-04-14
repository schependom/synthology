#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-paper-visual-report
#BSUB -n 2
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/exp3_paper_visual_%J.out
#BSUB -e logs/exp3_paper_visual_%J.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$PWD}}"
. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
mkdir -p logs

UNIVERSITIES="${UNIVERSITIES:-5}"
EXP2_SYNTH_TARGETS="${EXP2_SYNTH_TARGETS:-data/exp2/synthology/family_tree/train/targets.csv}"
EXP2_PARITY_SUMMARY="${EXP2_PARITY_SUMMARY:-data/exp2/baseline/parity_runs/parity_loop_summary.json}"
EXP3_TARGETS="${EXP3_TARGETS:-data/owl2bench/output/owl2bench_${UNIVERSITIES}/train/targets.csv}"
EXP3_ABOX="${EXP3_ABOX:-data/owl2bench/output/raw/owl2bench_${UNIVERSITIES}/OWL2RL-${UNIVERSITIES}.owl}"
EXP3_INFERRED="${EXP3_INFERRED:-data/exp3/baseline/owl2bench_${UNIVERSITIES}/inferred.nt}"
PAPER_OUT_DIR="${PAPER_OUT_DIR:-reports/paper}"
PAPER_REPORT_ARGS="${PAPER_REPORT_ARGS:-}"

synthology_load_modules python3/3.9.19
synthology_activate_python_env 0
synthology_sync_deps

echo "Starting paper visual report"
echo "  universities=${UNIVERSITIES}"
echo "  output_dir=${PAPER_OUT_DIR}"

cmd=(
  uv run invoke paper-visual-report
  --exp2-synth-targets="${EXP2_SYNTH_TARGETS}"
  --exp2-parity-summary="${EXP2_PARITY_SUMMARY}"
  --exp3-targets="${EXP3_TARGETS}"
  --exp3-abox="${EXP3_ABOX}"
  --exp3-inferred="${EXP3_INFERRED}"
  --out-dir="${PAPER_OUT_DIR}"
)

if [[ -n "${PAPER_REPORT_ARGS}" ]]; then
  cmd+=(--args="${PAPER_REPORT_ARGS}")
fi

"${cmd[@]}"

echo "Finished paper visual report at:"
date

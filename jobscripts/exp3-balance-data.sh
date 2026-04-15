#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-balance-data
#BSUB -n 4
#BSUB -W 04:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/exp3_balance_data_%J.out
#BSUB -e logs/exp3_balance_data_%J.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$PWD}}"
. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
mkdir -p logs

UNIVERSITIES="${UNIVERSITIES:-5}"
BASELINE_DIR="${BASELINE_DIR:-}"
SYNTHOLOGY_DIR="${SYNTHOLOGY_DIR:-}"
BALANCED_OUTPUT_DIR="${BALANCED_OUTPUT_DIR:-}"
BALANCE_SEED="${BALANCE_SEED:-23}"

synthology_load_modules python3/3.9.19
synthology_activate_python_env 0
synthology_sync_deps

echo "Starting Exp3 balance-data"
echo "  universities=${UNIVERSITIES}"
echo "  baseline_dir=${BASELINE_DIR}"
echo "  synthology_dir=${SYNTHOLOGY_DIR}"
echo "  balanced_output_dir=${BALANCED_OUTPUT_DIR}"
echo "  balance_seed=${BALANCE_SEED}"

cmd=(uv run invoke exp3-balance-data --universities="${UNIVERSITIES}" --seed="${BALANCE_SEED}")
if [[ -n "${BASELINE_DIR}" ]]; then
  cmd+=(--baseline-dir="${BASELINE_DIR}")
fi
if [[ -n "${SYNTHOLOGY_DIR}" ]]; then
  cmd+=(--synthology-dir="${SYNTHOLOGY_DIR}")
fi
if [[ -n "${BALANCED_OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir="${BALANCED_OUTPUT_DIR}")
fi

"${cmd[@]}"

echo "Finished Exp3 balance-data at:"
date

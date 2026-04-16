#!/usr/bin/env bash

#BSUB -q gpua100
#BSUB -J exp3-train-baseline
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/exp3_train_baseline_%J.out
#BSUB -e logs/exp3_train_baseline_%J.err

set -euo pipefail

if [ -f "${PWD}/jobscripts/common.sh" ]; then
	REPO_ROOT="${PWD}"
elif [ -n "${LS_SUBCWD:-}" ] && [ -f "${LS_SUBCWD}/jobscripts/common.sh" ]; then
	REPO_ROOT="${LS_SUBCWD}"
else
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
mkdir -p logs checkpoints/exp3/baseline

synthology_load_modules python3/3.9.19 cuda/11.7
synthology_activate_python_env 0
synthology_sync_deps

CONFIG_PATH="${1:-configs/experiments/exp3_hpc.yaml}"
synthology_require_file "${CONFIG_PATH}" "Exp3 HPC config"

echo "Starting Exp3 RRN training (baseline) with config: ${CONFIG_PATH}"
uv run invoke exp3-train-rrn-hpc --dataset=baseline --config-path="${CONFIG_PATH}"

echo "Finished Exp3 baseline training at:"
date

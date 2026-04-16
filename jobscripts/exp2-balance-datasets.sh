#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp2-balance-datasets
#BSUB -n 4
#BSUB -W 08:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/exp2_balance_datasets_%J.out
#BSUB -e logs/exp2_balance_datasets_%J.err

set -euo pipefail

# When submitted via `bsub < jobscripts/exp2-balance-datasets.sh`, BASH_SOURCE
# points to an LSF temp script in $HOME. Prefer the LSF working dir ($PWD).
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
mkdir -p logs

synthology_load_modules python3/3.9.19
synthology_activate_python_env 0
synthology_sync_deps

CONFIG_PATH="${1:-configs/experiments/exp2_balance_hpc.yaml}"
synthology_require_file "${CONFIG_PATH}" "Exp2 balance config"

echo "Starting Exp2 matched-budget generation with config: ${CONFIG_PATH}"
uv run invoke exp2-balance-datasets-hpc --config-path="${CONFIG_PATH}"

echo
echo "Exp2 matched-budget generation complete at:"
date

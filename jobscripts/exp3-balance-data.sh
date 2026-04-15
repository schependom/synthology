#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-balance-data
#BSUB -n 4
#BSUB -W 04:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/exp3_balance_data_%J.out
#BSUB -e logs/exp3_balance_data_%J.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
mkdir -p logs

synthology_load_modules python3/3.9.19
synthology_activate_python_env 0
synthology_sync_deps

echo "Starting Exp3 balance-data"
uv run invoke exp3-balance-data-hpc

echo "Finished Exp3 balance-data at:"
date

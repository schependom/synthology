#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-analyze-latest-baseline
#BSUB -n 2
#BSUB -W 01:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/exp3_analyze_latest_baseline_%J.out
#BSUB -e logs/exp3_analyze_latest_baseline_%J.err

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
mkdir -p logs

synthology_load_modules python3/3.9.19
synthology_activate_python_env 0
synthology_sync_deps

echo "Starting Exp3 latest-baseline analysis"
uv run invoke exp3-analyze-latest-baseline

echo "Finished Exp3 latest-baseline analysis at:"
date

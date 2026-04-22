#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-rep-smk
#BSUB -n 4
#BSUB -W 02:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/exp3_report_and_analyze_smoke_%J.out
#BSUB -e logs/exp3_report_and_analyze_smoke_%J.err

set -euo pipefail

# When submitted via `bsub < jobscripts/...`, BASH_SOURCE can point to an LSF temp script.
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

CONFIG_PATH="${1:-configs/experiments/exp3_smoke.yaml}"
synthology_require_file "${CONFIG_PATH}" "Exp3 HPC config"

echo "Starting Exp3 report + analysis smoke with config: ${CONFIG_PATH}"
uv run invoke exp3-report-and-analyze-hpc --config-path="${CONFIG_PATH}"

echo
echo "Exp3 report + analysis smoke complete at:"
date

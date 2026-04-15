#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-paper-visual-report
#BSUB -n 2
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/exp3_paper_visual_%J.out
#BSUB -e logs/exp3_paper_visual_%J.err

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

echo "Starting paper visual report"
uv run invoke exp3-paper-visual-report-hpc

echo "Finished paper visual report at:"
date

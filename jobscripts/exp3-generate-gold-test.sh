#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-generate-gold-test
#BSUB -n 2
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/exp3_generate_gold_test_%J.out
#BSUB -e logs/exp3_generate_gold_test_%J.err

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

echo "Starting Exp3 gold-test freeze"
uv run invoke exp3-generate-gold-test-hpc

echo "Finished Exp3 gold-test freeze at:"
date

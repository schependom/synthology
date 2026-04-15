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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
mkdir -p logs checkpoints/exp3/baseline

synthology_load_modules python3/3.9.19 cuda/11.7
synthology_activate_python_env 0
synthology_sync_deps

echo "Starting Exp3 RRN training (baseline)"
uv run --package rrn python -m rrn.train --config-name=exp3_baseline_u5_hpc

echo "Finished Exp3 baseline training at:"
date

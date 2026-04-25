#!/usr/bin/env bash

### General options
#BSUB -J exp1-rrn-mixed
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -o logs/exp1_mixed_%J.out
#BSUB -e logs/exp1_mixed_%J.err
#BSUB -u vincent.vanschependom@student.kuleuven.be
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

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
mkdir -p logs checkpoints/exp1/mixed
synthology_setup_runtime_storage

echo "Activating environment"
synthology_load_modules python3/3.9.19 cuda/11.7
synthology_activate_python_env 1

echo "Syncing dependencies"
synthology_sync_deps

echo "Starting Exp1 training: mixed"
uv run --package rrn python -m rrn.train --config-name=exp1_mixed_hpc

echo "Job finished at:"
date
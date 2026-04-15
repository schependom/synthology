#!/usr/bin/env bash

### General options
#BSUB -q gpua100
#BSUB -J exp1-test-constrained
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -o logs/exp1_test_constrained_%J.out
#BSUB -e logs/exp1_test_constrained_%J.err
### -- set the email address --
#BSUB -u vincent.vanschependom@student.kuleuven.be
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
mkdir -p logs
synthology_setup_runtime_storage

echo "Activating environment"
synthology_load_modules python3/3.9.19 cuda/11.7
synthology_activate_python_env 1

echo "Syncing dependencies"
synthology_sync_deps

echo "Starting Exp1 checkpoint test: constrained"
uv run --package rrn python -m rrn.test_checkpoint --config-name=exp1_constrained_hpc_test

echo "Job finished at:"
date

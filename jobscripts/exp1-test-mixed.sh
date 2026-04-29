#!/usr/bin/env bash

### General options
#BSUB -q hpc
#BSUB -J exp1-test-mixed
#BSUB -n 4
#BSUB -W 12:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/exp1_test_mixed_%J.out
#BSUB -e logs/exp1_test_mixed_%J.err
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
mkdir -p logs
synthology_setup_runtime_storage

echo "Activating environment"
synthology_load_modules python3/3.9.19
synthology_activate_python_env 1

echo "Syncing dependencies"
synthology_sync_deps

echo "Starting Exp1 checkpoint test: mixed"
uv run --package rrn python -m rrn.test_checkpoint --config-name=exp1_mixed_hpc_test

echo "Job finished at:"
date

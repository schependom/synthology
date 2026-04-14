#!/bin/sh

### General options
#BSUB -q gpua100
#BSUB -J exp1-rrn-constrained
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/exp1_constrained_%J.out
#BSUB -e logs/exp1_constrained_%J.err
#BSUB -u vincent.vanschependom@student.kuleuven.be
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

set -e

REPO_ROOT="${LSB_SUBCWD:-$PWD}"
. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
mkdir -p logs checkpoints/exp1/constrained
synthology_setup_runtime_storage

echo "Activating environment"
synthology_load_modules python3/3.9.19 cuda/11.7
synthology_activate_python_env 1

echo "Syncing dependencies"
synthology_sync_deps

echo "Starting Exp1 training: constrained"
uv run --package rrn python -m rrn.train --config-name=exp1_constrained_hpc

echo "Job finished at:"
date

#!/bin/sh

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

cd "${LSB_SUBCWD:-$PWD}"

mkdir -p logs

echo "Activating environment"
module load python3/3.9.19
module load cuda/11.7
source .env
source .venv/bin/activate

echo "Syncing dependencies"
uv sync

echo "Starting Exp1 checkpoint test: constrained"
uv run --package rrn python -m rrn.test_checkpoint --config-name=exp1_constrained_hpc_test

echo "Job finished at:"
date

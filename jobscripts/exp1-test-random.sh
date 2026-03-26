#!/bin/sh

### General options
#BSUB -q gpua100
#BSUB -J exp1-test-random
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -o logs/exp1_test_random_%J.out
#BSUB -e logs/exp1_test_random_%J.err

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

echo "Starting Exp1 checkpoint test: random"
uv run --package rrn python -m rrn.test_checkpoint --config-name=exp1_random_hpc_test

echo "Job finished at:"
date

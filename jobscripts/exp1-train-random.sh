#!/bin/sh

### General options
#BSUB -q gpua100
#BSUB -J exp1-rrn-random
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/exp1_random_%J.out
#BSUB -e logs/exp1_random_%J.err
#BSUB -u vincent.vanschependom@student.kuleuven.be
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

set -e

cd "${LSB_SUBCWD:-$PWD}"

mkdir -p logs checkpoints/exp1/random

echo "Activating environment"
module load python3/3.9.19
module load cuda/11.7
source .env
source .venv/bin/activate

echo "Syncing dependencies"
uv sync

echo "Starting Exp1 training: random"
uv run --package rrn python -m rrn.train --config-name=exp1_random_hpc

echo "Job finished at:"
date

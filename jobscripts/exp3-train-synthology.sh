#!/usr/bin/env bash

#BSUB -q gpua100
#BSUB -J exp3-train-synthology
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/exp3_train_synthology_%J.out
#BSUB -e logs/exp3_train_synthology_%J.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$PWD}}"
. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
mkdir -p logs checkpoints/exp3/synthology

UNIVERSITIES="${UNIVERSITIES:-5}"
EXP3_TRAIN_ARGS="${EXP3_TRAIN_ARGS:-}"

synthology_load_modules python3/3.9.19 cuda/11.7
synthology_activate_python_env 0
synthology_sync_deps

echo "Starting Exp3 RRN training (synthology)"
echo "  universities=${UNIVERSITIES}"
echo "  EXP3_TRAIN_ARGS=${EXP3_TRAIN_ARGS}"

if [[ -n "${EXP3_TRAIN_ARGS}" ]]; then
  uv run invoke exp3-train-rrn --dataset=synthology --universities="${UNIVERSITIES}" --args="${EXP3_TRAIN_ARGS}"
else
  uv run invoke exp3-train-rrn --dataset=synthology --universities="${UNIVERSITIES}"
fi

echo "Finished Exp3 synthology training at:"
date

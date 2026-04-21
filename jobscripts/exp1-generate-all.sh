#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp1-generate-trainval-sets
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/exp1_generate_trainval_sets_%J.out
#BSUB -e logs/exp1_generate_trainval_sets_%J.err

set -euo pipefail

# When submitted via `bsub < jobscripts/...`, BASH_SOURCE can point to an LSF temp script.
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
synthology_setup_runtime_storage
mkdir -p logs

synthology_load_modules python3/3.9.19 openjdk/21

synthology_activate_python_env 0
synthology_sync_deps

echo "Generating all train/val sets for exp1 for all negative sampling strategies."
uv run invoke exp1-generate-trainval-sets

echo "Finished Exp1 train/val set generation at:"
date

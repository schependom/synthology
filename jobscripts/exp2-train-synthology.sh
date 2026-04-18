#!/usr/bin/env bash

#BSUB -q gpua100
#BSUB -J exp2-train-synthology
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/exp2_train_synthology_%J.out
#BSUB -e logs/exp2_train_synthology_%J.err

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
synthology_setup_runtime_storage
mkdir -p logs checkpoints/exp2/synthology

synthology_load_modules python3/3.9.19 cuda/11.7
synthology_activate_python_env 0
synthology_sync_deps

echo "=== Data provenance ==="
for split in train val; do
	f="${REPO_ROOT}/data/exp2/synthology/family_tree/${split}/facts.csv"
	[ -f "$f" ] && echo "  ${split}: $(stat -c '%y' "$f" | cut -d. -f1) — $f" || echo "  ${split}: MISSING — $f"
done
f="${REPO_ROOT}/data/exp2/frozen_test/facts.csv"
[ -f "$f" ] && echo "  test: $(stat -c '%y' "$f" | cut -d. -f1) — $f" || echo "  test: MISSING — $f"
echo "======================"

echo "Starting Exp2 RRN training (synthology)"
uv run invoke exp2-train-rrn --dataset=synthology

echo "Finished Exp2 synthology training at:"
date

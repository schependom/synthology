#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-generate-gold-test
#BSUB -n 2
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/exp3_generate_gold_test_%J.out
#BSUB -e logs/exp3_generate_gold_test_%J.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$PWD}}"
. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
mkdir -p logs

UNIVERSITIES="${UNIVERSITIES:-5}"
SOURCE_TEST_DIR="${SOURCE_TEST_DIR:-}"
OUTPUT_TEST_DIR="${OUTPUT_TEST_DIR:-}"

synthology_load_modules python3/3.9.19
synthology_activate_python_env 0
synthology_sync_deps

echo "Starting Exp3 gold-test freeze"
echo "  universities=${UNIVERSITIES}"
echo "  source_test_dir=${SOURCE_TEST_DIR}"
echo "  output_test_dir=${OUTPUT_TEST_DIR}"

cmd=(uv run invoke exp3-generate-gold-test --universities="${UNIVERSITIES}")
if [[ -n "${SOURCE_TEST_DIR}" ]]; then
  cmd+=(--source-test-dir="${SOURCE_TEST_DIR}")
fi
if [[ -n "${OUTPUT_TEST_DIR}" ]]; then
  cmd+=(--output-test-dir="${OUTPUT_TEST_DIR}")
fi

"${cmd[@]}"

echo "Finished Exp3 gold-test freeze at:"
date

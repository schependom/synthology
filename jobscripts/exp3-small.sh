#!/usr/bin/env bash
# Full Exp3 small run in a single job: generate → balance → report.
# Uses configs/experiments/exp3_small.yaml (2 universities, owl_full, small caps).
# Expected wall time: ~30-90 min depending on Jena owl_full materialisation.
#
# Memory: owl_full on 2-university ABox (≈70 K triples) needs ~8-12 GB JVM heap.
# abox_jena_heap_mb=8192 → request 12 GB/slot × 8 slots = 96 GB headroom total.

#BSUB -q hpc
#BSUB -J exp3-small
#BSUB -n 8
#BSUB -W 03:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -o logs/exp3_small_%J.out
#BSUB -e logs/exp3_small_%J.err

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
mkdir -p logs

synthology_load_modules python3/3.9.19 openjdk/21

export MAVEN_HOME="apache-maven-3.9.13"
export PATH="${MAVEN_HOME}/bin:${PATH}"
export MAVEN_EXECUTABLE="${MAVEN_HOME}/bin/mvn"
export MAVEN_REPO_LOCAL="/dtu/blackhole/16/221590/.m2/repository"
mkdir -p "${MAVEN_REPO_LOCAL}"

synthology_activate_python_env 0
synthology_sync_deps

CONFIG_PATH="${1:-configs/experiments/exp3_small.yaml}"
synthology_require_file "${CONFIG_PATH}" "Exp3 small config"

echo "=== Exp3 small: baseline generation ==="
uv run invoke exp3-generate-baseline-hpc --config-path="${CONFIG_PATH}"

echo "=== Exp3 small: synthology generation ==="
uv run invoke exp3-generate-synthology-hpc --config-path="${CONFIG_PATH}"

echo "=== Exp3 small: balance data ==="
uv run invoke exp3-balance-data-hpc --config-path="${CONFIG_PATH}"

echo "=== Exp3 small: report and analyze ==="
uv run invoke exp3-report-and-analyze-hpc --config-path="${CONFIG_PATH}"

echo "Finished Exp3 small at:"
date

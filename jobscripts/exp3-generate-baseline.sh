#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-generate-baseline
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -o logs/exp3_generate_baseline_%J.out
#BSUB -e logs/exp3_generate_baseline_%J.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# Override these at submit time if needed, e.g.:
# UNIVERSITIES=1 SYNTHOLOGY_UDM_BASELINE_XMX_MB=57344 bsub < jobscripts/exp3-generate-baseline.sh
UNIVERSITIES="${UNIVERSITIES:-1}"
SYNTHOLOGY_UDM_BASELINE_XMX_MB="${SYNTHOLOGY_UDM_BASELINE_XMX_MB:-49152}"
SYNTHOLOGY_JENA_XMX_MB="${SYNTHOLOGY_JENA_XMX_MB:-${SYNTHOLOGY_UDM_BASELINE_XMX_MB}}"
SYNTHOLOGY_HEAP_MB="${SYNTHOLOGY_HEAP_MB:-${SYNTHOLOGY_UDM_BASELINE_XMX_MB}}"

if command -v module >/dev/null 2>&1; then
  module load python3/3.9.19 || true
  module load openjdk/21 || true
fi

export MAVEN_HOME="${MAVEN_HOME:-apache-maven-3.9.13}"
export PATH="${MAVEN_HOME}/bin:${PATH}"
export MAVEN_EXECUTABLE="${MAVEN_EXECUTABLE:-${MAVEN_HOME}/bin/mvn}"

# Keep baseline heap sizing explicit and aligned across wrappers.
export SYNTHOLOGY_UDM_BASELINE_XMX_MB
export SYNTHOLOGY_JENA_XMX_MB
export SYNTHOLOGY_HEAP_MB

source .venv/bin/activate
uv sync

echo "Starting Exp3 baseline generation"
echo "  universities=${UNIVERSITIES}"
echo "  SYNTHOLOGY_UDM_BASELINE_XMX_MB=${SYNTHOLOGY_UDM_BASELINE_XMX_MB}"
echo "  MAVEN_EXECUTABLE=${MAVEN_EXECUTABLE}"

uv run invoke exp3-generate-baseline --universities="${UNIVERSITIES}"

echo "Finished Exp3 baseline generation at:"
date

#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-generate-synthology
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/exp3_generate_synthology_%J.out
#BSUB -e logs/exp3_generate_synthology_%J.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${LSB_SUBCWD:-$PWD}}"
. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
mkdir -p logs

UNIVERSITIES="${UNIVERSITIES:-5}"
SYNTHOLOGY_JENA_XMX_MB="${SYNTHOLOGY_JENA_XMX_MB:-16384}"
EXP3_SYNTH_ARGS="${EXP3_SYNTH_ARGS:-}"

synthology_load_modules python3/3.9.19 openjdk/21

export MAVEN_HOME="${MAVEN_HOME:-apache-maven-3.9.13}"
export PATH="${MAVEN_HOME}/bin:${PATH}"
export MAVEN_EXECUTABLE="${MAVEN_EXECUTABLE:-${MAVEN_HOME}/bin/mvn}"
export MAVEN_REPO_LOCAL="${MAVEN_REPO_LOCAL:-/dtu/blackhole/16/221590/.m2/repository}"
mkdir -p "${MAVEN_REPO_LOCAL}"

export SYNTHOLOGY_HEAP_MB="${SYNTHOLOGY_JENA_XMX_MB}"

synthology_activate_python_env 0
synthology_sync_deps

echo "Starting Exp3 synthology generation"
echo "  universities=${UNIVERSITIES}"
echo "  SYNTHOLOGY_JENA_XMX_MB=${SYNTHOLOGY_JENA_XMX_MB}"
echo "  EXP3_SYNTH_ARGS=${EXP3_SYNTH_ARGS}"

if [[ -n "${EXP3_SYNTH_ARGS}" ]]; then
  uv run invoke exp3-generate-synthology --universities="${UNIVERSITIES}" --args="${EXP3_SYNTH_ARGS}"
else
  uv run invoke exp3-generate-synthology --universities="${UNIVERSITIES}"
fi

echo "Finished Exp3 synthology generation at:"
date

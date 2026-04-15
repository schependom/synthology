#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp3-generate-synthology
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/exp3_generate_synthology_%J.out
#BSUB -e logs/exp3_generate_synthology_%J.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

echo "Starting Exp3 synthology generation"
uv run invoke exp3-generate-synthology-hpc

echo "Finished Exp3 synthology generation at:"
date

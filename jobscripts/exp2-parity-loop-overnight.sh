#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp2-parity-loop
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/exp2_parity_%J.out
#BSUB -e logs/exp2_parity_%J.err

set -uo pipefail  # no -e: parity loop non-zero exit must not kill the job

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage
synthology_require_file "configs/udm_baseline/exp2_parity_loop.json" "parity config" || exit 1

synthology_load_modules python3/3.9.19 openjdk/21

export MAVEN_HOME="apache-maven-3.9.13"
export PATH="${MAVEN_HOME}/bin:${PATH}"

synthology_activate_python_env 0
synthology_sync_deps

echo "Starting Exp2 parity loop - $(date)"
echo "Parity config: configs/udm_baseline/exp2_parity_loop.json"

# Run parity loop - non-zero exit (exhausted attempts) must not abort the job
uv run invoke exp2-parity-loop \
  && echo "PARITY ACHIEVED" \
  || echo "PARITY NOT REACHED - will still run report on best attempt"

# Resolve the actual attempts root from the latest run archive
# (tasks.py ignores --attempts-root and always uses the archive path)
ACTUAL_ATTEMPTS_ROOT=$(find reports/experiment_runs -type d -name "attempts" \
  -path "*/exp2/parity_loop/*" | sort | tail -1)

echo "Actual attempts root: ${ACTUAL_ATTEMPTS_ROOT}"

if [[ -z "${ACTUAL_ATTEMPTS_ROOT}" ]]; then
  echo "ERROR: could not find attempts directory in run archive"
  exit 1
fi

# Run parity report pointing at the real attempts location
uv run invoke exp2-parity-report \
  --attempts-root="${ACTUAL_ATTEMPTS_ROOT}" \
  || echo "WARNING: parity report failed - check ${ACTUAL_ATTEMPTS_ROOT} manually"

echo "Done - $(date)"
echo "Check run archive: reports/experiment_runs/$(date +%Y-%m-%d)/exp2/parity_loop/"

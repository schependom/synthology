#!/usr/bin/env bash

#BSUB -q hpc
#BSUB -J exp2-parity-loop
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o exp2_parity_%J.out
#BSUB -e exp2_parity_%J.err

set -uo pipefail  # no -e: parity loop non-zero exit must not kill the job

REPO_ROOT="/dtu/blackhole/16/221590/synthology"
cd "${REPO_ROOT}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-250}"
MIN_DEEP_HOPS="${MIN_DEEP_HOPS:-3}"
DEEP_COUNT_MODE="${DEEP_COUNT_MODE:-tolerance}"
TOLERANCE_PCT="${TOLERANCE_PCT:-10.0}"
NODE_TOLERANCE_PCT="${NODE_TOLERANCE_PCT:-10}"
EDGE_DENSITY_TOLERANCE_PCT="${EDGE_DENSITY_TOLERANCE_PCT:-15}"
TARGET_RATIO_TOLERANCE_PCT="${TARGET_RATIO_TOLERANCE_PCT:-10}"
INFERRED_SHARE_TOLERANCE_PCT="${INFERRED_SHARE_TOLERANCE_PCT:-10}"

if command -v module >/dev/null 2>&1; then
  module load python3/3.9.19 || true
  module load openjdk/21 || true
fi

export MAVEN_HOME="apache-maven-3.9.13"
export PATH="${MAVEN_HOME}/bin:${PATH}"

source .venv/bin/activate
uv sync

echo "Starting Exp2 parity loop - $(date)"

# Run parity loop - non-zero exit (exhausted attempts) must not abort the job
uv run invoke exp2-parity-loop \
  --max-attempts="${MAX_ATTEMPTS}" \
  --min-deep-hops="${MIN_DEEP_HOPS}" \
  --deep-count-mode="${DEEP_COUNT_MODE}" \
  --tolerance-pct="${TOLERANCE_PCT}" \
  --node-tolerance-pct="${NODE_TOLERANCE_PCT}" \
  --edge-density-tolerance-pct="${EDGE_DENSITY_TOLERANCE_PCT}" \
  --target-ratio-tolerance-pct="${TARGET_RATIO_TOLERANCE_PCT}" \
  --inferred-share-tolerance-pct="${INFERRED_SHARE_TOLERANCE_PCT}" \
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

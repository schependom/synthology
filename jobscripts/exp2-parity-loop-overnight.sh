#!/usr/bin/env bash

set -euo pipefail

### General options
#BSUB -q hpc
#BSUB -J exp2-parity-loop
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o exp2_parity_%J.out
#BSUB -e exp2_parity_%J.err

# ---------- Tunables ----------
MAX_ATTEMPTS="${MAX_ATTEMPTS:-250}"
MIN_DEEP_HOPS="${MIN_DEEP_HOPS:-3}"
DEEP_COUNT_MODE="${DEEP_COUNT_MODE:-tolerance}"
TOLERANCE_PCT="${TOLERANCE_PCT:-10}"
NODE_TOLERANCE_PCT="${NODE_TOLERANCE_PCT:-10}"
EDGE_DENSITY_TOLERANCE_PCT="${EDGE_DENSITY_TOLERANCE_PCT:-15}"
TARGET_RATIO_TOLERANCE_PCT="${TARGET_RATIO_TOLERANCE_PCT:-10}"
INFERRED_SHARE_TOLERANCE_PCT="${INFERRED_SHARE_TOLERANCE_PCT:-10}"
RUN_PARITY_REPORT="${RUN_PARITY_REPORT:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ATTEMPTS_ROOT="${ATTEMPTS_ROOT:-data/exp2/baseline/parity_runs/${STAMP}}"

run_cmd() {
  echo
  echo "> $*"
  "$@"
}

echo "Exp2 parity overnight job"
echo "  repo_root=${REPO_ROOT}"
echo "  attempts_root=${ATTEMPTS_ROOT}"
echo "  max_attempts=${MAX_ATTEMPTS}"
echo "  min_deep_hops=${MIN_DEEP_HOPS}"
echo "  deep_count_mode=${DEEP_COUNT_MODE}"
echo "  tolerance_pct=${TOLERANCE_PCT}"

# Environment bootstrap
if command -v module >/dev/null 2>&1; then
  module load python3/3.9.19 || true
  module load openjdk/21 || true
fi

# Export mvn
export MAVEN_HOME="apache-maven-3.9.13"
export PATH="${MAVEN_HOME}/bin:${PATH}" 

source .venv/bin/activate
run_cmd uv sync

# Run parity loop with explicit attempts root for reproducible follow-up reporting.
run_cmd uv run invoke exp2-parity-loop --args="--attempts-root ${ATTEMPTS_ROOT} --max-attempts ${MAX_ATTEMPTS} --min-deep-hops ${MIN_DEEP_HOPS} --deep-count-mode ${DEEP_COUNT_MODE} --tolerance-pct ${TOLERANCE_PCT} --node-tolerance-pct ${NODE_TOLERANCE_PCT} --edge-density-tolerance-pct ${EDGE_DENSITY_TOLERANCE_PCT} --target-ratio-tolerance-pct ${TARGET_RATIO_TOLERANCE_PCT} --inferred-share-tolerance-pct ${INFERRED_SHARE_TOLERANCE_PCT}"

# Optional: build parity report immediately for the same attempts_root.
if [[ "${RUN_PARITY_REPORT}" == "1" ]]; then
  run_cmd uv run invoke exp2-parity-report --attempts-root="${ATTEMPTS_ROOT}"
fi

echo
echo "Finished Exp2 parity overnight job"
echo "Attempts root: ${ATTEMPTS_ROOT}"
echo "Summary JSON: ${ATTEMPTS_ROOT}/parity_loop_summary.json"

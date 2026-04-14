#!/usr/bin/env bash

set -euo pipefail

# Exp2 sweep over target caps and seeds.
#
# Defaults can be overridden as environment variables, e.g.:
# FACT_CAP=50000 TARGET_CAPS="80000 120000" SEEDS="23 42" \
#   bash jobscripts/exp2-sweep-targetcaps-seeds.sh

FACT_CAP="${FACT_CAP:-50000}"
TARGET_CAPS="${TARGET_CAPS:-80000 120000 160000}"
SEEDS="${SEEDS:-23 42}"
BASELINE_BASE_FACTS="${BASELINE_BASE_FACTS:-20}"
SYNTHOLOGY_PROOF_ROOTS="${SYNTHOLOGY_PROOF_ROOTS:-10}"
RUN_REPORT="${RUN_REPORT:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage

synthology_activate_python_env 0

run_cmd() {
  echo
  echo "> $*"
  "$@"
}

echo "Exp2 sweep configuration"
echo "  FACT_CAP=${FACT_CAP}"
echo "  TARGET_CAPS=${TARGET_CAPS}"
echo "  SEEDS=${SEEDS}"
echo "  BASELINE_BASE_FACTS=${BASELINE_BASE_FACTS}"
echo "  SYNTHOLOGY_PROOF_ROOTS=${SYNTHOLOGY_PROOF_ROOTS}"

for target_cap in ${TARGET_CAPS}; do
  echo
  echo "============================================================"
  echo "Running dataset generation for target_cap=${target_cap}"
  echo "============================================================"

  run_cmd uv run invoke exp2-balance-datasets \
    --fact-cap="${FACT_CAP}" \
    --target-cap="${target_cap}" \
    --baseline-base-facts="${BASELINE_BASE_FACTS}" \
    --synthology-proof-roots="${SYNTHOLOGY_PROOF_ROOTS}"

  if [[ "${RUN_REPORT}" == "1" ]]; then
    run_cmd uv run invoke exp2-report-data
  fi

  for seed in ${SEEDS}; do
    run_name_baseline="exp2_baseline_tc${target_cap}_seed${seed}"
    run_name_synthology="exp2_synthology_tc${target_cap}_seed${seed}"

    run_cmd uv run invoke exp2-train-rrn \
      --dataset=baseline \
      --args="+seed=${seed} +logger.name=${run_name_baseline} +logger.group=exp2_multihop +logger.tags=[exp2,baseline,target_cap_${target_cap},seed${seed}]"

    run_cmd uv run invoke exp2-train-rrn \
      --dataset=synthology \
      --args="+seed=${seed} +logger.name=${run_name_synthology} +logger.group=exp2_multihop +logger.tags=[exp2,synthology,target_cap_${target_cap},seed${seed}]"
  done
done

echo
echo "Exp2 target-cap + seed sweep complete."

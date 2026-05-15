#!/usr/bin/env bash
#BSUB -q hpc
#BSUB -J exp3-scale-bl-u10
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=14GB] span[hosts=1]"
#BSUB -o /dtu/blackhole/16/221590/synthology/logs/exp3_scale_bl_u10_%J.out
#BSUB -e /dtu/blackhole/16/221590/synthology/logs/exp3_scale_bl_u10_%J.err

# Baked-in parameters (set at script-generation time by the launcher)
REPO_ROOT="/dtu/blackhole/16/221590/synthology"
SCALE_U="10"
SCALE_CONFIG="/dtu/blackhole/16/221590/synthology/configs/experiments/scalability/exp3_scale_u10.yaml"
SCALE_RESULTS_DIR="/dtu/blackhole/16/221590/synthology/results/exp3_scalability"
set -uo pipefail

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

RESULT_FILE="${SCALE_RESULTS_DIR}/u${SCALE_U}_baseline.csv"
TIME_FILE="${SCALE_RESULTS_DIR}/u${SCALE_U}_baseline_time.txt"

echo "==================================================================="
echo "  Exp3 scalability: BASELINE  u=${SCALE_U}"
echo "  Config : ${SCALE_CONFIG}"
echo "==================================================================="

START=$(date +%s)
STATUS="OK"

set +e
/usr/bin/time -v -o "${TIME_FILE}" \
	uv run invoke exp3-generate-baseline-hpc --config-path="${SCALE_CONFIG}"
RC=$?
set -e

END=$(date +%s)
ELAPSED=$((END - START))

if [ "${RC}" -ne 0 ]; then
	STATUS="FAILED_RC${RC}"
	echo "ERROR: baseline u=${SCALE_U} exited with code ${RC} — result marked FAILED"
fi

MEM_MB=0
if [ -f "${TIME_FILE}" ]; then
	MEM_KB=$(grep -oP "(?<=Maximum resident set size \(kbytes\): )\d+" "${TIME_FILE}" 2>/dev/null || echo 0)
	[ "${MEM_KB:-0}" -gt 0 ] && MEM_MB=$((MEM_KB / 1024))
fi

printf '%s,baseline,%d,%d,%s\n' "${SCALE_U}" "${ELAPSED}" "${MEM_MB}" "${STATUS}" \
	> "${RESULT_FILE}"

echo ""
echo "SCALE_RESULT u=${SCALE_U} method=baseline time_sec=${ELAPSED} mem_mb=${MEM_MB} status=${STATUS}"
echo "Finished at:"; date

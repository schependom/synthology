#!/usr/bin/env bash
# Collects Exp3 scalability sweep results into a single CSV.
#
# Run after all jobs submitted by exp3-scalability-launch.sh have finished:
#   bash jobscripts/exp3-scalability-collect.sh
#
# Reads:  results/exp3_scalability/u{N}_{baseline,synthology}.csv
# Writes: results/exp3_scalability/timing_summary.csv
#
# CSV format: u,method,time_sec,time_min,mem_mb,status
# Failed rows have time_sec=NaN and mem_mb=NaN so MATLAB can use them as
# missing values (the plot skips bars where data is NaN).

set -euo pipefail

if [ -f "${PWD}/jobscripts/common.sh" ]; then
	REPO_ROOT="${PWD}"
elif [ -n "${LS_SUBCWD:-}" ] && [ -f "${LS_SUBCWD}/jobscripts/common.sh" ]; then
	REPO_ROOT="${LS_SUBCWD}"
else
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

RESULTS_DIR="${REPO_ROOT}/results/exp3_scalability"
SUMMARY="${RESULTS_DIR}/timing_summary.csv"

if [ ! -d "${RESULTS_DIR}" ]; then
	echo "ERROR: results directory not found: ${RESULTS_DIR}"
	echo "Run exp3-scalability-launch.sh first."
	exit 1
fi

U_VALUES=(2 4 6 8 10 12 14)

echo "u,method,time_sec,time_min,mem_mb,status" > "${SUMMARY}"

# ── Parse per-job result files ──────────────────────────────────────────────
MISSING=0
for U in "${U_VALUES[@]}"; do
	for METHOD in baseline synthology; do
		RESULT="${RESULTS_DIR}/u${U}_${METHOD}.csv"
		if [ ! -f "${RESULT}" ]; then
			echo "${U},${METHOD},NaN,NaN,NaN,MISSING" >> "${SUMMARY}"
			echo "  u=${U} ${METHOD}: result file missing — marked MISSING"
			MISSING=$((MISSING + 1))
			continue
		fi

		LINE=$(cat "${RESULT}")
		U_VAL=$(echo "${LINE}" | cut -d',' -f1)
		METHOD_VAL=$(echo "${LINE}" | cut -d',' -f2)
		TIME_SEC=$(echo "${LINE}" | cut -d',' -f3)
		MEM_MB=$(echo "${LINE}" | cut -d',' -f4)
		STATUS=$(echo "${LINE}" | cut -d',' -f5)

		TIME_MIN=$(awk "BEGIN {printf \"%.2f\", ${TIME_SEC}/60}")
		echo "${U},${METHOD},${TIME_SEC},${TIME_MIN},${MEM_MB},${STATUS}" >> "${SUMMARY}"
		if [ "${STATUS}" != "OK" ]; then
			# Keep timing: it is "time/memory at failure", which is meaningful for
			# the reasoning-wall graph (monotonically growing even for failed runs).
			# MATLAB will hatch these bars to distinguish them from successful runs.
			echo "  u=${U} ${METHOD}: ${TIME_MIN} min, ${MEM_MB} MB  [${STATUS}]"
		else
			echo "  u=${U} ${METHOD}: ${TIME_MIN} min, ${MEM_MB} MB  [OK]"
		fi
	done
done

echo ""
echo "Wrote: ${SUMMARY}"
[ "${MISSING}" -gt 0 ] && echo "WARNING: ${MISSING} result file(s) missing — those jobs may still be running."

# ── Print paper-ready table ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Exp3 Scalability — paper data summary"
echo "================================================================"
printf "  %-4s  %-12s  %-12s  %-10s  %-10s\n" "u" "BL time" "Synth time" "BL mem" "Synth mem"
printf "  %-4s  %-12s  %-12s  %-10s  %-10s\n" "----" "------------" "------------" "----------" "----------"

for U in "${U_VALUES[@]}"; do
	BL_LINE=$(grep "^${U},baseline," "${SUMMARY}" || echo "${U},baseline,NaN,NaN,NaN,MISSING")
	SB_LINE=$(grep "^${U},synthology," "${SUMMARY}" || echo "${U},synthology,NaN,NaN,NaN,MISSING")

	BL_MIN=$(echo "${BL_LINE}" | cut -d',' -f4)
	SB_MIN=$(echo "${SB_LINE}" | cut -d',' -f4)
	BL_MEM=$(echo "${BL_LINE}" | cut -d',' -f5)
	SB_MEM=$(echo "${SB_LINE}" | cut -d',' -f5)
	BL_STATUS=$(echo "${BL_LINE}" | cut -d',' -f6)
	SB_STATUS=$(echo "${SB_LINE}" | cut -d',' -f6)

	# Append status in parentheses if not OK
	[ "${BL_STATUS}" != "OK" ] && BL_MIN="${BL_MIN} (${BL_STATUS})"
	[ "${SB_STATUS}" != "OK" ] && SB_MIN="${SB_MIN} (${SB_STATUS})"

	printf "  %-4s  %-12s  %-12s  %-10s  %-10s\n" \
		"${U}" "${BL_MIN} min" "${SB_MIN} min" "${BL_MEM} MB" "${SB_MEM} MB"
done
echo "================================================================"
echo ""
echo "Run matlab/exp3_scalability.m to generate the figure."

#!/usr/bin/env bash

### General options
#BSUB -q gpua100
#BSUB -J exp1-test-proof-based-only
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -o logs/exp1_test_pb_only_%J.out
#BSUB -e logs/exp1_test_pb_only_%J.err
#BSUB -u vincent.vanschependom@student.kuleuven.be
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

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
mkdir -p logs
synthology_setup_runtime_storage

synthology_load_modules python3/3.9.19 cuda/11.7
synthology_activate_python_env 1
synthology_sync_deps

echo "================================================================"
echo "  Phase 1: Generating Pure Proof-Based Test Set"
echo "================================================================"
uv run --package ont_generator python -m ont_generator.create_data --config-name=exp1_test_proof_based

RESULTS_DIR="${REPO_ROOT}/results/exp1_test_pb_only_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

VARIANTS=(random constrained proof_based mixed)
# Use existing test configs, but override the test_path to point to the new pure proof_based test set
CONFIG_NAMES=(exp1_random_hpc_test exp1_constrained_hpc_test exp1_proof_based_hpc_test exp1_mixed_hpc_test)
TEST_PATH_OVERRIDE="data.dataset.test_path=data/exp1/test_set_proof_based/test"

declare -A CKPT_PATH
declare -A TEST_STATUS

# ── Phase 2: locate latest checkpoint per variant ──────────────────────────
echo "================================================================"
echo "  Phase 2: Locating latest checkpoints"
echo "================================================================"
for variant in "${VARIANTS[@]}"; do
	ckpt=$(find "${REPO_ROOT}/outputs" -name "best-checkpoint-${variant}*" \
		-printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
	CKPT_PATH[$variant]="${ckpt:-}"
	if [ -z "${ckpt}" ]; then
		echo "  ${variant}: NO CHECKPOINT — will be skipped"
		TEST_STATUS[$variant]="SKIPPED (no checkpoint)"
	else
		echo "  ${variant}: ${ckpt}"
		TEST_STATUS[$variant]="pending"
	fi
done
echo ""

# ── Phase 3: run each test sequentially ───────────────────────────────────
for i in "${!VARIANTS[@]}"; do
	variant="${VARIANTS[$i]}"
	config="${CONFIG_NAMES[$i]}"
	ckpt="${CKPT_PATH[$variant]}"

	if [ -z "${ckpt}" ]; then
		continue
	fi

	echo "================================================================"
	echo "  Testing: ${variant} on Pure Proof-Based Test Set"
	echo "  Checkpoint: ${ckpt}"
	echo "  Config: ${config}"
	echo "================================================================"

	log_file="${RESULTS_DIR}/${variant}.log"

	set +e
	uv run --package rrn python -m rrn.test_checkpoint \
		--config-name="${config}" \
		"test.checkpoint_path=${ckpt}" \
		"${TEST_PATH_OVERRIDE}" \
		2>&1 | tee "${log_file}"
	rc="${PIPESTATUS[0]}"
	set -e

	if [ "${rc}" -ne 0 ]; then
		echo "FAILED: ${variant} (exit code ${rc})"
		TEST_STATUS[$variant]="FAILED (exit ${rc})"
	else
		TEST_STATUS[$variant]="OK"
	fi
	echo ""
done

# ── Phase 4: parse metrics and print report ───────────────────────────────
REPORT="${RESULTS_DIR}/report.txt"
{
python3 - "${RESULTS_DIR}" "${VARIANTS[@]}" << 'PYEOF'
import sys, re, os

results_dir = sys.argv[1]
variants    = sys.argv[2:]

METRICS = [
    ("test/triple_acc_type_neg_inf_root", "Neg. Inf. Root Acc ↑"),
    ("test/triple_pr_auc",  "PR-AUC ↑"),
    ("test/triple_auc_roc", "AUC-ROC ↑"),
]

def extract_metric(text, metric_name):
    for sep in ["│", "|"]:
        pat = (
            rf"{re.escape(sep)}\s*{re.escape(metric_name)}\s*"
            rf"{re.escape(sep)}\s*([0-9]+\.[0-9]+(?:e[+\-]?[0-9]+)?)\s*"
            rf"{re.escape(sep)}"
        )
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return "N/A"

data = {}
for variant in variants:
    log_file = os.path.join(results_dir, f"{variant}.log")
    if os.path.exists(log_file):
        with open(log_file, "r", errors="replace") as fh:
            content = fh.read()
        data[variant] = {key: extract_metric(content, key) for key, _ in METRICS}
    else:
        data[variant] = {key: "N/A" for key, _ in METRICS}

print("=" * 85)
print("  Exp1 Pure Proof-Based Test Report (Proof of Reasoning)")
print("=" * 85)
print()
print(f"  {'Method':<22} {'Neg Inf Root Acc ↑':<20} {'PR-AUC ↑':<16} {'AUC-ROC ↑'}")
print("  " + "-" * 78)
for variant in variants:
    d   = data.get(variant, {})
    inf = d.get("test/triple_acc_type_neg_inf_root", "N/A")
    pr  = d.get("test/triple_pr_auc",  "N/A")
    roc = d.get("test/triple_auc_roc", "N/A")
    print(f"  {variant:<22} {inf:<20} {pr:<16} {roc}")
print()
print("=" * 85)
PYEOF
} | tee "${REPORT}"

echo ""
echo "Report: ${REPORT}"
echo "Logs:   ${RESULTS_DIR}/"
echo "Job finished at:"
date

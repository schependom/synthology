#!/usr/bin/env bash

### General options
#BSUB -q hpc
#BSUB -J exp1-test-all
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/exp1_test_all_%J.out
#BSUB -e logs/exp1_test_all_%J.err
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

synthology_load_modules python3/3.9.19
synthology_activate_python_env 1
synthology_sync_deps

RESULTS_DIR="${REPO_ROOT}/results/exp1_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

VARIANTS=(random constrained proof_based mixed)
CONFIG_NAMES=(exp1_random_hpc_test exp1_constrained_hpc_test exp1_proof_based_hpc_test exp1_mixed_hpc_test)

declare -A CKPT_PATH
declare -A TEST_STATUS

# ── Phase 1: locate latest checkpoint per variant ──────────────────────────
echo "================================================================"
echo "  Locating latest checkpoints"
echo "================================================================"
for variant in "${VARIANTS[@]}"; do
	ckpt=$(find "${REPO_ROOT}/outputs" -name "best-checkpoint-${variant}.ckpt" \
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

# ── Phase 2: run each test sequentially ───────────────────────────────────
for i in "${!VARIANTS[@]}"; do
	variant="${VARIANTS[$i]}"
	config="${CONFIG_NAMES[$i]}"
	ckpt="${CKPT_PATH[$variant]}"

	if [ -z "${ckpt}" ]; then
		continue
	fi

	echo "================================================================"
	echo "  Testing: ${variant}"
	echo "  Checkpoint: ${ckpt}"
	echo "  Config: ${config}"
	echo "================================================================"

	log_file="${RESULTS_DIR}/${variant}.log"

	set +e
	uv run --package rrn python -m rrn.test_checkpoint \
		--config-name="${config}" \
		"+test.checkpoint_path=${ckpt}" \
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

# ── Phase 3: parse metrics and print paper-ready report ───────────────────
REPORT="${RESULTS_DIR}/report.txt"
{
python3 - "${RESULTS_DIR}" "${VARIANTS[@]}" << 'PYEOF'
import sys, re, os

results_dir = sys.argv[1]
variants    = sys.argv[2:]

METRICS = [
    ("test/triple_pr_auc",  "PR-AUC ↑"),
    ("test/triple_auc_roc", "AUC-ROC ↑"),
    ("test/triple_fpr",     "FPR ↓"),
    ("test/triple_f1",      "F1 ↑"),
    ("test/triple_acc_pos", "Pos Acc ↑"),
    ("test/triple_acc_neg", "Neg Acc ↑"),
    ("test/triple_recall",  "Recall ↑"),
]

def extract_metric(text, metric_name):
    # PL rich table uses Unicode │ (U+2502); fall back to plain |
    for sep in ["│", "|"]:
        pat = (
            rf"{re.escape(sep)}\s*{re.escape(metric_name)}\s*"
            rf"{re.escape(sep)}\s*([0-9]+\.[0-9]+(?:e[+\-]?[0-9]+)?)\s*"
            rf"{re.escape(sep)}"
        )
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None

data = {}
for variant in variants:
    log_file = os.path.join(results_dir, f"{variant}.log")
    if os.path.exists(log_file):
        with open(log_file, "r", errors="replace") as fh:
            content = fh.read()
        data[variant] = {key: extract_metric(content, key) for key, _ in METRICS}
    else:
        data[variant] = {key: None for key, _ in METRICS}

def fmt_pct(val):
    return f"{val * 100:.1f}\\%" if val is not None else "N/A"

def fmt_pct_raw(val):
    return f"{val * 100:.1f}%" if val is not None else "N/A"

print("=" * 110)
print("  Exp1 Test Report — tab:overall_performance")
print("=" * 110)
print()
print(f"  {'Method':<16} {'PR-AUC ↑':<10} {'AUC-ROC ↑':<10} {'FPR ↓':<10} {'F1 ↑':<10} {'Pos Acc ↑':<10} {'Neg Acc ↑':<10} {'Recall ↑'}")
print("  " + "-" * 104)
for variant in variants:
    d   = data.get(variant, {})
    pr  = fmt_pct_raw(d.get("test/triple_pr_auc"))
    roc = f"{d.get('test/triple_auc_roc'):.3f}" if d.get("test/triple_auc_roc") is not None else "N/A"
    fpr = fmt_pct_raw(d.get("test/triple_fpr"))
    f1  = f"{d.get('test/triple_f1'):.3f}" if d.get("test/triple_f1") is not None else "N/A"
    pos = fmt_pct_raw(d.get("test/triple_acc_pos"))
    neg = fmt_pct_raw(d.get("test/triple_acc_neg"))
    rec = fmt_pct_raw(d.get("test/triple_recall"))
    print(f"  {variant:<16} {pr:<10} {roc:<10} {fpr:<10} {f1:<10} {pos:<10} {neg:<10} {rec}")
print()
print("  LaTeX rows (paste into tabular):")
print()
LABELS = {"random": "Random", "constrained": "Constrained", "proof_based": "Proof-based", "mixed": "Mixed"}
n = len(variants)
for i, variant in enumerate(variants):
    d   = data.get(variant, {})
    pr  = fmt_pct(d.get("test/triple_pr_auc"))
    roc = f"{d.get('test/triple_auc_roc'):.3f}" if d.get("test/triple_auc_roc") is not None else "N/A"
    fpr = fmt_pct(d.get("test/triple_fpr"))
    f1  = f"{d.get('test/triple_f1'):.3f}" if d.get("test/triple_f1") is not None else "N/A"
    pos = fmt_pct(d.get("test/triple_acc_pos"))
    neg = fmt_pct(d.get("test/triple_acc_neg"))
    rec = fmt_pct(d.get("test/triple_recall"))
    label = LABELS.get(variant, variant.capitalize())
    exp_col = f"\\multirow{{{n}}}{{*}}{{1}}" if i == 0 else ""
    print(f"  {exp_col:<22} & {label:<16} & {pr} & {roc} & {fpr} & {f1} & {pos} & {neg} & {rec} \\\\")
print()
print("=" * 110)
PYEOF
} | tee "${REPORT}"

echo ""
echo "Report: ${REPORT}"
echo "Logs:   ${RESULTS_DIR}/"
echo "Job finished at:"
date

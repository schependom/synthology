#!/usr/bin/env bash

### General options
#BSUB -q hpc
#BSUB -J exp2-test-all
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/exp2_test_all_%J.out
#BSUB -e logs/exp2_test_all_%J.err
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

RESULTS_DIR="${REPO_ROOT}/results/exp2_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

VARIANTS=(baseline synthology)
CONFIG_NAMES=(exp2_baseline_hpc_test exp2_synthology_hpc_test)

declare -A CKPT_PATH
declare -A TEST_STATUS

# ── Phase 1: locate latest checkpoint per variant ──────────────────────────
echo "================================================================"
echo "  Locating latest checkpoints"
echo "================================================================"
for variant in "${VARIANTS[@]}"; do
	ckpt=$(find "${REPO_ROOT}/reports/experiment_runs" -name "best-checkpoint-exp2-${variant}.ckpt" \
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

# ── Phase 3: parse metrics, print paper-ready report, update model_results.json ───
REPORT="${RESULTS_DIR}/report.txt"
{
python3 - "${RESULTS_DIR}" "${REPO_ROOT}" "${VARIANTS[@]}" << 'PYEOF'
import sys, re, os, json

results_dir = sys.argv[1]
repo_root   = sys.argv[2]
variants    = sys.argv[3:]

OVERALL_METRICS = [
    ("test/triple_pr_auc",  "PR-AUC ↑"),
    ("test/triple_auc_roc", "AUC-ROC ↑"),
    ("test/triple_fpr",     "FPR ↓"),
    ("test/triple_f1",      "F1 ↑"),
    ("test/triple_acc_pos", "Pos Acc ↑"),
    ("test/triple_acc_neg", "Neg Acc ↑"),
    ("test/triple_recall",  "Recall ↑"),
]

# hops_0=base facts, hops_1=1-hop, hops_2=2-hop, hops_3=≥3-hop (capped in RRN)
HOP_METRICS = [
    ("test/triple_acc_hops_1", "1-hop Triple Acc"),
    ("test/triple_acc_hops_2", "2-hop Triple Acc"),
    ("test/triple_acc_hops_3", "≥3-hop Triple Acc"),
    ("test/class_acc_hops_1",  "1-hop Class Acc"),
    ("test/class_acc_hops_2",  "2-hop Class Acc"),
    ("test/class_acc_hops_3",  "≥3-hop Class Acc"),
    ("test/triple_f1_hops_1",  "1-hop F1"),
    ("test/triple_f1_hops_2",  "2-hop F1"),
    ("test/triple_f1_hops_3",  "≥3-hop F1"),
    ("test/triple_fpr_hops_1", "1-hop FPR"),
    ("test/triple_fpr_hops_2", "2-hop FPR"),
    ("test/triple_fpr_hops_3", "≥3-hop FPR"),
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
            return float(m.group(1))
    return None

data_overall = {}
data_hops    = {}
for variant in variants:
    log_file = os.path.join(results_dir, f"{variant}.log")
    if os.path.exists(log_file):
        with open(log_file, "r", errors="replace") as fh:
            text = fh.read()
        data_overall[variant] = {key: extract_metric(text, key) for key, _ in OVERALL_METRICS}
        data_hops[variant]    = {key: extract_metric(text, key) for key, _ in HOP_METRICS}
    else:
        data_overall[variant] = {key: None for key, _ in OVERALL_METRICS}
        data_hops[variant]    = {key: None for key, _ in HOP_METRICS}

def fmt_pct(val):
    return f"{val * 100:.1f}\\%" if val is not None else "N/A"

def fmt_pct_raw(val):
    return f"{val * 100:.1f}%" if val is not None else "N/A"

def fmt_roc(val):
    return f"{val:.3f}" if val is not None else "N/A"

LABELS = {"baseline": "UDM", "synthology": "\\textsc{Synth}."}

# ── Overall performance table ──────────────────────────────────────────────
print("=" * 110)
print("  Exp2 Test Report — tab:exp1-rrn-performance (overall metrics)")
print("=" * 110)
print()
print(f"  {'Method':<16} {'PR-AUC ↑':<10} {'AUC-ROC ↑':<10} {'FPR ↓':<10} {'F1 ↑':<10} {'Pos Acc ↑':<10} {'Neg Acc ↑':<10} {'Recall ↑'}")
print("  " + "-" * 104)
for variant in variants:
    d = data_overall.get(variant, {})
    print(f"  {variant:<16} {fmt_pct_raw(d.get('test/triple_pr_auc')):<10} "
          f"{fmt_roc(d.get('test/triple_auc_roc')):<10} "
          f"{fmt_pct_raw(d.get('test/triple_fpr')):<10} "
          f"{fmt_roc(d.get('test/triple_f1')):<10} "
          f"{fmt_pct_raw(d.get('test/triple_acc_pos')):<10} "
          f"{fmt_pct_raw(d.get('test/triple_acc_neg')):<10} "
          f"{fmt_pct_raw(d.get('test/triple_recall'))}")
print()
print("  LaTeX rows (paste into tab:exp1-rrn-performance):")
print()
n = len(variants)
for i, variant in enumerate(variants):
    d = data_overall.get(variant, {})
    label   = LABELS.get(variant, variant.capitalize())
    exp_col = f"\\multirow{{{n}}}{{*}}{{2}}" if i == 0 else ""
    print(f"  {exp_col:<22} & {label:<20} & "
          f"{fmt_pct(d.get('test/triple_pr_auc'))} & "
          f"{fmt_roc(d.get('test/triple_auc_roc'))} & "
          f"{fmt_pct(d.get('test/triple_fpr'))} & "
          f"{fmt_roc(d.get('test/triple_f1'))} & "
          f"{fmt_pct(d.get('test/triple_acc_pos'))} & "
          f"{fmt_pct(d.get('test/triple_acc_neg'))} & "
          f"{fmt_pct(d.get('test/triple_recall'))} \\\\")
print()
print("=" * 110)

# ── Per-hop performance table (tab:per-hop-performance) ───────────────────
print()
print("=" * 80)
print("  Exp2 Per-Hop RRN Performance — tab:per-hop-performance")
print("  Rows = hop buckets (1, 2, ≥3).  hops_3 = all hops≥3 (RRN cap).")
print("=" * 80)
BUCKETS = [("1", "hops_1", "1"), ("2", "hops_2", "2"), ("≥3", "hops_3", "≥3")]
for variant in variants:
    d = data_hops.get(variant, {})
    print()
    print(f"  {variant.upper()}")
    print(f"  {'Hops':<6} {'Triple Acc':<12} {'Class Acc':<12} {'F1':<10} {'FPR'}")
    print("  " + "-" * 52)
    for label, suffix, _ in BUCKETS:
        print(f"  {label:<6} "
              f"{fmt_pct_raw(d.get(f'test/triple_acc_{suffix}')):<12} "
              f"{fmt_pct_raw(d.get(f'test/class_acc_{suffix}')):<12} "
              f"{fmt_pct_raw(d.get(f'test/triple_f1_{suffix}')):<10} "
              f"{fmt_pct_raw(d.get(f'test/triple_fpr_{suffix}'))}")
    print()
    print(f"  LaTeX rows for {variant} (tab:per-hop-performance):")
    for label, suffix, tex_label in BUCKETS:
        print(f"    {tex_label} & "
              f"{fmt_pct(d.get(f'test/triple_acc_{suffix}'))} & "
              f"{fmt_pct(d.get(f'test/class_acc_{suffix}'))} & "
              f"{fmt_pct(d.get(f'test/triple_f1_{suffix}'))} & "
              f"{fmt_pct(d.get(f'test/triple_fpr_{suffix}'))} \\\\")
print()
print("=" * 80)

# ── Write per_hop section to paper/metrics/model_results.json ─────────────
metrics_path = os.path.join(repo_root, "paper", "metrics", "model_results.json")
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
existing = {}
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as fh:
        existing = json.load(fh)

def _val(d, key):
    v = d.get(key)
    return round(v, 6) if v is not None else None

per_hop = existing.get("per_hop", {})
for variant in variants:
    d = data_hops.get(variant, {})
    per_hop[variant] = {
        "hop_1": {
            "triple_acc": _val(d, "test/triple_acc_hops_1"),
            "class_acc":  _val(d, "test/class_acc_hops_1"),
            "f1":         _val(d, "test/triple_f1_hops_1"),
            "fpr":        _val(d, "test/triple_fpr_hops_1"),
        },
        "hop_2": {
            "triple_acc": _val(d, "test/triple_acc_hops_2"),
            "class_acc":  _val(d, "test/class_acc_hops_2"),
            "f1":         _val(d, "test/triple_f1_hops_2"),
            "fpr":        _val(d, "test/triple_fpr_hops_2"),
        },
        "hop_3p": {
            "triple_acc": _val(d, "test/triple_acc_hops_3"),
            "class_acc":  _val(d, "test/class_acc_hops_3"),
            "f1":         _val(d, "test/triple_f1_hops_3"),
            "fpr":        _val(d, "test/triple_fpr_hops_3"),
        },
    }

existing["per_hop"] = per_hop
with open(metrics_path, "w", encoding="utf-8") as fh:
    json.dump(existing, fh, indent=2)
print(f"\n  Wrote per_hop metrics to: {metrics_path}")
PYEOF
} | tee "${REPORT}"

echo ""
echo "Generating paper table snippets..."
uv run invoke paper-export-tables

echo ""
echo "Report: ${REPORT}"
echo "Logs:   ${RESULTS_DIR}/"
echo "Job finished at:"
date

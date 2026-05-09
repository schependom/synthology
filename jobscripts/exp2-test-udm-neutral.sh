#!/usr/bin/env bash

### General options
#BSUB -q hpc
#BSUB -J exp2-test-udm-neutral
#BSUB -n 4
#BSUB -W 06:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/exp2_test_udm_neutral_%J.out
#BSUB -e logs/exp2_test_udm_neutral_%J.err
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

synthology_load_modules python3/3.9.19 openjdk/21
synthology_activate_python_env 0
synthology_sync_deps

# ── Phase 1: generate UDM neutral test set (skip if already done) ──────────
UDM_TEST_DIR="${REPO_ROOT}/data/exp2/udm_neutral_test/test"
if [ -f "${UDM_TEST_DIR}/facts.csv" ] && [ -f "${UDM_TEST_DIR}/targets.csv" ]; then
	echo "================================================================"
	echo "  UDM neutral test data already exists — skipping generation."
	echo "  ${UDM_TEST_DIR}"
	echo "================================================================"
else
	echo "================================================================"
	echo "  Phase 1: Generating UDM neutral test set"
	echo "  Config: exp2_udm_neutral_test (seed=99, n_test=500)"
	echo "  Output: ${UDM_TEST_DIR}"
	echo "================================================================"
	export SYNTHOLOGY_UDM_BASELINE_XMX_MB=8192
	uv run invoke exp2-generate-baseline --config-name=exp2_udm_neutral_test
	echo "  Generation complete."
fi
echo ""

RESULTS_DIR="${REPO_ROOT}/results/exp2_udm_neutral_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

VARIANTS=(baseline synthology)
CONFIG_NAMES=(exp2_baseline_udm_neutral_test exp2_synthology_udm_neutral_test)

declare -A CKPT_PATH
declare -A TEST_STATUS

# ── Phase 2: locate latest checkpoint per variant ──────────────────────────
echo "================================================================"
echo "  Phase 2: Locating latest checkpoints (exp2 models)"
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

# ── Phase 3: run each test sequentially ───────────────────────────────────
for i in "${!VARIANTS[@]}"; do
	variant="${VARIANTS[$i]}"
	config="${CONFIG_NAMES[$i]}"
	ckpt="${CKPT_PATH[$variant]}"

	if [ -z "${ckpt}" ]; then
		continue
	fi

	echo "================================================================"
	echo "  UDM Neutral Test: ${variant}"
	echo "  Checkpoint: ${ckpt}"
	echo "  Config: ${config}"
	echo "  Test set: data/exp2/udm_neutral_test/test  (Jena, seed=99)"
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

# ── Phase 4: parse metrics and print report ────────────────────────────────
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

def fmt_pct(val):
    return f"{val * 100:.1f}\\%" if val is not None else "N/A"

def fmt_pct_raw(val):
    return f"{val * 100:.1f}%" if val is not None else "N/A"

def fmt_roc(val):
    return f"{val:.3f}" if val is not None else "N/A"

LABELS = {"baseline": "UDM", "synthology": "\\textsc{Synth}."}

data = {}
for variant in variants:
    log_file = os.path.join(results_dir, f"{variant}.log")
    if os.path.exists(log_file):
        with open(log_file, "r", errors="replace") as fh:
            text = fh.read()
        data[variant] = {key: extract_metric(text, key) for key, _ in OVERALL_METRICS}
    else:
        data[variant] = {key: None for key, _ in OVERALL_METRICS}

print("=" * 110)
print("  Exp2 — UDM NEUTRAL TEST SET (Apache Jena, seed=99)")
print("  Reviewer-requested bias check: test KGs generated by Jena forward-chaining, not Synthology.")
print("=" * 110)
print()
print(f"  {'Method':<16} {'PR-AUC ↑':<10} {'AUC-ROC ↑':<10} {'FPR ↓':<10} {'F1 ↑':<10} {'Pos Acc ↑':<10} {'Neg Acc ↑':<10} {'Recall ↑'}")
print("  " + "-" * 104)
for variant in variants:
    d = data.get(variant, {})
    print(f"  {variant:<16} {fmt_pct_raw(d.get('test/triple_pr_auc')):<10} "
          f"{fmt_roc(d.get('test/triple_auc_roc')):<10} "
          f"{fmt_pct_raw(d.get('test/triple_fpr')):<10} "
          f"{fmt_roc(d.get('test/triple_f1')):<10} "
          f"{fmt_pct_raw(d.get('test/triple_acc_pos')):<10} "
          f"{fmt_pct_raw(d.get('test/triple_acc_neg')):<10} "
          f"{fmt_pct_raw(d.get('test/triple_recall'))}")
print()
print("  LaTeX rows (add as additional block in tab:exp1-rrn-performance):")
print()
n = len(variants)
for i, variant in enumerate(variants):
    d = data.get(variant, {})
    label   = LABELS.get(variant, variant.capitalize())
    exp_col = f"\\multirow{{{n}}}{{*}}{{2\\,\\dag}}" if i == 0 else ""
    print(f"  {exp_col:<28} & {label:<20} & "
          f"{fmt_pct(d.get('test/triple_pr_auc'))} & "
          f"{fmt_roc(d.get('test/triple_auc_roc'))} & "
          f"{fmt_pct(d.get('test/triple_fpr'))} & "
          f"{fmt_roc(d.get('test/triple_f1'))} & "
          f"{fmt_pct(d.get('test/triple_acc_pos'))} & "
          f"{fmt_pct(d.get('test/triple_acc_neg'))} & "
          f"{fmt_pct(d.get('test/triple_recall'))} \\\\")
print()
print("  (†) = evaluated on Jena-generated neutral test set (seed=99, different from training seed=23).")
print("=" * 110)

# Persist to paper/metrics
metrics_path = os.path.join(repo_root, "paper", "metrics", "model_results.json")
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
existing = {}
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as fh:
        existing = json.load(fh)

udm_neutral = existing.get("exp2_udm_neutral", {})
for variant in variants:
    d = data.get(variant, {})
    udm_neutral[variant] = {
        k.replace("test/triple_", "").replace("test/", ""): (round(v, 6) if v is not None else None)
        for k, _ in OVERALL_METRICS
        for _k, v in [(k, d.get(k))]
        if _k == k
    }

existing["exp2_udm_neutral"] = udm_neutral
with open(metrics_path, "w", encoding="utf-8") as fh:
    json.dump(existing, fh, indent=2)
print(f"\n  Results written to: {metrics_path}")
PYEOF
} | tee "${REPORT}"

echo ""
echo "Report: ${REPORT}"
echo "Logs:   ${RESULTS_DIR}/"
echo "Job finished at:"
date

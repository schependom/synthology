#!/usr/bin/env bash
# analyze_synthology_targets.sh
# Run from repository root after exp2-generate-synthology completes.
# Usage: bash analyze_synthology_targets.sh

set -euo pipefail

TARGETS="data/exp2/synthology/family_tree/train/targets.csv"

if [[ ! -f "$TARGETS" ]]; then
  echo "ERROR: File not found: $TARGETS"
  echo "Make sure you ran: uv run invoke exp2-generate-synthology"
  exit 1
fi

echo "========================================"
echo "  Synthology Exp2 Parity Target Audit"
echo "========================================"
echo ""

# ── 1. Header inspection ────────────────────────────────────────────────────
echo "--- Column Headers ---"
HEADER=$(head -1 "$TARGETS")
echo "$HEADER"
echo ""

# Print each column with its index for easy reference
echo "--- Column Index Map ---"
echo "$HEADER" | tr ',' '\n' | awk '{printf "  col %-3d -> %s\n", NR, $0}'
echo ""

# ── 2. Auto-detect depth column ─────────────────────────────────────────────
# Look for a column whose name contains 'depth', 'hop', or 'hops' (case-insensitive)
DEPTH_COL=$(echo "$HEADER" | tr ',' '\n' | \
  awk 'tolower($0) ~ /depth|hop/ {print NR; exit}')

if [[ -z "$DEPTH_COL" ]]; then
  echo "WARNING: No column matching 'depth' or 'hop' found in header."
  echo "Please inspect the column index map above and re-run with:"
  echo "  DEPTH_COL=<N> bash analyze_synthology_targets.sh"
  echo ""
  # Allow manual override via environment variable
  DEPTH_COL="${DEPTH_COL_OVERRIDE:-}"
  if [[ -z "$DEPTH_COL" ]]; then
    exit 1
  fi
fi

DEPTH_COL_NAME=$(echo "$HEADER" | tr ',' '\n' | sed -n "${DEPTH_COL}p")
echo "--- Depth Column Detected ---"
echo "  Column $DEPTH_COL: '$DEPTH_COL_NAME'"
echo ""

# ── 3. Basic counts ──────────────────────────────────────────────────────────
TOTAL_ROWS=$(tail -n +2 "$TARGETS" | wc -l)
echo "--- Row Counts ---"
echo "  Total targets (excl. header): $TOTAL_ROWS"
echo ""

# ── 4. Depth distribution ────────────────────────────────────────────────────
echo "--- Depth Distribution ---"
tail -n +2 "$TARGETS" | awk -F',' -v col="$DEPTH_COL" '
{
  val = $col + 0
  if      (val == 0) d0++
  else if (val == 1) d1++
  else if (val == 2) d2++
  else if (val >= 3 && val <= 4) d34++
  else if (val >= 5) d5plus++
  total++
}
END {
  printf "  depth = 0:        %7d  (%5.1f%%)\n", d0,    d0/total*100
  printf "  depth = 1:        %7d  (%5.1f%%)\n", d1,    d1/total*100
  printf "  depth = 2:        %7d  (%5.1f%%)\n", d2,    d2/total*100
  printf "  depth = 3-4:      %7d  (%5.1f%%)\n", d34,   d34/total*100
  printf "  depth >= 5:       %7d  (%5.1f%%)\n", d5plus,d5plus/total*100
  printf "  ----------------------------------------\n"
  printf "  depth >= 3 TOTAL: %7d  (%5.1f%%)\n", d34+d5plus, (d34+d5plus)/total*100
}'
echo ""

# ── 5. Compute parity target (d>=3 count) and tolerance thresholds ───────────
DEEP_COUNT=$(tail -n +2 "$TARGETS" | awk -F',' -v col="$DEPTH_COL" \
  'NR>0 && ($col+0) >= 3 {count++} END {print count+0}')

echo "--- Parity Target ---"
echo "  K_deep (d >= 3): $DEEP_COUNT"
echo ""

echo "--- Tolerance Thresholds (baseline must reach) ---"
awk -v k="$DEEP_COUNT" 'BEGIN {
  tolerances[1] = 5.0
  tolerances[2] = 10.0
  tolerances[3] = 12.5
  tolerances[4] = 15.0
  for (i = 1; i <= 4; i++) {
    t = tolerances[i]
    threshold = int(k * (1 - t/100))
    printf "  tol=%5.1f%%  ->  baseline needs >= %d deep-hop targets\n", t, threshold
  }
}'
echo ""

# ── 6. Recommendation ────────────────────────────────────────────────────────
echo "--- Recommendation ---"
awk -v k="$DEEP_COUNT" 'BEGIN {
  if      (k < 10000)  { tol="10.0";  note="Low target. Should converge quickly." }
  else if (k < 25000)  { tol="10.0";  note="Moderate target. Should converge within ~100 attempts." }
  else if (k < 40000)  { tol="12.5";  note="Borderline. Start at 12.5%, have 15% ready as fallback." }
  else                  { tol="REGEN"; note="Target too high. Re-run exp2-generate-synthology with --proof-roots-per-rule=8 first." }

  if (tol == "REGEN") {
    printf "  K_deep=%d is too high for practical convergence.\n", k
    printf "  ACTION: %s\n", note
  } else {
    printf "  Suggested parity loop command:\n\n"
    printf "    uv run invoke exp2-parity-loop \\\n"
    printf "      --deep-count-mode=tolerance \\\n"
    printf "      --tolerance-pct=%s\n\n", tol
    printf "  Reason: %s\n", note
  }
}'
echo ""
echo "========================================"
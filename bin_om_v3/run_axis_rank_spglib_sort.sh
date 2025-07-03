#!/usr/bin/env bash
# run_axis_rank_spglib_sort.sh
STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"

THETA0=10
MAXH=3
TOP=200

SCRIPT_DIR="$(dirname "$0")"
PY="${SCRIPT_DIR}/rank_axes_spglib_sort.py"

BASE="${STREAM_FILE%.*}"
CSV_OUT="${BASE}_axis_rank_maxh${MAXH}.csv"
SORTED_OUT="${BASE}_sorted_by_danger.stream"

set -euo pipefail
echo "[info] zone-axis ranking for $STREAM_FILE"
python "$PY" "$STREAM_FILE" \
       --theta0 "$THETA0"   \
       --maxh   "$MAXH"     \
       --csv    "$CSV_OUT"  \
       --top    "$TOP"      \
       --sorted-stream "$SORTED_OUT"

echo "[done] CSV   → $CSV_OUT"
echo "[done] STREAM→ $SORTED_OUT"

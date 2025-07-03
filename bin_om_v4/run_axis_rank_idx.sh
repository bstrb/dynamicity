#!/usr/bin/env bash
# run_axis_rank_idx.sh – index-weighted zone-axis ranking

STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"

THETA0=3      # angular width
ALPHA=0.10    # index-boost factor
MAXH=3        # axis depth
TOP=100       # rows printed to screen

SCRIPT_DIR="$(dirname "$0")"
PY="$SCRIPT_DIR/rank_axes_spglib_idx.py"

BASE="${STREAM_FILE%.*}"
CSV_OUT="${BASE}_axis_rank_idx.csv"
SORTED_OUT="${BASE}_sorted_by_danger_idx.stream"

set -euo pipefail
echo "[info] ranking with index weight α=$ALPHA, θ0=$THETA0, maxh=$MAXH"
python "$PY" "$STREAM_FILE" \
       --theta0 "$THETA0"   \
       --alpha  "$ALPHA"    \
       --maxh   "$MAXH"     \
       --top    "$TOP"      \
       --csv    "$CSV_OUT"  \
       --sorted-stream "$SORTED_OUT"

echo "[done] CSV   → $CSV_OUT"
echo "[done] Stream→ $SORTED_OUT"

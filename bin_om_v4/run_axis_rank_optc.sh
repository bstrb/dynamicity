#!/usr/bin/env bash
# run_axis_rank_optc.sh  – option C composite danger metric

STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"

THETA0=20      # angular width (deg)
K0=100          # index width
MAXH=3        # axis depth
TOP=100       # rows printed

SCRIPT_DIR="$(dirname "$0")"
PY="$SCRIPT_DIR/rank_axes_spglib_optc.py"

BASE="${STREAM_FILE%.*}"
CSV_OUT="${BASE}_axis_rank_optc.csv"
SORTED_OUT="${BASE}_sorted_by_danger_optc.stream"

set -euo pipefail
echo "[info] option-C ranking  θ0=$THETA0  k0=$K0  maxh=$MAXH"
python "$PY" "$STREAM_FILE" \
       --theta0 "$THETA0"  \
       --k0     "$K0"     \
       --maxh   "$MAXH"   \
       --top    "$TOP"    \
       --csv    "$CSV_OUT" \
       --sorted-stream "$SORTED_OUT"

echo "[done] CSV   → $CSV_OUT"
echo "[done] Stream→ $SORTED_OUT"

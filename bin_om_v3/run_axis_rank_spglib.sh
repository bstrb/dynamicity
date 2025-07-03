#!/usr/bin/env bash
# run_axis_rank_spglib.sh
# -----------------------
# Generic zone-axis ranking for *any* crystal system.

STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"

THETA0=5        # Gaussian width (deg)
MAXH=2          # axis depth (1=>100/110/001 only; 2 adds 101/111/210…)
TOP=50          # rows printed

SCRIPT_DIR="$(dirname "$0")"
PY_SCRIPT="${SCRIPT_DIR}/rank_axes_spglib.py"

CSV_OUT="${STREAM_FILE%.*}_axis_rank_maxh${MAXH}.csv"

set -euo pipefail
echo "[info] zone-axis ranking for $STREAM_FILE"
python "$PY_SCRIPT" "$STREAM_FILE" \
       --theta0 "$THETA0" \
       --maxh  "$MAXH"    \
       --csv   "$CSV_OUT" \
       --top   "$TOP"

echo "[done] CSV → $CSV_OUT"

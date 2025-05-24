#!/usr/bin/env bash
# run_zone_axis.sh
# ----------------
# Wrapper for misorientation_to_zone_axis.py

# ─── USER SETTINGS ────────────────────────────────────────────────────────
STREAM_FILE="/home/bubl3932/files/simulations/cP_LTA/sim_001/xgandalf_iterations_max_radius_0.0_step_0.1/LTAsim_0.0_0.0.stream"

HMAX=1          # search limit |h|≤HMAX
GMX=1.2         # reciprocal-space radius (nm-1) for crowdedness
TOP=20           # keep TOP most crowded axes
THETA0=5.0      # θ0 (deg) in danger formula
# -------------------------------------------------------------------------

SCRIPT_DIR="$(dirname "$0")"
PY_SCRIPT="${SCRIPT_DIR}/misorientation_to_zone_axis.py"

BASE="${STREAM_FILE%.*}"
CSV_OUT="${BASE}_zone_axes_hmax_${HMAX}.csv"
PLOT_OUT="${BASE}_zone_axes_hmax_${HMAX}.png"
SORTED_OUT="${BASE}_sorted_${HMAX}.stream"

set -euo pipefail

echo "[info] Processing $STREAM_FILE"
python "$PY_SCRIPT" "$STREAM_FILE"          \
       --hmax "$HMAX"                      \
       --gmax "$GMX"                       \
       --top-crowded "$TOP"                \
       --theta0 "$THETA0"                  \
       --csv "$CSV_OUT"                    \
       --plot "$PLOT_OUT"                  \
       --sorted-stream "$SORTED_OUT"

echo "[done]  CSV   : $CSV_OUT"
echo "[done]  Plot  : $PLOT_OUT"
echo "[done]  Stream: $SORTED_OUT"

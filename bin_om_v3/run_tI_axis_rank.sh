#!/usr/bin/env bash
# run_tI_axis_rank.sh – call rank_tI_axes.py

STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"

THETA0=5        # Gaussian width (deg)
TOP=40          # rows printed to screen

SCRIPT_DIR="$(dirname "$0")"
PY="${SCRIPT_DIR}/rank_tI_axes.py"

CSV_OUT="${STREAM_FILE%.*}_tI_axis_rank.csv"

set -euo pipefail
echo "[info] ranking tI zone-axis danger for $STREAM_FILE"
python "$PY" "$STREAM_FILE" \
       --theta0 "$THETA0"   \
       --csv "$CSV_OUT"     \
       --top "$TOP"
echo "[done] CSV  → $CSV_OUT"

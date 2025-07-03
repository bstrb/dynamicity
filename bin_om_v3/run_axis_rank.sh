#!/usr/bin/env bash
# run_axis_rank.sh
# ----------------
# Wrapper for rank_problematic_axes.py
# (automatic zone-axis ranking from orientation matrices)

# ─── USER SETTINGS ────────────────────────────────────────────────────────
STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"

THETA0=5          # Gaussian width θ₀ (deg) in danger score
TOP=40            # print TOP rows to console (CSV is always full)
# -------------------------------------------------------------------------

SCRIPT_DIR="$(dirname "$0")"
PY_SCRIPT="${SCRIPT_DIR}/rank_problematic_axes.py"

BASE="${STREAM_FILE%.*}"
CSV_OUT="${BASE}_axis_rank.csv"

set -euo pipefail

echo "[info] Ranking zone-axis danger for $STREAM_FILE"
python "$PY_SCRIPT" "$STREAM_FILE" \
       --theta0 "$THETA0" \
       --csv   "$CSV_OUT" \
       --top   "$TOP"

echo "[done] CSV written to: $CSV_OUT"

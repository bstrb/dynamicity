#!/usr/bin/env bash
# run_zone_axis_v2.sh
# -------------------
# Wrapper for misorientation_to_zone_axis_v2.py
#
# New in v2:
#   •  --crowded-every N   (re-compute crowded axes every N patterns)
#   •  --sf FILE           (MTZ/CIF with |F|; activates structure-factor weighting)
#   •  --sf-col COL        (column name for |F|; default F)

# ─── USER SETTINGS ────────────────────────────────────────────────────────
STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"

HMAX=6          # search limit |h|≤HMAX
GMX=0.6            # reciprocal-space radius (nm⁻¹) for crowdedness
TOP=100         # keep TOP most crowded axes
THETA0=5.0      # θ0 (deg) in danger formula

CROWDED_EVERY=5  # recompute crowded axes every N patterns (1 = every pattern)

# Optional structure-factor file (comment out if you don't want SF weighting)
#SF_FILE="/path/to/your/amplitudes.mtz"   # accepts .mtz or .cif
#SF_COL="F"                               # column holding |F| (or I½)

# -------------------------------------------------------------------------

SCRIPT_DIR="$(dirname "$0")"
PY_SCRIPT="${SCRIPT_DIR}/misorientation_to_zone_axis_v2.py"

BASE="${STREAM_FILE%.*}"
CSV_OUT="${BASE}_zone_axes_hmax_${HMAX}.csv"
PLOT_OUT="${BASE}_zone_axes_hmax_${HMAX}.png"
SORTED_OUT="${BASE}_sorted_${HMAX}.stream"

set -euo pipefail

echo "[info] Processing $STREAM_FILE"
python "$PY_SCRIPT" "$STREAM_FILE"               \
       --hmax "$HMAX"                            \
       --gmax "$GMX"                             \
       --top-crowded "$TOP"                      \
       --theta0 "$THETA0"                        \
       --crowded-every "$CROWDED_EVERY"          \
${SF_FILE:+  --sf "$SF_FILE"}                    \
${SF_COL:+  --sf-col "$SF_COL"}                  \
       --csv "$CSV_OUT"                          \
       --plot "$PLOT_OUT"                        \
       --sorted-stream "$SORTED_OUT"

echo "[done]  CSV   : $CSV_OUT"
echo "[done]  Plot  : $PLOT_OUT"
echo "[done]  Stream: $SORTED_OUT"

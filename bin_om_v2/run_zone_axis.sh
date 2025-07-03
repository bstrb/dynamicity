#!/usr/bin/env bash
# run_zone_axis.sh
# ----------------
# Run the zone-axis deviation calculator on one CrystFEL stream file and
# produce (i) a CSV, (ii) a θ-rank plot and (iii) a sorted *.stream* copy.

# ---- user-editable variables -----------------------------------------------
STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"
# STREAM_FILE="/home/bubl3932/files/MFM300_VIII/MFM_spot3_streams/filtered_metrics/filtered_metrics.stream"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"      # folder that holds this script
STREAM_DIR="$(dirname "${STREAM_FILE}")"         # folder that holds the stream
PY_SCRIPT="${SCRIPT_DIR}/misorientation_to_zone_axis.py"

HMAX=5        # search [uvw] from –HMAX…+HMAX

CSV_OUT="${STREAM_DIR}/$(basename "${STREAM_FILE%.*}")_zone_axes_hmax_${HMAX}.csv"
PLOT_OUT="${CSV_OUT%.csv}.png"
SORTED_STREAM_OUT="${STREAM_DIR}/$(basename "${STREAM_FILE%.*}")_sorted_${HMAX}.stream"

# ---------------------------------------------------------------------------

# Abort on errors or unset vars
set -euo pipefail

echo "[info] Processing ${STREAM_FILE}"
python "${PY_SCRIPT}" "${STREAM_FILE}" \
       --hmax "${HMAX}" \
       --csv  "${CSV_OUT}" \
       --plot "${PLOT_OUT}" \
       --sorted-stream "${SORTED_STREAM_OUT}"

echo "[done]  CSV:    ${CSV_OUT}"
echo "[done]  Plot:   ${PLOT_OUT}"
echo "[done]  Stream: ${SORTED_STREAM_OUT}"

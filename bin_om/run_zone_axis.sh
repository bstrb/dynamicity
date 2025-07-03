#!/usr/bin/env bash
# run_zone_axis.sh
# ----------------
# Run the zone-axis deviation calculator on one CrystFEL stream file.

# ---- user-editable variables ----------------------------------------------
# STREAM_FILE="/Users/xiaodong/Desktop/MFM300-VIII/MFM300_VIII_spot9/xgandalf_iterations_max_radius_0.5_step_0.1/MFM300_0.0_0.0.stream"
STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"          # folder that holds the python script
STREAM_DIR="$(dirname "${STREAM_FILE}")"          # folder that holds the stream file
PY_SCRIPT="${SCRIPT_DIR}/misorientation_to_zone_axis.py"
CSV_OUT="${STREAM_DIR}/$(basename "${STREAM_FILE%.*}").zone_axes.csv"
HMAX=3        # search [uvw] from –HMAX…+HMAX
# ---------------------------------------------------------------------------

# Make sure errors abort the script
set -euo pipefail

echo "[info] Activating conda env 'bkind'…"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bkind

echo "[info] Processing ${STREAM_FILE}"
python "${PY_SCRIPT}" "${STREAM_FILE}" \
       --hmax "${HMAX}" \
       --csv  "${CSV_OUT}" \
       --plot "${CSV_OUT%.csv}.png"


echo "[done]  CSV written to ${CSV_OUT}"

#!/usr/bin/env bash
# run_zone_axis.sh
# ----------------
# Run the zone-axis deviation calculator on one CrystFEL stream file.

# ---- user-editable variables ----------------------------------------------
STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/from_file_-512.5_-512.5.stream"
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

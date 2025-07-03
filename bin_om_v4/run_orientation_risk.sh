#!/usr/bin/env bash
# run_orientation_risk.sh  – rank orientations by dynamical-scattering risk
# Adapted for orientation_scoring_stream.py

# ---------------------------------------------------------------------------
# INPUTS  (edit as needed)
# ---------------------------------------------------------------------------

# Exact stream file you pasted earlier
# STREAM_FILE="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_iterations_max_radius_0.0_step_0.1/mfm300sim_0.0_0.0.stream"
STREAM_FILE="/Users/xiaodong/Desktop/simulations/LTA_cP/sim_001/xgandalf_iterations_max_radius_3.0_step_0.5/LTA_sim_0.0_0.0.stream"
# Scoring parameters
RESO=0.20     # d_min in Å
THICK=50     # crystal thickness in nm
STEP=5        # orientation-grid step in degrees
TOP=100        # rows printed (worst-TOP orientations)

# Optional: override default electron wavelength (Å) if not 200 kV
WL=0.019687

# ---------------------------------------------------------------------------
# DERIVED PATHS / SCRIPT INVOCATION
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(dirname "$0")"
PY="$SCRIPT_DIR/orientation_scoring_stream.py"

BASE="${STREAM_FILE%.*}"
TXT_OUT="${BASE}_orientation_risk.txt"

set -euo pipefail

echo "[info] Orientation scoring  d_min=${RESO}Å  t=${THICK} nm  step=${STEP}°"
python "$PY" "$STREAM_FILE" \
       --resolution "$RESO"  \
       --thickness  "$THICK" \
       --step       "$STEP"  \
       --top        "$TOP"   \
       --wavelength "$WL"   # uncomment if you set WL above \
    # | tee "$TXT_OUT"

echo "[done] Results → $TXT_OUT"

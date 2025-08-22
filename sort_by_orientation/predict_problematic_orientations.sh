#!/usr/bin/env bash
set -euo pipefail

python3 predict_problematic_axes_numba.py \
  --stream "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.0_step_0.1/MFM300_0.0_0.0.stream" \
  --ring-mult-min 2 \
  --nrows 50 \
  --csv 

  # --uvw-max 15 \
  # --g-max 3 \
  # --g-enum-bound 1.1 \
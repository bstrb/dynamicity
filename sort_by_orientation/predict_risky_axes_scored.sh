#!/usr/bin/env bash
set -euo pipefail

python3 predict_risky_axes_scored.py \
  --stream "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/zero_fitering/filtered_metrics.stream" \
  --nrows 20 \
  --thickness-nm 20 \
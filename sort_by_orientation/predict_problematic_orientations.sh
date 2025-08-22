#!/usr/bin/env bash
set -euo pipefail

# python3 predict_problematic_axes_numba.py \
#   --stream "/home/bubl3932/files/UOX1/xgandalf_iterations_max_radius_1.8_step_0.5/filtered_metrics/filtered_metrics.stream" \
#   --uvw-max 3 \
#   --n-min 1616 \
#   --m-min 885 \
#   --progress \
#   --csv 
 
# python3 predict_problematic_axes_numba.py \
#   --stream "/home/bubl3932/files/simulations/aP_C23H22O5/sim_000/1515178.stream" \
#   --uvw-max 5 \
#   --ring-mult-min 2 \
#   --g-max 0.8 \
#   --n-min 38 \
#   --m-min 885 \
#   --i-min-rel 0.0 \
#   --tol-g 5e-4 \
#   --progress \
#   --csv 
#   # --g-enum-bound 1.1 \

python3 predict_problematic_axes_numba.py \
  --stream "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/MFM300.stream" \
  --uvw-max 5 \
  --ring-mult-min 2 \
  --g-max 3 \
  --g-enum-bound 1.1 \
  --n-min 122 \
  --m-min 0 \
  --progress \
  --csv 
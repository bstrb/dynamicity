#!/usr/bin/env bash
set -euo pipefail

python3 predict_problematic_orientations.py \
  --cell "/home/bubl3932/files/simulations/cP_LTA/sim_000/7108314.cell" \
  --geom "/home/bubl3932/files/simulations/cP_LTA/sim_000/7108314.geom" \
  --uvw-max 5 \
  --ring-mult-min 2 \
  --g-max 0.9 \
  --g-enum-bound 1.1 \
  --n-min 200 \
  --m-min 20 \
  --i-min-rel 0.0 \
  --tol-g 5e-4 \
  --csv "/home/bubl3932/files/simulations/cP_LTA/sim_000/problematic_orientations.csv" \
  # --listuvw \

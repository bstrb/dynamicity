#!/usr/bin/env bash
set -euo pipefail

# /Users/xiaodong/Desktop/simulations/LTA_cP/sim_000/7108314.cell
# /Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/4135627.cell
# /Users/xiaodong/Desktop/simulations/RbS2Sb_aP/sim_000/1530127.cell

python3 predict_problematic_orientations.py \
  --cell "/Users/xiaodong/Desktop/simulations/RbS2Sb_aP/sim_000/1530127.cell" \
  --geom "/Users/xiaodong/Desktop/simulations/RbS2Sb_aP/sim_000/1530127.geom" \
  --uvw-max 5 \
  --ring-mult-min 2 \
  --g-max 0.9 \
  --g-enum-bound 1.1 \
  --n-min 40 \
  --m-min 20 \
  --i-min-rel 0.0 \
  --tol-g 5e-4 \
  --listuvw
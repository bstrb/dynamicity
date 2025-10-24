#!/usr/bin/env bash
set -euo pipefail

python3 predict_risky_axes_scored.py \
  --stream "/home/bubl3932/files/LauraPacoste_Dynamical-filtering-Buster/no-refine/ReOx-M.stream" \
  --nrows 150 \
  --thickness-nm 20 \
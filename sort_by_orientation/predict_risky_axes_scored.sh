#!/usr/bin/env bash
set -euo pipefail

python3 predict_risky_axes_scored.py \
  --stream "/home/bubl3932/files/COF300/COF300.stream" \
  --nrows 150 \
  --thickness-nm 20 \
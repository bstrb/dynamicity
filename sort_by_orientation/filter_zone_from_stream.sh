#!/bin/sh
IN="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_run/MFM300VIII_sim_sorted.stream"
ZONE_LIST="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_run/MFM300VIII_sim_report.txt"
NUM_CRYSTALS=50
TOL=0.1
OUT="/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_000/xgandalf_run/MFM300VIII_sim_sorted_first${NUM_CRYSTALS}_noZOLZ_tol${TOL}.stream"

python3 filter_zone_from_stream.py \
  --in "$IN" \
  --out "$OUT" \
  --limit-crystals "$NUM_CRYSTALS" \
  --tolerance "$TOL" \
  --zone-list "$ZONE_LIST"

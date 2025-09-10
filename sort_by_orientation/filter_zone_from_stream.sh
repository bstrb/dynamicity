#!/bin/sh
IN="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/xgandalf_iterations_max_radius_0.5_step_0.2/metrics_run_20250905-084938/filtered_metrics_sorted.stream"
ZONE_LIST="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/xgandalf_iterations_max_radius_0.5_step_0.2/metrics_run_20250905-084938/filtered_metrics_report.txt"
NUM_CRYSTALS=1000
TOL=0.1
IN_BASE=$(basename "$IN" .stream)
OUT="$(dirname "$IN")/${IN_BASE}_${NUM_CRYSTALS}_first_xtals_noZOLZ_tol${TOL}.stream"


python3 filter_zone_from_stream.py \
  --in "$IN" \
  --out "$OUT" \
  --limit-crystals "$NUM_CRYSTALS" \
  --tolerance "$TOL" \
  --zone-list "$ZONE_LIST"

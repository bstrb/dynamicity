#!/usr/bin/env bash
# run_qc.sh — minimal hard-coded wrapper

DIR="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/xgandalf_iterations_max_radius_1.0_step_0.1_old/filtered_metrics_for_poster/filtered_metrics_merge"
CELL="$DIR/cell.cell"
# CELL="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/MFM.cell"
SYM="4/mmm"
OUTDIR="qc_stats"     # change if you like
LOWRES=4
HIGHRES=0.4
WILSON="--wilson"     # set to "" to skip

python3 "$(dirname "$0")/qc_report.py" \
  --dir "$DIR" \
  --cell "$CELL" --symmetry "$SYM" \
  --outdir "$OUTDIR" --lowres "$LOWRES" --highres "$HIGHRES" \
  $WILSON

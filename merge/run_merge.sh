#!/bin/sh
set -eu

# Input stream file
STREAM="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/xgandalf_iterations_max_radius_1.0_step_0.1_old/filtered_metrics_for_poster/filtered_metrics.stream"

# Make an output directory based on the stream's stem
STEM="${STREAM%.stream}"
OUTDIR="${STEM}_merge"
SYM="4/mmm"  # Change as needed
ITERATIONS=5   # Change as needed


# Ensure output dirs exist
mkdir -p "$OUTDIR" 

python3 stream_to_cell.py \
  --stream "$STREAM" \
  --outdir "$OUTDIR"

# Run partialator
partialator \
  "$STREAM" \
  --model=offset \
  -j 24 \
  -o "$OUTDIR/crystfel.hkl" \
  -y "$SYM" \
  --polarisation=none \
  --min-measurements=2 \
  --max-adu=inf \
  --min-res=inf \
  --push-res=inf \
  --no-Bscale \
  --no-pr \
  --no-logs \
  --iterations="$ITERATIONS" \
  --harvest-file="$OUTDIR/parameters.json" \
  --log-folder="$OUTDIR/pr-logs" \
  >"$OUTDIR/stdout.log" \
  2>"$OUTDIR/stderr.log"

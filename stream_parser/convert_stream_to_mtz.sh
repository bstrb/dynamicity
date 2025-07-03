#!/usr/bin/env bash

# Path to your CrystFEL stream file
STREAM="/Users/xiaodong/Desktop/dynamicity/bin_om_v2/mfm300sim_0.0_0.0.stream"

# Number of frames per MTZ chunk
CHUNK_SIZE=50

# # Output directory (will be created if it doesn't exist)
# OUTDIR="/path/to/mtz_output"

# Invoke the converter
python3 convert_stream_to_mtz.py \
  --stream "$STREAM" \
  --chunk-size "$CHUNK_SIZE" #\
  # --out-dir "$OUTDIR"

# echo "All done – MTZ files are in $OUTDIR"

#!/bin/bash
set -euo pipefail

# --- USER SETTINGS ---

IN="/home/bubl3932/files/CAU23/crystfel/CAU23_sorted.stream"
ZONE_LIST="/home/bubl3932/files/CAU23/crystfel/CAU23_report.txt"
NUM_CRYSTALS=all   # "all" or an integer like 0, 1, 200
TOL=0.1            # tolerance, e.g. 0, 0.1, 1.5
# ---------------------

# Basic checks
[ -f "$IN" ] || { echo "Input .stream not found: $IN" >&2; exit 1; }
[ -f "$ZONE_LIST" ] || { echo "Zone list not found: $ZONE_LIST" >&2; exit 1; }
[ -s "$ZONE_LIST" ] || { echo "Zone list file is empty: $ZONE_LIST" >&2; exit 1; }

IN_DIR=$(dirname "$IN")
IN_BASE=$(basename "$IN" .stream)

# Validate NUM_CRYSTALS
if [ "$NUM_CRYSTALS" != "all" ]; then
  case "$NUM_CRYSTALS" in
    ''|*[!0-9]*)
      echo "NUM_CRYSTALS must be 'all' or a non-negative integer. Got: '$NUM_CRYSTALS'" >&2
      exit 1
      ;;
  esac
fi

if [ "$NUM_CRYSTALS" = "all" ]; then
  OUT="${IN_DIR}/${IN_BASE}_noZOLZ_tol${TOL}.stream"
  set -- python3 filter_zone_from_stream.py \
    --in "$IN" \
    --out "$OUT" \
    --tolerance "$TOL" \
    --zone-list "$ZONE_LIST"
else
  OUT="${IN_DIR}/${IN_BASE}_${NUM_CRYSTALS}_first_xtals_noZOLZ_tol${TOL}.stream"
  set -- python3 filter_zone_from_stream.py \
    --in "$IN" \
    --out "$OUT" \
    --limit-crystals "$NUM_CRYSTALS" \
    --tolerance "$TOL" \
    --zone-list "$ZONE_LIST"
fi

echo "Running: $*"
time "$@"

echo "Done."
echo "Output: $OUT"


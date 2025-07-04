#!/usr/bin/env bash
# run_crowded_axes.sh
# -------------------
# Quick wrapper for crowded_axes.py
# ─── USER SETTINGS ────────────────────────────────────────────────────────
SYSTEM="tetragonal"  # system name, used for output file names
CENTERING="I"
HMAX=2
TOP=50        # report top-10 axes
# Optionally give metric and |g| cut-off
GMX=1         # nm^-1   (comment out to disable)
CELL="15 15 12 90 90 90"   # Å Å Å ° ° °  (comment out to disable)
# -------------------------------------------------------------------------

SCRIPT_DIR="$(dirname "$0")"
PY="${SCRIPT_DIR}/crowded_axes.py"

CMD=(python "$PY" --system "$SYSTEM" --centering "$CENTERING"
     --hmax "$HMAX" --top "$TOP")

if [[ -n "${GMX:-}" ]]; then
  CMD+=(--gmax "$GMX")
fi
if [[ -n "${CELL:-}" ]]; then
  CMD+=(--cell $CELL)
fi

echo "[info] ${CMD[*]}"
"${CMD[@]}"

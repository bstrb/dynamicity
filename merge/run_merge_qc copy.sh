#!/usr/bin/env bash
# Robust, reproducible merge + QC pipeline for a CrystFEL stream
# - Preserves your existing defaults and hardcoded paths
# - Adds validation, better logging, and clearer variable names
# - Avoids conflicting flags; fail-fast with useful error messages

set -euo pipefail

#######################################
# Defaults (kept exactly as you had them)
#######################################
STREAM="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.0_step_0.1/MFM300VIII_512_0.0_0.0_sorted_part1.stream"
THREADS=24
SYM="4/mmm"       # Change as needed
ITERATIONS=5      # Change as needed

# QC defaults (kept)
LOWRES=4
HIGHRES=0.4
WILSON="--wilson"   # set to "" to skip

#######################################
# Derived paths
#######################################
STEM="${STREAM%.stream}"
MERGE_OUTDIR="${STEM}_merge"
PR_LOG_DIR="${MERGE_OUTDIR}/pr-logs"
CRYSTFEL_HKL="${MERGE_OUTDIR}/crystfel.hkl"
PARAMS_JSON="${MERGE_OUTDIR}/parameters.json"
STDOUT_LOG="${MERGE_OUTDIR}/partialator_stdout.log"
STDERR_LOG="${MERGE_OUTDIR}/partialator_stderr.log"
METADATA_LOG="${MERGE_OUTDIR}/metadata_and_outputs.txt"

#######################################
# qc_report inputs/outputs
#######################################
QC_DIR="${MERGE_OUTDIR}"        # directory containing outputs to QC
CELL_FILE="${QC_DIR}/cell.cell" # expected output of stream_to_cell.py
QC_OUTDIR="${MERGE_OUTDIR}/qc_stats"  # keep QC under merge dir to keep things together

#######################################
# Create output dirs BEFORE starting global tee
#######################################
mkdir -p "$MERGE_OUTDIR" "$PR_LOG_DIR" "$QC_OUTDIR"

#######################################
# Mirror all stdout + stderr to terminal and metadata_and_outputs.txt
# (partialator itself will be redirected away from terminal below)
#######################################
exec > >(tee -a "$METADATA_LOG") 2>&1

#######################################
# Helpers
#######################################
die() { echo "ERROR: $*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found in PATH"
}

timestamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }

#######################################
# Basic checks
#######################################
[ -f "$STREAM" ] || die "Stream file not found: $STREAM"
need python3
need partialator

#######################################
# Optional: record versions for reproducibility (best effort)
#######################################
CRYSTFEL_VER="$(partialator --version 2>/dev/null || true)"
PY_VER="$(python3 -c 'import sys; print(sys.version.replace("\n"," "))' 2>/dev/null || true)"

#######################################
# Log metadata (truncate file to start clean, despite global tee)
#######################################
{
  echo "Run started: $(timestamp)"
  echo "Host: $(hostname || echo unknown)"
  echo "CWD: $(pwd)"
  echo "STREAM: $STREAM"
  echo "THREADS: $THREADS"
  echo "SYM: $SYM"
  echo "ITERATIONS: $ITERATIONS"
  echo "LOWRES: $LOWRES"
  echo "HIGHRES: $HIGHRES"
  echo "WILSON: $WILSON"
  echo "Merge Outdir: $MERGE_OUTDIR"
  echo "QC Outdir: $QC_OUTDIR"
  echo "CrystFEL: ${CRYSTFEL_VER:-unknown}"
  echo "Python: ${PY_VER:-unknown}"
  echo "PATH: $PATH"
} > "$METADATA_LOG"

#######################################
# On error, point to logs
#######################################
trap 'echo "Failure at $(timestamp). See logs in: $MERGE_OUTDIR" >&2' ERR

#######################################
# 1) Convert stream to cell (as in your script)
#######################################
python3 stream_to_cell.py \
  --stream "$STREAM" \
  --outdir "$MERGE_OUTDIR"

# Sanity-check the expected cell file
if [ ! -s "$CELL_FILE" ]; then
  echo "WARNING: Expected cell file not found or empty: $CELL_FILE" | tee -a "$METADATA_LOG"
fi

#######################################
# 2) Run partialator (no terminal output)
#    - Use array to avoid line-continuation issues
#    - Redirect stdout/stderr ONLY to per-run logs
#######################################
echo "[$(timestamp)] Running partialator... (output suppressed; see $STDOUT_LOG and $STDERR_LOG)"

partialator_args=(
  "$STREAM"
  --model=offset
  -j "$THREADS"
  -o "$CRYSTFEL_HKL"
  -y "$SYM"
  --polarisation=none
  --min-measurements=2
  --max-adu=inf
  --min-res=inf
  --push-res=inf
  --no-Bscale
  --no-pr
  --iterations="$ITERATIONS"
  --harvest-file="$PARAMS_JSON"
  --log-folder="$PR_LOG_DIR"
)

# Redirect raw tool output away from terminal
partialator "${partialator_args[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG"

echo "[$(timestamp)] partialator finished."

#######################################
# Quick post-checks
#######################################
[ -s "$CRYSTFEL_HKL" ] || echo "WARNING: Expected merged HKL not found or empty: $CRYSTFEL_HKL" | tee -a "$METADATA_LOG"
[ -s "$PARAMS_JSON" ] || echo "WARNING: parameters.json missing/empty: $PARAMS_JSON" | tee -a "$METADATA_LOG"

#######################################
# 3) QC report
#######################################
python3 "$(dirname "$0")/qc_report.py" \
  --dir "$QC_DIR" \
  --cell "$CELL_FILE" --symmetry "$SYM" \
  --outdir "$QC_OUTDIR" --lowres "$LOWRES" --highres "$HIGHRES" \
  ${WILSON:+$WILSON}

echo "Run completed: $(timestamp)"

#!/usr/bin/env bash
# Robust, reproducible merge + QC pipeline for multiple CrystFEL streams
# - Hardcoded list of .stream files
# - Per-stream outputs in <stream>_merge dirs
# - partialator output suppressed from terminal

set -euo pipefail

#######################################
# Hardcoded list of stream files
#######################################
STREAMS=(
  "/home/bubl3932/files/Zheting/1DNEA3D_pHASE1_5_0.0_0.0_TOLERANCE3_origcell_origgeom_sorted_noZOLZ_tol0.1.stream"
)

#######################################
# Defaults
#######################################

THREADS=24
# SYM="4/mmm"
SYM="mmm"
ITERATIONS=5

LOWRES=4
HIGHRES=0.6
WILSON="--wilson"   # set to "" to skip

#######################################
# Helpers
#######################################
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found in PATH"; }
timestamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }

need python3
need partialator

#######################################
# Function: process a single stream file
#######################################
process_stream() (
  set -euo pipefail
  local STREAM_FILE="$1"
  [ -f "$STREAM_FILE" ] || die "Stream file not found: $STREAM_FILE"

  local STEM="${STREAM_FILE%.stream}"
  local MERGE_OUTDIR="${STEM}_merge_res_${LOWRES}-${HIGHRES}"
  local PR_LOG_DIR="${MERGE_OUTDIR}/pr-logs"
  local CRYSTFEL_HKL="${MERGE_OUTDIR}/crystfel.hkl"
  local PARAMS_JSON="${MERGE_OUTDIR}/parameters.json"
  local STDOUT_LOG="${MERGE_OUTDIR}/partialator_stdout.log"
  local STDERR_LOG="${MERGE_OUTDIR}/partialator_stderr.log"
  local METADATA_LOG="${MERGE_OUTDIR}/metadata_and_outputs.txt"
  local QC_DIR="${MERGE_OUTDIR}"
  local CELL_FILE="${QC_DIR}/cell.cell"
  local QC_OUTDIR="${MERGE_OUTDIR}/qc_stats"

  mkdir -p "$MERGE_OUTDIR" "$QC_OUTDIR" #"$PR_LOG_DIR"

  # Mirror messages for this stream into its metadata log
  exec > >(tee -a "$METADATA_LOG") 2>&1

  CRYSTFEL_VER="$(partialator --version 2>/dev/null || true)"
  PY_VER="$(python3 -c 'import sys; print(sys.version.replace("\n"," "))' 2>/dev/null || true)"

  {
    echo "Run started: $(timestamp)"
    echo "STREAM: $STREAM_FILE"
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
  } > "$METADATA_LOG"

  trap 'echo "Failure at $(timestamp). See logs in: $MERGE_OUTDIR" >&2' ERR

  echo "[$(timestamp)] stream_to_cell.py: start"
  python3 stream_to_cell.py --stream "$STREAM_FILE" --outdir "$MERGE_OUTDIR"
  echo "[$(timestamp)] stream_to_cell.py: done"

  if [ ! -s "$CELL_FILE" ]; then
    echo "WARNING: cell file missing: $CELL_FILE" | tee -a "$METADATA_LOG"
  fi

  echo "[$(timestamp)] Running partialator... (output suppressed)"
  partialator_args=(
    "$STREAM_FILE"
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
  partialator "${partialator_args[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG"
  echo "[$(timestamp)] partialator finished."

  WARN=0
  [ -s "$CRYSTFEL_HKL" ] || { echo "WARNING: HKL missing: $CRYSTFEL_HKL" | tee -a "$METADATA_LOG"; WARN=1; }
  [ -s "$PARAMS_JSON" ] || { echo "WARNING: parameters.json missing: $PARAMS_JSON" | tee -a "$METADATA_LOG"; WARN=1; }

  echo "[$(timestamp)] qc_report.py: start"
  python3 "$(dirname "$0")/qc_report.py" \
    --dir "$QC_DIR" --cell "$CELL_FILE" --symmetry "$SYM" \
    --outdir "$QC_OUTDIR" --lowres "$LOWRES" --highres "$HIGHRES" \
    ${WILSON:+$WILSON}
  echo "[$(timestamp)] qc_report.py: done"

  echo "[$(timestamp)] convert_hkl_crystfel_to_shelx.py: start"
  python3 "$(dirname "$0")/convert_hkl_crystfel_to_shelx.py" \
    --input-dir "$MERGE_OUTDIR"
  echo "[$(timestamp)] convert_hkl_crystfel_to_shelx.py: done"

  echo "Run completed: $(timestamp)"
  return "$WARN"
)

#######################################
# Run all hardcoded streams
#######################################
EXITCODE=0
for s in "${STREAMS[@]}"; do
  echo "=== Processing stream: $s ==="
  if process_stream "$s"; then
    echo "[OK] $s" | tee -a "$SUMMARY_FILE"
  else
    echo "[WARN] $s" | tee -a "$SUMMARY_FILE"
    EXITCODE=1
  fi
done

echo >> "$SUMMARY_FILE"
echo "Multi-stream run finished: $(timestamp)" >> "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE"

exit "$EXITCODE"

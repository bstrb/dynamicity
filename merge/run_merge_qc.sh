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
  "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/enh_feed_weak_positive_poc_scaling_20260622_strict_with_resolution/MFM300-VIII_cut_20-0_3_enh_feed_weak_positive_poc.stream"
  "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300-VIII_cut_20-0_3.stream"
)
#COF300
#--model=offset -y 4/m --iterations=25 --polarisation=none --min-measurements=2 --no-Bscale -j 8
#######################################
# Defaults
#######################################

THREADS=24
SYM="4/mmm"
ITERATIONS=1
MIN_MEASUREMENTS=1
PUSH_RES=inf
MODEL="offset"   # partialator model to use (e.g. "unity", "offset", "scale", "scale+offset")
OUTPUT_UNMERGED=false   # whether to output unmerged reflections (for manual merging downstream)
DISABLE_PR=true   # whether to disable partialator's post-refinement (PR) step, which can be unstable for some datasets and is not required for a basic QC report

# LOWRES & HIGHRES only used for qc report and is not a resolution cutoff & Wilson scaling can be unstable for some datasets, so it's optional
LOWRES=20.0
HIGHRES=0.4
WILSON=""   # set to "" to skip

#######################################
# Helpers
#######################################
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found in PATH"; }
timestamp() { date +"%Y-%m-%dT%H:%M"; }
timestamp_fs() { date +"%Y%m%dT%H%M"; }

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
  local RUN_ID
  RUN_ID="$(timestamp_fs)"
  local RUN_STARTED
  RUN_STARTED="$(timestamp)"
  local MERGE_OUTDIR="${STEM}_partialator_results_${RUN_ID}"
  local PR_LOG_DIR="${MERGE_OUTDIR}/pr-logs"
  local CRYSTFEL_HKL="${MERGE_OUTDIR}/crystfel.hkl"
  local PARAMS_JSON="${MERGE_OUTDIR}/parameters.json"
  local STDOUT_LOG="${MERGE_OUTDIR}/partialator_stdout.log"
  local STDERR_LOG="${MERGE_OUTDIR}/partialator_stderr.log"
  local METADATA_LOG="${MERGE_OUTDIR}/metadata_and_outputs.txt"
  local QC_DIR="${MERGE_OUTDIR}"
  local CELL_FILE="${QC_DIR}/cell.cell"
  local QC_OUTDIR="${MERGE_OUTDIR}/qc_stats"
  local UNMERGED_HKL="${MERGE_OUTDIR}/unmerged.hkl"
  local PARTIALATOR_POLARISATION="none"
  local PARTIALATOR_MAX_ADU="inf"
  local PARTIALATOR_MIN_RES="inf"
  local PARTIALATOR_NO_BSCALE=true
  local OUTPUT_UNMERGED_NORMALIZED
  local DISABLE_PR_NORMALIZED
  local PARTIALATOR_ARGS_SERIALIZED=""

  mkdir -p "$MERGE_OUTDIR" "$QC_OUTDIR" #"$PR_LOG_DIR"

  # Mirror messages for this stream into its metadata log
  exec > >(tee -a "$METADATA_LOG") 2>&1

  CRYSTFEL_VER="$(partialator --version 2>/dev/null || true)"
  PY_VER="$(python3 -c 'import sys; print(sys.version.replace("\n"," "))' 2>/dev/null || true)"

  {
    echo "Run started: $RUN_STARTED"
    echo "Run ID: $RUN_ID"
    echo "STREAM: $STREAM_FILE"
    echo "THREADS: $THREADS"
    echo "SYM: $SYM"
    echo "ITERATIONS: $ITERATIONS"
    echo "MIN_MEASUREMENTS: $MIN_MEASUREMENTS"
    echo "PUSH_RES: $PUSH_RES"
    echo "MODEL: $MODEL"
    echo "OUTPUT_UNMERGED: $OUTPUT_UNMERGED"
    echo "DISABLE_PR: $DISABLE_PR"
    echo "LOWRES: $LOWRES"
    echo "HIGHRES: $HIGHRES"
    echo "WILSON: $WILSON"
    echo "PARTIALATOR_POLARISATION: $PARTIALATOR_POLARISATION"
    echo "PARTIALATOR_MAX_ADU: $PARTIALATOR_MAX_ADU"
    echo "PARTIALATOR_MIN_RES: $PARTIALATOR_MIN_RES"
    echo "PARTIALATOR_NO_BSCALE: $PARTIALATOR_NO_BSCALE"
    echo "Merge Outdir: $MERGE_OUTDIR"
    echo "QC Outdir: $QC_OUTDIR"
    echo "PR Log Dir: $PR_LOG_DIR"
    echo "CrystFEL HKL: $CRYSTFEL_HKL"
    echo "Unmerged HKL: $UNMERGED_HKL"
    echo "Parameters JSON: $PARAMS_JSON"
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
    --model="$MODEL"
    -j "$THREADS"
    -o "$CRYSTFEL_HKL"
    -y "$SYM"
    --min-measurements="$MIN_MEASUREMENTS"
    --push-res="$PUSH_RES"
    --iterations="$ITERATIONS"
    --harvest-file="$PARAMS_JSON"
    --log-folder="$PR_LOG_DIR"
    --polarisation="$PARTIALATOR_POLARISATION"
    --max-adu="$PARTIALATOR_MAX_ADU"
    --min-res="$PARTIALATOR_MIN_RES"
    --no-Bscale
  )

  OUTPUT_UNMERGED_NORMALIZED="$(printf '%s' "$OUTPUT_UNMERGED" | tr '[:upper:]' '[:lower:]')"
  case "$OUTPUT_UNMERGED_NORMALIZED" in
    true|1|yes|y)
      partialator_args+=(--unmerged-output "$UNMERGED_HKL")
      echo "[$(timestamp)] partialator unmerged output enabled: $UNMERGED_HKL"
      ;;
    false|0|no|n)
      echo "[$(timestamp)] partialator unmerged output disabled"
      ;;
    *)
      die "OUTPUT_UNMERGED must be one of: true/false, 1/0, yes/no (got '$OUTPUT_UNMERGED')"
      ;;
  esac
  
  DISABLE_PR_NORMALIZED="$(printf '%s' "$DISABLE_PR" | tr '[:upper:]' '[:lower:]')"
  case "$DISABLE_PR_NORMALIZED" in
    true|1|yes|y)
      partialator_args+=(--no-pr)
      echo "[$(timestamp)] partialator PR disabled"
      ;;
    false|0|no|n)
      echo "[$(timestamp)] partialator PR enabled"
      ;;
    *)
      die "DISABLE_PR must be one of: true/false, 1/0, yes/no (got '$DISABLE_PR')"
      ;;
  esac

  printf -v PARTIALATOR_ARGS_SERIALIZED '%q ' partialator "${partialator_args[@]}"
  echo "PARTIALATOR_COMMAND: ${PARTIALATOR_ARGS_SERIALIZED% }" >> "$METADATA_LOG"

  partialator "${partialator_args[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG"
  echo "[$(timestamp)] partialator finished."

  WARN=0
  [ -s "$CRYSTFEL_HKL" ] || { echo "WARNING: HKL missing: $CRYSTFEL_HKL" | tee -a "$METADATA_LOG"; WARN=1; }
  [ -s "$PARAMS_JSON" ] || { echo "WARNING: parameters.json missing: $PARAMS_JSON" | tee -a "$METADATA_LOG"; WARN=1; }

  if [ -s "$PARAMS_JSON" ]; then
    python3 - "$PARAMS_JSON" \
      "$RUN_STARTED" \
        "$RUN_ID" \
        "$STREAM_FILE" \
        "$MERGE_OUTDIR" \
        "$QC_OUTDIR" \
        "$PR_LOG_DIR" \
        "$CRYSTFEL_HKL" \
        "$UNMERGED_HKL" \
        "$STDOUT_LOG" \
        "$STDERR_LOG" \
        "$CELL_FILE" \
        "$SYM" \
        "$THREADS" \
        "$ITERATIONS" \
        "$MIN_MEASUREMENTS" \
        "$PUSH_RES" \
        "$MODEL" \
        "$OUTPUT_UNMERGED_NORMALIZED" \
        "$DISABLE_PR_NORMALIZED" \
        "$LOWRES" \
        "$HIGHRES" \
        "$WILSON" \
        "$PARTIALATOR_POLARISATION" \
        "$PARTIALATOR_MAX_ADU" \
        "$PARTIALATOR_MIN_RES" \
        "$PARTIALATOR_NO_BSCALE" \
        "$PARTIALATOR_ARGS_SERIALIZED" <<'PY'
import json
import pathlib
import sys

(
  params_json,
  run_started,
  run_id,
  stream_file,
  merge_outdir,
  qc_outdir,
  pr_log_dir,
  crystfel_hkl,
  unmerged_hkl,
  stdout_log,
  stderr_log,
  cell_file,
  sym,
  threads,
  iterations,
  min_measurements,
  push_res,
  model,
  output_unmerged,
  disable_pr,
  lowres,
  highres,
  wilson,
  polarisation,
  max_adu,
  min_res,
  no_bscale,
  partialator_command,
) = sys.argv[1:]

path = pathlib.Path(params_json)
with path.open("r", encoding="utf-8") as handle:
  data = json.load(handle)

data["merge_wrapper"] = {
  "run_started": run_started,
  "run_id": run_id,
  "stream_file": stream_file,
  "merge_outdir": merge_outdir,
  "qc_outdir": qc_outdir,
  "pr_log_dir": pr_log_dir,
  "crystfel_hkl": crystfel_hkl,
  "unmerged_hkl": unmerged_hkl,
  "stdout_log": stdout_log,
  "stderr_log": stderr_log,
  "cell_file": cell_file,
  "symmetry": sym,
  "threads": int(threads),
  "iterations": int(iterations),
  "min_measurements": int(min_measurements),
  "push_res": push_res,
  "model": model,
  "output_unmerged": output_unmerged,
  "disable_pr": disable_pr,
  "lowres": lowres,
  "highres": highres,
  "wilson": wilson,
  "polarisation": polarisation,
  "max_adu": max_adu,
  "min_res": min_res,
  "no_bscale": no_bscale.lower() == "true",
  "partialator_command": partialator_command.strip(),
}

with path.open("w", encoding="utf-8") as handle:
  json.dump(data, handle, indent=2, sort_keys=True)
  handle.write("\n")
PY
    echo "[$(timestamp)] parameters.json updated with merge wrapper settings."
  fi

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
    echo "[OK] $s"
  else
    echo "[WARN] $s"
    EXITCODE=1
  fi
done

echo
echo "Multi-stream run finished: $(timestamp)"

exit "$EXITCODE"

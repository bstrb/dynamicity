#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
STREAM=$BASE/MFM300-VIII_cut_20-0_5.stream
SCORES=$BASE/oridyn_v2_scoring_filtering_20_0p5_20260623/v3_target_gated_scores/geometry_coupling_v3_target_gated_scores.csv
OUTDIR=$BASE/oridyn_v3_target_gated_filtering_20_0p5_20260704
RISK_COLUMN=trust_risk_v3_target_gated_shell_norm

PREFERRED_FILTER_SCRIPT=$REPO/tools/filter_stream_by_geometry_trust_keep_fraction.py
FILTER_SCRIPT=$REPO/tools/filter_stream_by_geometry_coupling_v2_keep_fraction.py
OUTPUT_ROOT_FLAG=--output-root
PROGRESS_EVERY=1000000
KEEP_FRACTIONS=(0.80 0.60 0.40 0.20)

DRY_RUN=false
FORCE=false

usage() {
  cat <<'EOF'
Usage: tools/run_v3_target_gated_filter_sweep.sh [--dry-run] [--force]

Filter the MFM300-VIII stream by the v3 target-gated risk column at keep80,
keep60, keep40, and keep20. Dry-run mode prints commands without running the
filtering step or writing outputs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  echo "[$(timestamp)] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

need_dir() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

need_executable() {
  [[ -x "$1" ]] || die "required executable not found or not executable: $1"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

fraction_to_tag() {
  "$PYTHON" - "$1" <<'PY'
import sys
print(f"keep{int(round(float(sys.argv[1]) * 100.0)):02d}")
PY
}

score_label() {
  local label=$RISK_COLUMN
  label=${label#trust_risk_}
  label=${label#manybeam_coupling_}
  label=${label%_norm}
  label=${label%_raw}
  label=${label//[^A-Za-z0-9]/_}
  label=${label##_}
  label=${label%%_}
  [[ -n "$label" ]] || label=score
  printf '%s' "$label"
}

tool_stream_name() {
  local tag=$1
  printf 'geometry_coupling_%s_%s.stream' "$(score_label)" "$tag"
}

check_risk_column() {
  "$PYTHON" - "$SCORES" "$RISK_COLUMN" <<'PY'
import csv
import sys
path, column = sys.argv[1], sys.argv[2]
with open(path, newline="") as handle:
    header = next(csv.reader(handle))
if column not in header:
    print(f"risk column {column!r} not found in {path}", file=sys.stderr)
    print("available columns:", ", ".join(header), file=sys.stderr)
    sys.exit(1)
print(f"risk column present: {column}")
PY
}

check_filter_cli() {
  local preferred_help
  preferred_help=$("$PYTHON" "$PREFERRED_FILTER_SCRIPT" --help 2>&1 || true)
  if [[ "$preferred_help" != *"--score-column"* ]]; then
    log "Preferred geometry-trust filter has no --score-column option; using score-column keep-fraction filter for direct v3 ranking."
  fi

  local help_text
  help_text=$("$PYTHON" "$FILTER_SCRIPT" --help 2>&1)
  [[ "$help_text" == *"--score-column"* ]] || die "$FILTER_SCRIPT does not expose --score-column"
  [[ "$help_text" == *"$OUTPUT_ROOT_FLAG"* ]] || die "$FILTER_SCRIPT does not expose $OUTPUT_ROOT_FLAG"
}

prepare_output_paths() {
  local tag=$1
  local workroot=$OUTDIR/tool_outputs/$tag
  local outstream=$OUTDIR/MFM300_VIII_v3_target_gated_${tag}.stream
  local logfile=$OUTDIR/logs/filter_v3_target_gated_${tag}.log

  if [[ "$FORCE" != true ]]; then
    [[ ! -e "$outstream" ]] || die "output stream exists; re-run with --force to replace: $outstream"
    [[ ! -e "$workroot" ]] || die "tool output directory exists; re-run with --force to replace: $workroot"
  fi
}

run_keep_fraction() {
  local keep_fraction=$1
  local tag=$2
  local workroot=$OUTDIR/tool_outputs/$tag
  local outstream=$OUTDIR/MFM300_VIII_v3_target_gated_${tag}.stream
  local logfile=$OUTDIR/logs/filter_v3_target_gated_${tag}.log
  local generated_stream=$workroot/$(tool_stream_name "$tag")

  local cmd=(
    "$PYTHON" "$FILTER_SCRIPT"
    --stream "$STREAM"
    --v2-scores-csv "$SCORES"
    "$OUTPUT_ROOT_FLAG" "$workroot"
    --score-column "$RISK_COLUMN"
    --keep-fractions "$keep_fraction"
    --progress-every "$PROGRESS_EVERY"
  )

  log "Starting v3 target-gated filtering: $tag ($keep_fraction)"
  log "Output stream: $outstream"
  log "Log file: $logfile"

  if [[ "$DRY_RUN" == true ]]; then
    print_command mkdir -p "$OUTDIR" "$OUTDIR/logs" "$OUTDIR/tool_outputs"
    if [[ "$FORCE" == true ]]; then
      print_command rm -rf "$workroot"
      print_command rm -f "$outstream" "$logfile"
    fi
    print_command "${cmd[@]}"
    echo "  2>&1 | tee $(printf '%q' "$logfile")"
    print_command cp "$generated_stream" "$outstream"
    return
  fi

  if [[ "$FORCE" == true ]]; then
    rm -rf "$workroot"
    rm -f "$outstream" "$logfile"
  fi

  mkdir -p "$OUTDIR/logs" "$OUTDIR/tool_outputs"
  "${cmd[@]}" 2>&1 | tee "$logfile"

  need_file "$generated_stream"
  cp "$generated_stream" "$outstream"
  log "Finished $tag: $outstream"
}

summarize_outputs() {
  echo
  echo "V3 target-gated filtering sweep summary"
  echo "Output directory: $OUTDIR"
  echo "Risk column: $RISK_COLUMN"
  echo "Filter script: $FILTER_SCRIPT"
  echo "Output flag: $OUTPUT_ROOT_FLAG"
  echo "Keep fractions: ${KEEP_FRACTIONS[*]}"
  echo "Output streams:"
  for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
    local tag
    tag=$(fraction_to_tag "$keep_fraction")
    echo "  $OUTDIR/MFM300_VIII_v3_target_gated_${tag}.stream"
  done
}

need_dir "$REPO"
need_executable "$PYTHON"
need_file "$STREAM"
need_file "$SCORES"
need_file "$PREFERRED_FILTER_SCRIPT"
need_file "$FILTER_SCRIPT"
check_filter_cli
check_risk_column

if [[ "$DRY_RUN" == true ]]; then
  log "Dry-run mode: commands will be printed, filtering will not run."
else
  mkdir -p "$OUTDIR" "$OUTDIR/logs" "$OUTDIR/tool_outputs"
fi

for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
  tag=$(fraction_to_tag "$keep_fraction")
  prepare_output_paths "$tag"
done

for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
  tag=$(fraction_to_tag "$keep_fraction")
  run_keep_fraction "$keep_fraction" "$tag"
done

summarize_outputs
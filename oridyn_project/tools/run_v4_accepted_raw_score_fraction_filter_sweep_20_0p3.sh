#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
STREAM=$BASE/MFM300-VIII_cut_20-0_3.stream
ACCEPTED_SCORES=$BASE/oridyn_v4_local_crowding_raw_20_0p3_20260704/partialator_survivor_mask/p1_iter1_20260705T1214/v4_p1_iter1_partialator_survivors_only_scores.csv
OUTDIR=$BASE/oridyn_v4_raw_accepted_fraction_filter_sweep_20_0p3_20260705
FILTER_SCRIPT=$REPO/tools/filter_stream_by_v4_accepted_raw_score_fraction.py
SCORE_COLUMN=local_crowding_target_gated_raw
KEEP_FRACTIONS=(0.90 0.80 0.70 0.60 0.50)
MIN_ACCEPTED_KEEP=20
PROGRESS_EVERY=1000000
SCORES_CHUNKSIZE=500000

DRY_RUN=false
SUMMARIZE_ONLY=false
FORCE=false
MAX_EVENTS=

usage() {
  cat <<'EOF'
Usage: tools/run_v4_accepted_raw_score_fraction_filter_sweep_20_0p3.sh [--dry-run] [--summarize-only] [--force] [--max-events N]

Run the v4 raw local-crowding keep-fraction stream filter on P1 iter1 partialator-accepted observations only.
Only accepted high raw-score observations are removed; nonaccepted stream observations are left unchanged.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --summarize-only)
      SUMMARIZE_ONLY=true
      shift
      ;;
    --force|--overwrite)
      FORCE=true
      shift
      ;;
    --max-events)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-events requires a value" >&2; exit 2; }
      MAX_EVENTS=$2
      shift 2
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

timestamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(timestamp)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "required file not found: $1"; }
need_dir() { [[ -d "$1" ]] || die "required directory not found: $1"; }
need_executable() { [[ -x "$1" ]] || die "required executable not found or not executable: $1"; }

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

check_score_column() {
  "$PYTHON" - "$ACCEPTED_SCORES" "$SCORE_COLUMN" <<'PY'
import csv
import sys
path, column = sys.argv[1], sys.argv[2]
with open(path, newline="") as handle:
    header = next(csv.reader(handle))
if column not in header:
    print(f"score column {column!r} not found in {path}", file=sys.stderr)
    print("available columns:", ", ".join(header), file=sys.stderr)
    sys.exit(1)
print(f"score column present: {column}")
PY
}

build_command() {
  local cmd=(
    "$PYTHON" "$FILTER_SCRIPT"
    --stream "$STREAM"
    --accepted-scores "$ACCEPTED_SCORES"
    --output-root "$OUTDIR"
    --score-column "$SCORE_COLUMN"
    --keep-fractions "${KEEP_FRACTIONS[@]}"
    --min-accepted-keep "$MIN_ACCEPTED_KEEP"
    --progress-every "$PROGRESS_EVERY"
    --scores-chunksize "$SCORES_CHUNKSIZE"
  )
  if [[ "$SUMMARIZE_ONLY" == true ]]; then
    cmd+=(--summarize-only)
  fi
  if [[ "$FORCE" == true ]]; then
    cmd+=(--overwrite)
  fi
  if [[ -n "$MAX_EVENTS" ]]; then
    cmd+=(--max-events "$MAX_EVENTS")
  fi
  printf '%s\0' "${cmd[@]}"
}

print_outputs() {
  echo
  echo "V4 accepted raw-score fraction filter sweep"
  echo "Output directory: $OUTDIR"
  echo "Input stream: $STREAM"
  echo "Accepted-only scores: $ACCEPTED_SCORES"
  echo "Score column: $SCORE_COLUMN"
  echo "Keep fractions: ${KEEP_FRACTIONS[*]}"
  echo "Minimum accepted observations retained per signed HKL: $MIN_ACCEPTED_KEEP"
  echo "Output streams:"
  for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
    local tag
    tag=$(fraction_to_tag "$keep_fraction")
    echo "  $OUTDIR/MFM300_VIII_v4_accepted_raw_score_${tag}.stream"
  done
}

need_dir "$REPO"
need_executable "$PYTHON"
need_file "$STREAM"
need_file "$ACCEPTED_SCORES"
need_file "$FILTER_SCRIPT"
check_score_column

LOGDIR=$OUTDIR/logs
LOGFILE=$LOGDIR/filter_v4_accepted_raw_score_fraction_20_0p3.log
mapfile -d '' CMD < <(build_command)

if [[ "$DRY_RUN" == true ]]; then
  log "Dry-run mode: command will be printed, filtering will not run."
  print_command mkdir -p "$OUTDIR" "$LOGDIR"
  print_command "${CMD[@]}"
  echo "  2>&1 | tee $(printf '%q' "$LOGFILE")"
  print_outputs
  exit 0
fi

mkdir -p "$OUTDIR" "$LOGDIR"
"${CMD[@]}" 2>&1 | tee "$LOGFILE"
print_outputs
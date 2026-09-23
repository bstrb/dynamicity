#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
STREAM=$BASE/MFM300-VIII_cut_20-0_3.stream
V5_SCORES=$BASE/oridyn_v5_nonself_local_excitation_raw_20_0p3_20260705/geometry_coupling_v5_nonself_local_excitation_raw_scores.csv
ACCEPTED_ONLY_TABLE=$BASE/oridyn_v4_local_crowding_raw_20_0p3_20260704/partialator_survivor_mask/p1_iter1_20260705T1214/v4_p1_iter1_partialator_survivors_only_scores.csv
OUTDIR=$BASE/oridyn_v5_nonself_positive_accepted_filter_sweep_20_0p3_20260706
FILTER_SCRIPT=$REPO/tools/filter_stream_by_v5_nonself_positive_accepted_fraction.py
SCORE_COLUMN=nonself_local_excitation_raw
POSITIVE_THRESHOLD=0.0
KEEP_FRACTIONS=(0.90 0.80 0.70 0.60 0.50)
MIN_ACCEPTED_KEEP=20
PROGRESS_EVERY=1000000
SCORES_CHUNKSIZE=500000

DRY_RUN=false
SELECTION_ONLY=false
SUMMARIZE_ONLY=false
FORCE=false
MAX_EVENTS=

usage() {
  cat <<'EOF'
Usage: tools/run_v5_nonself_positive_accepted_fraction_filter_sweep_20_0p3.sh [--dry-run] [--selection-only] [--summarize-only] [--force] [--max-events N]

Run the v5 positive-risk non-self keep-fraction stream filter on P1 iter1 partialator-accepted observations only.
Keep fractions apply only to accepted observations with nonself_local_excitation_raw > positive_threshold.
Zero-risk accepted observations and nonaccepted stream observations are left unchanged.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --selection-only)
      SELECTION_ONLY=true
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

check_input_columns() {
  "$PYTHON" - "$V5_SCORES" "$ACCEPTED_ONLY_TABLE" "$SCORE_COLUMN" <<'PY'
import csv
import sys
v5_path, accepted_path, score_column = sys.argv[1:]
key_columns = ["source_filename", "event", "h", "k", "l"]
with open(v5_path, newline="") as handle:
    v5_header = next(csv.reader(handle))
with open(accepted_path, newline="") as handle:
    accepted_header = next(csv.reader(handle))
missing_v5 = [column for column in [*key_columns, score_column] if column not in v5_header]
missing_accepted = [column for column in key_columns if column not in accepted_header]
if missing_v5:
    print(f"v5 score table missing columns: {missing_v5}", file=sys.stderr)
    sys.exit(1)
if missing_accepted:
    print(f"accepted-only table missing columns: {missing_accepted}", file=sys.stderr)
    sys.exit(1)
print(f"v5 score column present: {score_column}")
print("accepted-only exact key columns present")
PY
}

build_command() {
  local cmd=(
    "$PYTHON" "$FILTER_SCRIPT"
    --stream "$STREAM"
    --v5-scores "$V5_SCORES"
    --accepted-only-table "$ACCEPTED_ONLY_TABLE"
    --output-root "$OUTDIR"
    --score-column "$SCORE_COLUMN"
    --positive-threshold "$POSITIVE_THRESHOLD"
    --keep-fractions "${KEEP_FRACTIONS[@]}"
    --min-accepted-keep "$MIN_ACCEPTED_KEEP"
    --progress-every "$PROGRESS_EVERY"
    --scores-chunksize "$SCORES_CHUNKSIZE"
  )
  if [[ "$SUMMARIZE_ONLY" == true ]]; then
    cmd+=(--summarize-only)
  fi
  if [[ "$SELECTION_ONLY" == true ]]; then
    cmd+=(--selection-only)
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
  echo "V5 positive-risk accepted-observation fraction filter sweep"
  echo "Output directory: $OUTDIR"
  echo "Input stream: $STREAM"
  echo "V5 scores: $V5_SCORES"
  echo "Accepted-only table: $ACCEPTED_ONLY_TABLE"
  echo "Score column: $SCORE_COLUMN"
  echo "Positive threshold: $POSITIVE_THRESHOLD"
  echo "Keep fractions over positive-risk subset: ${KEEP_FRACTIONS[*]}"
  echo "Minimum accepted observations retained per signed HKL: $MIN_ACCEPTED_KEEP"
  if [[ "$SELECTION_ONLY" == true || "$SUMMARIZE_ONLY" == true ]]; then
    echo "Selection-only outputs:"
    echo "  $OUTDIR/filter_sweep_summary.csv"
    echo "  $OUTDIR/filter_sweep_summary.md"
    echo "  $OUTDIR/accepted_hkl_summary.csv"
    for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
      local tag
      tag=$(fraction_to_tag "$keep_fraction")
      echo "  $OUTDIR/v5_nonself_positive_accepted_${tag}_removed_by_hkl.csv"
      echo "  $OUTDIR/v5_nonself_positive_accepted_${tag}_removed_positive_accepted_keys.csv"
      echo "  $OUTDIR/v5_nonself_positive_accepted_${tag}_kept_positive_keys.csv"
    done
  else
    echo "Output streams:"
    for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
      local tag
      tag=$(fraction_to_tag "$keep_fraction")
      echo "  $OUTDIR/MFM300_VIII_v5_nonself_positive_accepted_${tag}.stream"
    done
  fi
}

need_dir "$REPO"
need_executable "$PYTHON"
need_file "$STREAM"
need_file "$V5_SCORES"
need_file "$ACCEPTED_ONLY_TABLE"
need_file "$FILTER_SCRIPT"
check_input_columns

LOGDIR=$OUTDIR/logs
LOGFILE=$LOGDIR/filter_v5_nonself_positive_accepted_fraction_20_0p3.log
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
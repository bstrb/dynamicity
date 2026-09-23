#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
STREAM=$BASE/MFM300-VIII_cut_20-0_3.stream
V5_SCORES=$BASE/oridyn_v5_nonself_local_excitation_raw_20_0p3_20260705/geometry_coupling_v5_nonself_local_excitation_raw_scores.csv
ACCEPTED_SCORES=$BASE/oridyn_v4_local_crowding_raw_20_0p3_20260704/partialator_survivor_mask/p1_iter1_20260705T1214/v4_p1_iter1_partialator_survivors_only_scores.csv
OUTDIR=$BASE/oridyn_v5_allscore_filter_and_50split_20_0p3_20260706
SCRIPT=$REPO/tools/run_v5_allscore_filter_and_50split_20_0p3.py
SCORE_COLUMN=nonself_local_excitation_raw
KEEP_FRACTIONS=(0.90 0.80 0.70 0.60 0.50)
MIN_ACCEPTED_KEEP=20
MIN_ACCEPTED_FOR_50SPLIT=40
SEED=1
PROGRESS_EVERY=1000000
SCORES_CHUNKSIZE=500000

DRY_RUN=false
FORCE=false
MAX_EVENTS=

usage() {
  cat <<'EOF'
Usage: tools/run_v5_allscore_filter_and_50split_20_0p3.sh [--dry-run] [--max-events N] [--force]

Run v5 all-score accepted-observation low-risk keep filters plus low50/high50/random50 diagnostic stream splits.
Uses the existing v5 score CSV only; does not recompute scores, run partialator, merge, or refine.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
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

check_input_columns() {
  "$PYTHON" - "$V5_SCORES" "$ACCEPTED_SCORES" "$SCORE_COLUMN" <<'PY'
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
    print(f"accepted score table missing columns: {missing_accepted}", file=sys.stderr)
    sys.exit(1)
print(f"v5 score column present: {score_column}")
print("accepted exact key columns present")
PY
}

build_command() {
  local cmd=(
    "$PYTHON" "$SCRIPT"
    --stream "$STREAM"
    --v5-scores "$V5_SCORES"
    --accepted-scores "$ACCEPTED_SCORES"
    --output-root "$OUTDIR"
    --score-column "$SCORE_COLUMN"
    --keep-fractions "${KEEP_FRACTIONS[@]}"
    --min-accepted-keep "$MIN_ACCEPTED_KEEP"
    --min-accepted-for-50split "$MIN_ACCEPTED_FOR_50SPLIT"
    --seed "$SEED"
    --progress-every "$PROGRESS_EVERY"
    --scores-chunksize "$SCORES_CHUNKSIZE"
  )
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
  echo "V5 all-score accepted-observation filter and 50-split workflow"
  echo "Output directory: $OUTDIR"
  echo "Input stream: $STREAM"
  echo "V5 scores: $V5_SCORES"
  echo "Accepted-only scores: $ACCEPTED_SCORES"
  echo "Score column: $SCORE_COLUMN"
  echo "Keep fractions: ${KEEP_FRACTIONS[*]}"
  echo "Minimum accepted retained per low-risk filter HKL: $MIN_ACCEPTED_KEEP"
  echo "Minimum accepted for 50% split eligibility: $MIN_ACCEPTED_FOR_50SPLIT"
  echo "Random seed: $SEED"
  echo "Output streams:"
  echo "  $OUTDIR/MFM300_VIII_v5_allscore_lowrisk_keep90.stream"
  echo "  $OUTDIR/MFM300_VIII_v5_allscore_lowrisk_keep80.stream"
  echo "  $OUTDIR/MFM300_VIII_v5_allscore_lowrisk_keep70.stream"
  echo "  $OUTDIR/MFM300_VIII_v5_allscore_lowrisk_keep60.stream"
  echo "  $OUTDIR/MFM300_VIII_v5_allscore_lowrisk_keep50.stream"
  echo "  $OUTDIR/MFM300_VIII_v5_allscore_low50_accepted.stream"
  echo "  $OUTDIR/MFM300_VIII_v5_allscore_high50_accepted.stream"
  echo "  $OUTDIR/MFM300_VIII_v5_allscore_random50_seed1_accepted.stream"
  echo "Summary outputs:"
  echo "  $OUTDIR/v5_allscore_filter_and_50split_summary.csv"
  echo "  $OUTDIR/v5_allscore_filter_and_50split_summary.md"
  echo "  $OUTDIR/v5_allscore_selected_hkl_summary.csv"
}

need_dir "$REPO"
need_executable "$PYTHON"
need_file "$STREAM"
need_file "$V5_SCORES"
need_file "$ACCEPTED_SCORES"
need_file "$SCRIPT"
check_input_columns

LOGDIR=$OUTDIR/logs
LOGFILE=$LOGDIR/v5_allscore_filter_and_50split_20_0p3.log
mapfile -d '' CMD < <(build_command)

if [[ "$DRY_RUN" == true ]]; then
  log "Dry-run mode: command will be printed, workflow will not run."
  print_command mkdir -p "$OUTDIR" "$LOGDIR"
  print_command "${CMD[@]}"
  echo "  2>&1 | tee $(printf '%q' "$LOGFILE")"
  print_outputs
  exit 0
fi

mkdir -p "$OUTDIR" "$LOGDIR"
"${CMD[@]}" 2>&1 | tee "$LOGFILE"
print_outputs
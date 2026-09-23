#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
STREAM=$BASE/MFM300-VIII_cut_20-0_5.stream
PREFERRED_SCORES=$BASE/oridyn_v2_scoring_filtering_20_0p5_20260623/v3_excited_lowcoupling_scores/geometry_coupling_v3_excited_lowcoupling_scores.csv
FALLBACK_SCORES=$BASE/oridyn_v2_scoring_filtering_20_0p5_20260623/v3_target_gated_scores/geometry_coupling_v3_target_gated_scores.csv
OUTDIR=$BASE/oridyn_v3_conservative_spread_filtering_20_0p5_20260704
FILTER_SCRIPT=$REPO/tools/filter_stream_by_conservative_risk_spread.py
SCORE_COLUMN=badness_v3_excited_lowcoupling_shell_norm
FALLBACK_SCORE_COLUMN=trust_risk_v3_target_gated_shell_norm
KEEP_FRACTIONS=(0.90 0.80 0.70 0.60 0.50)
MIN_OBS_TO_FILTER=20
MIN_KEEP_OBS=10
MIN_RISK_SPREAD_Q90_Q10=0.20
MIN_RISK_IQR=0.05
MAX_FILTER_INV_NM=14.0
PROGRESS_EVERY=1000000

DRY_RUN=false
SUMMARIZE_ONLY=false

usage() {
  cat <<'EOF'
Usage: tools/run_v3_conservative_spread_filter_sweep.sh [--dry-run] [--summarize-only]

Run the conservative v3 spread-aware OriDyn stream filter for keep90/80/70/60/50.
The filter removes only high-badness observations from signed HKLs that pass
resolution, count, minimum-kept, and risk-spread safety gates.
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

choose_scores_and_column() {
  "$PYTHON" - "$PREFERRED_SCORES" "$FALLBACK_SCORES" "$SCORE_COLUMN" "$FALLBACK_SCORE_COLUMN" <<'PY'
import csv
import sys
from pathlib import Path

preferred, fallback, score_column, fallback_column = sys.argv[1:]

def header(path):
    with open(path, newline="") as handle:
        return next(csv.reader(handle))

choices = []
if Path(preferred).is_file():
    cols = header(preferred)
    if score_column in cols:
        choices.append((preferred, score_column, "preferred score table with primary score column"))
    if fallback_column in cols:
        choices.append((preferred, fallback_column, "preferred score table with fallback score column"))
if Path(fallback).is_file():
    cols = header(fallback)
    if fallback_column in cols:
        choices.append((fallback, fallback_column, "fallback score table with fallback score column"))

if not choices:
    print("No usable score table/column combination found", file=sys.stderr)
    print(f"preferred: {preferred}", file=sys.stderr)
    print(f"fallback: {fallback}", file=sys.stderr)
    print(f"primary score column: {score_column}", file=sys.stderr)
    print(f"fallback score column: {fallback_column}", file=sys.stderr)
    sys.exit(1)

print("\t".join(choices[0]))
PY
}

check_filter_cli() {
  local help_text
  help_text=$("$PYTHON" "$FILTER_SCRIPT" --help 2>&1)
  [[ "$help_text" == *"--score-column"* ]] || die "$FILTER_SCRIPT does not expose --score-column"
  [[ "$help_text" == *"--fallback-score-column"* ]] || die "$FILTER_SCRIPT does not expose --fallback-score-column"
  [[ "$help_text" == *"--summarize-only"* ]] || die "$FILTER_SCRIPT does not expose --summarize-only"
}

build_command() {
  local scores=$1
  local score_column=$2
  local cmd=(
    "$PYTHON" "$FILTER_SCRIPT"
    --stream "$STREAM"
    --scores "$scores"
    --output-root "$OUTDIR"
    --score-column "$score_column"
    --fallback-score-column "$FALLBACK_SCORE_COLUMN"
    --keep-fractions "${KEEP_FRACTIONS[@]}"
    --min-obs-to-filter "$MIN_OBS_TO_FILTER"
    --min-keep-obs "$MIN_KEEP_OBS"
    --min-risk-spread-q90-q10 "$MIN_RISK_SPREAD_Q90_Q10"
    --min-risk-iqr "$MIN_RISK_IQR"
    --filter-low-resolution-only
    --max-filter-inv-nm "$MAX_FILTER_INV_NM"
    --higher-score-is-worse
    --progress-every "$PROGRESS_EVERY"
  )
  if [[ "$SUMMARIZE_ONLY" == true ]]; then
    cmd+=(--summarize-only)
  fi
  printf '%s\0' "${cmd[@]}"
}

print_outputs() {
  echo
  echo "V3 conservative spread filtering sweep summary"
  echo "Output directory: $OUTDIR"
  echo "Score table: $ACTIVE_SCORES"
  echo "Score column: $ACTIVE_SCORE_COLUMN"
  echo "Fallback score column: $FALLBACK_SCORE_COLUMN"
  echo "Filter script: $FILTER_SCRIPT"
  echo "Keep fractions: ${KEEP_FRACTIONS[*]}"
  echo "Safety gates: min_obs_to_filter=$MIN_OBS_TO_FILTER min_keep_obs=$MIN_KEEP_OBS min_spread_q90_q10=$MIN_RISK_SPREAD_Q90_Q10 min_iqr=$MIN_RISK_IQR"
  echo "Filtering only reflections with 1/d <= $MAX_FILTER_INV_NM nm^-1"
  echo "Output streams:"
  for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
    local tag
    tag=$(fraction_to_tag "$keep_fraction")
    echo "  $OUTDIR/MFM300_VIII_v3_conservative_spread_${tag}.stream"
  done
}

need_dir "$REPO"
need_executable "$PYTHON"
need_file "$STREAM"
need_file "$FILTER_SCRIPT"
check_filter_cli

IFS=$'\t' read -r ACTIVE_SCORES ACTIVE_SCORE_COLUMN ACTIVE_CHOICE < <(choose_scores_and_column)
need_file "$ACTIVE_SCORES"
log "Using $ACTIVE_CHOICE"
log "Score table: $ACTIVE_SCORES"
log "Score column: $ACTIVE_SCORE_COLUMN"

LOGDIR=$OUTDIR/logs
LOGFILE=$LOGDIR/filter_v3_conservative_spread_sweep.log

if [[ "$DRY_RUN" == true ]]; then
  log "Dry-run mode: command will be printed, filtering will not run."
  print_command mkdir -p "$OUTDIR" "$LOGDIR"
  mapfile -d '' CMD < <(build_command "$ACTIVE_SCORES" "$ACTIVE_SCORE_COLUMN")
  print_command "${CMD[@]}"
  echo "  2>&1 | tee $(printf '%q' "$LOGFILE")"
  print_outputs
  exit 0
fi

mkdir -p "$OUTDIR" "$LOGDIR"
mapfile -d '' CMD < <(build_command "$ACTIVE_SCORES" "$ACTIVE_SCORE_COLUMN")
"${CMD[@]}" 2>&1 | tee "$LOGFILE"
print_outputs
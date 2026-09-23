#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
STREAM=$BASE/MFM300-VIII_cut_20-0_3.stream
SCORES=$BASE/oridyn_v3_scoring_filtering_20_0p3_20260704/v3_excited_lowcoupling_scores/geometry_coupling_v3_excited_lowcoupling_scores.csv
OUTDIR=$BASE/oridyn_v3_conservative_spread_filtering_20_0p3_20260704
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
Usage: tools/run_v3_conservative_spread_filter_sweep_20_0p3.sh [--dry-run] [--summarize-only]

Run the conservative v3 spread-aware filter for the 20-0.3 Å production stream.
This wrapper intentionally uses 0p3-specific scores, stream, and output folders.
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

check_score_column() {
  "$PYTHON" - "$SCORES" "$SCORE_COLUMN" <<'PY'
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
    --scores "$SCORES"
    --output-root "$OUTDIR"
    --score-column "$SCORE_COLUMN"
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
  echo "V3 conservative spread 20_0p3 filtering summary"
  echo "Output directory: $OUTDIR"
  echo "Input stream: $STREAM"
  echo "Score table: $SCORES"
  echo "Score column: $SCORE_COLUMN"
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
need_file "$SCORES"
need_file "$FILTER_SCRIPT"
check_score_column

LOGDIR=$OUTDIR/logs
LOGFILE=$LOGDIR/filter_v3_conservative_spread_20_0p3.log
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
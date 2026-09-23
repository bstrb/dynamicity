#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
V2_SCORES=$BASE/geometry_coupling_v2_full_filter_keep90_80_20260623/v2_scores/geometry_coupling_v2_scores.csv
BASE_SCORES=$BASE/oridyn_large_ABC_min1_mildweights/oridyn_scores_mild/reflection_scores.csv
OUTDIR=$BASE/oridyn_v4_local_crowding_raw_20_0p3_20260704
COMPUTE_SCRIPT=$REPO/tools/compute_v4_local_crowding_raw_scores_20_0p3.py
PLOT_SCRIPT=$REPO/tools/plot_v4_local_crowding_distributions.py
SCORE_CSV=$OUTDIR/geometry_coupling_v4_local_crowding_raw_scores.csv
PLOTS_DIR=$OUTDIR/plots
SG0=0.01
CHUNKSIZE=500000
MAX_ROWS=
DRY_RUN=false
FORCE=false

usage() {
  cat <<'EOF'
Usage: tools/run_v4_local_crowding_raw_20_0p3.sh [--dry-run] [--max-rows N] [--force]

Compute raw score-only v4 local-crowding diagnostics for the 20-0.3 Å production dataset,
then generate distribution plots. This wrapper does not filter observations or create streams.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --max-rows)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-rows requires a value" >&2; exit 2; }
      MAX_ROWS=$2
      shift 2
      ;;
    --force|--overwrite)
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

need_dir "$REPO"
need_executable "$PYTHON"
need_file "$V2_SCORES"
need_file "$BASE_SCORES"
need_file "$COMPUTE_SCRIPT"
need_file "$PLOT_SCRIPT"

compute_cmd=(
  "$PYTHON" "$COMPUTE_SCRIPT"
  --v2-scores "$V2_SCORES"
  --base-scores "$BASE_SCORES"
  --outdir "$OUTDIR"
  --sg0 "$SG0"
  --chunksize "$CHUNKSIZE"
)
if [[ -n "$MAX_ROWS" ]]; then
  compute_cmd+=(--max-rows "$MAX_ROWS")
fi
if [[ "$FORCE" == true ]]; then
  compute_cmd+=(--overwrite)
fi

plot_cmd=(
  "$PYTHON" "$PLOT_SCRIPT"
  --scores "$SCORE_CSV"
  --outdir "$PLOTS_DIR"
)
if [[ -n "$MAX_ROWS" ]]; then
  plot_cmd+=(--max-sample-rows "$MAX_ROWS")
fi

log "Output directory: $OUTDIR"
log "Score CSV: $SCORE_CSV"
log "Plots directory: $PLOTS_DIR"
if [[ "$DRY_RUN" == true ]]; then
  log "Dry-run mode: commands will be printed only."
  print_command mkdir -p "$OUTDIR" "$PLOTS_DIR"
  print_command "${compute_cmd[@]}"
  print_command "${plot_cmd[@]}"
  exit 0
fi

mkdir -p "$OUTDIR" "$PLOTS_DIR"
"${compute_cmd[@]}"
need_file "$SCORE_CSV"
"${plot_cmd[@]}"
log "V4 local-crowding score-only diagnostic complete"
log "Wrote scores: $SCORE_CSV"
log "Wrote plots: $PLOTS_DIR"
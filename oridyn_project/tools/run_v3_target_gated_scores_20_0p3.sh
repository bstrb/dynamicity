#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
STREAM=$BASE/MFM300-VIII_cut_20-0_3.stream
V2_SCORES=$BASE/geometry_coupling_v2_full_filter_keep90_80_20260623/v2_scores/geometry_coupling_v2_scores.csv
OUTDIR=$BASE/oridyn_v3_scoring_filtering_20_0p3_20260704/v3_target_gated_scores
SCRIPT=$REPO/tools/compute_geometry_coupling_v3_target_gated_scores.py
OUTPUT_CSV=$OUTDIR/geometry_coupling_v3_target_gated_scores.csv

DRY_RUN=false
FORCE=false

usage() {
  cat <<'EOF'
Usage: tools/run_v3_target_gated_scores_20_0p3.sh [--dry-run] [--force]

Generate v3 target-gated geometry-coupling scores for the 20-0.3 Å production stream.
This wrapper intentionally uses 0p3-specific inputs and output folders.
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
need_file "$SCRIPT"
need_file "$STREAM"
need_file "$V2_SCORES"

cmd=(
  "$PYTHON" "$SCRIPT"
  --v2-scores "$V2_SCORES"
  --stream "$STREAM"
  --out "$OUTDIR"
)
if [[ "$FORCE" == true ]]; then
  cmd+=(--overwrite)
fi

log "0p3 target-gated score output: $OUTPUT_CSV"
if [[ "$DRY_RUN" == true ]]; then
  log "Dry-run mode: command will be printed, scoring will not run."
  print_command mkdir -p "$OUTDIR"
  print_command "${cmd[@]}"
  exit 0
fi

if [[ "$FORCE" != true && -e "$OUTPUT_CSV" ]]; then
  die "output exists; re-run with --force to replace: $OUTPUT_CSV"
fi

mkdir -p "$OUTDIR"
"${cmd[@]}"
need_file "$OUTPUT_CSV"
log "Wrote: $OUTPUT_CSV"
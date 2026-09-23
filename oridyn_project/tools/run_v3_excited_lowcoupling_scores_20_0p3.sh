#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
V3_TARGET=$BASE/oridyn_v3_scoring_filtering_20_0p3_20260704/v3_target_gated_scores/geometry_coupling_v3_target_gated_scores.csv
OUTDIR=$BASE/oridyn_v3_scoring_filtering_20_0p3_20260704/v3_excited_lowcoupling_scores
SCRIPT=$REPO/tools/add_v3_excited_lowcoupling_trust_scores.py
OUTPUT_CSV=$OUTDIR/geometry_coupling_v3_excited_lowcoupling_scores.csv

DRY_RUN=false
FORCE=false

usage() {
  cat <<'EOF'
Usage: tools/run_v3_excited_lowcoupling_scores_20_0p3.sh [--dry-run] [--force]

Generate excited-lowcoupling trust/badness scores from the 20-0.3 Å v3 target-gated table.
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
need_file "$V3_TARGET"

cmd=(
  "$PYTHON" "$SCRIPT"
  --v3-scores "$V3_TARGET"
  --out "$OUTDIR"
)
if [[ "$FORCE" == true ]]; then
  cmd+=(--overwrite)
fi

log "0p3 excited-lowcoupling score output: $OUTPUT_CSV"
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
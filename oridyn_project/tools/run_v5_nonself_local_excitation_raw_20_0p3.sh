#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bubl3932/projects/dynamicity/oridyn_project
PYTHON=/home/bubl3932/anaconda3/bin/python
BASE=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
STREAM=$BASE/MFM300-VIII_cut_20-0_3.stream
V4_SCORES=$BASE/oridyn_v4_local_crowding_raw_20_0p3_20260704/geometry_coupling_v4_local_crowding_raw_scores.csv
ACCEPTED_V4_SCORES=$BASE/oridyn_v4_local_crowding_raw_20_0p3_20260704/partialator_survivor_mask/p1_iter1_20260705T1214/v4_p1_iter1_partialator_survivors_only_scores.csv
OUTDIR=$BASE/oridyn_v5_nonself_local_excitation_raw_20_0p3_20260705
COMPUTE_SCRIPT=$REPO/tools/compute_v5_nonself_local_excitation_raw_scores_20_0p3.py

EDGE_WEIGHT=0.1
SG_UNITS=A^-1
KERNEL=gaussian
SIGMA_C=0.05
Q0=0.05
R_CUT=0.15
CHUNKSIZE=500000
TARGET_BATCH_SIZE=256
PROGRESS_EVERY_FRAMES=1000

DRY_RUN=false
FORCE=false
MAX_ROWS=
MAX_FRAMES=

usage() {
  cat <<'EOF'
Usage: tools/run_v5_nonself_local_excitation_raw_20_0p3.sh [--dry-run] [--max-rows N] [--max-frames N] [--force]

Compute score-only v5 non-self local excitation environment raw scores for the 20-0.3 Å production dataset.
This wrapper does not filter streams, run partialator, or run merging.
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
    --max-rows)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-rows requires a value" >&2; exit 2; }
      MAX_ROWS=$2
      shift 2
      ;;
    --max-frames)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-frames requires a value" >&2; exit 2; }
      MAX_FRAMES=$2
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

print_resolved_parameters() {
  "$PYTHON" - "$STREAM" "$EDGE_WEIGHT" "$SG_UNITS" "$KERNEL" "$SIGMA_C" "$Q0" "$R_CUT" <<'PY'
import math
import re
import sys
from pathlib import Path

stream, edge_weight, sg_units, kernel, sigma_c, q0, r_cut = sys.argv[1:]
edge_weight = float(edge_weight)
pattern = re.compile(r"profile_radius\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-zÅ0-9\^-]+)?")
profile = None
units = "nm^-1"
with Path(stream).open(encoding="utf-8", errors="replace") as handle:
    for line in handle:
        match = pattern.search(line)
        if match:
            profile = float(match.group(1))
            units = (match.group(2) or "nm^-1").replace("Å", "A")
            break
if profile is None:
    print("profile_radius: not parsed; compute script will require --profile-radius or --sg0-override")
else:
    profile_nm = profile if units == "nm^-1" else 10.0 * profile
    profile_A = 0.1 * profile_nm
    radius_sg = profile_A if sg_units == "A^-1" else profile_nm
    sg0 = radius_sg / math.sqrt(-math.log(edge_weight))
    print(f"profile_radius_nm_inv: {profile_nm:.8g}")
    print(f"profile_radius_A_inv: {profile_A:.8g}")
    print(f"sg_units: {sg_units}")
    print(f"edge_weight: {edge_weight:.8g}")
    print(f"sg0_profile_derived: {sg0:.8g}")
print(f"kernel: {kernel}")
print(f"sigma_c_A_inv: {float(sigma_c):.8g}")
print(f"q0_A_inv: {float(q0):.8g}")
print(f"r_cut_A_inv: {float(r_cut):.8g}")
PY
}

build_command() {
  local cmd=(
    "$PYTHON" "$COMPUTE_SCRIPT"
    --stream "$STREAM"
    --v4-scores "$V4_SCORES"
    --accepted-v4-scores "$ACCEPTED_V4_SCORES"
    --outdir "$OUTDIR"
    --edge-weight "$EDGE_WEIGHT"
    --sg-units "$SG_UNITS"
    --kernel "$KERNEL"
    --sigma-c "$SIGMA_C"
    --q0 "$Q0"
    --r-cut "$R_CUT"
    --chunksize "$CHUNKSIZE"
    --target-batch-size "$TARGET_BATCH_SIZE"
    --progress-every-frames "$PROGRESS_EVERY_FRAMES"
  )
  if [[ -n "$MAX_ROWS" ]]; then
    cmd+=(--max-rows "$MAX_ROWS")
  fi
  if [[ -n "$MAX_FRAMES" ]]; then
    cmd+=(--max-frames "$MAX_FRAMES")
  fi
  if [[ "$FORCE" == true ]]; then
    cmd+=(--overwrite)
  fi
  printf '%s\0' "${cmd[@]}"
}

need_dir "$REPO"
need_executable "$PYTHON"
need_file "$STREAM"
need_file "$V4_SCORES"
need_file "$ACCEPTED_V4_SCORES"
need_file "$COMPUTE_SCRIPT"

mapfile -d '' CMD < <(build_command)

log "Resolved v5 non-self local excitation parameters:"
print_resolved_parameters
log "Output directory: $OUTDIR"
log "Output CSV: $OUTDIR/geometry_coupling_v5_nonself_local_excitation_raw_scores.csv"

if [[ "$DRY_RUN" == true ]]; then
  log "Dry-run mode: command will be printed only."
  print_command mkdir -p "$OUTDIR"
  print_command "${CMD[@]}"
  exit 0
fi

mkdir -p "$OUTDIR"
"${CMD[@]}"
log "V5 non-self local excitation score-only diagnostic complete"
#!/usr/bin/env bash
# Graph/frame contrast sweep with nonpositive weak-reference guards.
# This writes corrected streams only. It does not run partialator or SHELXL.

set -euo pipefail

ROOT="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
PY="/home/bubl3932/anaconda3/envs/pyxem-env/bin/python"
SCALE_SCRIPT="/home/bubl3932/projects/dynamicity/oridyn_project/tools/scale_stream_by_graph_frame_shift_from_unmerged.py"
BIDIR_SCRIPT="/home/bubl3932/projects/dynamicity/oridyn_project/tools/shift_stream_bidirectional_graph_frame_to_lowrisk.py"

STREAM="$ROOT/MFM300-VIII_cut_20-0_3.stream"
UNMERGED="$ROOT/model_free_nonself_diagnostics/minimal_unmerged_y1_unity_nopr_n1/unmerged_minimal_y1.hkl"
SCORES="$ROOT/oridyn_large_ABC_min1_mildweights/oridyn_scores_mild/reflection_scores.csv"
OUTROOT="$ROOT/model_free_nonself_diagnostics/graph_frame_skip_nonpositive_iref_contrast_sweep"
PROMISING_STREAM="$ROOT/model_free_nonself_diagnostics/graph_frame_scaled_stream_lambda05/scaled_graph_frame_weak_posshift_lambda05.stream"

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
  FORCE=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--force]" >&2
  exit 2
fi

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  echo "[$(timestamp)] $*"
}

need_file() {
  [[ -f "$1" ]] || {
    echo "ERROR: required file not found: $1" >&2
    exit 1
  }
}

check_outstream() {
  local outstream="$1"
  if [[ "$outstream" == "$PROMISING_STREAM" ]]; then
    echo "ERROR: refusing to touch old promising stream: $PROMISING_STREAM" >&2
    exit 1
  fi
  if [[ -e "$outstream" && "$FORCE" != true ]]; then
    echo "ERROR: output stream exists. Re-run with --force to overwrite: $outstream" >&2
    exit 1
  fi
}

append_manifest() {
  local variant="$1"
  local correction_kind="$2"
  local output_stream="$3"
  local shift_tail="$4"
  local scale_top="$5"
  local correct_top="$6"
  local lambda_value="$7"
  local min_scale="$8"
  local output_dir="$9"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$correction_kind" "$output_stream" "$shift_tail" "$scale_top" \
    "$correct_top" "$lambda_value" "$min_scale" "true" "$output_dir" >> "$MANIFEST"
}

run_one_sided() {
  local variant="$1"
  local scale_top="$2"
  local outdir="$OUTROOT/$variant"
  local outstream="$outdir/scaled_graph_frame_${variant}.stream"
  local logfile="$outdir/scale_stream.log"

  check_outstream "$outstream"
  mkdir -p "$outdir"
  append_manifest "$variant" "one_sided_weak_down" "$outstream" "0.10" "$scale_top" "" "1.0" "0.10" "$outdir"

  log "Starting one-sided guarded variant: $variant"
  "$PY" "$SCALE_SCRIPT" \
    --stream "$STREAM" \
    --scores "$SCORES" \
    --unmerged "$UNMERGED" \
    --output-stream "$outstream" \
    --baseline-low-risk-fraction 0.20 \
    --shift-tail-fraction 0.10 \
    --scale-top-risk-fraction "$scale_top" \
    --min-relative-shift 0.05 \
    --lambda-scale 1.0 \
    --min-scale 0.10 \
    --max-scale 1.0 \
    --min-obs-all 50 \
    --local-neighbor-count 30 \
    --local-min-neighbors 10 \
    --min-denominator 1.0 \
    --exclude-partiality-too-small \
    --skip-weak-nonpositive-iref \
    --progress-every 100000 \
    2>&1 | tee "$logfile"
  log "Finished one-sided guarded variant: $variant"
}

run_bidirectional() {
  local variant="$1"
  local correct_top="$2"
  local outdir="$OUTROOT/$variant"
  local outstream="$outdir/bidirectional_graph_frame_${variant}.stream"
  local logfile="$outdir/bidirectional_shift.log"

  check_outstream "$outstream"
  mkdir -p "$outdir"
  append_manifest "$variant" "bidirectional" "$outstream" "0.10" "" "$correct_top" "1.0" "" "$outdir"

  log "Starting bidirectional guarded variant: $variant"
  "$PY" "$BIDIR_SCRIPT" \
    --stream "$STREAM" \
    --scores "$SCORES" \
    --unmerged "$UNMERGED" \
    --output-stream "$outstream" \
    --baseline-low-risk-fraction 0.20 \
    --shift-tail-fraction 0.10 \
    --correct-top-risk-fraction "$correct_top" \
    --min-relative-shift 0.05 \
    --lambda-shift 1.0 \
    --min-obs-all 50 \
    --local-neighbor-count 30 \
    --local-min-neighbors 10 \
    --min-denominator 1.0 \
    --exclude-partiality-too-small \
    --skip-weak-down-nonpositive-iref \
    --progress-every 100000 \
    2>&1 | tee "$logfile"
  log "Finished bidirectional guarded variant: $variant"
}

need_file "$PY"
need_file "$SCALE_SCRIPT"
need_file "$BIDIR_SCRIPT"
need_file "$STREAM"
need_file "$UNMERGED"
need_file "$SCORES"

mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/guarded_contrast_manifest.tsv"
cat > "$MANIFEST" <<'EOF'
variant	correction_kind	output_stream	shift_tail_fraction	scale_top_risk_fraction	correct_top_risk_fraction	lambda	min_scale	skip_weak_nonpositive_iref	output_folder
EOF
log "Wrote manifest header: $MANIFEST"

run_one_sided "weakdown_scale20_skip_nonpositive_iref" "0.20"
run_one_sided "weakdown_scale30_skip_nonpositive_iref" "0.30"
run_bidirectional "bidirectional_correct20_skip_weakdown_nonpositive_iref" "0.20"
run_bidirectional "bidirectional_correct30_skip_weakdown_nonpositive_iref" "0.30"

log "All guarded contrast variants finished."

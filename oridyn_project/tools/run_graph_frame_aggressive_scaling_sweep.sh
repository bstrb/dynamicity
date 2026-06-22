#!/usr/bin/env bash
# Aggressive proof-of-concept graph/frame stream-scaling sweep only.
# This does not run partialator, merging, SHELXL, or refinement.

set -euo pipefail

SCRIPT="/home/bubl3932/projects/dynamicity/oridyn_project/tools/scale_stream_by_graph_frame_shift_from_unmerged.py"
PY="/home/bubl3932/anaconda3/envs/pyxem-env/bin/python"

STREAM="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300-VIII_cut_20-0_3.stream"
UNMERGED="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/minimal_unmerged_y1_unity_nopr_n1/unmerged_minimal_y1.hkl"
SCORES="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/oridyn_large_ABC_min1_mildweights/oridyn_scores_mild/reflection_scores.csv"
OUTROOT="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling"
PROMISING_STREAM="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_scaled_stream_lambda05/scaled_graph_frame_weak_posshift_lambda05.stream"

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

run_variant() {
  local variant="$1"
  local shift_tail="$2"
  local scale_top="$3"
  local lambda="$4"
  local min_scale="$5"
  local outdir="$OUTROOT/$variant"
  local outstream="$outdir/scaled_graph_frame_${variant}.stream"
  local logfile="$outdir/scale_stream.log"

  if [[ "$outstream" == "$PROMISING_STREAM" ]]; then
    echo "ERROR: refusing to touch old promising stream: $PROMISING_STREAM" >&2
    exit 1
  fi

  if [[ -e "$outstream" && "$FORCE" != true ]]; then
    echo "ERROR: output stream exists. Re-run with --force to overwrite: $outstream" >&2
    exit 1
  fi

  mkdir -p "$outdir"
  log "Starting variant: $variant"
  log "Output stream: $outstream"

  "$PY" "$SCRIPT" \
    --stream "$STREAM" \
    --scores "$SCORES" \
    --unmerged "$UNMERGED" \
    --output-stream "$outstream" \
    --baseline-low-risk-fraction 0.20 \
    --shift-tail-fraction "$shift_tail" \
    --scale-top-risk-fraction "$scale_top" \
    --min-relative-shift 0.05 \
    --lambda-scale "$lambda" \
    --min-scale "$min_scale" \
    --max-scale 1.0 \
    --min-obs-all 50 \
    --local-neighbor-count 30 \
    --local-min-neighbors 10 \
    --min-denominator 1.0 \
    --exclude-partiality-too-small \
    --progress-every 100000 \
    2>&1 | tee "$logfile"

  log "Finished variant: $variant"
}

need_file "$PY"
need_file "$SCRIPT"
need_file "$STREAM"
need_file "$UNMERGED"
need_file "$SCORES"

mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/scaling_manifest.tsv"
cat > "$MANIFEST" <<'EOF'
variant	output_stream	shift_tail_fraction	scale_top_risk_fraction	lambda_scale	min_scale	output_folder
poc_shift10_scale10_lambda10_minscale10	/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale10_lambda10_minscale10/scaled_graph_frame_poc_shift10_scale10_lambda10_minscale10.stream	0.10	0.10	1.0	0.10	/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale10_lambda10_minscale10
poc_shift10_scale10_lambda10_minscale00	/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale10_lambda10_minscale00/scaled_graph_frame_poc_shift10_scale10_lambda10_minscale00.stream	0.10	0.10	1.0	0.0	/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale10_lambda10_minscale00
poc_shift10_scale20_lambda10_minscale10	/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale20_lambda10_minscale10/scaled_graph_frame_poc_shift10_scale20_lambda10_minscale10.stream	0.10	0.20	1.0	0.10	/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale20_lambda10_minscale10
EOF
log "Wrote manifest: $MANIFEST"

run_variant "poc_shift10_scale10_lambda10_minscale10" "0.10" "0.10" "1.0" "0.10"
run_variant "poc_shift10_scale10_lambda10_minscale00" "0.10" "0.10" "1.0" "0.0"
run_variant "poc_shift10_scale20_lambda10_minscale10" "0.10" "0.20" "1.0" "0.10"

log "All aggressive scaling variants finished."

#!/usr/bin/env bash
# merge_with_careless.sh
set -euo pipefail

input_mtz="mfm300sim_0.0_0.0.mtz"

[ -f "$input_mtz" ] || { echo "❌ MTZ not found: $input_mtz" >&2; exit 1; }
input_dir=$(dirname "$input_mtz")
input_base=$(basename "$input_mtz" .mtz)
stamp=$(date +%Y%m%d_%H%M%S)
out_dir="$input_dir/careless/${input_base}_$stamp"
mkdir -p "$out_dir"
out_base="$out_dir/$input_base"

#   --disable-gpu \
echo "▶ Running Careless merge + half-dataset split…"
careless mono \
  --image-key BATCH \
  --intensity-key I \
  --uncertainty-key SigI \
  --iterations 100 \
  --dmin 0.3 \
  --studentt-likelihood-dof 2 \
  --test-fraction 0.05 \
  --merge-half-datasets \
  --half-dataset-repeats 1 \
  BATCH \
  "$input_mtz" \
  "$out_base" \
  2>&1 | tee "$out_dir/careless.log"

echo -e "\n✅ Merge complete. You should now have:"
ls -1 "$out_dir"

#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_SRC="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300-VIII_cut_20-0_3_partialator_results_20260622T1211/shelx"
CORRECTED_SRC="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/enh_feed_weak_positive_poc_scaling_20260622_strict_with_resolution/MFM300-VIII_cut_20-0_3_enh_feed_weak_positive_poc_partialator_results_20260622T1412/shelx"
DEST_ROOT="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/enh_feed_strict_refinement_comparison_20260622"

DEST_A="${DEST_ROOT}/A_original_same_settings"
DEST_B="${DEST_ROOT}/B_enh_feed_strict_weak_positive"

FORCE=0

usage() {
  cat <<'EOF'
Usage: prepare_enh_feed_strict_refinement_comparison.sh [--force]

Copies two SHELX refinement input folders into a clean comparison folder:
  A_original_same_settings
  B_enh_feed_strict_weak_positive

Options:
  --force   Remove and recreate only the two destination folders if they exist.
  -h, --help
            Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

check_source_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing ${label} source folder: $path" >&2
    exit 1
  fi
}

prepare_destination_dir() {
  local path="$1"
  if [[ -e "$path" ]]; then
    if [[ "$FORCE" -eq 1 ]]; then
      rm -rf -- "$path"
    else
      echo "Destination already exists: $path" >&2
      echo "Refusing to overwrite. Re-run with --force to remove and recreate this folder." >&2
      exit 1
    fi
  fi
}

check_source_dir "$ORIGINAL_SRC" "original"
check_source_dir "$CORRECTED_SRC" "corrected strict enhancement-feed"

mkdir -p -- "$DEST_ROOT"
prepare_destination_dir "$DEST_A"
prepare_destination_dir "$DEST_B"

cp -a -- "$ORIGINAL_SRC" "$DEST_A"
cp -a -- "$CORRECTED_SRC" "$DEST_B"

cat > "${DEST_ROOT}/README.md" <<EOF
# Strict Enhancement-Feed Refinement Comparison

This folder contains two SHELX refinement input folders copied for a clean
side-by-side comparison.

## Folders

- \`A_original_same_settings\`: original partialator/SHELX inputs copied from:
  \`${ORIGINAL_SRC}\`
- \`B_enh_feed_strict_weak_positive\`: strict enhancement-feed weak-positive
  correction partialator/SHELX inputs copied from:
  \`${CORRECTED_SRC}\`

Both datasets were merged with the same settings. Folder B is the strict
enhancement-feed weak-positive correction.

The correction mass was concentrated almost entirely in the 20-1.09 A shell,
mostly around 1.5-2.4 A.

## Recommended Refinement Comparison

Use the same model, same SHELXL/Olex settings, same resolution cutoff, and the
same WGHT, EXTI, and anisotropic refinement choices for A and B. The goal is to
isolate the strict enhancement-feed weak-positive correction rather than mix it
with refinement-protocol changes.
EOF

echo "Copied SHELX comparison folders:"
echo "  A: $DEST_A"
echo "  B: $DEST_B"
echo "README: ${DEST_ROOT}/README.md"

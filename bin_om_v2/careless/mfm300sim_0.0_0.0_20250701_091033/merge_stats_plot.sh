#!/usr/bin/env bash
# --------------------------------------------------------------------------
# merge_stats_plot.sh
#
# ▸ Computes CC½, Rsplit, and completeness on existing Careless output
# ▸ Generates plots for each statistic using Python + matplotlib
#
# Usage:   ./merge_stats_plot.sh [output_directory]
# --------------------------------------------------------------------------
set -euo pipefail

### ---- 0. User-adjustable defaults ----------------------------------------
N_BINS=${N_BINS:-10}  # Number of resolution bins for stats and plots
### -------------------------------------------------------------------------

# ---- 1.  Locate and cd into output directory --------------------------------
WORKDIR="${1:-$(pwd)}"
[ -d "$WORKDIR" ] || { echo "❌ Directory not found: $WORKDIR" >&2; exit 1; }
cd "$WORKDIR"

# ---- 2.  Derive base prefix from directory name -----------------------------
DIRBASE="$(basename "$WORKDIR")"
PREFIX="${DIRBASE%_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*}"
OUT_BASE="$WORKDIR/$PREFIX"

# ---- 3.  Find half-dataset MTZ files ---------------------------------------
shopt -s nullglob
XVAL_FILES=( "${OUT_BASE}"_xval_*.mtz )
if (( ${#XVAL_FILES[@]} == 0 )); then
  echo "❌ No half-dataset MTZs found (looking for ${OUT_BASE}_xval_*.mtz)" >&2
  exit 1
fi

# ---- 4.  Find full-merge MTZ ------------------------------------------------
FULL_MTZ="${OUT_BASE}_0.mtz"
if [ ! -f "$FULL_MTZ" ]; then
  echo "❌ Full-merge MTZ not found: $FULL_MTZ" >&2
  exit 1
fi

# ---- 5.  Compute statistics ------------------------------------------------
echo "▶ Running CC½..."
careless.cchalf -b "$N_BINS" "${XVAL_FILES[@]}" \
  --output "${OUT_BASE}_cchalf.csv" \
  | tee "${OUT_BASE}_cchalf.log"

echo "▶ Running Rsplit..."
careless.rsplit -b "$N_BINS" "${XVAL_FILES[@]}" \
  --output "${OUT_BASE}_rsplit.csv" \
  | tee "${OUT_BASE}_rsplit.log"

echo "▶ Running completeness..."
careless.completeness -b "$N_BINS" "$FULL_MTZ" \
  --output "${OUT_BASE}_completeness.csv" \
  | tee "${OUT_BASE}_completeness.log"

echo "✅ Statistics computed."

# ---- 6.  Optional plotting ------------------------------------------------
if command -v python3 &>/dev/null; then
  echo "▶ Generating plots..."
  python3 - <<'PY' "$OUT_BASE" "$N_BINS"
import sys, pathlib, pandas as pd, matplotlib.pyplot as plt

# Retrieve arguments
out_base = pathlib.Path(sys.argv[1])
nbins = int(sys.argv[2])
# Directory and base name
parent_dir = out_base.parent
base_name = out_base.name

# Define files
files = {
    'cchalf':    parent_dir / f"{base_name}_cchalf.csv",
    'rsplit':    parent_dir / f"{base_name}_rsplit.csv",
    'completeness': parent_dir / f"{base_name}_completeness.csv",
}

# Plot each if present
for key, path in files.items():
    if not path.exists():
        print(f"⚠️ {path.name} not found; skipping.", file=sys.stderr)
        continue
    # Adjust labels
    if key == 'completeness':
        df = pd.read_csv(path)
        x = df.iloc[:,0]
        y = df.iloc[:,1]
        y *= 100.0
        ylabel = 'Completeness (%)'
        title = 'Completeness vs Resolution'
    elif key == 'rsplit':
        df = pd.read_csv(path)
        x = df.iloc[:,3]
        y = df.iloc[:,6]
        ylabel = 'R_split (%)'
        title = 'R_split vs Resolution'
    else:   
    
        df = pd.read_csv(path)
        x = df.iloc[:,3]
        y = df.iloc[:,6]
        ylabel = 'CC1/2'
        title = 'CC1/2 vs Resolution'
    # Plot
    plt.figure()
    plt.plot(x, y, marker='o')
    # plt.gca().invert_xaxis()
    plt.xlabel('Resolution (Å)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = path.with_suffix('.png')
    plt.savefig(out_png)
    plt.show()
    print(f"   ↪︎ {out_png.name} created")
print('🎉 Plotting complete')
PY
else
  echo "⚠️ python3 not found; skipping plot generation."
fi

#!/usr/bin/env python3
"""
scale_symmetry_hist.py – Scale, export ASU CSV, and interactively plot histograms
================================================================================
This single script can now **(1) scale & export** a reflection table where every
reflection is tagged by its asymmetric‑unit (ASU) triple, **(2) reload** such a
CSV without re‑scaling, and **(3) let you interactively plot intensity
histograms for *any* ASU you type at the prompt.  Command‑line flags let you mix
and match the modes; sensible defaults mirror the hard‑coded *Configuration*
section so you can still just edit a few paths and run the file.

Typical workflows
-----------------
1. **One‑shot scaling + CSV export + interactive histograms**
   ```bash
   python scale_symmetry_hist.py               # uses hard‑coded config
   # or explicitly
   python scale_symmetry_hist.py /data/run.stream --interactive
   ```
2. **Later: reload the CSV only** (no .stream or cctbx needed)
   ```bash
   python scale_symmetry_hist.py --from-csv /data/run_asu.csv --interactive
   ```
3. **Batch: scale + write CSV only (no plots)**
   ```bash
   python scale_symmetry_hist.py --no-interactive --no-gui
   ```

Dependencies
~~~~~~~~~~~~
``pip install numpy pandas matplotlib tqdm cctbx-xfel`` (cctbx only needed when
reading a *.stream*).
"""

from __future__ import annotations

###############################################################################
# Configuration – EDIT THESE PATHS / PARAMS                                  #
###############################################################################
INPUT_STREAM          = "/Users/xiaodong/Desktop/MFM300_VIII_spot9/xgandalf_iterations_max_radius_0.5_step_0.1/filtered_metrics/filtered_metrics.stream"   # Path to input .stream
SPACE_GROUP_OVERRIDE  = "I4(1)22"        # e.g. "I4_122"; blank → take from header / CLI

CYCLES                = 5            # OSF refinement iterations
HIST_BINS             = 50           # Histogram bin count
SHOW_GUI_DEFAULT      = True         # Default GUI behaviour (overridable)
WRITE_ASU_CSV_DEFAULT = True         # Write CSV after scaling?
ASU_CSV_NAME          = "asu_intensities.csv"  # filename if --csv-out not given
###############################################################################

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Import helper routines from scale_stream_v7.py (only needed if scaling)         #
# -----------------------------------------------------------------------------

def try_import_scale_stream_v7():
    try:
        import scale_stream_v7
        return scale_stream_v7
    except ModuleNotFoundError:
        sys.exit("[!] Could not import 'scale_stream_v7.py'. Put this script alongside it "
                 "or add its directory to PYTHONPATH.")

# cctbx (only needed for symmetry mapping)
try:
    from cctbx import crystal, miller
    from cctbx.array_family import flex
except ImportError:
    crystal = miller = flex = None  # handled later if required

###############################################################################
# Helper functions (scaling, mapping, plotting)                               #
###############################################################################

def scale_and_map(stream_path: Path, cycles: int, sg_override: str | None):
    """Scale intensities and return a DataFrame with ASU tags."""
    scale_stream_v7 = try_import_scale_stream_v7()

    # --- scaling ---
    header, chunks = scale_stream_v7.parse_stream(stream_path)
    unit_cell, sg_hdr = scale_stream_v7.extract_symmetry(header)
    target = scale_stream_v7.refine_osf(chunks, cycles)
    scale_stream_v7.apply_stats(chunks, target)

    # save scaled stream (always)
    scaled_path = stream_path.with_name(stream_path.stem + "_scaled.stream")
    scale_stream_v7.write_stream(header, chunks, scaled_path)
    print(f"    → Scaled stream written to {scaled_path}")

    # --- symmetry mapping ---
    if sg_override:
        sg = sg_override
    elif sg_hdr:
        sg = sg_hdr
        print(f"[*] Using space group from stream header: {sg}")
    else:
        sg = input("Enter space‑group symbol (e.g. P212121): ").strip()

    if crystal is None:
        sys.exit("[!] cctbx not installed – required for symmetry mapping.")
    if unit_cell is None:
        sys.exit("[!] Unit‑cell parameters missing – cannot perform symmetry mapping.")

    rows = [
        (r.h, r.k, r.l, r.I) for ch in chunks for r in ch.reflections
    ]
    df = pd.DataFrame(rows, columns=["h", "k", "l", "I"])

    cs = crystal.symmetry(unit_cell=unit_cell, space_group_symbol=sg)
    ms = miller.set(cs, flex.miller_index(list(zip(df.h, df.k, df.l))), False)
    df["asu_h"], df["asu_k"], df["asu_l"] = zip(*ms.map_to_asu().indices())
    return df


def save_asu_csv(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path, index=False)
    print(f"    → ASU CSV written to {csv_path}")


def plot_histogram(data: np.ndarray, bins: int, out_png: Path, show: bool, title: str):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.xlabel("Scaled intensity I")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close()


def interactive_prompt(df: pd.DataFrame, bins: int, base_path: Path, show_gui: bool):
    """Loop asking the user which ASU to plot until they quit."""
    print("\n[Interactive mode] Enter ASU triples like 0,0,5 or 'all' to plot everything."
          "  Blank / q / quit to exit.\n")
    while True:
        entry = input("ASU ⟹ ").strip().lower()
        if entry in {"", "q", "quit", "exit"}:
            print("Leaving interactive mode.✓")
            break
        if entry in {"all", "*"}:
            subset = df
            label = "all"
        else:
            try:
                h, k, l = map(int, entry.split(","))
            except ValueError:
                print("  ! Please enter three integers separated by commas, e.g. 0,0,5")
                continue
            subset = df[(df.asu_h == h) & (df.asu_k == k) & (df.asu_l == l)]
            if subset.empty:
                print(f"  ! No reflections for ASU ({h},{k},{l})")
                continue
            label = f"{h},{k},{l}"
        out_png = base_path.with_name(f"intensity_histogram_{label}.png")
        print(f"  → plotting {len(subset)} reflections …")
        plot_histogram(subset.I.values, bins, out_png, show_gui,
                       f"Histogram – ASU {label}")
        print(f"    saved as {out_png}\n")

###############################################################################
# CLI                                                                        #
###############################################################################

def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        prog="scale_symmetry_hist.py",
        description="Scale reflections, export ASU CSV, and interactively plot histograms.")

    # Input / output options
    p.add_argument("stream", nargs="?", type=Path, default=None,
                   help="Input .stream file (default: INPUT_STREAM in config)")
    p.add_argument("--from-csv", metavar="CSV", type=Path, default=None,
                   help="Skip scaling and load this ASU intensity CSV instead")
    p.add_argument("--csv-out", metavar="CSV", type=Path, default=None,
                   help="Filename for the ASU CSV (default: same folder / ASU_CSV_NAME)")

    # Processing overrides
    p.add_argument("--sg", metavar="SYMBOL", default=None,
                   help="Space‑group symbol (overrides config/header)")
    p.add_argument("--cycles", type=int, default=CYCLES,
                   help="OSF refinement cycles (when scaling)")
    p.add_argument("--bins", type=int, default=HIST_BINS, help="Histogram bins")

    # Behaviour toggles
    p.add_argument("--interactive", dest="interactive", action="store_true",
                   help="Enter interactive histogram prompt (default if no --asu)")
    p.add_argument("--no-interactive", dest="interactive", action="store_false",
                   help="Do not enter interactive prompt")
    p.set_defaults(interactive=None)  # None = decide later

    p.add_argument("--gui", dest="show_gui", action="store_true",
                   help="Always show Matplotlib windows (overrides config)")
    p.add_argument("--no-gui", dest="show_gui", action="store_false",
                   help="Never show Matplotlib windows")
    p.set_defaults(show_gui=None)

    args = p.parse_args(argv)

    # -------------------- Decide input source --------------------
    if args.from_csv is not None:
        csv_path = args.from_csv.expanduser()
        if not csv_path.exists():
            sys.exit(f"[!] CSV not found: {csv_path}")
        print(f"[✓] Loading ASU CSV {csv_path} …")
        df = pd.read_csv(csv_path)
        base_path = csv_path.with_suffix("")
    else:
        stream_path = (args.stream or Path(INPUT_STREAM)).expanduser()
        if not stream_path.exists():
            sys.exit(f"[!] Input stream not found: {stream_path}")
        print("[1/3] Scaling & symmetry mapping …")
        df = scale_and_map(stream_path, args.cycles, args.sg or SPACE_GROUP_OVERRIDE or None)
        # write csv?
        if args.csv_out is not None:
            csv_path = args.csv_out.expanduser()
        else:
            csv_path = stream_path.with_name(ASU_CSV_NAME)
        if WRITE_ASU_CSV_DEFAULT:
            save_asu_csv(df, csv_path)
        base_path = stream_path.with_suffix("")

    # -------------------- Plotting behaviour --------------------
    show_gui = args.show_gui if args.show_gui is not None else SHOW_GUI_DEFAULT

    # Enter interactive prompt unless user explicitly disabled it AND they passed
    # --from-csv or scaling finished without wanting interactivity.
    interactive = args.interactive
    if interactive is None:  # not specified
        interactive = True  # default: yes

    if interactive:
        interactive_prompt(df, args.bins, base_path, show_gui)
    else:
        print("✓ Done (no interactive plotting requested).")


if __name__ == "__main__":
    main()

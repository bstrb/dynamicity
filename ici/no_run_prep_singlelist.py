#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
no_run_prep_singlelist.py
Step 1 with "defaults-if-no-args":
- If launched with NO arguments, uses hardcoded defaults for root/geom/cell/h5 and a standard flag set.
- If launched WITH arguments, behaves as a normal CLI and appends any flags after `--`.
- Always writes:
    <run-root>/runs/run_000/lst_000.lst
    <run-root>/runs/run_000/sh_000.sh
- Does NOT execute indexamajig; only writes the exact command to sh_000.sh.

Default dataset (used only when no CLI args are provided):
    default_root = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
    default_geom = default_root + "/MFM300-VIII.geom"
    default_cell = default_root + "/MFM300-VIII.cell"
    default_h5   = default_root + "/sim.h5"
"""

from __future__ import annotations
import argparse, os, sys, shlex
from pathlib import Path
from typing import List
import h5py

IMAGES_DS = "/entry/data/images"

# -------- Default config MacOS (applies ONLY when run with NO CLI args) --------
DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_GEOM = DEFAULT_ROOT + "/MFM300-VIII.geom"
DEFAULT_CELL = DEFAULT_ROOT + "/MFM300-VIII.cell"
DEFAULT_H5   = DEFAULT_ROOT + "/sim.h5"

# -------- Default config WSL(applies ONLY when run with NO CLI args) --------
# DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
# DEFAULT_GEOM = DEFAULT_ROOT + "/MFM300.geom"
# DEFAULT_CELL = DEFAULT_ROOT + "/MFM300.cell"
# DEFAULT_H5   = DEFAULT_ROOT + "/MFM300.h5"

DEFAULT_FLAGS = [
    # Peakfinding
    # "--peaks=cxi",
    "--peaks=peakfinder9",
    "--min-snr-biggest-pix=1",
    "--min-snr-peak-pix=6",
    "--min-snr=1",
    "--min-sig=11",
    "--min-peak-over-neighbour=-inf",
    "--local-bg-radius=3",
    # Other
    "-j", "24",
    "--min-peaks=15",
    "--tolerance=10,10,10,5",
    "--xgandalf-sampling-pitch=5",
    "--xgandalf-grad-desc-iterations=1",
    "--xgandalf-tolerance=0.02",
    "--int-radius=4,5,9",
    "--no-half-pixel-shift",
    "--no-non-hits-in-stream",
    "--no-retry",
    "--fix-profile-radius=70000000",
    "--indexing=xgandalf",
    "--integration=rings",
]


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _gather_h5(paths: List[str]) -> List[str]:
    out = set()
    for p in paths:
        ap = _abs(p)
        if os.path.isfile(ap) and ap.lower().endswith(".h5"):
            out.add(ap)
        elif os.path.isdir(ap):
            for root, _, files in os.walk(ap):
                for fn in files:
                    if fn.lower().endswith(".h5"):
                        out.add(os.path.join(root, fn))
    return sorted(out)


def _count_images(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        if IMAGES_DS not in f:
            raise KeyError(f"{IMAGES_DS} not found in {h5_path}")
        return int(f[IMAGES_DS].shape[0])


def _write_combined_lst(lst_path: str, h5_files: List[str]) -> int:
    """Write combined .lst across all HDF5s; returns total image count."""
    total = 0
    Path(lst_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lst_path, "w", encoding="utf-8") as f:
        for h5 in h5_files:
            n = _count_images(h5)
            for i in range(n):
                f.write(f"{h5} //{i}\n")
            total += n
    return total


def _write_sh(sh_path: str, indexamajig_exec: str, geom: str, cell: str, lst: str, out_stream: str, extra_flags: List[str]) -> None:
    Path(sh_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [indexamajig_exec, "-g", geom, "-i", lst, "-o", out_stream, "-p", cell, *extra_flags]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Exact command that will be run:",
        " ".join(shlex.quote(c) for c in cmd),
    ]
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.chmod(sh_path, 0o755)

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Create a single combined .lst and a sh_000.sh with the exact indexamajig command (no execution).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-root", help="Root folder under which 'runs/run_000' will be created.")
    ap.add_argument("--geom", help="Path to .geom file.")
    ap.add_argument("--cell", help="Path to .cell file.")
    ap.add_argument("--indexamajig", default="indexamajig", help="indexamajig executable name/path to reference in the script.")
    ap.add_argument("sources", nargs="*", help="One or more .h5 files or directories to search recursively for .h5.")
    ap.add_argument("--", dest="idx_sep", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv: List[str]) -> int:
    # Split argv around '--' to forward everything after to indexamajig
    if "--" in argv:
        sep = argv.index("--")
        ours = argv[:sep]
        idx_flags = argv[sep + 1 :]
    else:
        ours = argv
        idx_flags = []

    ap = build_argparser()
    args = ap.parse_args(ours)

    using_defaults = (len(ours) == 0)

    if using_defaults:
        run_root = _abs(DEFAULT_ROOT)
        geom = _abs(DEFAULT_GEOM)
        cell = _abs(DEFAULT_CELL)
        idx_exec = "indexamajig"
        h5s = _gather_h5([DEFAULT_H5])
        # With defaults, start with DEFAULT_FLAGS and then append any flags after `--`
        eff_flags = list(DEFAULT_FLAGS) + idx_flags
    else:
        # Validate provided arguments
        if not args.run_root or not args.geom or not args.cell or not args.sources:
            print("ERROR: Missing required arguments. Provide --run-root, --geom, --cell and at least one source, or run with NO args to use defaults.", file=sys.stderr)
            return 2
        run_root = _abs(args.run_root)
        geom = _abs(args.geom)
        cell = _abs(args.cell)
        idx_exec = args.indexamajig
        h5s = _gather_h5(args.sources)
        eff_flags = idx_flags  # no default flags in CLI mode unless user passes them

    if not h5s:
        print("No .h5 sources found.", file=sys.stderr)
        return 2

    run_dir = os.path.join(run_root, "runs", "run_000")
    os.makedirs(run_dir, exist_ok=True)

    lst_path = os.path.join(run_dir, "lst_000.lst")
    sh_path  = os.path.join(run_dir, "sh_000.sh")
    stream_path = os.path.join(run_dir, "stream_000.stream")

    print("=== Step 1 (single-list): plan indexamajig run (no execution) ===")
    print(f"Mode:        {'DEFAULTS' if using_defaults else 'CLI'}")
    print(f"Run dir:     {run_dir}")
    print(f"Geom:        {geom}")
    print(f"Cell:        {cell}")
    print(f"HDF5 files:  {len(h5s)}")
    print(f"Extra flags: {' '.join(eff_flags) if eff_flags else '(none)'}")
    print("===============================================================")

    total_images = _write_combined_lst(lst_path, h5s)
    print(f"Wrote combined list: {lst_path}  (total images: {total_images})")

    _write_sh(sh_path, idx_exec, geom, cell, lst_path, stream_path, eff_flags)
    print(f"Wrote shell script:  {sh_path}")
    print("Inspect the .lst and .sh to verify before running.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

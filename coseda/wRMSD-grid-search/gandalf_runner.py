#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gandalf_runner.py
CLI entrypoint for Adaptive Center-Shift Optimization.

Usage examples:
  # Minimal (threads etc. passed after -- to indexamajig)
  python gandalf_runner.py \
      --run-root /path/to/run \
      --geom /path/to/setup.geom \
      --cell /path/to/cell.cell \
      /data/file1.h5 /data/file2.h5 \
      -- -j 32 --peaks peaks.conf

  # With custom knobs
  python gandalf_runner.py \
      --run-root ./run_adaptive \
      --geom geom.geom --cell cell.cell \
      /data/dir_with_h5s \
      --R-px 1.0 --s-init-px 0.2 --K-dir 10 \
      --s-refine-px 0.5 --s-min-px 0.1 \
      --eps-rel 0.007 --N-eval-max 16 \
      -- -j 48 --some-indexamajig-flag value
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List

from gandalfiterator import Params, gandalf_adaptive


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _gather_h5_sources(paths: List[str]) -> List[str]:
    """
    Accept mixed list of files and directories.
    - Files ending with .h5 are used directly.
    - Directories are searched recursively for *.h5.
    Returns unique absolute paths, sorted.
    """
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
        else:
            # ignore non-existent or non-h5 files
            continue
    return sorted(out)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Adaptive per-image center optimization (seed -> ring -> refine) with indexamajig waves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required core inputs
    ap.add_argument("--run-root", required=True, help="Run output root directory.")
    ap.add_argument("--geom", required=True, help="Path to .geom file.")
    ap.add_argument("--cell", required=True, help="Path to .cell file.")

    # Data sources (positional: one or more files/dirs)
    ap.add_argument("sources", nargs="+", help="One or more .h5 files or directories to search for .h5 files.")

    # Adaptive knobs (your defaults)
    ap.add_argument("--R-px", type=float, default=1.0, dest="R_px",
                    help="Trust-region radius around seed (pixels).")
    ap.add_argument("--s-init-px", type=float, default=0.2, dest="s_init_px",
                    help="Ring radius step (pixels).")
    ap.add_argument("--K-dir", type=int, default=10, dest="K_dir",
                    help="Directions per ring (golden-angle order).")
    ap.add_argument("--s-refine-px", type=float, default=0.5, dest="s_refine_px",
                    help="Initial refinement step (pixels).")
    ap.add_argument("--s-min-px", type=float, default=0.1, dest="s_min_px",
                    help="Minimum refinement step (stop threshold, pixels).")
    ap.add_argument("--eps-rel", type=float, default=0.007, dest="eps_rel",
                    help="Required relative improvement in wRMSD (e.g., 0.007 = 0.7%%).")
    ap.add_argument("--N-eval-max", type=int, default=16, dest="N_eval_max",
                    help="Hard cap on attempts per image.")
    ap.add_argument("--tie-tol-rel", type=float, default=0.01, dest="tie_tol_rel",
                    help="Relative tie tolerance on wRMSD (e.g., 0.01 = 1%%).")

    # Behavior toggles
    ap.add_argument("--no-8-connected", action="store_true",
                    help="Disable diagonal neighbors in refinement (use 4-connected only).")
    ap.add_argument("--no-directional-refine", action="store_true",
                    help="Disable directional prioritization even after two successes.")

    # Separator for indexamajig flags: anything after '--' is passed verbatim
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

    run_root = _abs(args.run_root)
    geom_path = _abs(args.geom)
    cell_path = _abs(args.cell)

    h5_sources = _gather_h5_sources(args.sources)
    if not h5_sources:
        print("No .h5 sources found. Provide .h5 files or directories.", file=sys.stderr)
        return 2

    params = Params(
        R_px=args.R_px,
        s_init_px=args.s_init_px,
        K_dir=args.K_dir,
        s_refine_px=args.s_refine_px,
        s_min_px=args.s_min_px,
        eps_rel=args.eps_rel,
        N_eval_max=args.N_eval_max,
        tie_tol_rel=args.tie_tol_rel,
        eight_connected=not args.no_8_connected,
        directional_refine=not args.no_directional_refine,
    )

    os.makedirs(run_root, exist_ok=True)

    print("=== Adaptive Center Optimization ===")
    print(f"Run root:    {run_root}")
    print(f"Geom:        {geom_path}")
    print(f"Cell:        {cell_path}")
    print(f"HDF5 files:  {len(h5_sources)}")
    print(f"Indexamajig flags: {' '.join(idx_flags) if idx_flags else '(none)'}")
    print(f"Defaults: R_px={params.R_px}, s_init_px={params.s_init_px}, K_dir={params.K_dir}, "
          f"s_refine_px={params.s_refine_px}, s_min_px={params.s_min_px}, eps_rel={params.eps_rel}, "
          f"N_eval_max={params.N_eval_max}, tie_tol_rel={params.tie_tol_rel}, "
          f"8-connected={params.eight_connected}, directional={params.directional_refine}")
    print("====================================")

    merged_path = gandalf_adaptive(
        run_root=run_root,
        geom_path=geom_path,
        cell_path=cell_path,
        h5_sources=h5_sources,
        params=params,
        indexamajig_flags_passthrough=idx_flags,
    )

    if merged_path:
        print(f"\nMerged stream written to: {merged_path}")
        return 0
    else:
        print("\nNo merged stream produced (no FINAL images).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

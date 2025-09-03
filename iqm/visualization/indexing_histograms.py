#!/usr/bin/env python3
# visualization/indexing_histograms.py
import os
import re
import sys
import glob
import math
import argparse
import mmap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import tri

# -------- Fast, memory-efficient counters (binary; no decoding) -------- #

# Compiled once; line-anchored, case-insensitive. Counts occurrences.
_RE_INDEXED = re.compile(br'(?mi)^[ \t]*num_reflections[ \t]*=')
_RE_HITS    = re.compile(br'(?mi)^[ \t]*num_peaks[ \t]*[:=][ \t]*\d+')

def _count_stream_file(path: Path) -> Optional[Tuple[float, float, int, int, float, str]]:
    """
    Return (x, y, num_indexed, num_hits, pct, filename) for a *.stream file.
    Skips files whose stem doesn't end with _<x>_<y>.
    """
    stem = path.stem
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return None
    try:
        x = float(parts[-2])
        y = float(parts[-1])
    except ValueError:
        return None

    try:
        with path.open("rb") as fh:
            with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                num_indexed = sum(1 for _ in _RE_INDEXED.finditer(mm))
                num_hits    = sum(1 for _ in _RE_HITS.finditer(mm))
    except Exception as e:
        sys.stderr.write(f"[warn] Skipping {path}: {e}\n")
        return None

    pct = (100.0 * num_indexed / num_hits) if num_hits > 0 else math.nan
    return (x, y, num_indexed, num_hits, pct, str(path))


def _make_plots(df: pd.DataFrame, title_prefix: str = "Indexing rate") -> None:
    """
    Generates a 3D trisurface and a 2D top-view heatmap using only Matplotlib.
    """
    if df.empty:
        raise RuntimeError("No valid data to plot.")

    finite_rates = df["rate"].replace([np.inf, -np.inf], np.nan).dropna()
    vmin = float(finite_rates.min()) if not finite_rates.empty else 0.0
    vmax = float(finite_rates.max()) if not finite_rates.empty else 1.0

    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.25)

    # ---- 3D surface ----
    ax3d = fig.add_subplot(gs[0], projection="3d")
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin, vmax)

    tri_obj = tri.Triangulation(df.x.values, df.y.values)
    surf = ax3d.plot_trisurf(
        tri_obj,
        df.rate.values,
        cmap=cmap,
        linewidth=0.2,
        antialiased=True,
        edgecolor="none",
        norm=norm,
        alpha=0.95,
    )

    ax3d.set_xlabel("X coordinate shift (pixels)", labelpad=8)
    ax3d.set_ylabel("Y coordinate shift (pixels)", labelpad=8)
    ax3d.set_zlabel("Indexing rate (%)", labelpad=8)
    ax3d.set_title(f"{title_prefix} surface", pad=12, fontsize=12)
    ax3d.view_init(elev=30, azim=135)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        try:
            pane.set_alpha(0.1)
        except Exception:
            pass
    cbar = fig.colorbar(surf, ax=ax3d, fraction=0.025, pad=0.08, shrink=0.9, aspect=15)
    cbar.set_label("Indexing rate (%)")

    # ---- 2D heatmap (tri-based interpolation; no SciPy) ----
    ax2d = fig.add_subplot(gs[1])
    xi = np.linspace(df.x.min(), df.x.max(), 200)
    yi = np.linspace(df.y.min(), df.y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    interp = tri.LinearTriInterpolator(tri_obj, df.rate.values)
    Zi = interp(Xi, Yi)

    hm = ax2d.pcolormesh(Xi, Yi, Zi, cmap=cmap, norm=norm, shading="auto")
    ax2d.set_xlabel("X coordinate shift (pixels)")
    ax2d.set_ylabel("Y coordinate shift (pixels)")
    ax2d.set_title("Heat-map (top view)")
    fig.colorbar(hm, ax=ax2d, fraction=0.046, pad=0.04).set_label("Indexing rate (%)")

    plt.tight_layout()
    plt.show()


def _find_streams(folder: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.stream" if recursive else "*.stream"
    return [Path(p) for p in glob.glob(str(folder / pattern), recursive=recursive)]


def run(folder: Path,
        jobs: int = 0,
        recursive: bool = False,
        csv_out: Optional[Path] = None,
        no_plot: bool = False) -> pd.DataFrame:
    """
    Scans *.stream files, computes counts and percentages, optionally plots, and returns a DataFrame.
    """
    files = _find_streams(folder, recursive)
    if not files:
        raise SystemExit(f"No *.stream files found under: {folder}")

    results = []
    with ProcessPoolExecutor(max_workers=jobs or None) as ex:
        fut_map = {ex.submit(_count_stream_file, p): p for p in files}
        for fut in as_completed(fut_map):
            rec = fut.result()
            if rec is not None:
                results.append(rec)

    if not results:
        raise SystemExit("No valid *.stream files matched the naming pattern ..._<x>_<y>.stream")

    df = pd.DataFrame(
        results,
        columns=["x", "y", "num_indexed", "num_hits", "rate", "file"]
    ).sort_values(["x", "y"], kind="mergesort")

    if csv_out:
        df.to_csv(csv_out, index=False)

    if not no_plot:
        _make_plots(df.assign(rate=df["rate"].astype(float)))

    return df


# ------------------ Backward-compatible wrapper for GUIs ------------------ #
def plot_indexing_rate(folder_path: str) -> None:
    """
    Backward-compatible API expected by existing GUIs.
    Equivalent to: run(Path(folder_path), jobs=0, recursive=False, csv_out=None, no_plot=False)
    """
    run(Path(folder_path), jobs=0, recursive=False, csv_out=None, no_plot=False)


# ------------------------------- CLI ------------------------------------- #
def _main():
    ap = argparse.ArgumentParser(
        description="Fast indexing-rate maps from large *.stream files (memory-mapped scanning)."
    )
    ap.add_argument("folder", help="Folder containing *.stream files")
    ap.add_argument("--jobs", "-j", type=int, default=0, help="Parallel workers (default: CPU count)")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--csv-out", type=Path, default=None, help="Write per-file stats as CSV")
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting; only compute/print")
    args = ap.parse_args()
    df = run(Path(args.folder), jobs=args.jobs, recursive=args.recursive, csv_out=args.csv_out, no_plot=args.no_plot)

    # Pretty print a brief summary if invoked via CLI
    summary = df[["file", "num_indexed", "num_hits", "rate"]]
    with pd.option_context("display.float_format", "{:0.4f}".format):
        print(summary.to_string(index=False))

if __name__ == "__main__":
    _main()

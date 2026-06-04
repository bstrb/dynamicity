#!/usr/bin/env python3
"""
Plot clean, separated rocking curves from PETS .rprofall files (Iobs only).

Expected fixed-width layout per data line:
- h, k, l: 4 chars each
- resolution, excitation, RSg, Iobs, sigma, Icalc: 14 chars each
- frame: 4 chars
- azimuth: 9 chars
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


@dataclass
class RProfallRow:
    h: int
    k: int
    l: int
    excitation: float
    iobs: float
    frame: int


def parse_fixed_width_line(line: str) -> RProfallRow | None:
    """Parse one fixed-width .rprofall data line."""
    if not line.strip() or line.startswith("#"):
        return None
    if len(line) < 109:
        return None

    try:
        h = int(line[0:4])
        k = int(line[4:8])
        l = int(line[8:12])
        excitation = float(line[26:40])
        iobs = float(line[54:68])
        frame = int(line[96:100])
    except ValueError:
        return None

    return RProfallRow(h=h, k=k, l=l, excitation=excitation, iobs=iobs, frame=frame)


def extract_reflection(path: Path, hkl: tuple[int, int, int]) -> list[RProfallRow]:
    """Return rows matching hkl from one .rprofall file."""
    h_target, k_target, l_target = hkl
    rows: list[RProfallRow] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            row = parse_fixed_width_line(raw_line)
            if row is None:
                continue
            if (row.h, row.k, row.l) == (h_target, k_target, l_target):
                rows.append(row)

    return rows


def build_palette(n: int) -> list[str]:
    """
    Colorblind-friendly starting colors, then tab20 fallback for many curves.
    """
    base = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # green
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ]
    if n <= len(base):
        return base[:n]

    cmap = plt.get_cmap("tab20")
    colors = list(base)
    for i in range(n - len(base)):
        colors.append(cmap(i % 20))
    return colors[:n]


def normalize(values: Iterable[float], mode: str) -> list[float]:
    arr = [max(v, 0.0) for v in values]
    if not arr:
        return arr
    if mode == "none":
        return arr
    vmax = max(arr)
    if vmax <= 0:
        return arr
    return [v / vmax for v in arr]


def default_prefix(hkl: tuple[int, int, int]) -> str:
    h, k, l = hkl
    return f"rocking_curves_hkl_{h}_{k}_{l}_iobs_clean_separated".replace("-", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot separated, clean rocking curves (Iobs only) from .rprofall files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input .rprofall files (pass as many as you want).",
    )
    parser.add_argument(
        "--hkl",
        nargs=3,
        required=True,
        type=int,
        metavar=("H", "K", "L"),
        help="Target reflection, e.g. --hkl -1 -1 0",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for files (same count as files).",
    )
    parser.add_argument(
        "--x",
        choices=["exc", "frame"],
        default="exc",
        help="X-axis choice: excitation ('exc') or frame number ('frame').",
    )
    parser.add_argument(
        "--normalize",
        choices=["max", "none"],
        default="max",
        help="Normalize each curve by its own max Iobs (default: max).",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=1.3,
        help="Vertical offset step between curves (default: 1.3).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI for PNG (default: 220).",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=10.2,
        help="Figure width in inches (default: 10.2).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=6.2,
        help="Figure height in inches (default: 6.2).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output prefix (default derived from hkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hkl = tuple(args.hkl)

    if args.labels is not None and len(args.labels) != len(args.files):
        raise SystemExit("ERROR: --labels count must match number of input files.")

    labels = args.labels or [p.stem for p in args.files]

    series = []
    for path, label in zip(args.files, labels):
        if not path.exists():
            print(f"WARNING: missing file, skipped: {path}")
            continue

        rows = extract_reflection(path, hkl)
        if not rows:
            print(f"WARNING: no {hkl} rows in {path}, skipped.")
            continue

        if args.x == "exc":
            rows.sort(key=lambda r: r.excitation)
            x_vals = [r.excitation * 1e3 for r in rows]
            x_label = "Excitation (x10^-3)"
        else:
            rows.sort(key=lambda r: r.frame)
            x_vals = [float(r.frame) for r in rows]
            x_label = "Frame Number"

        y_vals = normalize((r.iobs for r in rows), args.normalize)
        series.append((label, x_vals, y_vals, len(rows)))

    if not series:
        raise SystemExit("ERROR: no plottable series found.")

    colors = build_palette(len(series))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#222222",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)

    for idx, (label, x_vals, y_vals, npts) in enumerate(series):
        y_shift = [v + idx * args.offset for v in y_vals]
        color = colors[idx]

        ax.plot(
            x_vals,
            y_shift,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            label=label,
        )
        ax.scatter(
            x_vals,
            y_shift,
            color=color,
            s=14,
            zorder=3,
            edgecolors="white",
            linewidths=0.45,
        )

        ax.text(
            max(x_vals),
            y_shift[-1],
            f"{label} (n={npts})",
            color=color,
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="semibold",
        )

    title = args.title or f"Rocking Curves for hkl = {hkl}  [Iobs only]"
    ax.set_title(title, pad=12, fontsize=14)
    ax.set_xlabel(x_label)
    ylabel = "Iobs"
    if args.normalize == "max":
        ylabel = "Normalized Iobs"
    ax.set_ylabel(f"{ylabel} (offset for separation)")

    ax.grid(axis="x", color="#CFCFCF", alpha=0.45, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.13)

    fig.tight_layout()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or default_prefix(hkl)

    out_png = outdir / f"{prefix}.png"
    out_pdf = outdir / f"{prefix}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"SAVED {out_png}")
    print(f"SAVED {out_pdf}")
    for label, _, _, npts in series:
        print(f"{label}: {npts} points")


if __name__ == "__main__":
    main()


# =====================================================================
# file: cli.py  ── command‑line interface
# =====================================================================
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from dynio import load_frame
from scoring import add_metrics, combine_scores


def _parse_cell(text: str):
    parts = [float(x) for x in text.replace(",", " ").split()]
    if len(parts) == 3:
        return (*parts, 90.0, 90.0, 90.0)
    if len(parts) == 6:
        return tuple(parts)
    raise argparse.ArgumentTypeError("Unit cell must have 3 or 6 numbers")


def main(argv=None):
    p = argparse.ArgumentParser("dynscatt – per‑reflection dynamical weighting")
    p.add_argument("input", help="HDF5 frame or .stream file (single pattern)")
    p.add_argument("-g", "--spacegroup", required=True)
    p.add_argument("-c", "--cell", required=True, type=_parse_cell)
    p.add_argument("--min-I", type=float, default=0.0, help="intensity cutoff for Wilson fit")
    p.add_argument("--min-SN", type=float, default=0.0, help="I/σ cutoff for Wilson fit")
    p.add_argument("-o", "--out")
    ns = p.parse_args(argv)

    console = Console()
    df, meta = load_frame(ns.input)
    console.print(f"Loaded {len(df)} reflections from [cyan]{ns.input}[/] (format {meta['format']})")

    df = add_metrics(
        df,
        sg_symbol=ns.spacegroup,
        cell=ns.cell,
        min_I=ns.min_I,
        min_sn=ns.min_SN,
    )
    df = combine_scores(df)

    tbl = Table()
    tbl.add_column("metric"); tbl.add_column("mean"); tbl.add_column("std")
    for m in ("m_equiv", "m_wilson", "m_rank", "dyn_score", "w_dyn"):
        tbl.add_row(m, f"{df[m].mean():.3g}", f"{df[m].std():.3g}")
    console.print(tbl)

    out = Path(ns.out or f"{ns.input}.dynweights.csv")
    df.to_csv(out, index=False)
    console.print(f"[green]Saved → {out}")

if __name__ == "__main__":
    main()

# =====================================================================
# ... cli code identical ...

# =====================================================================
# requirements.txt
# =====================================================================
# cctbx-base   # optional but recommended; install via conda-forge
# gemmi>=0.6.3
# h5py
# numpy
# pandas
# scipy
# rich

#!/usr/bin/env python3
"""
stream_to_dataframe.py

Create pandas DataFrames from a CrystFEL *.stream* file using the
light‑weight ``StreamParser``.  No files are written; the script is meant
purely for in‑memory analysis or quick terminal inspection.

DataFrames returned (or shown):
    1. ``header_df``      – single‑row global header
    2. ``peaks_df``       – all peak‑search spots
    3. ``reflections_df`` – all indexed reflections

Example (Python REPL)
---------------------
    from stream_to_dataframe import stream_to_dfs
    hdr_df, peaks_df, refl_df = stream_to_dfs('run.stream')

Command‑line (quick peek)
-------------------------
    python stream_to_dataframe.py run.stream       # prints 5 rows of each
    python stream_to_dataframe.py run.stream --head 10
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

# local import (assumes this script sits next to parse_crystfel_stream.py)
from parse_crystfel_stream import StreamParser

__all__ = ["stream_to_dfs"]

# -----------------------------------------------------------------------------
# Public helper
# -----------------------------------------------------------------------------

def stream_to_dfs(stream_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse *stream_path* and return (header_df, peaks_df, reflections_df)."""
    parser = StreamParser(stream_path)
    parser.parse()

    header_df = pd.DataFrame([parser.header.__dict__])

    peaks_df = pd.DataFrame([
        {"event": f.event, **p.__dict__}
        for f in parser.frames for p in f.peaks
    ])

    reflections_df = pd.DataFrame([
        {"event": f.event, **r.__dict__}
        for f in parser.frames for r in f.reflections
    ])

    return header_df, peaks_df, reflections_df

# -----------------------------------------------------------------------------
# CLI: inspection‑only
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert .stream to in‑memory pandas DataFrames (no saving)")
    ap.add_argument("stream", help="Input .stream file")
    ap.add_argument("--head", type=int, default=5, help="Print first N rows of each DataFrame")
    args = ap.parse_args()

    hdr_df, peaks_df, refl_df = stream_to_dfs(args.stream)

    n = args.head
    print("HEADER (1 row):")
    print(hdr_df)
    print(f"\nPEAKS ({len(peaks_df)} rows, showing {n}):")
    print(peaks_df.head(n))
    print(f"\nREFLECTIONS ({len(refl_df)} rows, showing {n}):")
    print(refl_df.head(n))

if __name__ == "__main__":
    main()

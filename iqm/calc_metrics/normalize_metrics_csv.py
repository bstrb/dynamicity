#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize CrystFEL metrics CSV (global or per chunk), with minmax or zscore.

Input CSV layout (your current output):
  stream_file,weighted_rmsd,fraction_outliers,length_deviation,angle_deviation,peak_ratio,percentage_unindexed
  # Image filename: <path>
  # Event: //<event>
  <rows...>
  # Image filename: ...
  # Event: ...
  <rows...>

Behavior:
- Preserves header and comment lines.
- Normalizes numeric columns:
    weighted_rmsd, fraction_outliers, length_deviation,
    angle_deviation, peak_ratio, percentage_unindexed
- Handles 'NA' as missing (skipped in stats; remains 'NA' in output).
- Global scope: two-pass (stats, then apply), memory-safe.
- Per-chunk scope: one-pass (buffer a chunk, normalize, write).

Usage examples:
  # Global z-score
  python normalize_metrics_csv.py \
      --input /path/to/center_shift_metrics.csv \
      --output /path/to/center_shift_metrics.zscore.global.csv \
      --method zscore --scope global

  # Per-chunk minmax
  python normalize_metrics_csv.py \
      --input /path/to/center_shift_metrics.csv \
      --output /path/to/center_shift_metrics.minmax.perchunk.csv \
      --method minmax --scope per-chunk
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

NUMERIC_COLS = [
    "weighted_rmsd",
    "fraction_outliers",
    "length_deviation",
    "angle_deviation",
    "peak_ratio",
    "percentage_unindexed",
]

HEADER_PREFIX = "stream_file,"  # to spot header line quickly


class Welford:
    """Online mean/std (population std by default; we use sample std)."""

    __slots__ = ("n", "mean", "M2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def mean_std(self) -> Tuple[Optional[float], Optional[float]]:
        if self.n == 0:
            return None, None
        if self.n == 1:
            return self.mean, 0.0
        # sample std (ddof=1) to match most zscore expectations
        var = self.M2 / (self.n - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        return self.mean, std


def parse_row(line: str, header: List[str]) -> Optional[List[str]]:
    """Parse a CSV data row into a list; return None for comments/blank."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    # Use csv to be safe with commas in future
    return next(csv.reader([line]))


def to_float_or_none(x: str) -> Optional[float]:
    if x == "NA":
        return None
    try:
        return float(x)
    except Exception:
        return None


def fmt_val(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    # 6 decimals is a good general precision for normalized values
    return f"{x:.6f}"


def find_col_indices(header: List[str]) -> Dict[str, int]:
    idx = {}
    for name in NUMERIC_COLS:
        if name not in header:
            raise ValueError(f"Column '{name}' not found in header: {header}")
        idx[name] = header.index(name)
    return idx


def pass1_global_stats(path: Path, header: List[str]) -> Tuple[Dict[str, Welford], Dict[str, Tuple[float, float]]]:
    """Return z-score stats (mean/std) and min/max for each numeric column."""
    zstats = {k: Welford() for k in NUMERIC_COLS}
    minmax: Dict[str, Tuple[float, float]] = {k: (math.inf, -math.inf) for k in NUMERIC_COLS}
    idx = find_col_indices(header)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = parse_row(line, header)
            if row is None:
                continue
            for col in NUMERIC_COLS:
                v = to_float_or_none(row[idx[col]])
                if v is None:
                    continue
                zstats[col].add(v)
                mn, mx = minmax[col]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
                minmax[col] = (mn, mx)
    return zstats, minmax


def normalize_row(
    row: List[str],
    header: List[str],
    method: str,
    zstats: Optional[Dict[str, Welford]],
    minmax: Optional[Dict[str, Tuple[float, float]]],
) -> List[str]:
    """Return a new row with numeric columns normalized."""
    idx = find_col_indices(header)
    out = row[:]
    for col in NUMERIC_COLS:
        s = row[idx[col]]
        v = to_float_or_none(s)
        if v is None:
            out[idx[col]] = "NA"
            continue
        if method == "zscore":
            assert zstats is not None
            m, sd = zstats[col].mean_std()
            if m is None or sd is None or sd == 0.0:
                out[idx[col]] = fmt_val(0.0)
            else:
                out[idx[col]] = fmt_val((v - m) / sd)
        elif method == "minmax":
            assert minmax is not None
            mn, mx = minmax[col]
            if not math.isfinite(mn) or not math.isfinite(mx) or mx == mn:
                # No spread or no valid numbers → neutral midpoint
                out[idx[col]] = fmt_val(0.5)
            else:
                out[idx[col]] = fmt_val((v - mn) / (mx - mn))
        else:
            raise ValueError(f"Unknown method: {method}")
    return out


def write_rows_preserving_comments(
    in_path: Path,
    out_path: Path,
    header: List[str],
    method: str,
    scope: str,
    zstats: Optional[Dict[str, Welford]] = None,
    minmax: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """
    scope == 'global': use provided global stats (two-pass).
    scope == 'per-chunk': buffer rows within each (# Image filename/# Event) block,
                          compute per-chunk stats, write, then continue.
    """
    if scope == "global":
        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(header)  # header once
            for line in fin:
                s = line.rstrip("\n")
                if not s:
                    continue
                if s.startswith("#"):
                    # copy comments verbatim
                    fout.write(s + "\n")
                    continue
                row = next(csv.reader([s]))
                norm = normalize_row(row, header, method, zstats, minmax)
                w.writerow(norm)
        return

    # per-chunk: one pass, buffer rows until next chunk boundary
    assert scope == "per-chunk"
    buffer: List[List[str]] = []
    comments_buffer: List[str] = []
    current_chunk_started = False

    def flush_chunk(fout_csv):
        nonlocal buffer, comments_buffer
        if not buffer:
            # even if no rows, still flush comments so structure remains
            for c in comments_buffer:
                fout_csv.write(c + "\n")
            comments_buffer.clear()
            return

        # compute stats for this chunk
        zstats_chunk = {k: Welford() for k in NUMERIC_COLS}
        minmax_chunk: Dict[str, Tuple[float, float]] = {k: (math.inf, -math.inf) for k in NUMERIC_COLS}
        idx = find_col_indices(header)
        for row in buffer:
            for col in NUMERIC_COLS:
                v = to_float_or_none(row[idx[col]])
                if v is None:
                    continue
                zstats_chunk[col].add(v)
                mn, mx = minmax_chunk[col]
                minmax_chunk[col] = (v if v < mn else mn, v if v > mx else mx)

        # write comments then normalized rows
        for c in comments_buffer:
            fout_csv.write(c + "\n")
        comments_buffer.clear()

        w = csv.writer(fout_csv)
        for row in buffer:
            norm = normalize_row(
                row,
                header,
                method,
                zstats_chunk if method == "zscore" else None,
                minmax_chunk if method == "minmax" else None,
            )
            w.writerow(norm)
        buffer.clear()

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header)

        for raw in fin:
            s = raw.rstrip("\n")
            if not s:
                continue
            if s.startswith("#"):
                # A new chunk boundary is encountered when we see "# Image filename" AFTER we've already started a chunk
                if s.startswith("# Image filename:") and current_chunk_started:
                    flush_chunk(fout)
                comments_buffer.append(s)
                current_chunk_started = True
                continue

            # data row
            row = next(csv.reader([s]))
            buffer.append(row)

        # flush last chunk
        flush_chunk(fout)


def main():
    ap = argparse.ArgumentParser(description="Normalize metrics CSV globally or per chunk (minmax/zscore).")
    ap.add_argument("--input", required=True, help="Path to input center_shift_metrics.csv")
    ap.add_argument("--output", required=True, help="Path to write normalized CSV")
    ap.add_argument("--method", choices=["minmax", "zscore"], default="zscore", help="Normalization method")
    ap.add_argument("--scope", choices=["global", "per-chunk"], default="global", help="Normalization scope")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Find header
    with in_path.open("r", encoding="utf-8") as f:
        header_line = None
        for line in f:
            if line.startswith(HEADER_PREFIX):
                header_line = line.strip()
                break
        if not header_line:
            raise RuntimeError("Header line not found in input CSV.")
        header = next(csv.reader([header_line]))

    if args.scope == "global":
        zstats, minmax = pass1_global_stats(in_path, header)
        write_rows_preserving_comments(
            in_path, out_path, header, args.method, "global", zstats=zstats, minmax=minmax
        )
    else:
        write_rows_preserving_comments(
            in_path, out_path, header, args.method, "per-chunk"
        )

    print(f"Normalized CSV written → {out_path}")


if __name__ == "__main__":
    main()

# python normalize_metrics_csv.py --input /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/center_shift_metrics.csv --output /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/center_shift_metrics.zscore.global.csv --method zscore --scope global


# python normalize_metrics_csv.py --input /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/center_shift_metrics.csv --output /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/center_shift_metrics.zscore.per-chunk.csv --method zscore --scope per-chunk


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refine detector center shifts (dx, dy) per event directly from a CrystFEL stream.

Approach:
- Parse header geometry (single-panel 'p0') to get res (px/mm) and nominal center (fs0, ss0).
- For each indexed chunk:
    * Parse "Peaks from peak search" -> observed peaks (fs, ss, I, panel)
    * Parse "Reflections measured after indexing" -> reflections (fs, ss, panel)
    * Greedy bipartite match reflections <-> peaks within max_dist (px) AND same panel
    * Use the matched subset of (fs, ss, I) to robustly estimate the true center (cx, cy)
      via symmetric pairing across rows/columns + weighted median with MAD rejection.
    * Fallback to all peaks if matched subset is too small.
    * Compute dx_px = cx - fs0, dy_px = cy - ss0, then mm shifts via res.

Output CSV columns:
    image_filename,event,dx_px,dy_px,dx_mm,dy_mm,
    n_peaks,n_reflections,n_matched,n_pairs_x,n_pairs_y,
    predict_refine_dx_mm,predict_refine_dy_mm

Assumptions/Notes:
- Single-panel geometry ('p0') is assumed (as in your example). For multi-panel runs,
  you likely want to transform each panel into a common metric before combining.
- The matching threshold defaults to 0.75 px; adjust with --max-match-dist if needed.
"""

import argparse
import csv
import math
import re
import sys
from statistics import median
from typing import List, Tuple, Optional


# -------------------- Parsing: header & chunks --------------------

HEADER_GEOM_BEGIN = "----- Begin geometry file -----"
HEADER_GEOM_END   = "----- End geometry file -----"
CHUNK_BEGIN = "----- Begin chunk -----"
CHUNK_END   = "----- End chunk -----"

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def extract_geometry_block(stream_text: str) -> str:
    m = re.search(
        re.escape(HEADER_GEOM_BEGIN) + r"(.*?)" + re.escape(HEADER_GEOM_END),
        stream_text, re.S
    )
    if not m:
        sys.stderr.write("ERROR: Geometry block not found in stream header.\n")
        sys.exit(1)
    return m.group(1)

def parse_header_single_panel(geom_text: str):
    """
    Parse single-panel geometry:
        res (px/mm), p0/min_fs, p0/max_fs, p0/min_ss, p0/max_ss
    Compute nominal center as (min+max)/2 + 0.5 for fs, ss.
    """
    # res (px/mm)
    m = re.search(r'^\s*res\s*=\s*([0-9.eE+\-]+)\s*$', geom_text, re.M)
    if not m:
        sys.stderr.write("ERROR: 'res' not found in geometry block.\n")
        sys.exit(1)
    res = float(m.group(1))

    def get_int(name: str) -> int:
        mm = re.search(rf'^\s*p0/{name}\s*=\s*([0-9+\-]+)\s*$', geom_text, re.M)
        if not mm:
            sys.stderr.write(f"ERROR: 'p0/{name}' not found in geometry block.\n")
            sys.exit(1)
        return int(mm.group(1))

    min_fs = get_int("min_fs")
    max_fs = get_int("max_fs")
    min_ss = get_int("min_ss")
    max_ss = get_int("max_ss")

    # Nominal pixel center (half-pixel convention)
    fs0 = 0.5 * (min_fs + max_fs) +1
    ss0 = 0.5 * (min_ss + max_ss) +1

    return dict(res=res, fs0=fs0, ss0=ss0)

def iter_chunks(stream_text: str) -> List[str]:
    """
    Yield chunk texts (without the '----- Begin/End chunk -----' lines).
    """
    pattern = re.compile(
        re.escape(CHUNK_BEGIN) + r"(.*?)" + r"(?:" + re.escape(CHUNK_END) + r"|$)",
        re.S
    )
    for m in pattern.finditer(stream_text):
        yield m.group(1)


# -------------------- Parsing: within a chunk --------------------

def parse_chunk_ids(chunk: str) -> Tuple[str, str]:
    """
    Returns (image_filename, event_string) such as:
        ("/path/to/file.h5", "//3")
    If 'Event:' is missing, event_string is "".
    """
    img = ""
    ev = ""
    m1 = re.search(r'^\s*Image filename:\s*(.+?)\s*$', chunk, re.M)
    if m1:
        img = m1.group(1).strip()
    m2 = re.search(r'^\s*Event:\s*(.+?)\s*$', chunk, re.M)
    if m2:
        ev = m2.group(1).strip()
    return img, ev

def parse_predict_refine_shifts(chunk: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse 'predict_refine/det_shift x = <dx> y = <dy> mm' if present in the chunk.
    Returns (dx_mm, dy_mm) or (None, None).
    """
    m = re.search(
        r'predict_refine/det_shift\s*x\s*=\s*([\-0-9.eE]+)\s*y\s*=\s*([\-0-9.eE]+)\s*mm',
        chunk
    )
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            return None, None
    return None, None

def parse_peaks_block(chunk: str) -> List[Tuple[float, float, float, str]]:
    """
    Parse 'Peaks from peak search' block.
    Returns list of (fs, ss, intensity, panel).
    """
    peaks = []
    in_block = False
    for line in chunk.splitlines():
        if not in_block and line.strip().startswith("Peaks from peak search"):
            in_block = True
            continue
        if in_block and line.strip().startswith("End of peak list"):
            break
        if in_block:
            # Expect: fs  ss  (1/d)  Intensity  Panel
            m = re.match(
                r"\s*([0-9.]+)\s+([0-9.]+)\s+[0-9.\-eE]+\s+([0-9.\-eE]+)\s+(\S+)",
                line
            )
            if m:
                fs = float(m.group(1))
                ss = float(m.group(2))
                inten = float(m.group(3))
                panel = m.group(4)
                peaks.append((fs, ss, inten, panel))
    return peaks

def parse_reflections_block(chunk: str) -> List[Tuple[float, float, str]]:
    """
    Parse 'Reflections measured after indexing' block.
    Returns list of (fs, ss, panel).
    """
    refl = []
    in_block = False
    for line in chunk.splitlines():
        if not in_block and line.strip().startswith("Reflections measured after indexing"):
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("End of reflections") or line.strip().startswith("---"):
                break
            # We only need the last 3 columns: fs/px, ss/px, panel
            parts = line.split()
            if len(parts) >= 3:
                panel = parts[-1]
                try:
                    ss = float(parts[-2])
                    fs = float(parts[-3])
                    # Heuristic: panel looks like 'p0', 'p1', ...
                    if re.match(r"^p[0-9]+$", panel):
                        refl.append((fs, ss, panel))
                except ValueError:
                    pass
    return refl


# -------------------- Matching: reflections <-> peaks --------------------

def greedy_nearest_match(
    peaks: List[Tuple[float, float, float, str]],
    refls: List[Tuple[float, float, str]],
    max_dist_px: float = 4.0
) -> Tuple[List[Tuple[float, float, float]], int]:
    """
    Greedy bipartite matching of reflections to peaks (same panel), minimizing distance.
    Returns:
      matched_peaks: list[(fs, ss, I)] for matched pairs (peaks only; we use their intensities)
      n_pairs: number of matches used
    """
    if not peaks or not refls:
        return [], 0
    # Build candidates (i_peak, j_ref, dist2), panel filter
    cand = []
    for i, (pfs, pss, I, ppan) in enumerate(peaks):
        for j, (rfs, rss, rpan) in enumerate(refls):
            if ppan != rpan:
                continue
            d2 = (pfs - rfs) ** 2 + (pss - rss) ** 2
            if d2 <= max_dist_px ** 2:
                cand.append((d2, i, j))
    if not cand:
        return [], 0
    cand.sort(key=lambda t: t[0])

    used_p = set()
    used_r = set()
    matched = []
    for d2, i, j in cand:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        matched.append(i)

    matched_peaks = [(peaks[i][0], peaks[i][1], peaks[i][2]) for i in matched]
    return matched_peaks, len(matched)


# -------------------- Robust center refinement --------------------

def _mad(vals: List[float]) -> float:
    m = median(vals)
    return median([abs(v - m) for v in vals]) or 1.0

def _wmedian(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """
    Weighted median from (value, weight) pairs.
    """
    if not pairs:
        return None
    pairs = sorted(pairs, key=lambda x: x[0])
    tot = sum(w for _, w in pairs) or 1.0
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= 0.5 * tot:
            return v
    return pairs[-1][0]

def refine_center_from_points(
    points: List[Tuple[float, float, float]],
    bin_eps: float = 0.15,
    min_pair_sep: float = 5.0
) -> Tuple[float, float, int, int]:
    """
    points: list of (fs, ss, I)
    Returns (cx, cy, n_pairs_x, n_pairs_y).
    - Form pairs along same-ss (for cx) and same-fs (for cy) by binning to bin_eps.
    - Use weighted medians of midpoints with MAD-based outlier rejection.
    - Fallback to intensity-weighted medians if necessary.
    """
    if not points:
        return float("nan"), float("nan"), 0, 0

    def q(v: float) -> float:
        return round(v / bin_eps) * bin_eps

    by_ss = {}
    by_fs = {}
    for fs, ss, I in points:
        by_ss.setdefault(q(ss), []).append((fs, ss, I))
        by_fs.setdefault(q(fs), []).append((fs, ss, I))

    cx_cands = []
    for ssb, rows in by_ss.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda x: x[0])  # sort by fs
        i, j = 0, len(rows) - 1
        while i < j:
            fsi, ssi, Ii = rows[i]
            fsj, ssj, Ij = rows[j]
            if abs(fsj - fsi) >= min_pair_sep:
                mid = 0.5 * (fsi + fsj)
                w = math.sqrt(max(Ii, 0.0) * max(Ij, 0.0))
                cx_cands.append((mid, w))
            i += 1
            j -= 1

    cy_cands = []
    for fsb, cols in by_fs.items():
        if len(cols) < 2:
            continue
        cols.sort(key=lambda x: x[1])  # sort by ss
        i, j = 0, len(cols) - 1
        while i < j:
            fsi, ssi, Ii = cols[i]
            fsj, ssj, Ij = cols[j]
            if abs(ssj - ssi) >= min_pair_sep:
                mid = 0.5 * (ssi + ssj)
                w = math.sqrt(max(Ii, 0.0) * max(Ij, 0.0))
                cy_cands.append((mid, w))
            i += 1
            j -= 1

    def reject(pairs: List[Tuple[float, float]], k: float = 3.5) -> List[Tuple[float, float]]:
        if not pairs:
            return pairs
        vals = [v for v, _ in pairs]
        m = median(vals)
        s = _mad(vals)
        return [(v, w) for (v, w) in pairs if abs(v - m) <= k * s]

    cx_cands = reject(cx_cands)
    cy_cands = reject(cy_cands)

    cx = _wmedian(cx_cands) if cx_cands else None
    cy = _wmedian(cy_cands) if cy_cands else None
    n_pairs_x = len(cx_cands)
    n_pairs_y = len(cy_cands)

    # Fallback to intensity-weighted medians over raw points
    if cx is None or cy is None:
        pts = sorted([(fs, ss, max(I, 1e-9)) for fs, ss, I in points], key=lambda t: t[2])
        tot = sum(w for _, _, w in pts) or 1.0
        if cx is None:
            acc = 0.0
            for fs, _, w in pts:
                acc += w
                if acc >= 0.5 * tot:
                    cx = fs
                    break
        if cy is None:
            acc = 0.0
            for _, ss, w in pts:
                acc += w
                if acc >= 0.5 * tot:
                    cy = ss
                    break

    return float(cx), float(cy), n_pairs_x, n_pairs_y


# -------------------- Main pipeline --------------------

def process_stream(
    stream_path: str,
    out_csv: str,
    max_match_dist: float = 0.75,
    bin_eps: float = 0.15,
    min_pair_sep: float = 5.0
):
    text = load_text(stream_path)
    geom = extract_geometry_block(text)
    H = parse_header_single_panel(geom)
    res = H["res"]
    fs0 = H["fs0"]
    ss0 = H["ss0"]

    rows = []
    n_chunks = 0
    n_ok = 0

    for chunk in iter_chunks(text):
        n_chunks += 1
        img, ev = parse_chunk_ids(chunk)
        peaks = parse_peaks_block(chunk)
        refls = parse_reflections_block(chunk)

        # Default: prefer matched (indexed) peaks; fallback to all peaks
        matched_peaks, n_matched = greedy_nearest_match(peaks, refls, max_dist_px=max_match_dist)

        use_points = matched_peaks if len(matched_peaks) >= 8 else [(fs, ss, I) for (fs, ss, I, _) in peaks]
        cx, cy, n_px, n_py = refine_center_from_points(use_points, bin_eps=bin_eps, min_pair_sep=min_pair_sep)

        dx_px = cx - fs0
        dy_px = cy - ss0
        dx_mm = dx_px / res
        dy_mm = dy_px / res

        ok = math.isfinite(dx_px) and math.isfinite(dy_px)
        if ok:
            n_ok += 1
        if len(refls) >= 1:
            row = dict(
                image_filename=img,
                event=ev,
                dx_px=dx_px,
                dy_px=dy_px,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                n_peaks=len(peaks),
                n_reflections=len(refls),
                n_matched=n_matched,
                n_pairs_x=n_px,
                n_pairs_y=n_py,
            )
            rows.append(row)

    # Write CSV
    fieldnames = [
        "image_filename","event",
        "dx_px","dy_px","dx_mm","dy_mm",
        "n_peaks","n_reflections","n_matched","n_pairs_x","n_pairs_y"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_csv} (ok: {n_ok}/{n_chunks}).")
    print(f"res (px/mm): {res:.8g} | nominal center: fs0={fs0:.3f}, ss0={ss0:.3f}")

def main():
    ap = argparse.ArgumentParser(
        description="Refine per-event detector shifts (dx,dy) from a CrystFEL stream using indexed peaks."
    )
    ap.add_argument("stream", help="Path to CrystFEL .stream file")
    ap.add_argument("-o", "--output", default="refined_det_shifts.csv", help="Output CSV path")
    ap.add_argument("--max-match-dist", type=float, default=2.0, help="Max peak-reflection matching distance in pixels (default: 4.0)")
    ap.add_argument("--bin-eps", type=float, default=0.15, help="Quantization for row/col grouping in pixels (default: 0.15)")
    ap.add_argument("--min-pair-sep", type=float, default=5.0, help="Minimum separation for forming symmetric pairs (px)")
    args = ap.parse_args()

    try:
        process_stream(
            args.stream,
            args.output,
            max_match_dist=args.max_match_dist,
            bin_eps=args.bin_eps,
            min_pair_sep=args.min_pair_sep
        )
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted.\n")
        sys.exit(130)

if __name__ == "__main__":
    main()

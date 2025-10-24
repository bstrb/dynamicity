#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Updated) step3_evaluate_stream.py
Now includes det_shift_x_mm / det_shift_y_mm in the CSV.
"""
from __future__ import annotations
import argparse, os, sys, csv, re, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

# DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
run = "003"
DEFAULT_RUN_DIR = os.path.join(DEFAULT_ROOT, "runs", f"run_{run}")
DEFAULT_STREAM = f"stream_{run}.stream"
DEFAULT_MATCH_RADIUS = 4.0
DEFAULT_OUTLIER_SIGMA = 2.0

FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
RE_BEGIN_CHUNK   = re.compile(r"-{3,}\s*Begin\s+chunk\s*-{3,}", re.IGNORECASE)
RE_END_CHUNK     = re.compile(r"-{3,}\s*End(?:\s+of)?\s+chunk\s*-{3,}", re.IGNORECASE)
RE_IMG_FN        = re.compile(r"^\s*Image\s+filename\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_EVENT         = re.compile(r"^\s*Event\s*:\s*(?:/+)?\s*([0-9]+)\s*$", re.IGNORECASE)
RE_IMG_SERIAL    = re.compile(r"^\s*Image\s+serial\s+number\s*:\s*([0-9]+)\s*$", re.IGNORECASE)
RE_DET_DX        = re.compile(r"^\s*header/float//entry/data/det_shift_x_mm\s*=\s*(" + FLOAT_RE + r")\s*$", re.IGNORECASE)
RE_DET_DY        = re.compile(r"^\s*header/float//entry/data/det_shift_y_mm\s*=\s*(" + FLOAT_RE + r")\s*$", re.IGNORECASE)
RE_BEGIN_PEAKS   = re.compile(r"^\s*Peaks from peak search", re.IGNORECASE)
RE_END_PEAKS     = re.compile(r"^\s*End of peak list", re.IGNORECASE)
RE_PEAK_LINE     = re.compile(rf"^\s*({FLOAT_RE})\s+({FLOAT_RE})\s+{FLOAT_RE}\s+({FLOAT_RE})\s+(\S+)\s*$")
RE_BEGIN_CRYSTAL = re.compile(r"^\s*---\s*Begin\s+crystal", re.IGNORECASE)
RE_END_CRYSTAL   = re.compile(r"^\s*---\s*End\s+crystal", re.IGNORECASE)
RE_BEGIN_REFL    = re.compile(r"^\s*Reflections\s+measured\s+after\s+indexing", re.IGNORECASE)
RE_END_REFL      = re.compile(r"^\s*End\s+of\s+reflections", re.IGNORECASE)
RE_REFL_LINE     = re.compile(rf".*?\s({FLOAT_RE})\s+({FLOAT_RE})\s+(\S+)\s*$")

@dataclass
class Chunk:
    # cid: int
    image: Optional[str]
    event: Optional[str]
    indexed: bool
    peaks: List[Tuple[float, float, float, str]]
    refls: List[Tuple[float, float, str]]
    det_dx_mm: Optional[float]
    det_dy_mm: Optional[float]

def _sigma_mask(values: np.ndarray, sigma: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=bool)
    mu = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if sd == 0.0:
        return np.ones_like(values, dtype=bool)
    return values <= (mu + sigma * sd)

def _nn_dists_numpy(pfs: np.ndarray, pss: np.ndarray, rfs: np.ndarray, rss: np.ndarray) -> np.ndarray:
    if pfs.size == 0 or rfs.size == 0:
        return np.full(pfs.shape[0], np.inf, dtype=np.float32)
    df = pfs[:, None] - rfs[None, :]
    ds = pss[:, None] - rss[None, :]
    return np.sqrt((df * df + ds * ds).min(axis=1)).astype(np.float32, copy=False)

def compute_wrmsd_details(
    peaks: List[Tuple[float, float, float, str]],
    refls: List[Tuple[float, float, str]],
    match_radius: float,
    outlier_sigma: float
) -> Tuple[Optional[float], int, int, Optional[str]]:
    pmap: Dict[str, List[List[float]]] = {}
    rmap: Dict[str, List[List[float]]] = {}
    for fs, ss, inten, pan in peaks:
        lst = pmap.setdefault(pan, [[], [], []])
        lst[0].append(fs); lst[1].append(ss); lst[2].append(inten)
    for fs, ss, pan in refls:
        lst = rmap.setdefault(pan, [[], []])
        lst[0].append(fs); lst[1].append(ss)
    if not pmap or not rmap:
        return None, 0, 0, "too_few_peaks_or_refl"

    n_matches = 0
    md_all, w_all = [], []
    for pan, (pfs_list, pss_list, pint_list) in pmap.items():
        r = rmap.get(pan)
        if r is None:
            continue
        rfs_list, rss_list = r
        if not pfs_list or not rfs_list:
            continue
        pfs = np.asarray(pfs_list, dtype=np.float32)
        pss = np.asarray(pss_list, dtype=np.float32)
        pint= np.asarray(pint_list, dtype=np.float32)
        rfs = np.asarray(rfs_list, dtype=np.float32)
        rss = np.asarray(rss_list, dtype=np.float32)
        d = _nn_dists_numpy(pfs, pss, rfs, rss)
        within = (d <= float(match_radius))
        cnt = int(within.sum())
        if cnt > 0:
            n_matches += cnt
            md_all.append(d[within])
            w_all.append(pint[within])
    if not md_all:
        return None, n_matches, 0, "no_matches"
    md = np.concatenate(md_all)
    w  = np.concatenate(w_all)
    keep = _sigma_mask(md, float(outlier_sigma))
    n_kept = int(keep.sum())
    if n_kept == 0:
        return None, n_matches, 0, "all_clipped"
    kept = md[keep]; kw = w[keep]
    wsum = float(kw.sum())
    if wsum <= 0.0:
        return None, n_matches, n_kept, "zero_weight"
    wr = math.sqrt(float((kw * (kept ** 2)).sum()) / wsum)
    if math.isnan(wr) or math.isinf(wr):
        return None, n_matches, n_kept, "nan_inf"
    return float(wr), n_matches, n_kept, None

def parse_stream(stream_path: str) -> Iterable[Chunk]:
    # cid = 0
    ch: Optional[Chunk] = None
    in_chunk = False
    in_peaks = False
    in_crystal = False
    in_refl = False

    def flush():
        nonlocal ch
        if ch is not None:
            out = ch
            ch = None
            return out
        return None

    with open(stream_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if RE_BEGIN_CHUNK.search(line):
                # cid += 1
                in_chunk = True
                in_peaks = in_crystal = in_refl = False
                # ch = Chunk(cid, None, None, False, [], [], None, None)
                ch = Chunk(None, None, False, [], [], None, None)
                continue
            if not in_chunk:
                continue
            if RE_END_CHUNK.search(line):
                cc = flush()
                if cc is not None:
                    yield cc
                in_chunk = in_peaks = in_crystal = in_refl = False
                continue
            if ch is None:
                continue
            if ch.image is None:
                m = RE_IMG_FN.match(line)
                if m:
                    ch.image = m.group(1).strip(); continue
            if ch.event is None:
                m = RE_EVENT.match(line) or RE_IMG_SERIAL.match(line)
                if m:
                    ch.event = m.group(1).strip(); continue
            if ch.det_dx_mm is None:
                m = RE_DET_DX.match(line)
                if m:
                    ch.det_dx_mm = float(m.group(1)); continue
            if ch.det_dy_mm is None:
                m = RE_DET_DY.match(line)
                if m:
                    ch.det_dy_mm = float(m.group(1)); continue
            if RE_BEGIN_PEAKS.search(line):
                in_peaks = True;  continue
            if in_peaks:
                if RE_END_PEAKS.search(line):
                    in_peaks = False; continue
                mpk = RE_PEAK_LINE.match(line)
                if mpk:
                    fs = float(mpk.group(1)); ss = float(mpk.group(2))
                    inten = float(mpk.group(3)); panel = mpk.group(4)
                    ch.peaks.append((fs, ss, inten, panel))
                continue
            if RE_BEGIN_CRYSTAL.search(line):
                in_crystal = True; continue
            if in_crystal and RE_END_CRYSTAL.search(line):
                in_crystal = False; in_refl = False; continue
            if in_crystal and RE_BEGIN_REFL.search(line):
                in_refl = True; ch.indexed = True; continue
            if in_refl:
                if RE_END_REFL.search(line):
                    in_refl = False; continue
                mrf = RE_REFL_LINE.match(line)
                if mrf:
                    fs = float(mrf.group(1)); ss = float(mrf.group(2)); pan = mrf.group(3)
                    ch.refls.append((fs, ss, pan))
                continue
    cc = flush()
    if cc is not None:
        yield cc

def write_csv(path: str, rows: List[Dict[str, object]], header: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Parse stream_xxx.stream and compute per-chunk wRMSD (robust).")
    ap.add_argument("--run-root")
    ap.add_argument("--run-dir")
    ap.add_argument("--match-radius", type=float, default=DEFAULT_MATCH_RADIUS)
    ap.add_argument("--sigma", type=float, default=DEFAULT_OUTLIER_SIGMA)
    args = ap.parse_args(argv)

    if len(argv) == 0:
        run_dir = DEFAULT_RUN_DIR; mr = DEFAULT_MATCH_RADIUS; sg = DEFAULT_OUTLIER_SIGMA
    else:
        if args.run_dir:
            run_dir = args.run_dir
        elif args.run_root:
            run_dir = os.path.join(args.run_root, "runs", "run_{run}".format(run=run))
        else:
            print("Provide --run-root or --run-dir, or run with no args for defaults.", file=sys.stderr); return 2
        mr = float(args.match_radius); sg = float(args.sigma)

    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    stream_path = os.path.join(run_dir, DEFAULT_STREAM)
    if not os.path.isfile(stream_path):
        print(f"ERROR: not found: {stream_path}", file=sys.stderr); return 2

    rows: List[Dict[str, object]] = []
    wr_vals: List[float] = []
    n_chunks = n_any_peaks = n_found_refl_hdr = n_any_refls = n_indexed = 0

    for ch in parse_stream(stream_path):
        n_chunks += 1
        if ch.peaks: n_any_peaks += 1
        if ch.indexed: n_found_refl_hdr += 1
        if ch.refls: n_any_refls += 1
        if ch.indexed: n_indexed += 1

        if not ch.indexed:
            rows.append({
                "image": ch.image or "", "event": ch.event or "",
                "det_shift_x_mm": (f"{ch.det_dx_mm:.6f}" if ch.det_dx_mm is not None else ""),
                "det_shift_y_mm": (f"{ch.det_dy_mm:.6f}" if ch.det_dy_mm is not None else ""),
                "indexed": 0, "wrmsd": "", "n_matches": 0, "n_kept": 0, "reason": "unindexed"
            })
            continue
        wr, nm, nk, reason = compute_wrmsd_details(ch.peaks, ch.refls, mr, sg)
        if wr is not None: wr_vals.append(wr)
        rows.append({
            "image": ch.image or "", "event": ch.event or "",
            "det_shift_x_mm": (f"{ch.det_dx_mm:.6f}" if ch.det_dx_mm is not None else ""),
            "det_shift_y_mm": (f"{ch.det_dy_mm:.6f}" if ch.det_dy_mm is not None else ""),
            "indexed": 1, "wrmsd": (f"{wr:.6f}" if wr is not None else ""),
            "n_matches": nm, "n_kept": nk, "reason": (reason or "")
        })
    csv_path = os.path.join(run_dir, "chunk_metrics_{run}.csv".format(run=run))
    write_csv(csv_path, rows, header=[
        "image","event","det_shift_x_mm","det_shift_y_mm",
        "indexed","wrmsd","n_matches","n_kept","reason"
    ])
    sum_path = os.path.join(run_dir, "summary_{run}.txt".format(run=run))
    with open(sum_path, "w", encoding="utf-8") as f:
        def _summ(vs: List[float]) -> str:
            if not vs: return "n=0"
            a = np.asarray(vs, dtype=np.float64)
            return f"n={a.size} min={a.min():.3f} q25={np.quantile(a,0.25):.3f} med={np.median(a):.3f} q75={np.quantile(a,0.75):.3f} mean={a.mean():.3f} max={a.max():.3f}"
        f.write("=== Step 3 Summary ===\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Total chunks: {n_chunks}\n")
        f.write(f"Indexed (by header): {n_found_refl_hdr}\n")
        f.write(f"Chunks with refl lines: {n_any_refls}\n")
        f.write(f"Chunks with peak lines: {n_any_peaks}\n")
        f.write(f"Index rate: {(100.0*n_found_refl_hdr/n_chunks if n_chunks>0 else 0.0):.2f}%\n")
        f.write(f"WRMSD: {_summ(wr_vals)}\n")
    dbg_path = os.path.join(run_dir, "parse_debug_{run}.txt".format(run=run))
    with open(dbg_path, "w", encoding="utf-8") as f:
        f.write("Debug counters:\n")
        f.write(f"  Total chunks parsed:         {n_chunks}\n")
        f.write(f"  Chunks with 'Reflections...' header: {n_found_refl_hdr}\n")
        f.write(f"  Chunks with refl lines:      {n_any_refls}\n")
        f.write(f"  Chunks with peak lines:      {n_any_peaks}\n")
        f.write(f"  Indexed counted:             {n_indexed}\n")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {sum_path}")
    print(f"Wrote: {dbg_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

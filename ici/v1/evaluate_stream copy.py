#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Updated) step3_evaluate_stream.py
Now includes det_shift_x_mm / det_shift_y_mm in the CSV.
"""
from __future__ import annotations
import argparse, os, sys, csv, re, math
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

# Avoid BLAS/OMP oversubscription when using multiprocessing
import os as _os_env_patch
_os_env_patch.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os_env_patch.environ.setdefault("MKL_NUM_THREADS", "1")
_os_env_patch.environ.setdefault("OMP_NUM_THREADS", "1")
_os_env_patch.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
DEFAULT_RUN = "000"  # will be zero-padded to width 3 at runtime

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

def _normalize_run(run: str) -> str:
    """Return zero-padded run string (e.g. '0' -> '000'). If not numeric, return as-is."""
    try:
        return f"{int(run):03d}"
    except (TypeError, ValueError):
        return str(run)

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

# Helper for multiprocessing: simple wrapper for compute_wrmsd_details
def _compute_wrmsd_worker(args):
    peaks, refls, match_radius, outlier_sigma = args
    return compute_wrmsd_details(peaks, refls, match_radius, outlier_sigma)


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
    ap = argparse.ArgumentParser(description="Parse stream_<run>.stream and compute per-chunk wRMSD (robust).")
    ap.add_argument("--run-root", default=DEFAULT_ROOT, help='Root folder that contains runs/run_<run>')
    ap.add_argument("--run", default=DEFAULT_RUN, help='Run identifier (e.g., "0", "3", "12", "003"); will be zero-padded to width 3')
    ap.add_argument("--mr", type=float, default=4.0, help="Match radius for peak↔refl (pixels)")
    ap.add_argument("--sg", type=float, default=2.0, help="Sigma for outlier clipping")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                help="Worker processes for wRMSD compute (default: CPU count; use 1 to disable parallelism)")
    args = ap.parse_args(argv)

    # Resolve parameters (defaults if not provided)
    run_root = args.run_root if args.run_root else DEFAULT_ROOT
    run_str  = _normalize_run(args.run if args.run else DEFAULT_RUN)
    # mr = float(args.match_radius)
    # sg = float(args.sigma)
    mr = float(args.mr)
    sg = float(args.sg)

    # Construct paths from resolved values
    run_dir = os.path.join(run_root, "runs", f"run_{run_str}")
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    stream_path = os.path.join(run_dir, f"stream_{run_str}.stream")
    print(f"Run root : {os.path.abspath(os.path.expanduser(run_root))}")
    print(f"Run      : {run_str}")
    print(f"Run dir  : {run_dir}")

    if not os.path.isfile(stream_path):
        print(f"ERROR: not found: {stream_path}", file=sys.stderr)
        return 2

    
    rows: List[Dict[str, object]] = []
    wr_vals: List[float] = []
    n_chunks = n_any_peaks = n_found_refl_hdr = n_any_refls = n_indexed = 0

    indexed_payload = []   # (peaks, refls, mr, sg)
    indexed_meta = []      # (row_index, image, event, det_dx_mm, det_dy_mm)

    # 1) Parse and stage
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
        else:
            rows.append(None)  # placeholder to be filled after compute
            idx_row = len(rows) - 1
            indexed_meta.append((idx_row, ch.image, ch.event, ch.det_dx_mm, ch.det_dy_mm))
            indexed_payload.append((ch.peaks, ch.refls, mr, sg))

    # 2) Compute wRMSD (serial or parallel)
    workers = max(1, int(args.workers))
    if workers == 1 or len(indexed_payload) == 0:
        # serial path
        for (irow, img, ev, dxmm, dymm), payload in zip(indexed_meta, indexed_payload):
            wr, nm, nk, reason = _compute_wrmsd_worker(payload)
            if wr is not None: wr_vals.append(wr)
            rows[irow] = {
                "image": img or "", "event": ev or "",
                "det_shift_x_mm": (f"{dxmm:.6f}" if dxmm is not None else ""),
                "det_shift_y_mm": (f"{dymm:.6f}" if dymm is not None else ""),
                "indexed": 1, "wrmsd": (f"{wr:.6f}" if wr is not None else ""),
                "n_matches": nm, "n_kept": nk, "reason": (reason or "")
            }
    else:
            
    # Clamp workers to number of tasks and show a helpful log
        n_tasks = len(indexed_payload)
        workers = min(workers, n_tasks) if n_tasks > 0 else workers
        if workers > 1 and n_tasks > 0:
            print(f"[mp] Using {workers} workers for {n_tasks} indexed chunk(s)")
        with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_compute_wrmsd_worker, payload) for payload in indexed_payload]
                    for (irow, img, ev, dxmm, dymm), fut in zip(indexed_meta, futs):
                        wr, nm, nk, reason = fut.result()
                        if wr is not None: wr_vals.append(wr)
                        rows[irow] = {
                            "image": img or "", "event": ev or "",
                            "det_shift_x_mm": (f"{dxmm:.6f}" if dxmm is not None else ""),
                            "det_shift_y_mm": (f"{dymm:.6f}" if dymm is not None else ""),
                            "indexed": 1, "wrmsd": (f"{wr:.6f}" if wr is not None else ""),
                            "n_matches": nm, "n_kept": nk, "reason": (reason or "")
                    }
    csv_path = os.path.join(run_dir, f"chunk_metrics_{run_str}.csv")
    write_csv(csv_path, rows, header=[
            "image","event","det_shift_x_mm","det_shift_y_mm",
            "indexed","wrmsd","n_matches","n_kept","reason"
        ])
    sum_path = os.path.join(run_dir, f"summary_{run_str}.txt")
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
        dbg_path = os.path.join(run_dir, f"parse_debug_{run_str}.txt")
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

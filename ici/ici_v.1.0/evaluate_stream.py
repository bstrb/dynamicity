#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_stream.py 

Fast evaluator that preserves the correctness of your original parser:
- Memory-maps the .stream and finds chunks via your Begin/End patterns.
- Parses each chunk with your original regexes (incl. panel IDs).
- Computes peak↔reflection matches **per panel**, then aggregates wRMSD.
- Parallelizes per-chunk work (no heavy IPC; each worker receives only the chunk bytes).

Outputs (same names):
  chunk_metrics_<run>.csv
  summary_<run>.txt
  parse_debug_<run>.txt
"""

from __future__ import annotations

import argparse
import mmap
import os
import re
import sys
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Keep BLAS single-threaded inside workers
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
DEFAULT_RUN = "000"  # will be zero-padded to width 3 at runtime

# ---------------- Regexes (your originals) ----------------

FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

RE_BEGIN_CHUNK   = re.compile(r"-{3,}\s*Begin\s+chunk\s*-{3,}", re.IGNORECASE)
RE_END_CHUNK     = re.compile(r"-{3,}\s*End(?:\s+of)?\s+chunk\s*-{3,}", re.IGNORECASE)

RE_IMG_FN        = re.compile(r"^\s*Image\s+filename\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_IMG_FILE      = re.compile(r"^\s*Image\s+file\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_EVENT         = re.compile(r"^\s*Event\s*:\s*(?:/+)?\s*([0-9]+)\s*$", re.IGNORECASE)
RE_IMG_SERIAL    = re.compile(r"^\s*Image\s+serial\s+number\s*:\s*([0-9]+)\s*$", re.IGNORECASE)
RE_DET_DX        = re.compile(r"^\s*header/float//entry/data/det_shift_x_mm\s*=\s*(" + FLOAT_RE + r")\s*$", re.IGNORECASE)
RE_DET_DY        = re.compile(r"^\s*header/float//entry/data/det_shift_y_mm\s*=\s*(" + FLOAT_RE + r")\s*$", re.IGNORECASE)

RE_BEGIN_PEAKS   = re.compile(r"^\s*Peaks from peak search", re.IGNORECASE)
RE_END_PEAKS     = re.compile(r"^\s*End of peak list", re.IGNORECASE)
# fs ss I panel
RE_PEAK_LINE     = re.compile(rf"^\s*({FLOAT_RE})\s+({FLOAT_RE})\s+{FLOAT_RE}\s+({FLOAT_RE})\s+(\S+)\s*$")

RE_BEGIN_CRYSTAL = re.compile(r"^\s*---\s*Begin\s+crystal", re.IGNORECASE)
RE_END_CRYSTAL   = re.compile(r"^\s*---\s*End\s+crystal", re.IGNORECASE)

RE_BEGIN_REFL    = re.compile(r"^\s*Reflections\s+measured\s+after\s+indexing", re.IGNORECASE)
RE_END_REFL      = re.compile(r"^\s*End\s+of\s+reflections", re.IGNORECASE)
# ... with panel at end of line
RE_REFL_LINE     = re.compile(rf".*?\s({FLOAT_RE})\s+({FLOAT_RE})\s+(\S+)\s*$")

# Byte regex equivalents for chunk detection
RE_BEGIN_CHUNK_B = re.compile(br"-{3,}\s*Begin\s+chunk\s*-{3,}", re.IGNORECASE)
RE_END_CHUNK_B   = re.compile(br"-{3,}\s*End(?:\s+of)?\s+chunk\s*-{3,}", re.IGNORECASE)

@dataclass
class ChunkRow:
    image: str
    event: str
    det_dx_mm: float | None
    det_dy_mm: float | None
    indexed: int
    wrmsd: float | None
    n_matches: int
    n_kept: int
    reason: str

# ---------------- wRMSD helpers ----------------
# --- replace existing helpers with these ---

def _sigma_mask_upper(values: np.ndarray, sigma: float) -> np.ndarray:
    """Keep distances <= mean + sigma*std (matches original)."""
    if values.size == 0:
        return np.zeros((0,), dtype=bool)
    mu = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if sd == 0.0:
        return np.ones_like(values, dtype=bool)
    return values <= (mu + sigma * sd)

def _nn_dists_peaks_to_refl(pfs: np.ndarray, pss: np.ndarray,
                            rfs: np.ndarray, rss: np.ndarray) -> np.ndarray:
    """
    Peak-primary distances: for each peak, distance to nearest reflection.
    Vector length = #peaks. This matches the original script.
    """
    if pfs.size == 0 or rfs.size == 0:
        # no reflections → no matches, return +inf for each peak
        return np.full(pfs.shape[0], np.inf, dtype=np.float32)
    df = pfs[:, None] - rfs[None, :]
    ds = pss[:, None] - rss[None, :]
    return np.sqrt((df*df + ds*ds).min(axis=1)).astype(np.float32, copy=False)

def _wrmsd_one_panel_peak_primary(p_fs, p_ss, p_int, r_fs, r_ss,
                                  match_radius, outlier_sigma):
    """
    Compute matches & weighted RMS using peaks as primaries, with intensity weights.
    Returns (wr, n_matches, n_kept, kept_dists, kept_weights)
    """
    if p_fs.size == 0 or r_fs.size == 0:
        return None, 0, 0, np.empty((0,), float), np.empty((0,), float)

    d = _nn_dists_peaks_to_refl(p_fs, p_ss, r_fs, r_ss)   # length = #peaks
    within = (d <= float(match_radius))
    n_matches = int(within.sum())
    if n_matches == 0:
        return None, 0, 0, np.empty((0,), float), np.empty((0,), float)

    d_in = d[within]
    w_in = p_int[within]

    keep = _sigma_mask_upper(d_in, float(outlier_sigma))
    n_kept = int(keep.sum())
    if n_kept == 0:
        return None, n_matches, 0, np.empty((0,), float), np.empty((0,), float)

    kd = d_in[keep]
    kw = w_in[keep]
    wsum = float(kw.sum())
    if wsum <= 0.0:
        return None, n_matches, n_kept, kd, kw

    wr = float(np.sqrt((kw * (kd ** 2)).sum() / wsum))
    return wr, n_matches, n_kept, kd, kw

# ---------------- Per-chunk parser (panel-aware) ----------------

def _bytes_to_lines(b: bytes):
    return b.decode("utf-8", "ignore").splitlines()

def parse_chunk_text(b: bytes, mr: float, sg: float) -> ChunkRow:
    L = _bytes_to_lines(b)

    # image path
    img = ""
    for ln in L[:100]:
        m = RE_IMG_FN.match(ln) or RE_IMG_FILE.match(ln)
        if m:
            img = m.group(1).strip()
            break

    # event id
    ev = ""
    for ln in L[:150]:
        m = RE_EVENT.match(ln)
        if m:
            ev = m.group(1).strip()
            break

    # detector shift
    dx = dy = None
    for ln in L[:200]:
        if dx is None:
            mdx = RE_DET_DX.match(ln)
            if mdx: dx = float(mdx.group(1))
        if dy is None:
            mdy = RE_DET_DY.match(ln)
            if mdy: dy = float(mdy.group(1))
        if dx is not None and dy is not None:
            break

    # peaks: dict panel -> [(fs, ss)]
    peaks_by_panel = {}
    in_peaks = False
    for ln in L:
        if not in_peaks and RE_BEGIN_PEAKS.search(ln):
            in_peaks = True
            continue
        if in_peaks:
            if RE_END_PEAKS.search(ln) or ln.startswith("---") or ln.startswith("Begin chunk") or ln.startswith("End chunk"):
                in_peaks = False
                continue
            mp = RE_PEAK_LINE.match(ln)
            if mp:
                fs = float(mp.group(1)); ss = float(mp.group(2)); inten = float(mp.group(3))
                pan = mp.group(4)
                peaks_by_panel.setdefault(pan, []).append((fs, ss, inten))


    # reflections: dict panel -> [(fs, ss)]
    refl_by_panel = {}
    in_refl = False
    for ln in L:
        if not in_refl and RE_BEGIN_REFL.search(ln):
            in_refl = True
            continue
        if in_refl:
            if RE_END_REFL.search(ln) or ln.startswith("---") or ln.startswith("Begin chunk") or ln.startswith("End chunk"):
                in_refl = False
                continue
            mrline = RE_REFL_LINE.match(ln)
            if mrline:
                fs = float(mrline.group(1)); ss = float(mrline.group(2))
                pan = mrline.group(3)
                refl_by_panel.setdefault(pan, []).append((fs, ss))

    # any reflections?
    any_indexed = any(len(v) for v in refl_by_panel.values())
    if not any_indexed:
        return ChunkRow(img, ev, dx, dy, 0, None, 0, 0, "unindexed")

    # panel-wise matching, then aggregate
    total_matches = 0
    total_kept = 0
    kept_all = []

    for pan, rlist in refl_by_panel.items():
        plist = peaks_by_panel.get(pan, [])
        if plist:
            p_arr = np.asarray(plist, dtype=float)  # columns: fs, ss, inten
            p_fs, p_ss, p_int = p_arr[:,0], p_arr[:,1], p_arr[:,2]
        else:
            p_fs = p_ss = p_int = np.empty((0,), float)

        r_arr = np.asarray(rlist, dtype=float) if rlist else np.empty((0,2), float)
        r_fs = r_arr[:,0] if r_arr.size else np.empty((0,), float)
        r_ss = r_arr[:,1] if r_arr.size else np.empty((0,), float)

        wr_p, n_matches_p, n_kept_p, kd, kw = _wrmsd_one_panel_peak_primary(
            p_fs, p_ss, p_int, r_fs, r_ss, mr, sg
        )
        total_matches += n_matches_p
        total_kept += n_kept_p
        if kd.size:
            kept_all.append((kd, kw))


    if total_kept == 0:
        return ChunkRow(img, ev, dx, dy, 1, None, total_matches, 0, "no_within_radius_or_all_outliers")

    # concatenate distances and weights from all panels
    kd_all = np.concatenate([kd for (kd, kw) in kept_all])
    kw_all = np.concatenate([kw for (kd, kw) in kept_all])
    wsum = float(kw_all.sum())
    if wsum <= 0.0:
        return ChunkRow(img, ev, dx, dy, 1, None, total_matches, total_kept, "zero_weight")

    wr_all = float(np.sqrt((kw_all * (kd_all ** 2)).sum() / wsum))
    return ChunkRow(img, ev, dx, dy, 1, wr_all, total_matches, total_kept, "")

# ---------------- Chunk discovery via mmap ----------------

def get_chunks(path: str):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    begins = [m.start() for m in RE_BEGIN_CHUNK_B.finditer(mm)]
    ends   = [m.end()   for m in RE_END_CHUNK_B.finditer(mm)]

    # Pair each begin with the first end after it
    bounds = []
    j = 0
    for a in begins:
        while j < len(ends) and ends[j] <= a:
            j += 1
        if j < len(ends):
            b = ends[j]
            if a < b:
                bounds.append((a, b))
            j += 1
    return mm, bounds

# ---------------- Main ----------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Fast, panel-aware evaluator for CrystFEL .stream files.")
    ap.add_argument("--run-root", default=DEFAULT_ROOT, help="Experiment root containing 'runs/'")
    ap.add_argument("--run", default=DEFAULT_RUN, help="Run number, e.g. 000")
    ap.add_argument("--mr", type=float, default=4.0, help="Match radius for peak↔refl (pixels)")
    ap.add_argument("--sg", type=float, default=2.0, help="Sigma for outlier clipping")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Processes (default: cpu_count)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = os.path.abspath(os.path.expanduser(args.run_root))
    run_dir  = os.path.join(run_root, f"run_{int(args.run):03d}")
    stream_path = os.path.join(run_dir, f"stream_{int(args.run):03d}.stream")

    # print("Run root :", run_root)
    # print("Run      :", f"{int(args.run):03d}")
    # print("Run dir  :", run_dir)

    mm, bounds = get_chunks(stream_path)
    print(f"[scan] found {len(bounds)} chunks in stream")

    workers = max(1, int(args.workers))
    n_tasks = len(bounds)
    workers = min(workers, n_tasks) if n_tasks > 0 else workers
    if workers > 1 and n_tasks > 0:
        print(f"[mp] Using {workers} workers for {n_tasks} chunk(s)")

    rows = []
    if workers == 1:
        for (a, b) in bounds:
            rows.append(parse_chunk_text(mm[a:b], args.mr, args.sg))
    else:
        # Batch to limit outstanding futures
        BATCH = 4000
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i in range(0, len(bounds), BATCH):
                futs = [ex.submit(parse_chunk_text, mm[a:b], args.mr, args.sg) for (a, b) in bounds[i:i+BATCH]]
                for fut in futs:
                    try:
                        rows.append(fut.result())
                    except Exception as e:
                        rows.append(ChunkRow("", "", None, None, 0, None, 0, 0, f"worker_error:{e}"))

    # Write outputs
    import csv
    csv_path = os.path.join(run_dir, f"chunk_metrics_{int(args.run):03d}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image","event","det_shift_x_mm","det_shift_y_mm","indexed","wrmsd","n_matches","n_kept","reason"])
        for r in rows:
            w.writerow([r.image, r.event, f"{r.det_dx_mm:.6f}" if r.det_dx_mm is not None else "",
                        f"{r.det_dy_mm:.6f}" if r.det_dy_mm is not None else "",
                        r.indexed, f"{r.wrmsd:.6f}" if r.wrmsd is not None else "", r.n_matches, r.n_kept, r.reason])
    print(f"Wrote: {csv_path}")

    # Summary
    n_chunks = len(rows)
    n_indexed = sum(r.indexed for r in rows)
    finite_wr = [r.wrmsd for r in rows if (r.wrmsd is not None and np.isfinite(r.wrmsd))]
    wr_best = (min(finite_wr) if finite_wr else None)
    wr_med  = (float(np.median(finite_wr)) if finite_wr else None)

    sum_path = os.path.join(run_dir, f"summary_{int(args.run):03d}.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(f"chunks={n_chunks}\nindexed={n_indexed}\n")
        f.write(f"wrmsd_best={wr_best if wr_best is not None else ''}\n")
        f.write(f"wrmsd_median={wr_med if wr_med is not None else ''}\n")
    # print(f"Wrote: {sum_path}")


    return 0

if __name__ == "__main__":
    raise SystemExit(main())

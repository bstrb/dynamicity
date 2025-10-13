#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_best_wrmsd_stream.py

Fast, RAM-safe, parallel selector:
- Scans a directory of CrystFEL .stream files.
- For each (image basename, event), computes intensity-weighted RMSD (wRMSD)
  per chunk (per-panel nearest neighbor, optional sigma clipping), across all input streams.
- Keeps only the lowest-wRMSD chunk globally and writes a single merged .stream
  with exactly those winning chunks (header from the first input).
- Writes winners progressively once all streams that contain a given key have finished.

Usage:
  python3 select_best_wrmsd_stream.py /path/to/streams [--workers 0]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# SciPy KDTree (optional but faster for NN)
_HAVE_SCIPY = False
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------- CLI ---------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select lowest-wRMSD chunks across streams and write a single .stream")
    ap.add_argument("dir", type=Path, help="Directory containing .stream files")
    ap.add_argument("--workers", type=int, default=0,
                    help="Worker processes (0 = use all logical CPUs, capped by number of files)")
    ap.add_argument("--match-radius", type=float, default=4.0, help="NN match radius in pixels")
    ap.add_argument("--outlier-sigma", type=float, default=2.0, help="Sigma for outlier clipping")
    ap.add_argument("--min-peaks", type=int, default=1, help="Skip chunks with fewer peaks")
    ap.add_argument("--min-reflections", type=int, default=1, help="Skip chunks with fewer reflections")
    return ap.parse_args()


# ------------------------- Regex & parsing --------------------------

FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

RE_BEGIN_CHUNK   = re.compile(r"^-{5}\s+Begin chunk\s+-{5}")
RE_END_CHUNK     = re.compile(r"^-{5}\s+End(?: of)? chunk\s+-{5}")

RE_IMG_FN        = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
RE_EVENT         = re.compile(r"^\s*Event:\s*(?:/+)?\s*(\S+)\s*$")

RE_BEGIN_PEAKS   = re.compile(r"^\s*Peaks from peak search", re.IGNORECASE)
RE_END_PEAKS     = re.compile(r"^\s*End of peak list", re.IGNORECASE)
# IMPORTANT: matches your original peak line (5 tokens; 4th numeric is intensity)
RE_PEAK_LINE     = re.compile(rf"^\s*({FLOAT_RE})\s+({FLOAT_RE})\s+{FLOAT_RE}\s+({FLOAT_RE})\s+(\S+)\s*$")

RE_BEGIN_CRYSTAL = re.compile(r"^\s*---\s+Begin crystal", re.IGNORECASE)
RE_END_CRYSTAL   = re.compile(r"^\s*---\s+End crystal", re.IGNORECASE)

RE_BEGIN_REFL    = re.compile(r"^\s*Reflections measured after indexing", re.IGNORECASE)
RE_END_REFL      = re.compile(r"^\s*End of reflections", re.IGNORECASE)
# reflections: capture last two floats (fs, ss) and final token (panel)
RE_REFL_LINE     = re.compile(rf".*?\s({FLOAT_RE})\s+({FLOAT_RE})\s+(\S+)\s*$")


class ChunkLite:
    __slots__ = ("image", "event", "peaks", "refls")
    def __init__(self):
        self.image: Optional[str] = None
        self.event: Optional[str] = None
        self.peaks: List[Tuple[float, float, float, str]] = []  # (fs, ss, intensity, panel)
        self.refls: List[Tuple[float, float, str]] = []         # (fs, ss, panel)


def iter_chunks_for_metrics(stream_path: Path) -> Iterable[ChunkLite]:
    """
    Stream the file line-by-line and yield a minimal ChunkLite with peaks+refls only.
    """
    with stream_path.open("r", encoding="utf-8", errors="ignore") as f:
        in_chunk = in_peaks = in_crystal = in_refl = False
        ch: Optional[ChunkLite] = None

        for raw in f:
            line = raw.rstrip("\n")

            if RE_BEGIN_CHUNK.search(line):
                if in_chunk and ch and ch.image and ch.event:
                    yield ch
                in_chunk = True
                in_peaks = in_crystal = in_refl = False
                ch = ChunkLite()
                continue

            if not in_chunk:
                continue

            if RE_END_CHUNK.search(line):
                if ch and ch.image and ch.event:
                    yield ch
                in_chunk = in_peaks = in_crystal = in_refl = False
                ch = None
                continue

            if ch is None:
                continue

            if ch.image is None:
                m = RE_IMG_FN.match(line)
                if m:
                    ch.image = m.group(1).strip()
                    continue

            if ch.event is None:
                m = RE_EVENT.match(line)
                if m:
                    ch.event = m.group(1).strip()
                    continue

            # crystal boundaries
            if RE_BEGIN_CRYSTAL.search(line):
                in_crystal = True
                continue
            if in_crystal and RE_END_CRYSTAL.search(line):
                in_crystal = False
                in_refl = False
                continue

            # peaks
            if RE_BEGIN_PEAKS.search(line):
                in_peaks = True
                continue
            if in_peaks:
                if RE_END_PEAKS.search(line):
                    in_peaks = False
                    continue
                mpk = RE_PEAK_LINE.match(line)
                if mpk:
                    fs = float(mpk.group(1)); ss = float(mpk.group(2))
                    inten = float(mpk.group(3)); panel = mpk.group(4)
                    ch.peaks.append((fs, ss, inten, panel))
                continue

            # reflections inside crystal
            if in_crystal and RE_BEGIN_REFL.search(line):
                in_refl = True
                continue
            if in_refl:
                if RE_END_REFL.search(line):
                    in_refl = False
                    continue
                mrf = RE_REFL_LINE.match(line)
                if mrf:
                    fs = float(mrf.group(1)); ss = float(mrf.group(2)); panel = mrf.group(3)
                    ch.refls.append((fs, ss, panel))
                continue

        if in_chunk and ch and ch.image and ch.event:
            yield ch


def iter_chunk_keys(stream_path: Path) -> Iterable[Tuple[str, str]]:
    """
    Fast pass to collect (img_basename, event) per chunk; no peaks/reflections parsed.
    """
    with stream_path.open("r", encoding="utf-8", errors="ignore") as f:
        in_chunk = False
        img: Optional[str] = None
        ev:  Optional[str] = None
        for raw in f:
            line = raw.rstrip("\n")

            if RE_BEGIN_CHUNK.search(line):
                in_chunk = True
                img = ev = None
                continue

            if not in_chunk:
                continue

            if RE_END_CHUNK.search(line):
                if img and ev:
                    yield (os.path.basename(img), ev)
                in_chunk = False
                img = ev = None
                continue

            if img is None:
                m = RE_IMG_FN.match(line)
                if m:
                    img = m.group(1).strip()
                    continue
            if ev is None:
                m = RE_EVENT.match(line)
                if m:
                    ev = m.group(1).strip()
                    continue


# -------------------------- wRMSD core -------------------------------

def _nn_dists_numpy(pfs, pss, rfs, rss) -> np.ndarray:
    if pfs.size == 0 or rfs.size == 0:
        return np.full(pfs.shape[0], np.inf, dtype=np.float32)
    df = pfs[:, None] - rfs[None, :]
    ds = pss[:, None] - rss[None, :]
    return np.sqrt((df * df + ds * ds).min(axis=1)).astype(np.float32, copy=False)

def _nn_dists_kd(pfs, pss, rfs, rss) -> np.ndarray:
    tree = cKDTree(np.column_stack((rfs, rss)))
    dists, _ = tree.query(np.column_stack((pfs, pss)), k=1, workers=-1)
    return dists.astype(np.float32, copy=False)

def _sigma_mask(md: np.ndarray, sigma: float) -> np.ndarray:
    if md.size == 0:
        return np.zeros((0,), dtype=bool)
    mu = float(md.mean())
    sd = float(md.std(ddof=1)) if md.size > 1 else 0.0
    if sd == 0.0:
        return np.ones_like(md, dtype=bool)
    return md <= (mu + sigma * sd)

def compute_wrmsd(
    peaks: List[Tuple[float, float, float, str]],
    refls: List[Tuple[float, float, str]],
    match_radius: float,
    outlier_sigma: float
) -> Optional[float]:
    """
    Per-panel nearest neighbor between peaks and reflections, intensity-weighted RMSD,
    with sigma clipping (matches your calc_indexing_metrics logic).
    """
    if not peaks or not refls:
        return None

    # build per panel
    pmap: Dict[str, List[List[float]]] = {}
    rmap: Dict[str, List[List[float]]] = {}
    for fs, ss, inten, pan in peaks:
        lst = pmap.setdefault(pan, [[], [], []])
        lst[0].append(fs); lst[1].append(ss); lst[2].append(inten)
    for fs, ss, pan in refls:
        lst = rmap.setdefault(pan, [[], []])
        lst[0].append(fs); lst[1].append(ss)

    mr = float(match_radius)
    md_all = []
    w_all  = []

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

        d = _nn_dists_kd(pfs, pss, rfs, rss) if _HAVE_SCIPY else _nn_dists_numpy(pfs, pss, rfs, rss)
        within = (d <= mr)
        if np.any(within):
            md_all.append(d[within])
            w_all.append(pint[within])

    if not md_all:
        return None

    md = np.concatenate(md_all)
    w  = np.concatenate(w_all)
    keep = _sigma_mask(md, outlier_sigma)
    kept = md[keep]
    if kept.size == 0:
        return None
    kw = w[keep]
    wsum = float(kw.sum())
    if wsum <= 0.0:
        return None
    return math.sqrt(float((kw * (kept ** 2)).sum()) / wsum)


# -------------------- Pass 1: key presence per stream -----------------

def scan_stream_keys_worker(args) -> Tuple[int, set[Tuple[str, str]], int]:
    """
    Returns (stream_idx, set_of_keys_present, num_chunks)
    """
    stream_idx, path_str = args
    sp = Path(path_str)
    keys = set()
    num_chunks = 0
    with sp.open("r", encoding="utf-8", errors="ignore") as f:
        in_chunk = False
        img = ev = None
        for raw in f:
            line = raw.rstrip("\n")
            if RE_BEGIN_CHUNK.search(line):
                in_chunk = True
                img = ev = None
                num_chunks += 1
                continue
            if not in_chunk:
                continue
            if RE_END_CHUNK.search(line):
                if img and ev:
                    keys.add((os.path.basename(img), ev))
                in_chunk = False
                img = ev = None
                continue
            if img is None:
                m = RE_IMG_FN.match(line)
                if m:
                    img = m.group(1).strip()
                    continue
            if ev is None:
                m = RE_EVENT.match(line)
                if m:
                    ev = m.group(1).strip()
                    continue
    return (stream_idx, keys, num_chunks)


# -------------------- Pass 2: per-stream minima ----------------------

# Progress queue (inherited)
PROG_Q = None  # type: ignore

def _init_worker_progress(progress_q):
    global PROG_Q
    PROG_Q = progress_q

def process_stream_worker(args) -> Tuple[int, Dict[Tuple[str,str], Tuple[float, int]]]:
    """
    Returns (stream_idx, best_local) where best_local maps (img_base,event)
    -> (best_wrmsd_in_this_stream, occurrence_index_in_this_stream)
    The occurrence index lets us re-scan the file once and extract the
    correct chunk later without storing text.
    """
    stream_idx, path_str, match_radius, outlier_sigma, min_peaks, min_refl = args
    sp = Path(path_str)

    best_local: Dict[Tuple[str,str], Tuple[float,int]] = {}
    occ_counter: Dict[Tuple[str,str], int] = defaultdict(int)

    processed = 0
    BATCH = 1000  # progress cadence

    try:
        for ch in iter_chunks_for_metrics(sp):
            processed += 1
            if PROG_Q and (processed % BATCH == 0):
                PROG_Q.put(BATCH)

            if not (ch.image and ch.event):
                continue
            key = (os.path.basename(ch.image), ch.event)
            occ_counter[key] += 1

            if len(ch.peaks) < min_peaks or len(ch.refls) < min_refl:
                continue

            wr = compute_wrmsd(ch.peaks, ch.refls, match_radius, outlier_sigma)
            if wr is None or math.isnan(wr) or math.isinf(wr):
                continue

            cur = best_local.get(key)
            if (cur is None) or (wr < cur[0]):
                best_local[key] = (wr, occ_counter[key])

    except Exception as e:
        sys.stderr.write(f"[WARN] {sp.name}: {e}\n")

    if PROG_Q:
        rem = processed % BATCH
        if rem:
            PROG_Q.put(rem)

    return (stream_idx, best_local)


# -------------------- Writer helpers (RAM-safe) ----------------------

BEGIN = "----- Begin chunk -----"
END   = "----- End chunk -----"

def write_header_from_first_stream(all_streams: List[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for sp in all_streams:
            try:
                with sp.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith(BEGIN):
                            return  # header written (possibly empty)
                        out.write(line)
            except Exception:
                continue


def write_winners_for_stream(
    stream_path: Path,
    keys_to_occ: Dict[Tuple[str,str], int],
    out_path: Path
) -> int:
    """
    Re-scan one stream and append the exact winning chunks (by occurrence index per key)
    to out_path. No chunk text is kept in memory longer than necessary.
    """
    if not keys_to_occ:
        return 0

    wanted = dict(keys_to_occ)  # copy
    occ_now: Dict[Tuple[str,str], int] = defaultdict(int)
    written = 0

    with stream_path.open("r", encoding="utf-8", errors="ignore") as f, \
         out_path.open("a", encoding="utf-8") as out:

        in_chunk = False
        buf: List[str] = []
        img: Optional[str] = None
        ev:  Optional[str] = None

        for raw in f:
            line = raw.rstrip("\n")

            if RE_BEGIN_CHUNK.search(line):
                in_chunk = True
                buf = [line + "\n"]
                img = ev = None
                continue

            if not in_chunk:
                continue

            if img is None:
                m = RE_IMG_FN.match(line)
                if m:
                    img = os.path.basename(m.group(1).strip())
            if ev is None:
                m = RE_EVENT.match(line)
                if m:
                    ev = m.group(1).strip()

            buf.append(line + "\n")

            if RE_END_CHUNK.search(line):
                if img and ev:
                    key = (img, ev)
                    occ_now[key] += 1
                    want_idx = wanted.get(key)
                    if want_idx is not None and occ_now[key] == want_idx:
                        out.writelines(buf)
                        written += 1
                        del wanted[key]
                        if not wanted:
                            break
                in_chunk = False
                buf = []
                img = ev = None

    return written


# ------------------------------ Main -----------------------------------

def common_prefix(stems: Sequence[str]) -> str:
    if not stems:
        return "output"
    prefix = os.path.commonprefix(list(stems))
    while prefix and not prefix[-1].isalnum():
        prefix = prefix[:-1]
    return prefix or "output"

def main():
    args = parse_args()

    # Validate input directory & list streams
    if not args.dir.is_dir():
        sys.stderr.write(f"ERROR: {args.dir} is not a directory\n")
        sys.exit(1)
    streams = sorted([p for p in args.dir.iterdir() if p.suffix == ".stream"])
    if not streams:
        sys.stderr.write(f"ERROR: No .stream files found in {args.dir}\n")
        sys.exit(1)

    # Output path + header
    prefix = common_prefix([p.stem for p in streams])
    out_dir = args.dir / "best_wrmsd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}.stream"
    write_header_from_first_stream(streams, out_path)

    # ── PASS 1: discover which keys exist in which streams (also count chunks)
    method = "fork" if (sys.platform != "win32" and sys.platform != "darwin") else "spawn"
    ctx = mp.get_context(method)

    n_workers_p1 = min(max(1, os.cpu_count() or 1), len(streams))
    with ctx.Pool(processes=n_workers_p1) as pool:
        p1_jobs = [pool.apply_async(scan_stream_keys_worker, ((i, str(sp)),)) for i, sp in enumerate(streams)]
        per_stream_keys: List[set[Tuple[str,str]]] = [set() for _ in streams]
        keys_streams: Dict[Tuple[str,str], set[int]] = defaultdict(set)
        total_chunks = 0
        for r in p1_jobs:
            sidx, keys, n_chunks = r.get()
            per_stream_keys[sidx] = keys
            for k in keys:
                keys_streams[k].add(sidx)
            total_chunks += n_chunks

    remaining_per_key: Dict[Tuple[str,str], int] = {k: len(v) for k, v in keys_streams.items()}
    total_keys = len(remaining_per_key)

    # ── Progress bars (three stacked)
    #    1) chunks processed across all streams
    #    2) streams finished (pass 2)
    #    3) winners written to output
    pbar_chunks = tqdm(total=total_chunks, desc="Chunks processed", unit="chunk", position=0, dynamic_ncols=True) if tqdm else None
    pbar_streams = tqdm(total=len(streams), desc="Streams finished", unit="file", position=1, dynamic_ncols=True) if tqdm else None
    pbar_winners = tqdm(total=total_keys, desc="Winners written", unit="chunk", position=2, dynamic_ncols=True) if tqdm else None

    # ── PASS 2: compute per-chunk wRMSD, merge minima, write winners progressively
    progress_q = ctx.Queue()  # unbounded, supports get_nowait
    n_workers = (os.cpu_count() or 1) if args.workers == 0 else max(1, args.workers)
    n_workers = min(n_workers, len(streams))

    with ctx.Pool(processes=n_workers, initializer=_init_worker_progress, initargs=(progress_q,)) as pool:
        # Submit all workers
        pending: Dict[int, mp.pool.ApplyResult] = {
            i: pool.apply_async(
                process_stream_worker,
                ((i, str(sp), float(args.match_radius), float(args.outlier_sigma),
                  int(args.min_peaks), int(args.min_reflections)),)
            )
            for i, sp in enumerate(streams)
        }

        # Aggregation state
        global_best: Dict[Tuple[str,str], Tuple[float, int, int]] = {}   # key -> (wr, best_stream_idx, occ_idx_in_best)
        stream_finished = [False] * len(streams)
        winners_by_stream: Dict[int, Dict[Tuple[str,str], int]] = defaultdict(dict)  # sidx -> {key: occ_idx}
        written_keys: set[Tuple[str,str]] = set()

        def drain_chunks_progress():
            if not pbar_chunks:
                return
            advanced = 0
            while True:
                try:
                    advanced += int(progress_q.get_nowait())
                except Exception:
                    break
            if advanced:
                pbar_chunks.update(advanced)

        def maybe_write_ready():
            # Write winners for streams that have finished and have pending winners
            total_written_now = 0
            for sidx, mapping in list(winners_by_stream.items()):
                if mapping and stream_finished[sidx]:
                    written = write_winners_for_stream(streams[sidx], mapping, out_path)
                    total_written_now += written
                    if written:
                        for k in list(mapping.keys()):
                            written_keys.add(k)
                    winners_by_stream[sidx].clear()
            if pbar_winners and total_written_now:
                pbar_winners.update(total_written_now)

        remaining_jobs = set(pending.keys())

        # Central event loop: update progress continuously, reap any results, write ready winners
        while remaining_jobs:
            # 1) show chunk progress
            drain_chunks_progress()

            # 2) reap any finished workers (in any order)
            just_finished = []
            for sidx in list(remaining_jobs):
                r = pending[sidx]
                if r.ready():
                    try:
                        sidx_ret, best_local = r.get()
                        assert sidx_ret == sidx
                    except Exception as e:
                        sys.stderr.write(f"[WARN] worker {sidx} raised: {e}\n")
                        best_local = {}
                    stream_finished[sidx] = True
                    just_finished.append(sidx)
                    if pbar_streams:
                        pbar_streams.update(1)

                    # Merge local minima into global minima
                    for k, (wr, occ_idx) in best_local.items():
                        gb = global_best.get(k)
                        if (gb is None) or (wr < gb[0]):
                            global_best[k] = (wr, sidx, occ_idx)

                    # Decrement remaining counters for keys present in this stream;
                    # when they hit zero, we can finalize that key to its best stream.
                    for k in per_stream_keys[sidx]:
                        rem = remaining_per_key.get(k)
                        if rem is None:
                            continue
                        rem -= 1
                        if rem <= 0:
                            remaining_per_key.pop(k, None)
                            if k in global_best and k not in written_keys:
                                _, best_sidx, occ_idx = global_best[k]
                                winners_by_stream[best_sidx][k] = occ_idx
                        else:
                            remaining_per_key[k] = rem

            # 3) write any winners whose best stream is finished
            if just_finished:
                maybe_write_ready()
                remaining_jobs.difference_update(just_finished)
            else:
                # avoid hot spin; still drain progress a little later
                time.sleep(0.02)

        # Final drain and close bars
        drain_chunks_progress()
        # After all streams done, write any remaining winners (should be none, but safe)
        pending_total = 0
        for sidx, mapping in list(winners_by_stream.items()):
            if mapping:
                written = write_winners_for_stream(streams[sidx], mapping, out_path)
                pending_total += written
                if written:
                    for k in list(mapping.keys()):
                        written_keys.add(k)
                winners_by_stream[sidx].clear()
        if pbar_winners and pending_total:
            pbar_winners.update(pending_total)

    # Close progress bars neatly
    if pbar_chunks:  pbar_chunks.close()
    if pbar_streams: pbar_streams.close()
    if pbar_winners: pbar_winners.close()

    print(f"Done. Wrote winners to {out_path}")

if __name__ == "__main__":
    main()

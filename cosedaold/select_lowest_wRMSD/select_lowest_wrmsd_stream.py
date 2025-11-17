#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_lowest_wrmsd_stream.py (ETA + fast write via offsets + optional parallel write)

Three-stage, traceable pipeline for selecting lowest-wRMSD chunks across CrystFEL .stream files.

Pipeline:
  1) calc   — One pass over each .stream; compute per-chunk wRMSD (with details)
               and write newline-delimited JSON (gz) metrics per stream.
               ETA: optional fast pre-count of chunks so the progress bar shows time remaining.
               Also records byte offsets for each chunk for fast writing later.
  2) select — Read metrics only (no streams). Choose the lowest-wRMSD per (img_base,event)
               with deterministic tie-breaks and write winners.jsonl.gz (including offsets).
  3) write  — Copy winners directly by byte offsets (seek+copy) into a single merged .stream.
               Optional parallelization per stream. Header taken from the first .stream.

Additionally:
  * Only TWO progress bars: "Calculating wRMSD" (stage 1) and "Writing winners" (stage 3).
  * Metrics are fully auditable/replayable.

Usage:
  # Run all three stages in sequence (ETA enabled by default)
  python3 select_lowest_wrmsd_stream.py all /path/to/streams

  # Or stage-by-stage
  python3 select_lowest_wrmsd_stream.py calc   /path/to/streams [--workers 0] [--metrics-dir metrics] [--resume] [--no-precount]
  python3 select_lowest_wrmsd_stream.py select /path/to/streams [--metrics-dir metrics] [--winners winners.jsonl.gz]
  python3 select_lowest_wrmsd_stream.py write  /path/to/streams [--winners winners.jsonl.gz] [--out OUTPATH] [--write-workers 0]

Notes:
  - Requires only the standard library + numpy; will use SciPy cKDTree if available for faster NN.
  - tqdm is optional; if missing, the script still runs (just without bars/ETA).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Optional (faster) SciPy KD-Tree
_HAVE_SCIPY = False
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# --------------------------- Regex & Parsing ---------------------------

FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

RE_BEGIN_CHUNK   = re.compile(r"^-{5}\s+Begin chunk\s+-{5}")
RE_END_CHUNK     = re.compile(r"^-{5}\s+End(?: of)? chunk\s+-{5}")

RE_IMG_FN        = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
RE_EVENT         = re.compile(r"^\s*Event:\s*(?:/+)?\s*(\S+)\s*$")

RE_BEGIN_PEAKS   = re.compile(r"^\s*Peaks from peak search", re.IGNORECASE)
RE_END_PEAKS     = re.compile(r"^\s*End of peak list", re.IGNORECASE)
# Peak line: fs ss (3rd float ignored) intensity panel
RE_PEAK_LINE     = re.compile(rf"^\s*({FLOAT_RE})\s+({FLOAT_RE})\s+{FLOAT_RE}\s+({FLOAT_RE})\s+(\S+)\s*$")

RE_BEGIN_CRYSTAL = re.compile(r"^\s*---\s+Begin crystal", re.IGNORECASE)
RE_END_CRYSTAL   = re.compile(r"^\s*---\s+End crystal", re.IGNORECASE)

RE_BEGIN_REFL    = re.compile(r"^\s*Reflections measured after indexing", re.IGNORECASE)
RE_END_REFL      = re.compile(r"^\s*End of reflections", re.IGNORECASE)
# Reflections: capture last two floats (fs, ss) and final token (panel)
RE_REFL_LINE     = re.compile(rf".*?\s({FLOAT_RE})\s+({FLOAT_RE})\s+(\S+)\s*$")


class ChunkLite:
    __slots__ = ("image", "event", "peaks", "refls")
    def __init__(self):
        self.image: Optional[str] = None
        self.event: Optional[str] = None
        self.peaks: List[Tuple[float, float, float, str]] = []  # (fs, ss, intensity, panel)
        self.refls: List[Tuple[float, float, str]] = []         # (fs, ss, panel)


# ------------------------------ wRMSD ----------------------------------

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

class WRMSDResult:
    __slots__ = ("wrmsd","n_matches","n_kept","reason")
    def __init__(self, wrmsd: Optional[float], n_matches: int, n_kept: int, reason: Optional[str]):
        self.wrmsd = wrmsd
        self.n_matches = n_matches
        self.n_kept = n_kept
        self.reason = reason

def compute_wrmsd_details(
    peaks: List[Tuple[float, float, float, str]],
    refls: List[Tuple[float, float, str]],
    match_radius: float,
    outlier_sigma: float
) -> WRMSDResult:
    """
    Per-panel NN between peaks and reflections, intensity-weighted RMSD,
    with sigma clipping. Returns detail counts and a reason if wrmsd is None.
    """
    if not peaks or not refls:
        return WRMSDResult(None, 0, 0, "too_few_peaks_or_refl")

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
    n_matches = 0
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
        cnt = int(within.sum())
        if cnt > 0:
            n_matches += cnt
            md_all.append(d[within])
            w_all.append(pint[within])

    if not md_all:
        return WRMSDResult(None, n_matches, 0, "no_matches")

    md = np.concatenate(md_all)
    w  = np.concatenate(w_all)
    keep = _sigma_mask(md, outlier_sigma)
    n_kept = int(keep.sum())
    if n_kept == 0:
        return WRMSDResult(None, n_matches, 0, "all_clipped")

    kept = md[keep]
    kw = w[keep]
    wsum = float(kw.sum())
    if wsum <= 0.0:
        return WRMSDResult(None, n_matches, n_kept, "zero_weight")

    wr = math.sqrt(float((kw * (kept ** 2)).sum()) / wsum)
    if math.isnan(wr) or math.isinf(wr):
        return WRMSDResult(None, n_matches, n_kept, "nan_inf")
    return WRMSDResult(float(wr), n_matches, n_kept, None)


# ----------------------- IO helpers / header ---------------------------

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

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# --------------------- Fast pre-count for ETA (calc) --------------------

def _count_chunks_in_stream(path_str: str) -> int:
    """
    Return the number of chunks in a .stream by counting 'Begin chunk' lines.
    Extremely fast (single pass, no parsing).
    """
    sp = Path(path_str)
    n = 0
    try:
        with sp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if RE_BEGIN_CHUNK.search(line):
                    n += 1
    except Exception:
        pass
    return n

# --------- Offset-aware iterator (for calc) to capture byte ranges ------

def iter_chunks_for_metrics_with_offsets(stream_path: Path) -> Iterable[Tuple[ChunkLite, int, int]]:
    """
    Yield (ChunkLite, start_off, end_off) using *binary* I/O so the offsets
    are true byte positions. Lines are decoded per-read for regex matching.
    """
    def dec(b: bytes) -> str:
        # Decode once per line; strip only CR/LF so ^/$ anchors keep working.
        return b.decode("utf-8", errors="ignore").rstrip("\r\n")

    with stream_path.open("rb") as fb:
        in_chunk = in_peaks = in_crystal = in_refl = False
        ch: Optional[ChunkLite] = None
        chunk_start_off: Optional[int] = None

        while True:
            pos_before = fb.tell()
            raw = fb.readline()
            if not raw:
                # EOF: flush last chunk if open
                if in_chunk and ch and ch.image and ch.event and chunk_start_off is not None:
                    yield ch, chunk_start_off, fb.tell()
                break

            line = dec(raw)

            if RE_BEGIN_CHUNK.search(line):
                # close prior chunk if any
                if in_chunk and ch and ch.image and ch.event and chunk_start_off is not None:
                    # previous chunk ended at the start of this line
                    yield ch, chunk_start_off, pos_before
                in_chunk = True
                in_peaks = in_crystal = in_refl = False
                ch = ChunkLite()
                chunk_start_off = pos_before
                continue

            if not in_chunk:
                continue

            if RE_END_CHUNK.search(line):
                if ch and ch.image and ch.event and chunk_start_off is not None:
                    # end exactly after this line's bytes
                    yield ch, chunk_start_off, fb.tell()
                in_chunk = in_peaks = in_crystal = in_refl = False
                ch = None
                chunk_start_off = None
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

# -------------------------- Stage 1: calc ------------------------------

# Shared progress queue for calc workers
PROG_Q = None  # type: ignore

def _init_worker_progress(progress_q):
    global PROG_Q
    PROG_Q = progress_q

def process_stream_to_metrics(args) -> Tuple[int, int]:
    """
    Worker: stream -> metrics JSONL.GZ file.
    Returns (stream_idx, chunks_processed) for bookkeeping.
    """
    (stream_idx, path_str, metrics_dir, match_radius, outlier_sigma,
     min_peaks, min_refl, resume) = args

    sp = Path(path_str)
    metrics_dir = Path(metrics_dir)
    _ensure_dir(metrics_dir)
    out_path = metrics_dir / f"{sp.stem}.jsonl.gz"

    if resume and out_path.exists():
        # Skip this stream entirely in resume mode.
        return (stream_idx, 0)

    BATCH = 1000
    processed = 0
    occ_counter: Dict[Tuple[str,str], int] = defaultdict(int)

    try:
        with gzip.open(out_path, "wt", encoding="utf-8") as gz:
            for ch, start_off, end_off in iter_chunks_for_metrics_with_offsets(sp):
                processed += 1
                if PROG_Q and (processed % BATCH == 0):
                    PROG_Q.put(BATCH)

                if not (ch.image and ch.event):
                    continue

                key = (os.path.basename(ch.image), ch.event)
                occ_counter[key] += 1

                n_peaks = len(ch.peaks)
                n_refls = len(ch.refls)

                # Apply minimum thresholds early
                if n_peaks < min_peaks or n_refls < min_refl:
                    rec = {
                        "stream_idx": stream_idx,
                        "stream_path": str(sp),
                        "img_base": key[0],
                        "event": key[1],
                        "occ_idx": occ_counter[key],
                        "wrmsd": None,
                        "n_peaks": n_peaks,
                        "n_refl": n_refls,
                        "n_matches": 0,
                        "n_kept": 0,
                        "reason": "too_few_peaks_or_refl",
                        "start_off": int(start_off),
                        "end_off": int(end_off),
                    }
                    gz.write(json.dumps(rec) + "\n")
                    continue

                res = compute_wrmsd_details(ch.peaks, ch.refls, match_radius, outlier_sigma)

                rec = {
                    "stream_idx": stream_idx,
                    "stream_path": str(sp),
                    "img_base": key[0],
                    "event": key[1],
                    "occ_idx": occ_counter[key],
                    "wrmsd": res.wrmsd,
                    "n_peaks": n_peaks,
                    "n_refl": n_refls,
                    "n_matches": res.n_matches,
                    "n_kept": res.n_kept,
                    "reason": res.reason,
                    "start_off": int(start_off),
                    "end_off": int(end_off),
                }
                gz.write(json.dumps(rec) + "\n")
    except Exception as e:
        sys.stderr.write(f"[WARN] calc worker {sp.name}: {e}\n")

    if PROG_Q:
        rem = processed % BATCH
        if rem:
            PROG_Q.put(rem)

    return (stream_idx, processed)


def stage_calc(
    streams_dir: Path,
    metrics_dir: Path,
    workers: int,
    match_radius: float,
    outlier_sigma: float,
    min_peaks: int,
    min_reflections: int,
    resume: bool,
    precount: bool
) -> None:
    if not streams_dir.is_dir():
        sys.stderr.write(f"ERROR: {streams_dir} is not a directory\n")
        sys.exit(1)

    streams = sorted([p for p in streams_dir.iterdir() if p.suffix == ".stream"])
    if not streams:
        sys.stderr.write(f"ERROR: No .stream files found in {streams_dir}\n")
        sys.exit(1)

    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Determine which streams we will actually process (respect --resume)
    to_process: List[Path] = []
    for sp in streams:
        if resume:
            out_path = metrics_dir / f"{sp.stem}.jsonl.gz"
            if out_path.exists():
                continue
        to_process.append(sp)

    if not to_process:
        sys.stderr.write("[calc] Nothing to do (all metrics present or no streams).\n")
        return

    # Multiprocessing context
    method = "fork" if (sys.platform != "win32" and sys.platform != "darwin") else "spawn"
    ctx = mp.get_context(method)

    # Choose worker count
    if workers == 0:
        n_workers = min(max(1, os.cpu_count() or 1), len(to_process))
    else:
        n_workers = min(max(1, workers), len(to_process))

    # Optional pre-count for ETA
    total_chunks = None
    if precount and tqdm is not None:
        with ctx.Pool(processes=min(n_workers, len(to_process))) as pool:
            totals = pool.map(_count_chunks_in_stream, [str(p) for p in to_process])
        total_chunks = int(sum(totals))

    progress_q = ctx.Queue()
    with ctx.Pool(processes=n_workers, initializer=_init_worker_progress, initargs=(progress_q,)) as pool:
        jobs: Dict[int, mp.pool.ApplyResult] = {
            i: pool.apply_async(
                process_stream_to_metrics,
                ((i, str(sp), str(metrics_dir), float(match_radius), float(outlier_sigma),
                  int(min_peaks), int(min_reflections), bool(resume)),)
            )
            for i, sp in enumerate(to_process)
        }

        # Progress bar (known total if we pre-counted)
        if tqdm:
            pbar = tqdm(total=total_chunks, desc="Calculating wRMSD", unit="chunk", dynamic_ncols=True)
        else:
            pbar = None

        remaining = set(jobs.keys())
        try:
            while remaining:
                # Drain progress increments
                advanced = 0
                while True:
                    try:
                        advanced += int(progress_q.get_nowait())
                    except Exception:
                        break
                if pbar is not None and advanced:
                    pbar.update(advanced)

                just_finished = []
                for sidx in list(remaining):
                    r = jobs[sidx]
                    if r.ready():
                        try:
                            r.get()
                        except Exception as e:
                            sys.stderr.write(f"[WARN] calc worker {sidx} raised: {e}\n")
                        just_finished.append(sidx)
                for sidx in just_finished:
                    remaining.discard(sidx)

                if not just_finished:
                    time.sleep(0.02)
        finally:
            if pbar is not None:
                pbar.close()


# -------------------------- Stage 2: select ----------------------------

def stage_select(
    metrics_dir: Path,
    winners_path: Path
) -> None:
    if not metrics_dir.is_dir():
        sys.stderr.write(f"ERROR: metrics dir {metrics_dir} not found\n")
        sys.exit(1)

    metric_files = sorted([p for p in metrics_dir.iterdir()
                           if p.suffixes[-2:] == [".jsonl", ".gz"] or
                              (p.suffixes[-1:] == [".gz"] and p.name.endswith(".jsonl.gz"))])
    if not metric_files:
        sys.stderr.write(f"ERROR: No metrics *.jsonl.gz in {metrics_dir}\n")
        sys.exit(1)

    # value = (wrmsd, occ_idx, stream_path, stream_idx, start_off, end_off)
    best_by_key: Dict[Tuple[str,str], Tuple[float, int, str, int, int, int]] = {}

    seen_keys: set[Tuple[str,str]] = set()
    reasons_per_key: Dict[Tuple[str,str], Counter] = defaultdict(Counter)

    for mf in metric_files:
        try:
            with gzip.open(mf, "rt", encoding="utf-8") as gz:
                for line in gz:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    key = (str(rec["img_base"]), str(rec["event"]))
                    seen_keys.add(key)

                    wr = rec.get("wrmsd", None)
                    if wr is not None:
                        cand = (
                            float(wr),
                            int(rec["occ_idx"]),
                            str(rec["stream_path"]),
                            int(rec["stream_idx"]),
                            int(rec.get("start_off", -1)),
                            int(rec.get("end_off", -1)),
                        )
                        cur = best_by_key.get(key)
                        # Deterministic tie-break: lower wrmsd, then lower occ_idx, then lexicographic stream_path
                        if cur is None or (cand[0], cand[1], cand[2]) < (cur[0], cur[1], cur[2]):
                            best_by_key[key] = cand
                    else:
                        reason = rec.get("reason", "unknown")
                        reasons_per_key[key][reason] += 1
        except Exception as e:
            sys.stderr.write(f"[WARN] select reading {mf.name}: {e}\n")

    _ensure_dir(winners_path)
    total_seen = len(seen_keys)
    total_winners = len(best_by_key)
    unresolved_keys = seen_keys.difference(best_by_key.keys())

    with gzip.open(winners_path, "wt", encoding="utf-8") as gz:
        for (img_base, event), (wrmsd, occ_idx, stream_path, stream_idx, start_off, end_off) in best_by_key.items():
            gz.write(json.dumps({
                "img_base": img_base,
                "event": event,
                "stream_path": stream_path,
                "stream_idx": stream_idx,
                "occ_idx": occ_idx,
                "wrmsd": wrmsd,
                "start_off": start_off,
                "end_off": end_off,
            }) + "\n")

    reason_counts = Counter()
    for k in unresolved_keys:
        if reasons_per_key[k]:
            reason_counts[reasons_per_key[k].most_common(1)[0][0]] += 1
        else:
            reason_counts["unknown"] += 1

    sys.stderr.write(
        f"[select] keys seen: {total_seen}, winners: {total_winners}, unresolved: {len(unresolved_keys)}\n"
    )
    if unresolved_keys:
        sys.stderr.write("[select] unresolved by reason: " + ", ".join(f"{r}:{c}" for r,c in reason_counts.items()) + "\n")


# -------------------------- Stage 3: write -----------------------------

def common_prefix(stems: Sequence[str]) -> str:
    if not stems:
        return "output"
    prefix = os.path.commonprefix(list(stems))
    while prefix and not prefix[-1].isalnum():
        prefix = prefix[:-1]
    return prefix or "output"

def _write_winner_ranges_for_stream(
    stream_path: Path,
    ranges: List[Tuple[int, int]],
    part_path: Path
) -> int:
    """
    Write all [start,end) byte ranges from stream_path into part_path (binary).
    Returns total bytes copied. Coalesces overlapping/adjacent ranges.
    """
    _ensure_dir(part_path)

    if not ranges:
        part_path.touch()
        return 0

    # Coalesce ranges to minimize seeks/reads
    ranges = sorted(ranges, key=lambda ab: ab[0])
    merged: List[Tuple[int, int]] = []
    cs, ce = ranges[0]
    for s, e in ranges[1:]:
        if s <= ce:  # overlap or adjacency
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    total = 0
    with stream_path.open("rb") as src, part_path.open("wb") as outb:
        for s, e in merged:
            src.seek(s)
            remaining = e - s
            while remaining > 0:
                chunk = src.read(min(1 << 20, remaining))
                if not chunk:
                    break
                outb.write(chunk)
                remaining -= len(chunk)
                total += len(chunk)
    return total

def _offsets_part_worker(args) -> Path:
    """
    Top-level (picklable) worker: write all [start,end) ranges from a single stream
    into a .part file. Returns the part path.
    args = (stream_path: Path, ranges: List[Tuple[int,int]], part_path: Path)
    """
    stream_path, ranges, part_path = args
    ranges = sorted(ranges, key=lambda ab: ab[0])
    # Reuse helper that writes in binary
    _write_winner_ranges_for_stream(stream_path, ranges, part_path)
    return part_path
def stage_write(
    streams_dir: Path,
    winners_path: Path,
    out_path: Optional[Path],
    write_workers: int = 0
) -> None:
    if not streams_dir.is_dir():
        sys.stderr.write(f"ERROR: {streams_dir} is not a directory\n")
        sys.exit(1)
    if not winners_path.is_file():
        sys.stderr.write(f"ERROR: winners file not found: {winners_path}\n")
        sys.exit(1)

    # Load winners (prefer offsets; fall back to occ_idx scan if offsets missing)
    by_stream_offsets: Dict[Path, List[Tuple[int,int]]] = defaultdict(list)
    by_stream_occidx: Dict[Path, Dict[Tuple[str,str], int]] = defaultdict(dict)
    n_winners = 0

    with gzip.open(winners_path, "rt", encoding="utf-8") as gz:
        for line in gz:
            if not line.strip():
                continue
            rec = json.loads(line)
            sp = Path(rec["stream_path"])
            start_off = int(rec.get("start_off", -1))
            end_off = int(rec.get("end_off", -1))
            if start_off >= 0 and end_off >= 0 and end_off > start_off:
                by_stream_offsets[sp].append((start_off, end_off))
            else:
                key = (str(rec["img_base"]), str(rec["event"]))
                by_stream_occidx[sp][key] = int(rec["occ_idx"])
            n_winners += 1

    if n_winners == 0:
        sys.stderr.write("[write] No winners to write.\n")
        return

    # Build default out_path if not provided
    streams = sorted([p for p in streams_dir.iterdir() if p.suffix == ".stream"])
    if out_path is None:
        if not streams:
            sys.stderr.write(f"ERROR: No .stream files in {streams_dir}\n")
            sys.exit(1)
        prefix = common_prefix([p.stem for p in streams])
        out_dir = streams_dir / "lowest_wrmsd"
        out_path = out_dir / f"{prefix}.stream"
        write_header_from_first_stream(streams, out_path)
    else:
        if streams and (not out_path.exists() or out_path.stat().st_size == 0):
            write_header_from_first_stream(streams, out_path)
        else:
            _ensure_dir(out_path)
            out_path.touch(exist_ok=True)

    # Progress bar over winners (we update from parent process)
    pbar = tqdm(total=n_winners, desc="Writing winners", unit="chunk", dynamic_ncols=True) if tqdm else None

    # Part directory
    parts_dir = (out_path.parent / ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    part_paths: List[Path] = []

    # --------- Offsets path (fast) with optional parallelism ----------
    if by_stream_offsets:
        # Prepare tasks for a top-level worker
        tasks = []
        total_ranges = 0
        for sp, ranges in by_stream_offsets.items():
            tasks.append((sp, ranges, parts_dir / (sp.stem + ".part")))
            total_ranges += len(ranges)

        if write_workers == 0:
            n_workers = min(max(1, os.cpu_count() or 1), len(tasks))
        else:
            n_workers = min(max(1, write_workers), len(tasks))

        if n_workers > 1:
            ctx = mp.get_context("fork" if (sys.platform != "win32" and sys.platform != "darwin") else "spawn")
            with ctx.Pool(processes=n_workers) as pool:
                for part_path in pool.imap_unordered(_offsets_part_worker, tasks, chunksize=1):
                    part_paths.append(Path(part_path))
                    # Update pbar approximately by the number of ranges in this stream
                    # (we don't have it here; update by 1 stream and do a second pass below)
                    if pbar is not None:
                        # defer precise updates; do nothing here
                        pass
        else:
            for sp, ranges, part_path in tasks:
                _offsets_part_worker((sp, ranges, part_path))
                part_paths.append(part_path)

        # Now accurately update pbar by the number of ranges (winners) handled via offsets
        if pbar is not None:
            pbar.update(total_ranges)

    # --------- Fallback path (no offsets for some streams) ------------
    def _fallback_scan_write(sp: Path, mapping: Dict[Tuple[str,str], int]) -> Path:
        part = parts_dir / (sp.stem + ".part")
        written = 0
        wanted = dict(mapping)
        occ_now: Dict[Tuple[str,str], int] = defaultdict(int)
        with sp.open("r", encoding="utf-8", errors="ignore") as f, part.open("w", encoding="utf-8") as out:
            in_chunk = False
            buf: List[str] = []
            img = ev = None
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
        if pbar is not None and written:
            pbar.update(written)
        return part

    for sp, mapping in by_stream_occidx.items():
        part_paths.append(_fallback_scan_write(sp, mapping))

    # Concatenate parts into the final output (deterministic: sort by stream path)
    with out_path.open("ab") as outb:
        for part in sorted(part_paths, key=lambda p: p.name):
            with part.open("rb") as pb:
                while True:
                    chunk = pb.read(1 << 20)
                    if not chunk:
                        break
                    outb.write(chunk)

    # Clean up parts
    for part in part_paths:
        try:
            part.unlink()
        except Exception:
            pass

    if pbar is not None:
        pbar.close()

    sys.stderr.write(f"[write] Merged winners into {out_path}\n")

# ------------------------------- CLI -----------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select lowest-wRMSD chunks across streams via (calc → select → write)")
    sub = ap.add_subparsers(dest="cmd", required=False)

    def add_common_tuning(p):
        p.add_argument("--match-radius", type=float, default=4.0, help="NN match radius in pixels")
        p.add_argument("--outlier-sigma", type=float, default=2.0, help="Sigma for outlier clipping")
        p.add_argument("--min-peaks", type=int, default=1, help="Skip chunks with fewer peaks")
        p.add_argument("--min-reflections", type=int, default=1, help="Skip chunks with fewer reflections")

    # calc
    ap_calc = sub.add_parser("calc", help="Compute per-chunk wRMSD and write metrics/*.jsonl.gz")
    ap_calc.add_argument("dir", type=Path, help="Directory containing .stream files")
    ap_calc.add_argument("--metrics-dir", type=Path, default=None, help="Directory to write metrics (default: DIR/metrics)")
    ap_calc.add_argument("--workers", type=int, default=0, help="Worker processes (0 = all logical CPUs, capped by number of files)")
    ap_calc.add_argument("--resume", action="store_true", help="Skip streams whose metrics file already exists")
    ap_calc.add_argument("--no-precount", action="store_true", help="Skip fast pre-count of chunks (no ETA for stage 1)")
    add_common_tuning(ap_calc)

    # select
    ap_sel = sub.add_parser("select", help="Select lowest-wRMSD per (img_base,event) from metrics")
    ap_sel.add_argument("dir", type=Path, help="Directory containing .stream files (used for defaults)")
    ap_sel.add_argument("--metrics-dir", type=Path, default=None, help="Directory containing metrics (default: DIR/metrics)")
    ap_sel.add_argument("--winners", type=Path, default=None, help="Path to write winners.jsonl.gz (default: DIR/winners.jsonl.gz)")

    # write
    ap_write = sub.add_parser("write", help="Write winning chunks to a merged .stream")
    ap_write.add_argument("dir", type=Path, help="Directory containing .stream files")
    ap_write.add_argument("--winners", type=Path, default=None, help="Path to winners.jsonl.gz (default: DIR/winners.jsonl.gz)")
    ap_write.add_argument("--out", type=Path, default=None, help="Output merged .stream path (default: DIR/lowest_wrmsd/<prefix>.stream)")
    ap_write.add_argument("--write-workers", type=int, default=0, help="Writer processes for per-stream parts (0 = auto)")

    # all
    ap_all = sub.add_parser("all", help="Run calc → select → write in one go")
    ap_all.add_argument("dir", type=Path, help="Directory containing .stream files")
    ap_all.add_argument("--metrics-dir", type=Path, default=None, help="Directory to write metrics (default: DIR/metrics)")
    ap_all.add_argument("--winners", type=Path, default=None, help="Path to write winners.jsonl.gz (default: DIR/winners.jsonl.gz)")
    ap_all.add_argument("--out", type=Path, default=None, help="Output merged .stream path (default: DIR/lowest_wrmsd/<prefix>.stream)")
    ap_all.add_argument("--workers", type=int, default=0, help="Worker processes (0 = all logical CPUs, capped by number of files)")
    ap_all.add_argument("--resume", action="store_true", help="Skip streams whose metrics file already exists")
    ap_all.add_argument("--no-precount", action="store_true", help="Skip fast pre-count of chunks (no ETA for stage 1)")
    ap_all.add_argument("--write-workers", type=int, default=0, help="Writer processes for per-stream parts (0 = auto)")
    add_common_tuning(ap_all)

    # ---- key bit: inject default subcommand BEFORE argparse validates ----
    argv = sys.argv[1:]
    subcommands = {"calc", "select", "write", "all"}
    if not argv or argv[0].startswith("-") or argv[0] not in subcommands:
        argv = ["all", *argv]

    return ap.parse_args(argv)

def main() -> None:
    args = parse_args()

    if args.cmd in ("calc", "all"):
        streams_dir: Path = args.dir
        metrics_dir: Path = args.metrics_dir or (streams_dir / "metrics")
        stage_calc(
            streams_dir=streams_dir,
            metrics_dir=metrics_dir,
            workers=int(getattr(args, "workers", 0)),
            match_radius=float(args.match_radius),
            outlier_sigma=float(args.outlier_sigma),
            min_peaks=int(args.min_peaks),
            min_reflections=int(args.min_reflections),
            resume=bool(getattr(args, "resume", False)),
            precount=not bool(getattr(args, "no_precount", False))
        )

    if args.cmd in ("select", "all"):
        streams_dir: Path = args.dir
        metrics_dir: Path = args.metrics_dir or (streams_dir / "metrics")
        winners_path: Path = args.winners or (streams_dir / "winners.jsonl.gz")
        stage_select(metrics_dir=metrics_dir, winners_path=winners_path)

    if args.cmd in ("write", "all"):
        streams_dir: Path = args.dir
        winners_path: Path = args.winners or (streams_dir / "winners.jsonl.gz")
        out_path: Optional[Path] = getattr(args, "out", None)
        stage_write(
            streams_dir=streams_dir,
            winners_path=winners_path,
            out_path=out_path,
            write_workers=int(getattr(args, "write_workers", 0)),
        )


if __name__ == "__main__":
    main()

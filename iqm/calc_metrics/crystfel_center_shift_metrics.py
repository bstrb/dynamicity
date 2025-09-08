#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CrystFEL center-shift metrics (fast, parallel, RAM-safe)

- Progress bar shows number of CHUNKS processed (pre-counts first).
- Multiprocessing across .stream files (default: use all CPU cores).
- Workers compute metrics and STREAM rows to an on-disk SQLite DB (no big in-RAM list).
- KDTree (SciPy) for fast nearest neighbors if available; NumPy fallback otherwise.
- Final CSV is grouped by (Image filename, Event) and includes comment lines:
    # Image filename: <path>
    # Event: //<event>

Outputs columns:
  stream_file,weighted_rmsd,fraction_outliers,length_deviation,angle_deviation,peak_ratio,percentage_unindexed

Usage:
  python crystfel_center_shift_metrics.py \
    --dir /path/to/streams \
    --output /path/to/center_shift_metrics.csv \
    --match-radius 3.0 \
    --outlier-sigma 3.0 \
    --workers 0
"""

import argparse
import csv
import math
import os
import re
import sqlite3
import sys
import time
import subprocess  # add this near the other imports

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import multiprocessing as mp
import queue as queue_std

# ---- Worker-shared globals (set by pool initializer) ----
WORK_ROWS_Q = None

def _init_worker(rows_q):
    """
    Pool initializer: attach inherited queue to a module-global
    so workers don't receive Queue via pickling (which triggers warnings).
    """
    global WORK_ROWS_Q
    WORK_ROWS_Q = rows_q


# ----- Optional fast neighbors + progress -----
_HAVE_SCIPY = False
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# ------------------------ Regexes ------------------------
FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
INT_RE = r"[-+]?\d+"

RE_BEGIN_UNIT_CELL = re.compile(r"^-{5}\s+Begin unit cell\s+-{5}")
RE_END_UNIT_CELL   = re.compile(r"^-{5}\s+End unit cell\s+-{5}")
RE_A = re.compile(rf"^\s*a\s*=\s*({FLOAT_RE})\s*A", re.IGNORECASE)
RE_B = re.compile(rf"^\s*b\s*=\s*({FLOAT_RE})\s*A", re.IGNORECASE)
RE_C = re.compile(rf"^\s*c\s*=\s*({FLOAT_RE})\s*A", re.IGNORECASE)
RE_AL = re.compile(rf"^\s*al(?:pha)?\s*=\s*({FLOAT_RE})\s*deg", re.IGNORECASE)
RE_BE = re.compile(rf"^\s*be(?:ta)?\s*=\s*({FLOAT_RE})\s*deg", re.IGNORECASE)
RE_GA = re.compile(rf"^\s*ga(?:mma)?\s*=\s*({FLOAT_RE})\s*deg", re.IGNORECASE)

RE_BEGIN_CHUNK = re.compile(r"^-{5}\s+Begin chunk\s+-{5}")
RE_END_CHUNK = re.compile(r"^-{5}\s+End(?: of)? chunk\s+-{5}")

RE_IMG_FN      = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
RE_EVENT = re.compile(r"^\s*Event:\s*(?:/+)?\s*(\S+)\s*$")

RE_NUM_PEAKS   = re.compile(r"^\s*num_peaks\s*=\s*(" + INT_RE + r")\s*$")
RE_BEGIN_PEAKS = re.compile(r"^\s*Peaks from peak search", re.IGNORECASE)
RE_END_PEAKS   = re.compile(r"^\s*End of peak list", re.IGNORECASE)

RE_BEGIN_CRYSTAL = re.compile(r"^\s*---\s+Begin crystal", re.IGNORECASE)
RE_END_CRYSTAL   = re.compile(r"^\s*---\s+End crystal", re.IGNORECASE)
RE_CELL_PARAMS = re.compile(
    rf"^\s*Cell parameters\s+({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s+nm,\s+({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s+deg",
    re.IGNORECASE
)
RE_BEGIN_REFL = re.compile(r"^\s*Reflections measured after indexing", re.IGNORECASE)
RE_END_REFL   = re.compile(r"^\s*End of reflections", re.IGNORECASE)

RE_PEAK_LINE = re.compile(
    rf"^\s*({FLOAT_RE})\s+({FLOAT_RE})\s+{FLOAT_RE}\s+({FLOAT_RE})\s+(\S+)\s*$"
)
RE_REFL_LINE = re.compile(
    rf".*?\s({FLOAT_RE})\s+({FLOAT_RE})\s+(\S+)\s*$"
)

# ------------------------ Data structures ------------------------
class TargetCell:
    __slots__ = ("a","b","c","al","be","ga")
    def __init__(self, a_A: float, b_A: float, c_A: float, al: float, be: float, ga: float):
        self.a=a_A; self.b=b_A; self.c=c_A; self.al=al; self.be=be; self.ga=ga

class ChunkData:
    __slots__ = ("image","event","num_peaks_declared","peaks","cell_params_A","reflections")
    def __init__(self):
        self.image: Optional[str] = None
        self.event: Optional[str] = None
        self.num_peaks_declared: Optional[int] = None
        self.peaks: List[Tuple[float,float,float]] = []
        self.cell_params_A: Optional[Tuple[float,float,float,float,float,float]] = None
        self.reflections: List[Tuple[float,float]] = []

# ------------------------ Parsing ------------------------
def parse_target_cell(stream_path: Path) -> TargetCell:
    a=b=c=al=be=ga=None
    with stream_path.open("r", encoding="utf-8", errors="ignore") as f:
        in_unit=False
        for line in f:
            if not in_unit:
                if RE_BEGIN_UNIT_CELL.search(line):
                    in_unit=True
                continue
            if RE_END_UNIT_CELL.search(line):
                break
            m=RE_A.match(line);  a=float(m.group(1)) if m else a
            m=RE_B.match(line);  b=float(m.group(1)) if m else b
            m=RE_C.match(line);  c=float(m.group(1)) if m else c
            m=RE_AL.match(line); al=float(m.group(1)) if m else al
            m=RE_BE.match(line); be=float(m.group(1)) if m else be
            m=RE_GA.match(line); ga=float(m.group(1)) if m else ga
    if None in (a,b,c,al,be,ga):
        raise RuntimeError(f"Failed to parse target unit cell from {stream_path}")
    return TargetCell(a,b,c,al,be,ga)

def count_chunks_in_file(stream_path: Path) -> int:
    cnt=0
    with stream_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if RE_BEGIN_CHUNK.search(line):
                cnt+=1
    return cnt

def fast_count_chunks_grep(paths: List[Path]) -> int:
    """
    Very fast chunk pre-count using system grep:
    counts lines matching '----- Begin chunk -----' (case-insensitive, tolerant of spaces).
    Falls back to Python scanner on failure.
    """
    total = 0
    # POSIX ERE with character classes; accepts optional leading spaces
    pattern = r'^[[:space:]]*-{5}[[:space:]]*Begin[[:space:]]+chunk[[:space:]]*-{5}'
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # speed: byte-wise
    for p in paths:
        try:
            res = subprocess.run(
                ["grep", "-a", "-i", "-E", "-c", pattern, str(p)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                env=env,
            )
            n = int((res.stdout or "0").strip() or "0")
        except Exception:
            # Fallback to Python if grep not available
            n = count_chunks_in_file(p)
        total += n
    return total

def iterate_chunks(stream_path: Path) -> Iterable[ChunkData]:
    with stream_path.open("r", encoding="utf-8", errors="ignore") as f:
        in_chunk=in_peaks=in_crystal=in_refl=False
        ch: Optional[ChunkData]=None
        for raw in f:
            line = raw.rstrip("\n")

            # Handle a new Begin chunk regardless of current state
            if RE_BEGIN_CHUNK.search(line):
                if in_chunk and ch and ch.image and ch.event:
                    yield ch  # flush previous chunk even if no explicit end marker
                in_chunk = True
                in_peaks = in_crystal = in_refl = False
                ch = ChunkData()
                continue


            if RE_END_CHUNK.search(line):
                if ch and ch.image and ch.event:
                    yield ch
                in_chunk=in_peaks=in_crystal=in_refl=False
                ch=None
                continue

            if ch is None:
                continue

            if ch.image is None:
                m=RE_IMG_FN.match(line)
                if m:
                    ch.image=m.group(1).strip()
                    continue

            if ch.event is None:
                m=RE_EVENT.match(line)
                if m:
                    ch.event=m.group(1).strip()
                    continue

            if ch.num_peaks_declared is None:
                m=RE_NUM_PEAKS.match(line)
                if m:
                    ch.num_peaks_declared=int(m.group(1))
                    continue

            if not in_peaks and RE_BEGIN_PEAKS.search(line):
                in_peaks=True
                continue
            if in_peaks:
                if RE_END_PEAKS.search(line):
                    in_peaks=False
                    continue
                m=RE_PEAK_LINE.match(line)
                if m:
                    fs=float(m.group(1)); ss=float(m.group(2)); inten=float(m.group(3)); panel=m.group(4)
                    ch.peaks.append((fs, ss, inten, panel))
                continue

            if not in_crystal and RE_BEGIN_CRYSTAL.search(line):
                in_crystal=True
                continue
            if in_crystal and RE_END_CRYSTAL.search(line):
                in_crystal=False; in_refl=False
                continue
            if in_crystal:
                m=RE_CELL_PARAMS.match(line)
                if m and ch.cell_params_A is None:
                    aA=float(m.group(1))*10.0; bA=float(m.group(2))*10.0; cA=float(m.group(3))*10.0
                    al=float(m.group(4)); be=float(m.group(5)); ga=float(m.group(6))
                    ch.cell_params_A=(aA,bA,cA,al,be,ga)
                    continue
                if RE_BEGIN_REFL.search(line):
                    in_refl=True; continue
                if in_refl:
                    if RE_END_REFL.search(line):
                        in_refl = False
                        continue

                    m = RE_REFL_LINE.match(line)
                    if m:
                        fs = float(m.group(1))
                        ss = float(m.group(2))
                        panel = m.group(3)
                        ch.reflections.append((fs, ss, panel))
                    continue

        # EOF flush: yield an open chunk even if no explicit end marker
        if in_chunk and ch and ch.image and ch.event:
            yield ch


# ------------------------ Vector math ------------------------
def _sigma_clip_numpy(distances, sigma: float):
    import numpy as np
    if distances.size==0:
        return distances, 0, np.zeros((0,), dtype=bool)
    mu=distances.mean()
    sd=distances.std(ddof=1) if distances.size>1 else 0.0
    if sd==0.0:
        return distances, 0, np.ones_like(distances, dtype=bool)
    mask = distances <= (mu + sigma*sd)
    kept = distances[mask]
    n_out = distances.size - kept.size
    return kept, n_out, mask

def _nn_dists_numpy(pfs, pss, rfs, rss):
    import numpy as np
    if pfs.size==0 or rfs.size==0:
        return np.full(pfs.shape[0], np.inf, dtype=float)
    df = pfs[:,None]-rfs[None,:]
    ds = pss[:,None]-rss[None,:]
    d2 = df*df + ds*ds
    return (d2.min(axis=1))**0.5

def _nn_dists_kd(pfs, pss, rfs, rss):
    import numpy as np
    tree = cKDTree(np.column_stack((rfs,rss)))
    dists, _ = tree.query(np.column_stack((pfs,pss)), k=1, workers=-1)
    return dists.astype(float, copy=False)

def compute_metrics_np(
    peaks: List[Tuple[float,float,float,str]],
    refls: List[Tuple[float,float,str]],
    cell_params_A: Optional[Tuple[float,float,float,float,float,float]],
    tgt: TargetCell,
    match_radius: float,
    outlier_sigma: float,
) -> Dict[str, Optional[float]]:
    import numpy as np
    n_peaks = len(peaks); n_refl = len(refls)

    if n_peaks>0:
        pk = np.array(peaks, dtype=object)
        pfs, pss, pint, ppanel = pk[:,0].astype(float), pk[:,1].astype(float), pk[:,2].astype(float), pk[:,3].astype(str)
    else:
        pfs=pss=pint=np.empty((0,), dtype=float); ppanel=np.empty((0,), dtype=object)

    if n_refl>0:
        rf = np.array(refls, dtype=object)
        rfs, rss, rpanel = rf[:,0].astype(float), rf[:,1].astype(float), rf[:,2].astype(str)
    else:
        rfs=rss=np.empty((0,), dtype=float); rpanel=np.empty((0,), dtype=object)

    matched_count=0
    wrmsd = float("nan")
    frac_out = float("nan")

    if n_peaks>0 and n_refl>0:
        # match per panel
        d_all = []
        w_all = []
        for panel in np.unique(ppanel):
            pmask = (ppanel == panel)
            rmask = (rpanel == panel)
            if not pmask.any() or not rmask.any():
                continue
            _pfs = pfs[pmask]; _pss = pss[pmask]; _pint = pint[pmask]
            _rfs = rfs[rmask]; _rss = rss[rmask]
            d = _nn_dists_kd(_pfs,_pss,_rfs,_rss) if _HAVE_SCIPY else _nn_dists_numpy(_pfs,_pss,_rfs,_rss)
            within = d <= float(match_radius)
            if within.any():
                d_all.append(d[within])
                w_all.append(_pint[within])
        if d_all:
            md = np.concatenate(d_all)
            mw = np.concatenate(w_all)
        else:
            md = np.empty((0,), dtype=float)
            mw = np.empty((0,), dtype=float)
        matched_count = md.size
        if matched_count>0:
            kept, n_out, keep_mask = _sigma_clip_numpy(md, outlier_sigma)
            if kept.size>0:
                kw = mw[keep_mask]
                wsum = kw.sum()
                wrmsd = math.sqrt(float((kw*(kept**2)).sum()/wsum)) if wsum>0 else float("nan")
                frac_out = n_out/float(md.size)
            else:
                wrmsd = float("nan"); frac_out = 1.0

    percentage_unindexed = (100.0*(1.0 - matched_count/float(n_peaks))) if n_peaks>0 else float("nan")
    peak_ratio = (n_refl/float(n_peaks)) if n_peaks>0 else float("nan")

    if cell_params_A is not None:
        aA,bA,cA,al,be,ga = cell_params_A
        terms=[]
        for o,t in ((aA,tgt.a),(bA,tgt.b),(cA,tgt.c)):
            if t!=0.0:
                terms.append(((o-t)/t)**2)
        length_dev = math.sqrt(sum(terms)/3.0)*100.0 if terms else float("nan")
        angle_dev = math.sqrt(((al-tgt.al)**2 + (be-tgt.be)**2 + (ga-tgt.ga)**2)/3.0)
    else:
        length_dev=float("nan"); angle_dev=float("nan")

    return dict(
        weighted_rmsd=wrmsd,
        fraction_outliers=frac_out,
        length_deviation=length_dev,
        angle_deviation=angle_dev,
        peak_ratio=peak_ratio,
        percentage_unindexed=percentage_unindexed,
    )

# ------------------------ Worker ------------------------
# Rows sent to DB writer: tuples exactly matching INSERT order
# (image, event, stream, weighted_rmsd, fraction_outliers, length_deviation, angle_deviation, peak_ratio, percentage_unindexed)

def _worker_process_stream(args):
    # NOTE: rows queue comes from global set by _init_worker
    (stream_path_str, tgt_tuple, match_radius, outlier_sigma, batch_size) = args

    rq = WORK_ROWS_Q
    spath = Path(stream_path_str)
    sname = spath.name
    tgt = TargetCell(*tgt_tuple)

    batch = []
    pushed = 0

    last_flush = time.time()
    FLUSH_SECS = 0.5  # flush at least twice per second for responsive progress

    try:
        for ch in iterate_chunks(spath):
            m = compute_metrics_np(
                peaks=ch.peaks,
                refls=ch.reflections,
                cell_params_A=ch.cell_params_A,
                tgt=tgt,
                match_radius=match_radius,
                outlier_sigma=outlier_sigma,
            )
            row = (
                ch.image, ch.event, sname,
                m["weighted_rmsd"], m["fraction_outliers"],
                m["length_deviation"], m["angle_deviation"],
                m["peak_ratio"], m["percentage_unindexed"],
            )
            # optionally skip chunks with no peaks or reflections
            if len(ch.peaks) == 0 or len(ch.reflections) == 0:
                 continue

            batch.append(row)

            now = time.time()
            if len(batch) >= batch_size or (now - last_flush) >= FLUSH_SECS:
                if batch:
                    rq.put(batch)
                    pushed += len(batch)
                    batch = []
                last_flush = now

        if batch:
            rq.put(batch)
            pushed += len(batch)

    except Exception as e:
        sys.stderr.write(f"[WARN] {sname}: {e}\n")

    # Signal this worker is done (no payload)
    rq.put("__WORKER_DONE__")
    return pushed


# ------------------------ Main ------------------------
def main():
    ap = argparse.ArgumentParser(description="CrystFEL metrics with progress bar, parallel, and low RAM")
    ap.add_argument("--dir", required=True, help="Directory containing .stream files")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--match-radius", type=float, default=4.0, help="Match radius in pixels")
    ap.add_argument("--outlier-sigma", type=float, default=2.0, help="Outlier sigma")
    ap.add_argument("--reference-stream", default=None, help="Optional .stream path for the target cell")
    ap.add_argument("--workers", type=int, default=0, help="Worker processes (0 = all cores)")
    ap.add_argument("--db", default=None, help="Optional sqlite path (default: <dir>/center_shift_metrics.sqlite3)")
    ap.add_argument("--batch", type=int, default=200, help="Rows per IPC batch from worker to DB (default 2000)")
    ap.add_argument("--skip-precounter", action="store_true", help="Skip pre-counting chunks (progress shows unknown total)")
    args = ap.parse_args()

    base_dir = Path(args.dir).expanduser().resolve()
    if not base_dir.is_dir():
        sys.stderr.write(f"ERROR: --dir {base_dir} is not a directory\n")
        sys.exit(1)

    stream_files = sorted([p for p in base_dir.iterdir() if p.suffix == ".stream"])
    if not stream_files:
        sys.stderr.write(f"ERROR: No .stream files found in {base_dir}\n")
        sys.exit(1)

    # Target unit cell
    ref_path = Path(args.reference_stream).resolve() if args.reference_stream else stream_files[0]
    tgt = parse_target_cell(ref_path)
    tgt_tuple = (tgt.a, tgt.b, tgt.c, tgt.al, tgt.be, tgt.ga)

    # DB setup
    db_path = Path(args.db) if args.db else (base_dir / "center_shift_metrics.sqlite3")
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Speedy pragmas
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=FILE;")
    conn.commit()

    cur.execute("""
        CREATE TABLE metrics (
            image TEXT NOT NULL,
            event TEXT NOT NULL,
            stream TEXT NOT NULL,
            weighted_rmsd REAL,
            fraction_outliers REAL,
            length_deviation REAL,
            angle_deviation REAL,
            peak_ratio REAL,
            percentage_unindexed REAL
        )
    """)
    cur.execute("CREATE INDEX idx_chunk ON metrics(image, event)")
    cur.execute("CREATE INDEX idx_stream ON metrics(stream)")
    conn.commit()

    # Pre-count total chunks for progress
    total_chunks = None
    if not args.skip_precounter:
        t0=time.time()
        tc = fast_count_chunks_grep(stream_files)

        total_chunks = tc
        t1=time.time()
        # optional: print timing
        print(f"Pre-counted chunks: {tc} in {t1-t0:.1f}s")

    # Progress bar
    def _pbar(total):
        if tqdm is None:
            class Dummy: 
                def update(self, *a, **k): pass
                def close(self): pass
            return Dummy()
        return tqdm(total=total, unit="chunk", smoothing=0.05, desc="Processing", dynamic_ncols=True)
    pbar = _pbar(total_chunks)

    manager = mp.Manager()
    progress_q = None  # drive progress from DB inserts
    rows_q = manager.Queue(maxsize=50)

    # Launch pool
    n_workers = os.cpu_count() if args.workers==0 else max(1, args.workers)
    jobs = [(str(sp), tgt_tuple, float(args.match_radius), float(args.outlier_sigma), progress_q, rows_q, int(args.batch))
            for sp in stream_files]

    # Process context, queue, pool, jobs
    ctx = mp.get_context("spawn")

    rows_q = ctx.Queue(maxsize=200)  # inherited by workers via initializer

    n_workers = os.cpu_count() if args.workers == 0 else max(1, args.workers)
    jobs = [
        (str(sp), tgt_tuple, float(args.match_radius), float(args.outlier_sigma), int(args.batch))
        for sp in stream_files
    ]

    # Important: pass the queue via initializer so it's inherited (no warnings)
    pool = ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(rows_q,))

    results = [pool.apply_async(_worker_process_stream, (job,)) for job in jobs]


    # Main loop: drain queues, write to DB, update progress
    done_workers = 0
    inserted_since_commit = 0
    COMMIT_EVERY = 20000

    try:
        while done_workers < len(jobs):
        

            # Drain result rows
            drained_something = False
            try:
                while True:
                    item = rows_q.get_nowait()
                    drained_something = True
                    if item == "__WORKER_DONE__":
                        done_workers += 1
                        continue
                    # item is a list of rows
                    cur.executemany("""
                        INSERT INTO metrics
                        (image, event, stream, weighted_rmsd, fraction_outliers,
                        length_deviation, angle_deviation, peak_ratio, percentage_unindexed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, item)
                    pbar.update(len(item))
                    inserted_since_commit += len(item)


                    if inserted_since_commit >= COMMIT_EVERY:
                        conn.commit()
                        inserted_since_commit = 0
            except queue_std.Empty:
                pass

            # Reap finished async calls to surface exceptions (don’t store results)
            new_results = []
            for r in results:
                if r.ready():
                    try:
                        _ = r.get()  # raises if worker crashed
                    except Exception as e:
                        sys.stderr.write(f"[WARN] worker raised: {e}\n")
                else:
                    new_results.append(r)
            results = new_results

            # Avoid tight loop
            if not drained_something:
                time.sleep(0.05)


        try:
            while True:
                item = rows_q.get_nowait()
                if item != "__WORKER_DONE__":
                    cur.executemany("""
                        INSERT INTO metrics
                        (image, event, stream, weighted_rmsd, fraction_outliers,
                        length_deviation, angle_deviation, peak_ratio, percentage_unindexed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, item)
                    pbar.update(len(item))
                    inserted_since_commit += len(item)


        except queue_std.Empty:
            pass

        if inserted_since_commit:
            conn.commit()

    finally:
        pbar.close()
        pool.close()
        pool.join()

    # Emit CSV grouped by (image,event)
    out_path = Path(args.output).expanduser().resolve()
    with out_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "stream_file",
            "weighted_rmsd",
            "fraction_outliers",
            "length_deviation",
            "angle_deviation",
            "peak_ratio",
            "percentage_unindexed",
        ])

        def fmt(x: Optional[float], ndig: int) -> str:
            if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))):
                return "NA"
            return f"{x:.{ndig}f}"

        # Retrieve all distinct (image,event) without loading full table into RAM
        cur.execute("SELECT DISTINCT image, event FROM metrics")
        chunks = cur.fetchall()
        def _ev_key(ev: str):
            try: return int(ev)
            except Exception: return ev
        chunks.sort(key=lambda t: (t[0], _ev_key(t[1])))

        # For each chunk, dump rows ordered by stream
        for image, event in chunks:
            fcsv.write(f"# Image filename: {image}\n")
            fcsv.write(f"# Event: //{event}\n")
            cur.execute("""
                SELECT stream, weighted_rmsd, fraction_outliers, length_deviation,
                       angle_deviation, peak_ratio, percentage_unindexed
                FROM metrics WHERE image=? AND event=? ORDER BY stream ASC
            """, (image, event))
            for (s, wr, fo, ld, ad, pr, pu) in cur:
                writer.writerow([
                    s, fmt(wr,6), fmt(fo,4), fmt(ld,3), fmt(ad,3), fmt(pr,6), fmt(pu,2)
                ])

    conn.close()
    print(f"Done. Wrote {out_path}")
    print(f"(Intermediate DB: {db_path})")

if __name__ == "__main__":
    main()
    
# python crystfel_center_shift_metrics.py --dir /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2 --output /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/center_shift_metrics.csv


# python crystfel_center_shift_metrics.py --dir /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/xgandalf_iterations_max_radius_0.5_step_0.2 --output /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/xgandalf_iterations_max_radius_0.5_step_0.2/center_shift_metrics.csv
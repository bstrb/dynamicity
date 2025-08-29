#!/usr/bin/env python3
"""
csv_to_stream_parallel.py

Fast + optionally parallel merger of large XDS .stream files
listed in a CSV (columns: stream_file, event_number).

Call signature remains:

    write_stream_from_filtered_csv(
        filtered_csv_path,
        output_stream_path,
        event_col="event_number",
        streamfile_col="stream_file",
        mode="auto",          # "auto" | "threads" | "processes" | "serial"
        max_workers=None,     # None => sensible default
    )
"""

from __future__ import annotations
import csv, os, sys, mmap
from collections import defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

BEGIN = b"----- Begin chunk -----"
END   = b"----- End chunk -----\n"
EVENT_PREFIX = b"Event:"

# ────────────────────────────────────────────────────────────────────────────────
# 1.  CSV → wanted events & row order
# ────────────────────────────────────────────────────────────────────────────────
def _collect_requests(csv_path:str,
                      event_col:str,
                      streamfile_col:str
                     ) -> Tuple[List[Tuple[str,str]],
                                Dict[str,Set[str]]]:
    wanted : Dict[str,Set[str]] = defaultdict(set)
    rows   : List[Tuple[str,str]] = []
    base   = os.path.dirname(csv_path)

    with open(csv_path, newline="") as fh:
        for rec in csv.DictReader(fh):
            evt   = rec[event_col].strip()
            fpath = os.path.join(base, rec[streamfile_col].strip())
            wanted[fpath].add(evt)
            rows.append((fpath, evt))
    return rows, wanted

# ────────────────────────────────────────────────────────────────────────────────
# 2.  Parse ONE .stream file – **mmap + bytes** (fast, picklable)
# ────────────────────────────────────────────────────────────────────────────────
def _scan_one_file(task:Tuple[str,Set[str]]
                  ) -> Tuple[str, str, Dict[str,str]]:
    """Return (stream_path, header_text(str), {event->chunk(str)})"""
    stream_path, wanted_events = task
    chunks : Dict[str,str] = {}
    header_bytes : bytes = b""

    with open(stream_path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

        # find first BEGIN ⇒ everything before = header
        first_begin = mm.find(BEGIN)
        if first_begin != -1:
            header_bytes = mm[:first_begin]
        else:                       # malformed file
            return stream_path, header_bytes.decode(), {}

        pos = first_begin
        mm_len = mm.size()

        while pos < mm_len:
            next_end = mm.find(END, pos)
            if next_end == -1:
                break
            chunk_end = next_end + len(END)
            chunk = mm[pos:chunk_end]

            # fast check for Event: line inside the chunk
            ev_pos = chunk.find(EVENT_PREFIX)
            if ev_pos != -1:
                nl = chunk.find(b"\n", ev_pos)
                event_bytes = chunk[ev_pos+len(EVENT_PREFIX): nl].strip()
                if event_bytes.startswith(b"//"):
                    event_bytes = event_bytes[2:].strip()
                ev = event_bytes.decode()

                if ev in wanted_events:
                    chunks[ev] = chunk.decode()  # store *text* in RAM

            # jump to next chunk
            pos = mm.find(BEGIN, chunk_end)
            if pos == -1:
                break

    return stream_path, header_bytes.decode(), chunks

# ────────────────────────────────────────────────────────────────────────────────
# 3.  Public API
# ────────────────────────────────────────────────────────────────────────────────
def write_stream_from_filtered_csv(
    filtered_csv_path : str,
    output_stream_path: str,
    event_col         : str = "event_number",
    streamfile_col    : str = "stream_file",
    *,
    mode              : str = "auto",     # "auto"|"threads"|"processes"|"serial"
    max_workers       : int | None = None,
):
    """
    Merge selected chunks into one .stream.  `mode`:
      serial      – one file at a time (default for HDDs)
      threads     – ThreadPoolExecutor            (good for SSD / NFS)
      processes   – ProcessPoolExecutor           (large CPU work)
      auto        – 'threads' if the device looks like SSD/NVMe, else 'serial'
    """
    # 3.1 CSV
    rows, wanted_per_file = _collect_requests(filtered_csv_path,
                                              event_col, streamfile_col)
    if not rows:
        print("CSV is empty:", filtered_csv_path)
        return

    # 3.2 Decide concurrency strategy
    if mode == "auto":
        mode = "threads" if os.statvfs(filtered_csv_path).f_bsize >= 4096 else "serial"

    files = list(wanted_per_file.items())

    cache : Dict[str,Tuple[str,Dict[str,str]]] = {}  # stream→(header,chunks)
    print(f"[csv_to_stream]  Parsing {len(files)} stream files  (mode={mode})")

    # Progress bar for scanning files
    if mode == "serial":
        iterator = files
    else:
        iterator = files

    if mode == "serial":
        for fpath, evs in tqdm(iterator, desc="Scanning files", unit="file"):
            cache[fpath] = _scan_one_file((fpath, evs))[1:]
    else:
        Executor = ThreadPoolExecutor if mode == "threads" else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as ex:
            futs = {ex.submit(_scan_one_file, (f, evs)): f for f, evs in files}
            for fut in tqdm(as_completed(futs), desc="Scanning files", total=len(futs), unit="file"):
                fpath, hdr, chunks = fut.result()
                cache[fpath] = (hdr, chunks)

    # 3.3 Write combined stream in CSV order
    header_written = False
    END_txt = END.decode()

    with open(output_stream_path, "w") as out:
        for fpath, evt in tqdm(rows, desc="Writing chunks", unit="chunk"):
            hdr, chunks = cache[fpath]

            if not header_written:
                out.write(hdr)
                header_written = True

            chunk = chunks.get(evt)
            if chunk is None:
                print(f"[csv_to_stream]  ! missing {evt} in {os.path.basename(fpath)}")
                continue

            out.write(chunk)
            if not chunk.rstrip().endswith(END_txt):
                out.write(END_txt + "\n")

    print("[csv_to_stream]  Done →", output_stream_path)

# ────────────────────────────────────────────────────────────────────────────────
# 4.  CLI for ad-hoc use
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, multiprocessing as mp

    ap = argparse.ArgumentParser(description="Combine chunks from many .stream files")
    ap.add_argument("csv",   help="CSV with stream_file,event_number columns")
    ap.add_argument("out",   help="output .stream path")
    ap.add_argument("--mode", choices=["auto","serial","threads","processes"],
                    default="auto")
    ap.add_argument("-j","--jobs", type=int, default=None,
                    help="max workers (threads or processes)")
    ns = ap.parse_args()

    if ns.mode == "processes" and os.name == "nt":
        mp.freeze_support()          # Windows fork quirk
    write_stream_from_filtered_csv(
        ns.csv, ns.out,
        mode=ns.mode,
        max_workers=ns.jobs,
    )

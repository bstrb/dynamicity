#!/usr/bin/env python3
import argparse, os, sys, subprocess, re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def run_one_event(ev_dir: str) -> tuple[str, int]:
    """
    Execute the single event shell script in ev_dir with cwd=ev_dir.
    Returns (ev_dir, rc).
    """
    # pick exactly one sh_XXXXXX.sh
    sh_candidates = sorted([p for p in os.listdir(ev_dir) if p.startswith("sh_") and p.endswith(".sh")])
    if len(sh_candidates) != 1:
        # 2 = bad/missing script
        return ev_dir, 2
    sh_file = sh_candidates[0]

    # open logs inside the event dir
    out_path = os.path.join(ev_dir, "idx.stdout")
    err_path = os.path.join(ev_dir, "idx.stderr")
    with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
        # run with cwd=ev_dir so relative paths in the script work
        rc = subprocess.call(["bash", sh_file], cwd=ev_dir, stdout=out, stderr=err)
    return ev_dir, rc
import re

def concat_streams(run_dir: str, run_str: str, ev_dirs: list[str]) -> int:
    """
    Concatenate event stream files into run_<run>/stream_<run>.stream.
    Keeps header only from the first file; removes headers from subsequent parts.
    """
    items = []
    for d in ev_dirs:
        parts = [p for p in os.listdir(d) if p.startswith("stream_") and p.endswith(".stream")]
        if len(parts) != 1:
            continue
        fname = parts[0]
        m = re.search(r"stream_(\d{6})\.stream$", fname)
        if not m:
            continue
        items.append((int(m.group(1)), os.path.join(d, fname)))

    if not items:
        print(f"[ERR] no event stream parts found under {run_dir}", file=sys.stderr)
        return 2

    items.sort(key=lambda t: t[0])  # deterministic order
    out_path = os.path.join(run_dir, f"stream_{run_str}.stream")

    header_written = False
    header_pattern = re.compile(r"^CrystFEL stream version", re.IGNORECASE)

    with open(out_path, "wb") as wf:
        for i, (_, path) in enumerate(items):
            with open(path, "rb") as rf:
                data = rf.read()
                if not data:
                    continue
                if i > 0:
                    # skip header lines for subsequent parts
                    # find first occurrence of "Begin chunk"
                    idx = data.find(b"Begin chunk")
                    if idx > 0:
                        data = data[idx:]
                wf.write(data)
    print(f"[run] wrote {out_path} ({len(items)} parts, single header)")
    return 0

def main():
    ap = argparse.ArgumentParser(
        description="Run each event's sh_XXXXXX.sh in parallel (cwd=event dir), then concatenate stream parts."
    )
    ap.add_argument("--run-root", required=True, help="Timestamped runs folder (contains run_XXX/)")
    ap.add_argument("--run", required=True, help="Run number (0, 7, 012 → zero-padded internally)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="Max parallel workers")
    args = ap.parse_args()

    run_str = f"{int(args.run):03d}"
    run_dir = os.path.join(os.path.abspath(os.path.expanduser(args.run_root)), f"run_{run_str}")

    print(f"Run root : {os.path.abspath(os.path.expanduser(args.run_root))}")
    print(f"Run      : {run_str}")
    print(f"Run dir  : {run_dir}")

    if not os.path.isdir(run_dir):
        print(f"[ERR] missing run dir: {run_dir}", file=sys.stderr)
        return 2

    # discover event dirs
    ev_dirs = [os.path.join(run_dir, d) for d in sorted(os.listdir(run_dir))
               if d.startswith("event_") and os.path.isdir(os.path.join(run_dir, d))]
    if not ev_dirs:
        print(f"[ERR] No event_* directories in {run_dir}", file=sys.stderr)
        return 2

    # run in parallel
    max_workers = max(1, int(args.jobs or 1))
    print(f"[mp] running {len(ev_dirs)} event jobs with {max_workers} workers...")
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one_event, d) for d in ev_dirs]
        for fut in as_completed(futs):
            ev_dir, rc = fut.result()
            name = os.path.basename(ev_dir)
            if rc != 0:
                print(f"[err] {name} failed (rc={rc}) — see {ev_dir}/idx.stderr", file=sys.stderr)
                failures.append(ev_dir)
            else:
                print(f"[ok]  {name}")

    if failures:
        print(f"[fail] {len(failures)} event(s) failed; not concatenating.", file=sys.stderr)
        return 1

    # concatenate per-event streams
    return concat_streams(run_dir, run_str, ev_dirs)

if __name__ == "__main__":
    sys.exit(main())

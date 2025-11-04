#!/usr/bin/env python3
import argparse, os, sys, subprocess, re
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, "extract_mille_shifts.py")

def run_one_event(ev_dir: str) -> tuple[str, int, int]:
    """
    Run per-event indexing and, if present, mille extraction in the same worker.
    Returns (ev_dir, rc_sh, rc_mille) where rc_mille==0 if ran OK,
    3 if mille-data.bin missing, and non-zero if extractor failed.
    """
    # 1) find sh_XXXXXX.sh
    sh_candidates = sorted([p for p in os.listdir(ev_dir)
                            if p.startswith("sh_") and p.endswith(".sh")])
    if len(sh_candidates) != 1:
        return ev_dir, 2, 3  # bad/missing script; mark mille as "missing"

    sh_file = sh_candidates[0]
    out_path = os.path.join(ev_dir, "idx.stdout")
    err_path = os.path.join(ev_dir, "idx.stderr")

    # run indexing with cwd=ev_dir
    with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
        rc_sh = subprocess.call(["bash", sh_file], cwd=ev_dir, stdout=out, stderr=err)

    # if indexing failed, do NOT attempt mille; return early
    if rc_sh != 0:
        return ev_dir, rc_sh, 3

    # 2) immediately run mille extractor (same worker, same cwd) if mille-data.bin exists
    bin_path = os.path.join(ev_dir, "mille-data.bin")
    if not os.path.isfile(bin_path):
        return ev_dir, rc_sh, 3  # no mille for this event; "missing"

    # mille_out = os.path.join(ev_dir, "per_frame_dx_dy.csv")
    with open(os.path.join(ev_dir, "mille.stdout"), "w", encoding="utf-8") as mout, \
         open(os.path.join(ev_dir, "mille.stderr"), "w", encoding="utf-8") as merr:
        # no flags required; adjust if you later want --globals/--scale
        rc_mille = subprocess.call(
            ["python3", EXTRACT_SCRIPT, "mille-data.bin"],
            cwd=ev_dir, stdout=mout, stderr=merr
        )

    return ev_dir, rc_sh, rc_mille

def concat_streams(run_dir: str, run_str: str, ev_dirs: list[str]) -> int:
    import re
    # collect stream parts (one per event)
    items = []
    for d in ev_dirs:
        parts = [p for p in os.listdir(d) if p.startswith("stream_") and p.endswith(".stream")]
        if len(parts) != 1:
            continue
        m = re.search(r"stream_(\d{6})\.stream$", parts[0])
        if not m:
            continue
        items.append((int(m.group(1)), os.path.join(d, parts[0])))

    if not items:
        print(f"[ERR] no event stream parts found under {run_dir}", file=sys.stderr)
        return 2

    items.sort(key=lambda t: t[0])
    out_path = os.path.join(run_dir, f"stream_{run_str}.stream")

    # keep header only from the first part
    with open(out_path, "wb") as wf:
        for i, (_, p) in enumerate(items):
            with open(p, "rb") as rf:
                data = rf.read()
            if not data:
                continue
            if i > 0:
                idx = data.find(b"Begin chunk")
                if idx > 0:
                    data = data[idx:]
            wf.write(data)

    print(f"[run] wrote {out_path} ({len(items)} parts, single header)")
    return 0

def main():
    ap = argparse.ArgumentParser(
        description="Run each event's sh_XXXXXX.sh in parallel (cwd=event dir), inline mille extraction, then concatenate."
    )
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    run_str = f"{int(args.run):03d}"
    run_dir = os.path.join(os.path.abspath(os.path.expanduser(args.run_root)), f"run_{run_str}")
    print(f"Run root : {os.path.abspath(os.path.expanduser(args.run_root))}")
    print(f"Run      : {run_str}")
    print(f"Run dir  : {run_dir}")

    if not os.path.isdir(run_dir):
        print(f"[ERR] missing run dir: {run_dir}", file=sys.stderr)
        return 2

    ev_dirs = [os.path.join(run_dir, d) for d in sorted(os.listdir(run_dir))
               if d.startswith("event_") and os.path.isdir(os.path.join(run_dir, d))]
    if not ev_dirs:
        print(f"[ERR] No event_* directories in {run_dir}", file=sys.stderr)
        return 2

    # parallel execution: indexing + inline mille per event
    workers = max(1, int(args.jobs or 1))
    print(f"[mp] running {len(ev_dirs)} event jobs with {workers} workers…")
    failures, mille_warn = [], []
    # from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one_event, d) for d in ev_dirs]
        for fut in as_completed(futs):
            ev_dir, rc_sh, rc_mille = fut.result()
            name = os.path.basename(ev_dir)
            if rc_sh != 0:
                print(f"[err] {name} (index rc={rc_sh}) — see {ev_dir}/idx.stderr", file=sys.stderr)
                failures.append(ev_dir)
            else:
                if rc_mille == 0:
                    print(f"[ok]  {name} (index + mille)")
                elif rc_mille == 3:
                    print(f"[ok]  {name} (index only; no mille-data.bin)")
                else:
                    print(f"[warn] {name} (mille rc={rc_mille}) — see {ev_dir}/mille.stderr", file=sys.stderr)
                    # do not fail the whole run if mille fails; still allow concatenation
                    mille_warn.append(ev_dir)

    if failures:
        print(f"[fail] {len(failures)} event(s) failed indexing; not concatenating.", file=sys.stderr)
        return 1

    # concatenate streams from successfully indexed events (all events attempted already)
    return concat_streams(run_dir, run_str, ev_dirs)

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse, os, sys, subprocess, re
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, "extract_mille_shifts.py")

def _check_index_success(ev_dir: str) -> tuple[bool, str]:
    """
    Determine if indexing truly succeeded for this event.
    Strategy:
      A) Try idx.stderr for '... indexable'
      B) Fallback: parse per-event stream for num_reflections > 0
    """
    err_path = os.path.join(ev_dir, "idx.stderr")

    # A) try reading idx.stderr
    try:
        with open(err_path, "r", encoding="utf-8", errors="replace") as f:
            s = f.read()
        m_all = re.findall(r'\b(\d+)\s+indexable\b', s, flags=re.I)
        if m_all:
            idxable = int(m_all[-1])
            if idxable > 0:
                return True, f"idx.stderr: indexable={idxable}"
    except FileNotFoundError:
        pass

    # B) fallback: parse stream
    try:
        parts = [p for p in os.listdir(ev_dir) if p.startswith("stream_") and p.endswith(".stream")]
        if len(parts) == 1:
            sp = os.path.join(ev_dir, parts[0])
            with open(sp, "r", encoding="utf-8", errors="replace") as f:
                t = f.read()
            m2 = re.search(r"\bnum_reflections\s*=\s*(\d+)", t, flags=re.I)
            if m2:
                nref = int(m2.group(1))
                if nref > 0:
                    return True, f"stream: num_reflections={nref}"
                else:
                    return False, f"stream: num_reflections={nref}"
    except FileNotFoundError:
        pass

    return False, "no conclusive success marker"


def run_one_event(ev_dir: str) -> tuple[str, int, int]:
    """
    Run per-event indexing, then optional mille extraction.
    Return (event_dir, rc_index, rc_mille)
    """
    sh_candidates = sorted([p for p in os.listdir(ev_dir)
                            if p.startswith("sh_") and p.endswith(".sh")])
    if len(sh_candidates) != 1:
        return ev_dir, 2, 3

    sh_file = sh_candidates[0]
    out_path = os.path.join(ev_dir, "idx.stdout")
    err_path = os.path.join(ev_dir, "idx.stderr")

    with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
        rc_sh_raw = subprocess.call(["bash", sh_file], cwd=ev_dir, stdout=out, stderr=err)

    ok, why = _check_index_success(ev_dir)
    if not ok:
        return ev_dir, (rc_sh_raw if rc_sh_raw != 0 else 10), 3

    rc_sh = 0  # indexing success

    # Mille extraction
    bin_path = os.path.join(ev_dir, "mille-data.bin")
    if not os.path.isfile(bin_path):
        return ev_dir, rc_sh, 3

    with open(os.path.join(ev_dir, "mille.stdout"), "w", encoding="utf-8") as mout, \
         open(os.path.join(ev_dir, "mille.stderr"), "w", encoding="utf-8") as merr:
        rc_mille = subprocess.call(
            ["python3", EXTRACT_SCRIPT, "mille-data.bin"],
            cwd=ev_dir, stdout=mout, stderr=merr
        )

    return ev_dir, rc_sh, rc_mille


def _expect_actual_report(run_dir: str) -> tuple[int, int]:
    """
    Count expected vs. actual stream parts.
    """
    import glob

    ev_dirs = sorted(
        d for d in glob.glob(os.path.join(run_dir, "event_*"))
        if os.path.isdir(d)
    )

    expected = []
    actual = []

    for d in ev_dirs:
        if glob.glob(os.path.join(d, "*.lst")):
            expected.append(os.path.basename(d))
        if glob.glob(os.path.join(d, "*.stream")):
            actual.append(os.path.basename(d))

    missing = sorted(set(expected) - set(actual))
    if missing:
        miss_report = os.path.join(run_dir, "missing_event_parts.txt")
        with open(miss_report, "w", encoding="utf-8") as f:
            f.write("# Event dirs that did not produce a .stream part\n")
            for m in missing:
                f.write(m + "\n")
        print(f"[warn] {len(missing)} event(s) missing a part (see {miss_report})")
    else:
        print("[ok] All expected events produced a .stream part")

    return len(expected), len(actual)


def concat_streams(run_dir: str, run_str: str, ev_dirs: list[str]) -> int:
    import shutil

    RE_BEGIN_CHUNK = re.compile(r"^-{3,}\s*Begin chunk\s*-{3,}\s*$")
    out_path = os.path.join(run_dir, f"stream_{run_str}.stream")

    items = []
    for d in sorted(ev_dirs):
        try:
            parts = [p for p in os.listdir(d) if p.startswith("stream_") and p.endswith(".stream")]
        except FileNotFoundError:
            continue
        if parts:
            parts.sort()
            items.append(os.path.join(d, parts[0]))

    if not items:
        print(f"[fail] no per-event streams found in {run_dir}", file=sys.stderr)
        return 1

    wrote_chunks = 0
    header_written = False
    bufsize = 1024 * 256

    with open(out_path, "w", encoding="utf-8") as out:
        for path in items:
            try:
                rf = open(path, "r", encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue

            saw_begin = False
            try:
                if not header_written:
                    header_lines = []
                    while True:
                        line = rf.readline()
                        if not line:
                            break
                        if RE_BEGIN_CHUNK.match(line):
                            saw_begin = True
                            if header_lines:
                                out.write("".join(header_lines).rstrip() + "\n\n")
                            header_written = True
                            out.write(line)
                            break
                        else:
                            header_lines.append(line)

                    if not saw_begin:
                        rf.close()
                        continue
                else:
                    while True:
                        line = rf.readline()
                        if not line:
                            break
                        if RE_BEGIN_CHUNK.match(line):
                            saw_begin = True
                            if wrote_chunks > 0:
                                out.write("\n")
                            out.write(line)
                            break
                    if not saw_begin:
                        rf.close()
                        continue

                shutil.copyfileobj(rf, out, length=bufsize)
                wrote_chunks += 1

            finally:
                rf.close()

    if wrote_chunks == 0:
        try: os.remove(out_path)
        except FileNotFoundError: pass
        print("[fail] no chunks to concatenate.", file=sys.stderr)
        return 1

    # print(f"[concat] wrote {out_path} from {wrote_chunks} parts")
    return 0

def main():
    ap = argparse.ArgumentParser(
        description="Run each event's sh_XXXXXX.sh in parallel; inline mille; concatenate."
    )
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    run_str = f"{int(args.run):03d}"
    run_dir = os.path.join(os.path.abspath(os.path.expanduser(args.run_root)), f"run_{run_str}")

    if not os.path.isdir(run_dir):
        print(f"[ERR] missing run dir: {run_dir}", file=sys.stderr)
        return 2

    import glob

    # Collect *all* physical event_* directories
    ev_dirs = sorted(
        d for d in glob.glob(os.path.join(run_dir, "event_*"))
        if os.path.isdir(d)
    )

    if not ev_dirs:
        print(f"[ERR] No event_* dirs in {run_dir}", file=sys.stderr)
        return 2
    
    if args.jobs >= int(os.cpu_count()):
        workers = os.cpu_count()
    else:
        workers = max(1, int(args.jobs or 1))


    print(f"[mp] running {len(ev_dirs)} event jobs with {workers} workers…", flush=True)

    failures, mille_warn = [], []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one_event, d) for d in ev_dirs]

        for fut in as_completed(futs):
            ev_dir, rc_sh, rc_mille = fut.result()

            if rc_sh != 0:
                failures.append(ev_dir)
            else:
                if rc_mille not in (0, 3):
                    mille_warn.append(ev_dir)

            # notify orchestrator that one event is done
            print("__EVENT_DONE__", flush=True)

    if failures:
        print(f"[fail] {len(failures)} event(s) failed indexing.", file=sys.stderr)

    n_exp, n_act = _expect_actual_report(run_dir)

    return concat_streams(run_dir, run_str, ev_dirs)

if __name__ == "__main__":
    sys.exit(main())

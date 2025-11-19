#!/usr/bin/env python3
import argparse, os, shlex, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Create per-event sh_event.sh for indexamajig.")
    ap.add_argument("--run-root", required=True, help="Timestamped runs folder (contains run_XXX/)")
    ap.add_argument("--geom", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--run", required=True, help="e.g. 0, 7, 012 -> zero-padded to 3")
    ap.add_argument("--indexamajig", default="indexamajig")
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        ours, extra = argv[:sep], argv[sep+1:]
    else:
        ours, extra = argv, []
    args = ap.parse_args(ours)

    run = f"{int(args.run):03d}"
    run_dir = os.path.join(os.path.abspath(os.path.expanduser(args.run_root)), f"run_{run}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    geom = os.path.abspath(args.geom)
    cell = os.path.abspath(args.cell)
    idx  = args.indexamajig

    n_events = 0
    for ev_name in sorted(os.listdir(run_dir)):
        ev_dir = os.path.join(run_dir, ev_name)
        if not (os.path.isdir(ev_dir) and ev_name.startswith("event_")):
            continue
        lsts = [fn for fn in os.listdir(ev_dir) if fn.endswith(".lst")]
        if len(lsts) != 1:
            continue

        lst_path = os.path.join(ev_dir, lsts[0])
        
        # Extract the numeric part from event_XXXXXX → 000000
        suffix = ev_name.split("_")[-1]
        sh_ev = os.path.join(ev_dir, f"sh_{suffix}.sh")
        out_part = os.path.join(ev_dir, f"stream_{suffix}.stream")

        cmd = [
            idx, "-g", geom, "-i", lst_path, "-o", out_part, "-p", cell, *extra
        ]

        with open(sh_ev, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
            f.write("# Auto-generated per-image job\n")
            f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
        os.chmod(sh_ev, 0o755)
        n_events += 1

    # print(f"[ok] wrote {n_events} event scripts under {run_dir}")

if __name__ == "__main__":
    main()

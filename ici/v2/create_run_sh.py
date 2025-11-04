#!/usr/bin/env python3
import argparse, os, shlex, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="Create sh_<run>.sh for indexamajig with run-scoped paths."
    )
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--run", required=True, help="e.g. 0, 7, 012 -> zero-padded")
    ap.add_argument("--indexamajig", default="indexamajig")
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        ours, extra = argv[:sep], argv[sep+1:]  # flags after '--' go to indexamajig
    else:
        ours, extra = argv, []
    args = ap.parse_args(ours)

    run = f"{int(args.run):03d}"
    run_dir = os.path.join(os.path.abspath(os.path.expanduser(args.run_root)), f"run_{run}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    lst_path = os.path.join(run_dir, f"lst_{run}.lst")
    sh_path  = os.path.join(run_dir, f"sh_{run}.sh")
    stream   = os.path.join(run_dir, f"stream_{run}.stream")

    cmd = [
        args.indexamajig,
        "-g", os.path.abspath(args.geom),
        "-i", lst_path,
        "-o", stream,
        "-p", os.path.abspath(args.cell),
        *extra,  # your DEFAULT_FLAGS etc. if you forward them after `--`
    ]

    with open(sh_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        f.write("# Exact command that will be run:\n")
        f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
    os.chmod(sh_path, 0o755)
    print(f"[ok] wrote {sh_path}")

if __name__ == "__main__":
    main()

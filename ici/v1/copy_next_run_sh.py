#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copy_next_run_sh.py

New behavior (no next run creation):
1) Auto-detect the latest run folder under <run_root>/runs/run_### (or use --run to target one).
2) If the target run already has a .sh, do nothing and exit.
3) If not, copy a .sh from the nearest earlier run that has one, and rewrite only
   the -i / -o paths inside to point to the target run.
4) Preserve permissions.

Usage examples:
  python3 copy_next_run_sh.py
  python3 copy_next_run_sh.py --run-root "/path/to/sim_004"
  python3 copy_next_run_sh.py --run-root "/path/to/sim_004" --run 5
"""
import os
import re
import sys
import argparse
from typing import Optional, List

DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
# DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"


def _normalize_run(run: str) -> str:
    """Zero-pad numeric runs to width 3 (e.g. '0' -> '000'); pass through non-numeric."""
    try:
        return f"{int(run):03d}"
    except (TypeError, ValueError):
        return str(run)


def _runs_dir(run_root: str) -> str:
    return os.path.join(run_root, "runs")


def _run_dir(run_root: str, n: int) -> str:
    return os.path.join(_runs_dir(run_root), f"run_{n:03d}")


def _sh_path_for_run(run_root: str, n: int) -> str:
    return os.path.join(_run_dir(run_root, n), f"sh_{n:03d}.sh")


def _list_run_numbers(run_root: str) -> List[int]:
    """Return sorted list of numeric run indices present under <run_root>/runs."""
    base = _runs_dir(run_root)
    if not os.path.isdir(base):
        return []
    out = []
    for name in os.listdir(base):
        m = re.fullmatch(r"run_(\d{3})", name)
        if m:
            try:
                out.append(int(m.group(1)))
            except ValueError:
                pass
    return sorted(out)


def _find_first_sh_in_dir(d: str) -> Optional[str]:
    """Return sh_###.sh if present, else the first *.sh in the directory, else None."""
    if not os.path.isdir(d):
        return None
    m = re.search(r"run_(\d{3})$", d)
    preferred = None
    if m:
        n = int(m.group(1))
        preferred = os.path.join(d, f"sh_{n:03d}.sh")
        if os.path.isfile(preferred):
            return preferred
    for f in os.listdir(d):
        if f.endswith(".sh"):
            return os.path.join(d, f)
    return None


def _rewrite_io_paths(text: str, old_n: int, new_n: int, run_root: str) -> str:
    """
    Replace only the -i / -o path payloads that reference run_old with run_new.

    Matches (whitespace)(-i|-o)(whitespace)<any_path>/runs/run_old/(lst/stream)_old.(lst/stream)
    and rewrites the path to <run_root>/runs/run_new/<lst/stream>_new.(lst/stream).
    """
    sep = r"[\\/]"
    pattern_i = re.compile(
        rf"(\s-i\s+).*(?:{sep}runs{sep}run_{old_n:03d}{sep})lst_{old_n:03d}\.lst\b"
    )
    pattern_o = re.compile(
        rf"(\s-o\s+).*(?:{sep}runs{sep}run_{old_n:03d}{sep})stream_{old_n:03d}\.stream\b"
    )

    new_text = re.sub(
        pattern_i,
        rf"\1{run_root}/runs/run_{new_n:03d}/lst_{new_n:03d}.lst",
        text,
    )
    new_text = re.sub(
        pattern_o,
        rf"\1{run_root}/runs/run_{new_n:03d}/stream_{new_n:03d}.stream",
        new_text,
    )
    return new_text


def _copy_permissions(src: str, dst: str) -> None:
    try:
        st = os.stat(src)
        os.chmod(dst, st.st_mode)
    except Exception as e:
        print(f"WARNING: could not copy permissions {src} -> {dst}: {e}", file=sys.stderr)


def autodetect_latest_run(run_root: str) -> int:
    runs = _list_run_numbers(run_root)
    if not runs:
        raise SystemExit(f"ERROR: No run_### folders found under: {_runs_dir(run_root)}")
    return runs[-1]


def ensure_sh_in_run(run_root: str, n: int) -> str:
    """
    Ensure run_n has a .sh. If present, return it unchanged.
    If missing:
      - find the nearest earlier run with a .sh,
      - copy it into run_n as sh_nnn.sh,
      - rewrite -i/-o from donor->n,
      - copy permissions.
    Returns the sh path inside run_n.
    """
    target_dir = _run_dir(run_root, n)
    os.makedirs(target_dir, exist_ok=True)

    existing = _find_first_sh_in_dir(target_dir)
    if existing:
        print(f"[ok] Found .sh in run_{n:03d}: {existing}")
        return existing

    # Need to backfill from an earlier run with a .sh
    run_numbers = _list_run_numbers(run_root)
    earlier = [m for m in run_numbers if m < n]
    for donor in reversed(earlier):
        donor_dir = _run_dir(run_root, donor)
        donor_sh = _find_first_sh_in_dir(donor_dir)
        if donor_sh:
            with open(donor_sh, "r", encoding="utf-8") as f:
                text = f.read()
            text_new = _rewrite_io_paths(text, donor, n, run_root)
            target_sh = _sh_path_for_run(run_root, n)
            with open(target_sh, "w", encoding="utf-8") as f:
                f.write(text_new)
            _copy_permissions(donor_sh, target_sh)
            print(f"[backfill] Created {target_sh} from {donor_sh}")
            return target_sh

    raise SystemExit(
        f"ERROR: run_{n:03d} has no .sh and no earlier runs with a .sh were found to copy."
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Ensure the latest (or specified) run has a .sh. "
            "If missing, copy from the nearest earlier run and rewrite -i/-o to the target run. "
            "Does NOT create a next run."
        )
    )
    ap.add_argument("--run-root", help='Root folder that contains runs/run_<run>')
    ap.add_argument(
        "--run",
        help='Target run identifier (e.g., "0", "3", "12", "003"); '
             'when omitted, the script auto-detects the latest run.',
    )
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = os.path.abspath(os.path.expanduser(args.run_root if args.run_root else DEFAULT_ROOT))

    # Determine the target run we ensure:
    if args.run is not None:
        run_str = _normalize_run(args.run)
        try:
            target_n = int(run_str)
        except ValueError:
            print(f"ERROR: --run must be numeric (got '{run_str}').", file=sys.stderr)
            sys.exit(2)
        print(f"[start] Using provided --run: {target_n:03d}")
    else:
        target_n = autodetect_latest_run(run_root)
        print(f"[start] Auto-detected latest run: {target_n:03d}")

    print(f"Run root : {run_root}")
    print(f"Target   : {_run_dir(run_root, target_n)}")

    target_sh = ensure_sh_in_run(run_root, target_n)
    print(f"[done] Ensured .sh present: {target_sh}")


if __name__ == "__main__":
    main()

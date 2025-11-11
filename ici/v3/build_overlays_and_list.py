#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_overlays_and_list.py

From image_run_log.csv (LATEST run only), create per-source overlay HDF5 files
(named with the *next* run number) and a lst_{next}.lst that lists lines like:

  /abs/path/to/<src_basename>_overlay_{next:03d}.h5 //event_number

Key behaviors:
- Groups rows by section header "#/<abs/source.h5> event <id>".
- Uses ONLY rows whose run_n == latest_run and whose next_dx,next_dy are numeric.
- Skips rows where next_* are "done" or blank.
- Writes one overlay H5 per source; packs that source's (event_id -> next_dx,next_dy) into it.
- Persists overlay→original mapping (JSON + TSV) in the next run folder.
- Optionally tags overlay H5 with attribute 'overlay_original_path' (best-effort).

Requires:
- overlay_elink.create_overlay(h5_path, out_path)  -> create an overlay/ELINK HDF5
- overlay_elink.write_shifts_mm(out_path, dict[event_id] = (dx_mm, dy_mm))

This script is idempotent for a given next run number.
"""
from __future__ import annotations
import argparse, os, sys, json, math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

try:
    # best-effort, only used to tag an attribute on the overlay
    import h5py  # type: ignore
except Exception:
    h5py = None  # optional

try:
    from overlay_elink import create_overlay, write_shifts_mm
except Exception as e:
    print(f"[overlay] FATAL: could not import overlay_elink: {e}", file=sys.stderr)
    sys.exit(2)

# ---------- helpers ----------

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _real_abs(p: str) -> str:
    # normalize to a real absolute path (dereference symlinks)
    return os.path.realpath(os.path.abspath(os.path.expanduser(p)))

def _parse_section_header(line: str) -> Optional[Tuple[str, int]]:
    """
    Header lines look like:
      #/abs/path/to/source.h5 event 123
    Returns (abs_source_path, event_id) or None if not a header.
    """
    if not line.startswith("#/") or " event " not in line:
        return None
    s = line[1:].strip()  # drop leading '#'
    # now "/abs/.../file.h5 event N"
    try:
        left, right = s.rsplit(" event ", 1)
        ev = int(right.strip())
        src = _real_abs(left.strip())
        return (src, ev)
    except Exception:
        return None

def _detect_latest_run(log_path: Path) -> int:
    latest = -1
    with log_path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("run_n"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if parts and parts[0].isdigit():
                latest = max(latest, int(parts[0]))
    return latest

def _collect_latest_numeric_proposals(log_path: Path, latest_run: int) -> Dict[Tuple[str,int], Tuple[float,float]]:
    """
    Walk the CSV, track the current section (src,event).
    Keep ONLY rows where run_n == latest_run and next_dx,next_dy are numeric.
    For a given (src,event) keep the LAST occurrence within the latest run.
    Returns map[(src_abs,event_id)] = (next_dx_mm, next_dy_mm).
    """
    proposals: Dict[Tuple[str,int], Tuple[float,float]] = {}
    current_key: Optional[Tuple[str,int]] = None

    with log_path.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("#/"):
                hdr = _parse_section_header(raw.rstrip("\n"))
                current_key = hdr
                continue
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("run_n"):
                continue
            if current_key is None:
                # ignore stray rows outside any section
                continue

            parts = [p.strip() for p in s.split(",")]
            # ensure at least 7 columns: run, dx, dy, idx, wrmsd, next_dx, next_dy
            if len(parts) < 7:
                parts += [""] * (7 - len(parts))

            run_s, next_dx_s, next_dy_s = parts[0], parts[5], parts[6]
            if not run_s.isdigit():
                continue
            rn = int(run_s)
            if rn != latest_run:
                # only the latest run proposals are relevant for the NEXT iteration
                continue

            # must be numeric proposals (skip "done", "", "nan", etc.)
            if not (_is_float(next_dx_s) and _is_float(next_dy_s)):
                continue

            ndx, ndy = float(next_dx_s), float(next_dy_s)
            proposals[current_key] = (ndx, ndy)

    return proposals

def _group_by_source(proposals: Dict[Tuple[str,int], Tuple[float,float]]) -> Dict[str, Dict[int, Tuple[float,float]]]:
    """
    Convert {(src,event)->(dx,dy)} into {src->{event:(dx,dy)}} and sort by event id.
    """
    by_src: Dict[str, Dict[int, Tuple[float,float]]] = {}
    for (src, ev), shift in proposals.items():
        by_src.setdefault(src, {})[ev] = shift
    # force deterministic event ordering
    for src in list(by_src.keys()):
        ev_map = by_src[src]
        by_src[src] = {ev: ev_map[ev] for ev in sorted(ev_map.keys())}
    return by_src

def _ensure_next_run_dir(run_root: Path) -> Tuple[int, Path]:
    """
    Find existing run directories (run_000, run_001, ...). Next = max+1.
    Return (next_num, next_dir_path).
    """
    nums: List[int] = []
    for name in os.listdir(run_root):
        if name.startswith("run_") and len(name) == 7 and name[4:].isdigit():
            nums.append(int(name[4:]))
    next_num = (max(nums) + 1) if nums else 0
    nxt = run_root / f"run_{next_num:03d}"
    nxt.mkdir(parents=True, exist_ok=True)
    return next_num, nxt

def _overlay_name_for(src_path: Path, next_num: int) -> Path:
    # /a/b/c/file.h5 -> file_overlay_XXX.h5
    stem = src_path.stem  # "file"
    return src_path.parent / f"{stem}_overlay_{next_num:03d}.h5"

def _write_overlay_map(next_run_dir: Path, overlay_to_original: Dict[str, str]) -> Path:
    # JSON
    map_json = next_run_dir / "overlay_to_original.json"
    map_json.write_text(json.dumps(overlay_to_original, indent=2), encoding="utf-8")
    # TSV (handy to eyeball)
    map_tsv = next_run_dir / "overlay_to_original.tsv"
    with map_tsv.open("w", encoding="utf-8") as f:
        f.write("overlay_path\toriginal_path\n")
        for ov, org in overlay_to_original.items():
            f.write(f"{ov}\t{org}\n")
    return map_json

def _tag_overlay_with_original(overlay_path: Path, original_abs: str) -> None:
    if h5py is None:
        return
    try:
        with h5py.File(str(overlay_path), "a") as h:
            h.attrs["overlay_original_path"] = original_abs
    except Exception as e:
        print(f"[overlay] warn: could not tag {overlay_path.name}: {e}")

def _fmt_list_line(overlay_path: Path, ev_id: int) -> str:
    # absolute path + " //<event>"
    return f"{_real_abs(str(overlay_path))} //{ev_id}\n"

# ---------- main work ----------

def build_overlays_and_list(run_root: Path) -> int:
    rd = run_root  # the 'runs_YYYYmmdd_HHMMSS' folder
    log_csv = rd / "image_run_log.csv"
    if not log_csv.exists():
        print(f"[overlay] ERROR: missing log: {log_csv}", file=sys.stderr)
        return 2

    latest_run = _detect_latest_run(log_csv)
    if latest_run < 0:
        print("[overlay] ERROR: could not determine latest run in image_run_log.csv", file=sys.stderr)
        return 2
    print(f"[overlay] latest_run detected: {latest_run}")

    proposals = _collect_latest_numeric_proposals(log_csv, latest_run)
    print(f"[overlay] collected {len(proposals)} (src,event) proposals with numeric next_* in latest run")

    # group into per-source dicts
    by_src = _group_by_source(proposals)
    print(f"[overlay] sources with proposals in latest run: {len(by_src)}")

    # next run folder/number
    next_num, next_run_dir = _ensure_next_run_dir(rd)
    lst_path = next_run_dir / f"lst_{next_num:03d}.lst"

    overlay_to_original: Dict[str, str] = {}
    n_lines = 0

    # build / update overlays and the list file
    with lst_path.open("w", encoding="utf-8") as out_lst:
        for src_abs, ev_map in by_src.items():
            src_p = Path(src_abs)
            # choose overlay name in the *same* folder as the source
            overlay_path = _overlay_name_for(src_p, next_num)

            # create / refresh overlay file
            create_overlay(str(src_p), str(overlay_path))
            _tag_overlay_with_original(overlay_path, src_abs)

            # write shifts for THIS source only (event_id -> (dx,dy))
            write_shifts_mm(str(overlay_path), ev_map)

            # remember mapping
            overlay_to_original[_real_abs(str(overlay_path))] = src_abs

            # emit one list line per event in this source
            for ev_id, (dx, dy) in ev_map.items():
                out_lst.write(_fmt_list_line(overlay_path, ev_id))
                n_lines += 1

    _write_overlay_map(next_run_dir, overlay_to_original)

    print(f"[overlay] wrote {n_lines} list line(s) into {lst_path.name}")
    print(f"[overlay] overlays created/updated: {len(overlay_to_original)}")
    if n_lines == 0:
        print("[overlay] NOTE: 0 eligible proposals for latest run "
              "(either all were 'done' or next_* were blank).")

    return 0

# ---------- CLI ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create per-source overlays and lst for the NEXT run from latest proposals.")
    ap.add_argument("--run-root", required=True, help="The 'runs_YYYYmmdd_HHMMSS' experiment folder that contains image_run_log.csv and run_XXX/...")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = Path(os.path.abspath(os.path.expanduser(args.run_root)))
    return build_overlays_and_list(run_root)

if __name__ == "__main__":
    sys.exit(main())

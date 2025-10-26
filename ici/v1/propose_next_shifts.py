#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py

Read runs/image_run_log.csv, reconstruct per-(real_h5_path,event) search state
from *all prior rows*, and fill in next_dx_mm,next_dy_mm for the *latest run*
rows only, using the same two-stage strategy:
- ring search until first indexed+wrmsd
- local Nelder–Mead thereafter

Keeps the existing CSV schema:
run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm

By default, the script rewrites the log file in place. Use --sidecar to write
a separate CSV with proposed shifts for the latest run only.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, math, os, re, sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
# DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"

# Defaults matching the original script
R_MAX_DEFAULT = 0.05
R_STEP_DEFAULT = 0.01
K_BASE_DEFAULT = 4.0
DELTA_LOCAL_DEFAULT = 0.01
LOCAL_PATIENCE_DEFAULT = 3
SEED_DEFAULT = 1337
CONVERGE_TOL_DEFAULT = 1e-4
 
# Nelder–Mead params (same as original)
NM_ALPHA = 1.0
NM_GAMMA = 2.0
NM_RHO   = 0.5
NM_SIGMA = 0.5

# ----------------- Data structures -----------------
@dataclass
class NMVertex:
    dx: float
    dy: float
    f: Optional[float] = None
    done: bool = False

@dataclass
class ImgState:
    phase: str = "ring"
    last_dx: float = 0.0
    last_dy: float = 0.0
    best_wrmsd: Optional[float] = None
    best_dx: Optional[float] = None
    best_dy: Optional[float] = None
    best_run: Optional[int] = None
    best_chunk_id: Optional[int] = None
    ring_step: int = 0
    ring_angle_idx: int = -1
    ring_angle_base: Optional[float] = None
    nm_step: float = DELTA_LOCAL_DEFAULT
    nm_vertices: List[NMVertex] = field(default_factory=list)
    nm_initialized: bool = False
    nm_last_proposal: Optional[Tuple[float,float]] = None
    local_tries_since_improve: int = 0
    done: bool = False
    give_up: bool = False

# ----------------- Utilities -----------------
def n_angles_for_radius(r: float, r_max: float, k_base: float) -> int:
    return max(1, math.ceil(k_base * (r / max(r_max, 1e-9))))

def _hash_angle(seed: int, key: Tuple[str, int]) -> float:
    """Deterministic base angle in [0, 2π) derived from seed+key."""
    h = hashlib.sha256()
    h.update(f"{seed}|{key[0]}|{key[1]}".encode("utf-8"))
    val = int.from_bytes(h.digest()[:8], "big")  # 64-bit
    frac = (val & ((1<<53)-1)) / float(1<<53)   # use 53 bits like double mantissa
    return 2.0 * math.pi * frac

def _fmt6(x: float) -> str:
    return f"{x:.6f}"

def _float_or_blank(s: str) -> Optional[float]:
    s = (s or "").strip()
    if s == "":
        return None
    try:
        v = float(s)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

# ------------- Ring & NM steppers (same behavior) -------------
def _ensure_nm_initialized(S: ImgState) -> None:
    if S.nm_initialized:
        return
    cx = S.best_dx if S.best_dx is not None else S.last_dx
    cy = S.best_dy if S.best_dy is not None else S.last_dy
    S.nm_vertices = [
        NMVertex(cx, cy, f=S.best_wrmsd, done=(S.best_wrmsd is not None)),
        NMVertex(cx + S.nm_step, cy, f=None, done=False),
        NMVertex(cy + 0*0, cy + S.nm_step, f=None, done=False),  # will set dx correctly below
    ]
    # fix second vertex: (cx, cy + nm_step)
    S.nm_vertices[2].dx = cx
    S.nm_initialized = True

def pick_ring_probe(
    S: ImgState, seed_dx: float, seed_dy: float, r_step: float, r_max: float, k_base: float
) -> Tuple[float, float, str]:
    r = (S.ring_step + 1) * r_step
    if r > r_max + 1e-12:
        S.give_up = True
        return seed_dx, seed_dy, "ring_exhausted"
    n = n_angles_for_radius(r, r_max, k_base)
    S.ring_angle_idx = (S.ring_angle_idx + 1) % n
    theta = (S.ring_angle_base or 0.0) + 2.0 * math.pi * S.ring_angle_idx / n
    ndx = seed_dx + r * math.cos(theta)
    ndy = seed_dy + r * math.sin(theta)
    if S.ring_angle_idx == n - 1:
        S.ring_step += 1
        S.ring_angle_idx = -1
    return ndx, ndy, f"ring_r={r:.5f}_n={n}"

def pick_local_nm_probe(S: ImgState) -> Tuple[float, float, str]:
    _ensure_nm_initialized(S)
    for v in S.nm_vertices:
        if not v.done:
            S.nm_last_proposal = (v.dx, v.dy)
            return v.dx, v.dy, "nm_eval_vertex"
    # all vertices evaluated: reflect worst
    verts = sorted(S.nm_vertices, key=lambda vv: (float("inf") if vv.f is None else vv.f))
    best, mid, worst = verts[0], verts[1], verts[2]
    xc = (best.dx + mid.dx) / 2.0
    yc = (best.dy + mid.dy) / 2.0
    xr = xc + NM_ALPHA * (xc - worst.dx)
    yr = yc + NM_ALPHA * (yc - worst.dy)
    S.nm_last_proposal = (xr, yr)
    return xr, yr, "nm_reflect"

def update_state_from_run(
    S: ImgState,
    run_n: int,
    chunk_id: int,
    was_indexed: bool,
    wrmsd: Optional[float],
    tried_dx: float,
    tried_dy: float,
    tol: float,
) -> None:
    S.last_dx = tried_dx
    S.last_dy = tried_dy
    wr_valid = (wrmsd is not None) and math.isfinite(wrmsd)

    if was_indexed and wr_valid:
        if S.phase == "ring":
            S.phase = "local"
        improved = (S.best_wrmsd is None) or (wrmsd < S.best_wrmsd * (1.0 - tol))
        if improved:
            S.best_wrmsd = wrmsd
            S.best_dx = tried_dx
            S.best_dy = tried_dy
            S.best_run = run_n
            S.best_chunk_id = chunk_id
            S.local_tries_since_improve = 0
        else:
            S.local_tries_since_improve += 1

        if S.nm_initialized and S.nm_last_proposal is not None:
            lx, ly = S.nm_last_proposal
            for v in S.nm_vertices:
                if (abs(v.dx - lx) < 1e-12) and (abs(v.dy - ly) < 1e-12) and (not v.done):
                    v.f = wrmsd
                    v.done = True
                    break

    elif was_indexed and S.best_run is None:
        # remember first indexed position even without wrmsd
        S.best_wrmsd = None
        S.best_dx = tried_dx
        S.best_dy = tried_dy
        S.best_run = run_n
        S.best_chunk_id = chunk_id

def should_stop_local(S: ImgState, local_patience: int) -> bool:
    return (S.phase == "local") and (S.local_tries_since_improve >= local_patience)

# ------------- Parsing & writing the log -------------
def parse_log(log_path: str) -> Tuple[List[Tuple[Optional[Tuple[str,int]], Optional[Tuple[str,...]]]], int]:
    """
    Parse the log into a list of entries preserving order.
    Each entry is either:
      ("section", (real_path, event))  -> represented as (key, None)
      ("row", (run_n, dx, dy, indexed, wrmsd, next_dx, next_dy)) -> represented as (None, tuple)
    Returns (entries, latest_run_n)
    """
    entries: List[Tuple[Optional[Tuple[str,int]], Optional[Tuple[str,...]]]] = []
    latest_run = -1
    with open(log_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for ln in f:
            if ln.startswith("#/"):
                try:
                    path_part, ev_part = ln[1:].rsplit(" event ", 1)
                    ev = int(ev_part.strip())
                    key = (os.path.abspath(path_part.strip()), ev)
                    entries.append((key, None))
                except Exception:
                    # keep the line as a raw row with sentinel
                    entries.append((None, ("RAW", ln.rstrip("\n"))))
                continue
            parts = [p.strip() for p in ln.rstrip("\n").split(",")]
            # Expect 7 columns; if not, pad.
            while len(parts) < 7:
                parts.append("")
            run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = parts[:7]
            try:
                run_n = int(run_s)
                latest_run = max(latest_run, run_n)
            except Exception:
                # Non-data line; keep raw
                entries.append((None, ("RAW", ln.rstrip("\n"))))
                continue
            entries.append((None, (run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s)))
    return entries, latest_run

def write_log(log_path: str, entries) -> None:
    tmp_path = log_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm\n")
        for key, row in entries:
            if key is not None:
                f.write("#" + key[0] + f" event {key[1]}\n")
            elif row is not None and len(row) >= 2 and row[0] == "RAW":
                f.write(row[1] + "\n")
            elif row is not None:
                f.write(",".join(row) + "\n")
    os.replace(tmp_path, log_path)
    
def propose_for_latest(entries, latest_run: int,
                       r_max, r_step, k_base,
                       delta_local, local_patience,
                       seed, tol,
                       sidecar_path: Optional[str]):
    """
    Reconstruct history per (real_h5_path,event) and compute next_dx_mm,next_dy_mm
    for rows belonging to the latest run. If no new point should be tried, mark
    as 'done' in both fields. Optionally writes a sidecar file. Prints a summary:
      [propose] <N> new proposals, <M> marked done
    """
    # Build history per key and list of row indices for the latest run
    history: Dict[Tuple[str,int], List[Tuple[int,float,float,int,Optional[float]]]] = {}
    latest_rows_by_key: Dict[Tuple[str,int], List[int]] = {}
    current_key: Optional[Tuple[str,int]] = None

    for idx, (key, row) in enumerate(entries):
        if key is not None:
            current_key = key
            continue
        if row is None or len(row) == 0:
            continue
        if row[0] == "RAW":
            continue

        run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = row[:7]
        try:
            run_n = int(run_s)
        except Exception:
            continue

        dx = float(dx_s) if dx_s else 0.0
        dy = float(dy_s) if dy_s else 0.0
        indexed = int(idx_s or "0")
        wr = _float_or_blank(wr_s)

        if current_key is None:
            # Rows without a preceding section header are ignored for proposals
            continue

        # Append to history
        history.setdefault(current_key, []).append((run_n, dx, dy, indexed, wr))

        # Track latest-run row indices
        if run_n == latest_run:
            latest_rows_by_key.setdefault(current_key, []).append(idx)

    # Generate proposals per key for latest run rows
    # Value type: Tuple[float,float] OR ("done","done")
    proposals: Dict[Tuple[str,int], Tuple[object, object]] = {}

    for key, trials in history.items():
        # Only generate for keys that appear in the latest run
        if key not in latest_rows_by_key:
            continue

        # Use ALL trials up to latest run
        trials_sorted = sorted(trials, key=lambda t: t[0])  # by run_n
        S = ImgState(ring_angle_base=_hash_angle(seed, key), nm_step=delta_local)
        tried_points = set((_fmt6(dx), _fmt6(dy)) for (_, dx, dy, _, _) in trials_sorted)

        # Seed last_dx/last_dy from the last trial
        if trials_sorted:
            _, last_dx, last_dy, _, _ = trials_sorted[-1]
            S.last_dx, S.last_dy = last_dx, last_dy

        # Update state by replaying all trials
        for (run_n, dx, dy, indexed, wr) in trials_sorted:
            update_state_from_run(S, run_n, run_n, bool(indexed), wr, dx, dy, tol)

        # Decide whether a NEW point is warranted
        last_trial = trials_sorted[-1] if trials_sorted else None
        was_indexed_latest = bool(last_trial[3]) if last_trial else False

        needs_more = False
        if (not was_indexed_latest) and (S.phase == "ring") and (not S.give_up):
            needs_more = True
        elif was_indexed_latest and (S.phase == "local") and (not should_stop_local(S, local_patience)):
            needs_more = True

        if needs_more:
            # Try to generate a novel candidate; skip repeats
            made_new = False
            for _ in range(1000):  # safety cap
                if S.phase == "ring" and not S.give_up:
                    ndx, ndy, _ = pick_ring_probe(S, S.last_dx, S.last_dy, r_step, r_max, k_base)
                else:
                    ndx, ndy, _ = pick_local_nm_probe(S)
                keyfmt = (_fmt6(ndx), _fmt6(ndy))
                if keyfmt not in tried_points:
                    proposals[key] = (ndx, ndy)  # NEW proposal
                    made_new = True
                    break
            if not made_new:
                # Could not find a novel point; still propose best/last as a fallback
                ndx = S.best_dx if S.best_dx is not None else S.last_dx
                ndy = S.best_dy if S.best_dy is not None else S.last_dy
                proposals[key] = (ndx, ndy)
        else:
            # No new point should be tried: mark as done
            proposals[key] = ("done", "done")

    # Count for summary
    n_new = 0
    n_done = 0
    for key, val in proposals.items():
        if isinstance(val[0], str):  # "done"
            n_done += 1
        else:
            n_new += 1

    # Apply proposals to entries (in-place) or write sidecar
    if sidecar_path:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write("real_h5_path,event,run_n,next_dx_mm,next_dy_mm\n")
            for key, idx_list in latest_rows_by_key.items():
                if key not in proposals:
                    continue
                ndx, ndy = proposals[key]
                for row_idx in idx_list:
                    run_n = int(entries[row_idx][1][0])
                    if isinstance(ndx, str):  # "done"
                        f.write(f"{key[0]},{key[1]},{run_n},done,done\n")
                    else:
                        f.write(f"{key[0]},{key[1]},{run_n},{_fmt6(ndx)},{_fmt6(ndy)}\n")
        print(f"[propose] {n_new} new proposals, {n_done} marked done")
        return entries

    # In-place: fill only latest-run rows' next_* (one proposal applies to all latest rows for that key)
    for key, idx_list in latest_rows_by_key.items():
        if key not in proposals:
            continue
        ndx, ndy = proposals[key]
        for row_idx in idx_list:
            row = list(entries[row_idx][1])
            if len(row) < 7:
                row += [""] * (7 - len(row))
            if row[0].isdigit():
                if isinstance(ndx, str):  # "done" marker
                    row[5] = "done"
                    row[6] = "done"
                else:
                    row[5] = _fmt6(ndx)
                    row[6] = _fmt6(ndy)
                entries[row_idx] = (None, tuple(row))

    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    return entries

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fill next_dx_mm,next_dy_mm for latest run using full history."
    )
    ap.add_argument(
        "--run-root",
        default=None,
        help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.",
    )
    ap.add_argument("--r-max", type=float, default=R_MAX_DEFAULT)
    ap.add_argument("--r-step", type=float, default=R_STEP_DEFAULT)
    ap.add_argument("--k-base", type=float, default=K_BASE_DEFAULT)
    ap.add_argument("--delta-local", type=float, default=DELTA_LOCAL_DEFAULT)
    ap.add_argument("--local-patience", type=int, default=LOCAL_PATIENCE_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--converge-tol", type=float, default=CONVERGE_TOL_DEFAULT)
    ap.add_argument("--sidecar", help="If provided, write proposals to this CSV instead of rewriting the log.")
    args = ap.parse_args(argv)

    # ✅ If --run-root is omitted or empty, fall back to DEFAULT_ROOT
    run_root = os.path.abspath(os.path.expanduser(args.run_root or DEFAULT_ROOT))
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print("ERROR: missing runs/image_run_log.csv", file=sys.stderr)
        return 2

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("ERROR: no data rows found in image_run_log.csv", file=sys.stderr)
        return 2

    updated_entries = propose_for_latest(
        entries=entries,
        latest_run=latest_run,
        r_max=float(args.r_max),
        r_step=float(args.r_step),
        k_base=float(args.k_base),
        delta_local=float(args.delta_local),
        local_patience=int(args.local_patience),
        seed=int(args.seed),
        tol=float(args.converge_tol),
        sidecar_path=args.sidecar,
    )

    if args.sidecar:
        print(f"[propose] Wrote proposals to {args.sidecar}")
        return 0

    write_log(log_path, updated_entries)
    print(f"[propose] Updated {log_path} with next_* for run_{latest_run:03d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

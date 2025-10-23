#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_prepare_next_run_overlay_elink.py  (ADJUSTED)

Changes vs original:
- Align metrics by (real_source_h5, event) instead of by chunk_id/gi
- Record the true chunk_id from CSV into state (so early-break picks correct chunks)
- Emit a per-run "used_shifts_<NNN>.csv" ledger for visibility
- Print a small mapping summary ("Matched X/Y images to CSV rows by (real_src, event)")

Defaults when run without flags:
  RUN_ROOT      = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
  r_max=0.05, r_step=0.01, k_base=4, delta_local=0.01, local_patience=3, seed=1337, converge_tl=1e-4
"""
from __future__ import annotations
import argparse, os, sys, csv, math, json, random, shlex, re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import h5py

from overlay_elink import create_overlay, write_shifts_mm

IMAGES_DS = "/entry/data/images"
DEFAULT_RUN_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"

R_MAX_DEFAULT = 0.05
R_STEP_DEFAULT = 0.01
K_BASE_DEFAULT = 4
DELTA_LOCAL_DEFAULT = 0.01
LOCAL_PATIENCE_DEFAULT = 3
SEED_DEFAULT = 1337
CONVERGE_TOL_DEFAULT = 1e-4

# Nelder–Mead parameters (fixed)
NM_ALPHA = 1.0    # reflection
NM_GAMMA = 2.0    # expansion
NM_RHO   = 0.5    # contraction
NM_SIGMA = 0.5    # shrink

@dataclass
class NMVertex:
    dx: float
    dy: float
    f: Optional[float] = None  # wrmsd
    done: bool = False         # evaluated

@dataclass
class ImgState:
    # Phase: 'ring' until first index seen; then 'local'
    phase: str = "ring"
    # Last tried shift (mm)
    last_dx: float = 0.0
    last_dy: float = 0.0
    # Best observed indexed result so far
    best_wrmsd: Optional[float] = None
    best_dx: Optional[float] = None
    best_dy: Optional[float] = None
    best_run: Optional[int] = None
    best_chunk_id: Optional[int] = None
    # Ring search bookkeeping
    ring_step: int = 0               # 0 => r = r_step, 1 => 2*r_step, ...
    ring_angle_idx: int = -1         # within current ring
    ring_angle_base: Optional[float] = None
    # Local search (Nelder–Mead) bookkeeping
    nm_step: float = DELTA_LOCAL_DEFAULT
    nm_vertices: List[NMVertex] = field(default_factory=list)  # length 3 when initialized
    nm_initialized: bool = False
    nm_last_proposal: Optional[Tuple[float,float]] = None      # dx,dy proposed last time
    # Local patience
    local_tries_since_improve: int = 0
    # Terminal flags
    done: bool = False
    give_up: bool = False

def resolve_real_source(h5_path: str) -> str:
    """
    If h5_path is an overlay (ELINK at /entry/data/images), return the
    ExternalLink filename (the original .h5). Otherwise return h5_path.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            link = f.get("/entry/data/images", getlink=True)
            if isinstance(link, h5py.ExternalLink):
                return os.path.abspath(link.filename)
    except Exception:
        pass
    return os.path.abspath(h5_path)

def parse_lst(lst_path: str) -> List[Tuple[str,int]]:
    pairs = []
    with open(lst_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or "//" not in ln: continue
            h5, rest = ln.split("//", 1)
            pairs.append((h5.strip(), int(rest.strip())))
    return pairs

def find_latest_run(run_root: str) -> Tuple[int, str]:
    runs_dir = os.path.join(run_root, "runs")
    max_n = -1; max_path = ""
    if not os.path.isdir(runs_dir):
        return -1, ""
    for name in os.listdir(runs_dir):
        m = re.match(r"^run_(\d{3})$", name)
        if not m: continue
        n = int(m.group(1))
        if n > max_n:
            max_n = n; max_path = os.path.join(runs_dir, name)
    return max_n, max_path

def load_state(path: str, N: int, seed: int) -> Dict[int, ImgState]:
    st: Dict[int, ImgState] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for k, v in raw.items():
                if "nm_vertices" in v and isinstance(v["nm_vertices"], list):
                    v["nm_vertices"] = [NMVertex(**vv) for vv in v["nm_vertices"]]
                st[int(k)] = ImgState(**v)
    rng = random.Random(seed)
    for i in range(N):
        if i not in st:
            st[i] = ImgState(ring_angle_base=rng.uniform(0.0, 2.0*math.pi))
        elif st[i].ring_angle_base is None:
            st[i].ring_angle_base = rng.uniform(0.0, 2.0*math.pi)
    return st

def save_state(path: str, st: Dict[int, ImgState]) -> None:
    enc = {str(k): (asdict(v) | {"nm_vertices": [asdict(nmv) for nmv in v.nm_vertices]}) for k, v in st.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(enc, f, indent=2, sort_keys=True)

def n_angles_for_radius(r: float, r_max: float, k_base: float) -> int:
    return max(1, math.ceil(k_base * (r / max(r_max, 1e-9))))

def pick_ring_probe(S: ImgState, seed_dx: float, seed_dy: float, r_step: float, r_max: float, k_base: float) -> Tuple[float,float,str]:
    r = (S.ring_step + 1) * r_step
    if r > r_max + 1e-12:
        S.give_up = True
        return seed_dx, seed_dy, "ring_exhausted"
    n = n_angles_for_radius(r, r_max, k_base)
    S.ring_angle_idx = (S.ring_angle_idx + 1) % n
    theta = (S.ring_angle_base or 0.0) + 2.0*math.pi * S.ring_angle_idx / n
    ndx = seed_dx + r * math.cos(theta)
    ndy = seed_dy + r * math.sin(theta)
    if S.ring_angle_idx == n - 1:
        S.ring_step += 1
        S.ring_angle_idx = -1
    return ndx, ndy, f"ring_r={r:.5f}_n={n}"

def _ensure_nm_initialized(S: ImgState) -> None:
    if S.nm_initialized:
        return
    cx = S.best_dx if S.best_dx is not None else S.last_dx
    cy = S.best_dy if S.best_dy is not None else S.last_dy
    S.nm_vertices = [
        NMVertex(cx, cy, f=S.best_wrmsd, done=(S.best_wrmsd is not None)),
        NMVertex(cx + S.nm_step, cy, f=None, done=False),
        NMVertex(cx, cy + S.nm_step, f=None, done=False),
    ]
    S.nm_initialized = True

def pick_local_nm_probe(S: ImgState, seed: int) -> Tuple[float,float,str]:
    _ensure_nm_initialized(S)
    for v in S.nm_vertices:
        if not v.done:
            S.nm_last_proposal = (v.dx, v.dy)
            return v.dx, v.dy, "nm_eval_vertex"
    verts = sorted(S.nm_vertices, key=lambda vv: (float("inf") if vv.f is None else vv.f))
    best, mid, worst = verts[0], verts[1], verts[2]
    xc = (best.dx + mid.dx) / 2.0
    yc = (best.dy + mid.dy) / 2.0
    xr = xc + NM_ALPHA * (xc - worst.dx)
    yr = yc + NM_ALPHA * (yc - worst.dy)
    S.nm_last_proposal = (xr, yr)
    return xr, yr, "nm_reflect"

def update_state_from_run(S: ImgState, run_n: int, chunk_id: int, was_indexed: bool, wrmsd: Optional[float],
                          tried_dx: float, tried_dy: float, tol: float) -> ImgState:
    S.last_dx = tried_dx
    S.last_dy = tried_dy
    if was_indexed and wrmsd is not None and math.isfinite(wrmsd):
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
                    v.f = wrmsd; v.done = True
                    break
    return S

def should_stop_local(S: ImgState, local_patience: int) -> bool:
    return (S.phase == "local") and (S.local_tries_since_improve >= local_patience)

def parse_sh(sh_path: str):
    geom = cell = None
    flags: List[str] = []
    with open(sh_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            toks = shlex.split(ln)
            if toks and "indexamajig" in toks[0]:
                i=1
                while i<len(toks):
                    t=toks[i]
                    if t=="-g": geom=toks[i+1]; i+=2; continue
                    if t=="-p": cell=toks[i+1]; i+=2; continue
                    if t in ("-i","-o"): i+=2; continue
                    flags.append(t); i+=1
                break
    return geom, cell, flags

def parse_stream_chunks(stream_path: str) -> List[Tuple[int,int]]:
    bounds = []
    start = None
    with open(stream_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if "Begin chunk" in ln:
            start = i
        elif "End chunk" in ln and start is not None:
            bounds.append((start, i+1)); start = None
    return bounds, lines

def write_early_break_stream(run_root: str, latest_run_n: int, state: Dict[int, ImgState]) -> str:
    runs_dir = os.path.join(run_root, "runs")
    out_path = os.path.join(runs_dir, f"early_break_{latest_run_n:03d}.stream")
    pieces: List[str] = []
    by_run: Dict[int, List[int]] = {}
    for gi, S in state.items():
        if S.best_run is None or S.best_chunk_id is None:
            continue
        by_run.setdefault(S.best_run, []).append(S.best_chunk_id)
    for run_n, cids in by_run.items():
        spath = os.path.join(runs_dir, f"run_{run_n:03d}", f"stream_{run_n:03d}.stream")
        try:
            bounds, lines = parse_stream_chunks(spath)
        except Exception:
            continue
        for cid in sorted(set(cids)):
            if 0 <= cid < len(bounds):
                a, b = bounds[cid]
                pieces.extend(lines[a:b])
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(pieces)
    return out_path

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Prepare next run overlays (ELINK) from latest run with NM/local, logs, early-break.")
    ap.add_argument("--run-root")
    ap.add_argument("--r-max", type=float, default=R_MAX_DEFAULT)
    ap.add_argument("--r-step", type=float, default=R_STEP_DEFAULT)
    ap.add_argument("--k-base", type=float, default=K_BASE_DEFAULT)
    ap.add_argument("--delta-local", type=float, default=DELTA_LOCAL_DEFAULT)
    ap.add_argument("--local-patience", type=int, default=LOCAL_PATIENCE_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--converge-tol", type=float, default=CONVERGE_TOL_DEFAULT)
    args = ap.parse_args(argv)

    using_defaults = (len(argv) == 0)
    run_root = (DEFAULT_RUN_ROOT if using_defaults else os.path.abspath(os.path.expanduser(args.run_root or DEFAULT_RUN_ROOT)))
    r_max = R_MAX_DEFAULT if using_defaults else float(args.r_max)
    r_step = R_STEP_DEFAULT if using_defaults else float(args.r_step)
    k_base = K_BASE_DEFAULT if using_defaults else float(args.k_base)
    delta_local = DELTA_LOCAL_DEFAULT if using_defaults else float(args.delta_local)
    local_patience = LOCAL_PATIENCE_DEFAULT if using_defaults else int(args.local_patience)
    seed = SEED_DEFAULT if using_defaults else int(args.seed)
    tol = CONVERGE_TOL_DEFAULT if using_defaults else float(args.converge_tol)

    print("=== Step 4 (ELINK): Prepare NEXT run overlays ===")
    print(f"Mode:     {'DEFAULTS' if using_defaults else 'CLI'}")
    print(f"Run root: {run_root}")

    last_n, last_run_dir = find_latest_run(run_root)
    if last_n < 0:
        print("ERROR: no run_* folders found under runs/", file=sys.stderr); return 2

    lst_path = os.path.join(last_run_dir, f"lst_{last_n:03d}.lst")
    metrics_path = os.path.join(last_run_dir, f"chunk_metrics_{last_n:03d}.csv")
    run0_sh = os.path.join(run_root, "runs", "run_000", "sh_000.sh")
    if not all(os.path.isfile(p) for p in [lst_path, metrics_path, run0_sh]):
        print("ERROR: missing lst/metrics/sh_000.sh", file=sys.stderr); return 2

    pairs = parse_lst(lst_path)             # [(src_h5, local_frame)]
    N = len(pairs)

    # Group by real source (useful for overlay write-out)
    by_src: Dict[str, List[Tuple[int,int]]] = {}
    for gi, (src, fr) in enumerate(pairs):
        real = resolve_real_source(src)
        by_src.setdefault(real, []).append((gi, fr))

    # --------- READ METRICS keyed by (real_source_from_image_col, event) ----------
    def _src_from_image_col(img: str) -> str:
        if "//" in img:
            h5, _ = img.split("//", 1)
            return h5.strip()
        return img.strip()

    rows_by_key: Dict[Tuple[str,int], Dict[str,str]] = {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            evs = (row.get("event") or "").strip()
            if not evs.isdigit():
                continue
            ev = int(evs)
            img = (row.get("image") or "").strip()
            src_from_img = _src_from_image_col(img)
            real_from_img = resolve_real_source(src_from_img)
            rows_by_key[(real_from_img, ev)] = row

    # persistent state
    runs_dir = os.path.join(run_root, "runs")
    state_path = os.path.join(runs_dir, "state.json")
    state: Dict[int, ImgState] = load_state(state_path, N, seed)

    # update state from latest results & write per-image logs
    logs_dir = os.path.join(runs_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    matched = 0
    for gi in range(N):
        src, fr = pairs[gi]                            # src is the .lst path (possibly overlay); fr is local frame (== event)
        real = resolve_real_source(src)                # normalize to real source (.h5)
        row = rows_by_key.get((real, fr), {})          # align by (real_source, event==frame)
        if row: matched += 1

        was_indexed = int(row.get("indexed","0") or 0) == 1
        wrs = row.get("wrmsd","")
        wr = float(wrs) if wrs not in ("", None, "") else None
        tried_dx = float(row.get("det_shift_x_mm","0") or 0.0)
        tried_dy = float(row.get("det_shift_y_mm","0") or 0.0)

        # Use the true chunk_id from CSV (fallback to gi if absent)
        cid = int(row.get("chunk_id", gi) or gi)
        state[gi] = update_state_from_run(state[gi], last_n, cid, was_indexed, wr, tried_dx, tried_dy, tol)

        # Append to per-image log
        logp = os.path.join(logs_dir, f"image_{gi:06d}.csv")
        first = not os.path.exists(logp)
        with open(logp, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if first:
                w.writerow(["run_n","chunk_id","phase","tried_dx_mm","tried_dy_mm","indexed","wrmsd","best_wrmsd","best_dx_mm","best_dy_mm"])
            w.writerow([last_n, cid, state[gi].phase, f"{tried_dx:.6f}", f"{tried_dy:.6f}", int(was_indexed),
                        (f"{wr:.6f}" if wr is not None else ""),
                        (state[gi].best_wrmsd if state[gi].best_wrmsd is not None else ""),
                        (state[gi].best_dx if state[gi].best_dx is not None else ""),
                        (state[gi].best_dy if state[gi].best_dy is not None else "")])

        if should_stop_local(state[gi], local_patience):
            state[gi].done = True

    print(f"Matched {matched}/{N} images to CSV rows by (real_src, event).")

    # Build / update early-break stream
    eb_path = write_early_break_stream(run_root, last_n, state)
    print(f"Early-break stream updated: {eb_path}")

    # decide candidates
    cand_dx = np.zeros((N,), dtype=np.float64)
    cand_dy = np.zeros((N,), dtype=np.float64)
    any_candidates = False
    for gi in range(N):
        S = state[gi]
        if S.done or S.give_up:
            cand_dx[gi] = (S.best_dx if S.best_dx is not None else S.last_dx)
            cand_dy[gi] = (S.best_dy if S.best_dy is not None else S.last_dy)
            continue
        if S.phase == "ring" and not S.give_up:
            ndx, ndy, _ = pick_ring_probe(S, S.last_dx, S.last_dy, r_step, r_max, k_base)
            if S.give_up:
                cand_dx[gi] = S.last_dx; cand_dy[gi] = S.last_dy
            else:
                cand_dx[gi] = ndx; cand_dy[gi] = ndy; any_candidates = True
            continue
        # Local NM probe
        ndx, ndy, _ = pick_local_nm_probe(S, seed + gi)
        cand_dx[gi] = ndx; cand_dy[gi] = ndy; any_candidates = True

    if not any_candidates:
        final_csv = os.path.join(runs_dir, "best_shifts_final.csv")
        with open(final_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["global_index","best_wrmsd","best_dx_mm","best_dy_mm","phase","done","give_up","best_run","best_chunk_id"])
            for gi in range(N):
                S = state[gi]
                w.writerow([gi,
                           (S.best_wrmsd if S.best_wrmsd is not None else ""),
                           (S.best_dx if S.best_dx is not None else ""),
                           (S.best_dy if S.best_dy is not None else ""),
                           S.phase, int(S.done), int(S.give_up),
                           (S.best_run if S.best_run is not None else ""),
                           (S.best_chunk_id if S.best_chunk_id is not None else "")])
        save_state(state_path, state)
        print("All images converged/exhausted; no new run created.")
        print(f"Wrote: {final_csv}")
        return 0

    # prepare next run dir
    next_n = last_n + 1
    next_run_dir = os.path.join(runs_dir, f"run_{next_n:03d}")
    os.makedirs(next_run_dir, exist_ok=True)

    # create/refresh overlays by source and write shifts for indices present in that source
    overlay_map: Dict[str, str] = {}
    overlays_dir = os.path.join(next_run_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    for src, idx_list in by_src.items():
        base = os.path.basename(src)
        overlay_path = os.path.join(overlays_dir, f"overlay_{base}")
        try:
            create_overlay(src, overlay_path)
        except Exception as e:
            print(f"ERROR creating overlay for {src}: {e}", file=sys.stderr); return 2
        overlay_map[src] = overlay_path
        local_indices = [fr for (_, fr) in idx_list]
        gi_indices    = [gi for (gi, _) in idx_list]
        dx_vals = cand_dx[gi_indices].tolist()
        dy_vals = cand_dy[gi_indices].tolist()
        write_shifts_mm(overlay_path, local_indices, dx_vals, dy_vals)

    # write lst for next run (pointing to overlays), preserving order
    lst_next = os.path.join(next_run_dir, f"lst_{next_n:03d}.lst")
    with open(lst_next, "w", encoding="utf-8") as f:
        for gi, (src, fr) in enumerate(pairs):
            real = resolve_real_source(src)
            f.write(f"{overlay_map[real]} //{fr}\n")

    # reuse geom/cell/flags from sh_000.sh
    geom, cell, flags = parse_sh(os.path.join(run_root, "runs", "run_000", "sh_000.sh"))
    sh_next = os.path.join(next_run_dir, f"sh_{next_n:03d}.sh")
    stream_next = os.path.join(next_run_dir, f"stream_{next_n:03d}.stream")
    with open(sh_next, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        cmd = ["indexamajig", "-g", geom, "-i", lst_next, "-o", stream_next, "-p", cell, *flags]
        f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
    os.chmod(sh_next, 0o755)

    # Save state for next iteration
    save_state(state_path, state)

    # --------------- Per-run shifts ledger (visibility) ---------------
    ledger_path = os.path.join(next_run_dir, f"used_shifts_{next_n:03d}.csv")
    with open(ledger_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gi","real_src_h5","frame","phase","done","give_up","cand_dx_mm","cand_dy_mm",
                    "best_wrmsd","best_dx_mm","best_dy_mm","last_dx_mm","last_dy_mm"])
        for gi in range(N):
            real = resolve_real_source(pairs[gi][0])
            fr = pairs[gi][1]
            S = state[gi]
            w.writerow([
                gi, real, fr, S.phase, int(S.done), int(S.give_up),
                f"{cand_dx[gi]:.6f}", f"{cand_dy[gi]:.6f}",
                ("" if S.best_wrmsd is None else f"{S.best_wrmsd:.6f}"),
                ("" if S.best_dx   is None else f"{S.best_dx:.6f}"),
                ("" if S.best_dy   is None else f"{S.best_dy:.6f}"),
                f"{S.last_dx:.6f}", f"{S.last_dy:.6f}",
            ])

    print(f"Prepared next run: run_{next_n:03d}")
    print(f"  lst:  {os.path.basename(lst_next)}")
    print(f"  sh:   {os.path.basename(sh_next)}")
    print(f"  overlays in: {overlays_dir}")
    print(f"  early-break: {eb_path}")
    print(f"  logs: {logs_dir}")
    print(f"  shifts ledger: {os.path.basename(ledger_path)}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))

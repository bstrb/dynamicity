#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict ZOLZ-centered, visually "problematic" zone axes from a CrystFEL stream header.

Enhancements vs. original:
- Robust header parsing tolerant to whitespace
- Progress/timing via --progress
- Parallel per-direction scanning via --jobs (multiprocessing)
- **Numba-accelerated** three-beam overlap counting (huge speedup on large in-zone sets)

Author: Buster Blomberg (parallel + Numba)
"""

import argparse, math, re, csv, sys, os, time
from math import sqrt, floor, gcd

import numpy as np
from numba import njit

# ----------------------------- Utilities -----------------------------

_FLOAT_RE = r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

def _read_text(path_or_dash: str) -> str:
    if path_or_dash == "-" or path_or_dash is None:
        return sys.stdin.read()
    with open(path_or_dash, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def _extract_block(txt: str, begin_marker: str, end_marker: str) -> str:
    # Tolerant anchored regex first
    begin_re = re.compile(rf'^\s*{re.escape(begin_marker)}\s*$', re.MULTILINE)
    end_re   = re.compile(rf'^\s*{re.escape(end_marker)}\s*$', re.MULTILINE)
    mb = begin_re.search(txt); me = end_re.search(txt) if mb else None
    if mb and me and me.start() > mb.end():
        return txt[mb.end():me.start()]
    # Fallback substring search
    bpos = txt.find(begin_marker)
    if bpos != -1:
        epos = txt.find(end_marker, bpos + len(begin_marker))
        if epos != -1 and epos > bpos:
            bend = txt.find("\n", bpos); bend = bend if bend != -1 else bpos + len(begin_marker)
            estart = txt.rfind("\n", 0, epos); estart = estart if estart != -1 else epos
            if estart > bend:
                return txt[bend:estart]
    raise ValueError(f"Could not find well-formed block: '{begin_marker}' .. '{end_marker}'")

# ----------------------------- Parsing -----------------------------

def parse_cell_text(txt):
    cell = {"lattice_type": None, "centering": "P", "a": None, "b": None, "c": None, "al": None, "be": None, "ga": None}
    def grab(key, unit=None, fallback=None):
        if key in ("lattice_type","centering"):
            m = re.search(rf'^{key}\s*=\s*([A-Za-z]+)', txt, re.MULTILINE)
            return (m.group(1) if m else fallback)
        else:
            if unit:
                m = re.search(rf'^{key}\s*=\s*{_FLOAT_RE}\s*{re.escape(unit)}', txt, re.MULTILINE)
            else:
                m = re.search(rf'^{key}\s*=\s*{_FLOAT_RE}', txt, re.MULTILINE)
            return (float(m.group(1)) if m else fallback)
    cell["lattice_type"] = (grab("lattice_type") or "").strip().lower()
    cell["centering"]    = (grab("centering") or "P").strip().upper()
    cell["a"]=grab("a","A"); cell["b"]=grab("b","A"); cell["c"]=grab("c","A")
    cell["al"]=grab("al","deg"); cell["be"]=grab("be","deg"); cell["ga"]=grab("ga","deg")
    for k in ("a","b","c","al","be","ga"):
        if cell[k] is None: raise ValueError(f"Missing '{k}' in unit cell header")
    return cell

def parse_geom_text(txt):
    def grab_float(key, unit=None):
        if unit:
            m = re.search(rf'^{re.escape(key)}\s*=\s*{_FLOAT_RE}\s*{re.escape(unit)}', txt, re.MULTILINE)
        else:
            m = re.search(rf'^{re.escape(key)}\s*=\s*{_FLOAT_RE}\s*$', txt, re.MULTILINE)
        return float(m.group(1)) if m else None
    def grab_int(key):
        m = re.search(rf'^p0/{re.escape(key)}\s*=\s*(\d+)', txt, re.MULTILINE)
        return int(m.group(1)) if m else None
    geom = {
        "wavelength_A": grab_float("wavelength","A"),
        "clen_m": grab_float("clen","m"),
        "res_px_per_m": grab_float("res"),
        "min_ss": grab_int("min_ss"), "max_ss": grab_int("max_ss"),
        "min_fs": grab_int("min_fs"), "max_fs": grab_int("max_fs"),
    }
    for k in ("wavelength_A","clen_m","res_px_per_m","min_ss","max_ss","min_fs","max_fs"):
        if geom[k] is None: raise ValueError(f"Missing '{k}' in geometry header")
    return geom

def parse_stream_headers(path_or_dash: str):
    txt = _read_text(path_or_dash)
    geom_txt = _extract_block(txt, "----- Begin geometry file -----", "----- End geometry file -----")
    cell_txt = _extract_block(txt, "----- Begin unit cell -----", "----- End unit cell -----")
    return parse_cell_text(cell_txt), parse_geom_text(geom_txt)

# ----------------------- Metric & extinctions -----------------------

def deg2rad(x): return x*math.pi/180.0

def reciprocal_metric(a,b,c,al_deg,be_deg,ga_deg):
    al = deg2rad(al_deg); be = deg2rad(be_deg); ga = deg2rad(ga_deg)
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    G = [[a*a, a*b*cg, a*c*cb],[a*b*cg, b*b, b*c*ca],[a*c*cb, b*c*ca, c*c]]
    detG = (G[0][0]*(G[1][1]*G[2][2]-G[1][2]*G[2][1]) - G[0][1]*(G[1][0]*G[2][2]-G[1][2]*G[2][0]) + G[0][2]*(G[1][0]*G[2][1]-G[1][1]*G[2][0]))
    if abs(detG) < 1e-18: raise ValueError("Singular metric")
    adj = [
        [ (G[1][1]*G[2][2]-G[1][2]*G[2][1]), -(G[0][1]*G[2][2]-G[0][2]*G[2][1]),  (G[0][1]*G[1][2]-G[0][2]*G[1][1]) ],
        [-(G[1][0]*G[2][2]-G[1][2]*G[2][0]),  (G[0][0]*G[2][2]-G[0][2]*G[2][0]), -(G[0][0]*G[1][2]-G[0][2]*G[1][0]) ],
        [ (G[1][0]*G[2][1]-G[1][1]*G[2][0]), -(G[0][0]*G[2][1]-G[0][1]*G[2][0]),  (G[0][0]*G[1][1]-G[0][1]*G[1][0]) ]
    ]
    return [[adj[i][j]/detG for j in range(3)] for i in range(3)]  # G*

def g_magnitude(h,k,l,Gstar):
    v0 = Gstar[0][0]*h+Gstar[0][1]*k+Gstar[0][2]*l
    v1 = Gstar[1][0]*h+Gstar[1][1]*k+Gstar[1][2]*l
    v2 = Gstar[2][0]*h+Gstar[2][1]*k+Gstar[2][2]*l
    val = h*v0 + k*v1 + l*v2
    return math.sqrt(val) if val > 0.0 else 0.0

def lattice_allowed(h,k,l, c):
    c = c.upper()
    if c == "P": return True
    if c == "I": return ((h+k+l)&1) == 0
    if c == "F": return ((h&1)==(k&1)==(l&1))
    if c == "A": return ((k+l)&1) == 0
    if c == "B": return ((h+l)&1) == 0
    if c == "C": return ((h+k)&1) == 0
    return True

# ------------------------- Core utilities -------------------------

def gcd3(x,y,z): return gcd(gcd(abs(x),abs(y)),abs(z))

def canonical_uvw(u,v,w):
    if u==v==w==0: return None
    g = gcd3(u,v,w)
    if g == 0: return None
    u,v,w = u//g, v//g, w//g
    if (w < 0) or (w == 0 and (v < 0 or (v == 0 and u < 0))):
        u,v,w = -u,-v,-w
    return (u,v,w)

def enumerate_hkl_up_to_g(Gstar, gmax, centering):
    a_s = sqrt(max(Gstar[0][0],1e-16))
    b_s = sqrt(max(Gstar[1][1],1e-16))
    c_s = sqrt(max(Gstar[2][2],1e-16))
    hmax = max(1, floor(gmax/a_s)+2)
    kmax = max(1, floor(gmax/b_s)+2)
    lmax = max(1, floor(gmax/c_s)+2)
    HKL=[]
    for h in range(-hmax,hmax+1):
        for k in range(-kmax,kmax+1):
            for l in range(-lmax,lmax+1):
                if (h==0 and k==0 and l==0): continue
                if not lattice_allowed(h,k,l,centering): continue
                g = g_magnitude(h,k,l,Gstar)
                if g <= gmax + 1e-12:
                    HKL.append((h,k,l,g))
    return HKL, (hmax,kmax,lmax)

def rad_to_pixels(g, clen_m, wavelength_A, res_px_per_m):
    return g * clen_m * wavelength_A * res_px_per_m

def zone_condition_zero(h,k,l,u,v,w):
    return (h*u + k*v + l*w) == 0

def rel_intensity(h,k,l,Gstar, g0=0.25):
    # crude |g|-only gate
    g = g_magnitude(h,k,l,Gstar)
    return 1.0 / (1.0 + (g/g0)**2)

# ---------------------- Numba three-beam engine ----------------------

# We map (h,k,l) to a 64-bit sortable key: pack 3 signed 21-bit lanes with a +bias
# Range ±1,048,575 is plenty for practical h,k,l at your bounds.
BITS = 21
BIAS = 1 << (BITS - 1)     # 2^20
SHIFT_K = BITS
SHIFT_H = 2*BITS
MAX_ABS = BIAS - 1         # 1,048,575

@njit(cache=True)
def _hkls_to_keys(hkls_int32):
    n = hkls_int32.shape[0]
    keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        h = hkls_int32[i,0]; k = hkls_int32[i,1]; l = hkls_int32[i,2]
        # if out-of-range, clamp to a sentinel outside valid domain
        if abs(h) > MAX_ABS or abs(k) > MAX_ABS or abs(l) > MAX_ABS:
            # This shouldn't happen with reasonable bounds; mark invalid
            keys[i] = np.int64(-1)
        else:
            keys[i] = (((np.int64(h)+BIAS) << SHIFT_H)
                      | ((np.int64(k)+BIAS) << SHIFT_K)
                      |  (np.int64(l)+BIAS))
    return keys

@njit(cache=True)
def _key_for(h,k,l):
    if abs(h) > MAX_ABS or abs(k) > MAX_ABS or abs(l) > MAX_ABS:
        return np.int64(-1)
    return (((np.int64(h)+BIAS) << SHIFT_H)
           | ((np.int64(k)+BIAS) << SHIFT_K)
           |  (np.int64(l)+BIAS))

@njit(cache=True)
def _contains(sorted_keys, key):
    # binary search for key in sorted 1D array
    left = 0
    right = sorted_keys.size
    while left < right:
        mid = (left + right) // 2
        v = sorted_keys[mid]
        if v < key:
            left = mid + 1
        else:
            right = mid
    if left < sorted_keys.size and sorted_keys[left] == key:
        return True
    return False

@njit(cache=True)
def three_beam_overlap_count_numba(hkls_int32):
    """
    hkls_int32: (N,3) int32 array of in-zone (h,k,l).
    Count pairs (i<=j) where (h1±h2, k1±k2, l1±l2) is also in the set (excluding 0,0,0).
    """
    # Precompute sorted 1D key array for membership tests
    keys = _hkls_to_keys(hkls_int32)
    # Filter out invalid keys (shouldn't happen; safety)
    valid = np.where(keys >= 0)[0]
    keys_sorted = np.sort(keys[valid])

    n = hkls_int32.shape[0]
    count = 0
    for i in range(n):
        h1 = hkls_int32[i,0]; k1 = hkls_int32[i,1]; l1 = hkls_int32[i,2]
        for j in range(i, n):
            h2 = hkls_int32[j,0]; k2 = hkls_int32[j,1]; l2 = hkls_int32[j,2]

            # sum
            hs = h1 + h2; ks = k1 + k2; ls = l1 + l2
            if (hs != 0) or (ks != 0) or (ls != 0):
                ks_key = _key_for(hs,ks,ls)
                if ks_key >= 0 and _contains(keys_sorted, ks_key):
                    count += 1

            # diff
            hd = h1 - h2; kd = k1 - k2; ld = l1 - l2
            if (hd != 0) or (kd != 0) or (ld != 0):
                kd_key = _key_for(hd,kd,ld)
                if kd_key >= 0 and _contains(keys_sorted, kd_key):
                    count += 1
    return count

# ---------------------- ZOLZ ring & crowding ----------------------

def first_zolz_ring(u,v,w, HKL, Gstar, tol_g=1e-4, I_min_rel=0.0):
    gmin = None
    for (h,k,l,g) in HKL:
        if not zone_condition_zero(h,k,l,u,v,w): continue
        if rel_intensity(h,k,l,Gstar) < I_min_rel: continue
        if g > 0 and (gmin is None or g < gmin): gmin = g
    if gmin is None: return None, None, []
    ring = [(h,k,l) for (h,k,l,g) in HKL
            if zone_condition_zero(h,k,l,u,v,w) and abs(g - gmin) <= tol_g
            and rel_intensity(h,k,l,Gstar) >= I_min_rel]
    return gmin, ring, ring

def in_zone_under_gmax(u,v,w, HKL, Gstar, gmax, I_min_rel=0.0):
    acc=[]
    for (h,k,l,g) in HKL:
        if g > gmax + 1e-12: continue
        if not zone_condition_zero(h,k,l,u,v,w): continue
        if rel_intensity(h,k,l,Gstar) < I_min_rel: continue
        acc.append((h,k,l,g))
    return acc

# ------------------- Per-direction worker (parallel) -------------------

# Globals for forked workers
_G = {
    "HKL": None, "Gstar": None,
    "g_crowd": None, "tol_g": None, "i_min_rel": None,
    "r_edge_px": None, "clen_m": None, "wavelength_A": None, "res_px_per_m": None,
    "ring_mult_min": None, "n_min": None, "m_min": None,
    "score_alpha": None, "score_beta": None
}

def _worker_init(HKL, Gstar, params):
    _G["HKL"] = HKL
    _G["Gstar"] = Gstar
    for k,v in params.items():
        _G[k] = v

def _eval_dir(uvw):
    u,v,w = uvw
    HKL   = _G["HKL"]
    Gstar = _G["Gstar"]
    tol_g = _G["tol_g"]
    Imin  = _G["i_min_rel"]
    r_edge= _G["r_edge_px"]
    clen  = _G["clen_m"]
    wlA   = _G["wavelength_A"]
    respm = _G["res_px_per_m"]
    gmax  = _G["g_crowd"]
    ring_mult_min = _G["ring_mult_min"]
    n_min = _G["n_min"]; m_min = _G["m_min"]
    a = _G["score_alpha"]; b = _G["score_beta"]

    # First ZOLZ ring
    gmin, _, ring_hkls = first_zolz_ring(u,v,w, HKL, Gstar, tol_g=tol_g, I_min_rel=Imin)
    if gmin is None:
        return None
    rpx = rad_to_pixels(gmin, clen, wlA, respm)
    if rpx > r_edge + 1e-9:
        return None

    ring_mult = len(ring_hkls)
    if ring_mult < ring_mult_min:
        return None

    inzone = in_zone_under_gmax(u,v,w, HKL, Gstar, gmax, I_min_rel=Imin)
    N = len(inzone)
    if N < n_min:
        return None

    # Numba path: convert to int32 Nx3 and call JIT counter
    hkls_int32 = np.empty((N,3), dtype=np.int32)
    for i,(h,k,l,_) in enumerate(inzone):
        hkls_int32[i,0] = h; hkls_int32[i,1] = k; hkls_int32[i,2] = l
    M = int(three_beam_overlap_count_numba(hkls_int32))

    if M < m_min:
        return None

    S = a * N + b * M
    return {"u":u,"v":v,"w":w,"ring_type":"ZOLZ","g_min_1_over_A":gmin,"r_px":rpx,
            "ring_mult": ring_mult,"N":N,"M":M,"Score":S,"inside":True}

# --------------------------------- CLI ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Predict visually problematic ZOLZ-centered zone axes from a CrystFEL stream header (parallel + Numba).")
    ap.add_argument("--stream", required=True, help="Path to .stream file, or '-' to read from stdin")
    ap.add_argument("--uvw-max", type=int, default=3)
    ap.add_argument("--g-enum-bound", type=float, default=None)
    ap.add_argument("--g-max", type=float, default=None)
    ap.add_argument("--zolz-only", action="store_true", default=True)
    ap.add_argument("--holz", dest="holz", action="store_true")
    ap.add_argument("--i-min-rel", type=float, default=0.0)
    ap.add_argument("--ring-mult-min", type=int, default=2)
    ap.add_argument("--n-min", type=int, default=12)
    ap.add_argument("--m-min", type=int, default=8)
    ap.add_argument("--score-alpha", type=float, default=0.4)
    ap.add_argument("--score-beta",  type=float, default=0.6)
    ap.add_argument("--margin-px", type=float, default=0.0)
    ap.add_argument("--tol-g", type=float, default=5e-4)
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--listuvw", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--printresults", action="store_true", help="Print results to stdout (default: dont print to stderr)")
    ap.add_argument("--progress-steps", type=int, default=20)
    # Parallel controls
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="Worker processes (default: all cores)")
    ap.add_argument("--chunksize", type=int, default=8, help="Items per task sent to each worker")
    args = ap.parse_args()

    t0 = time.time()
    if args.progress:
        eprint("[0/6] Reading headers...")

    # Parse stream
    cell, geom = parse_stream_headers(args.stream)

    # Reciprocal metric
    Gstar = reciprocal_metric(cell["a"], cell["b"], cell["c"], cell["al"], cell["be"], cell["ga"])

    # Geometry & edge-derived g limits
    nx = geom["max_fs"] - geom["min_fs"] + 1
    ny = geom["max_ss"] - geom["min_ss"] + 1
    r_edge_px = min(nx, ny)/2.0 - args.margin_px
    g_edge = (r_edge_px / geom["res_px_per_m"]) / (geom["clen_m"] * geom["wavelength_A"])
    g_enum = args.g_enum_bound if args.g_enum_bound is not None else 1.10 * g_edge
    g_crowd = args.g_max if args.g_max is not None else g_edge

    if args.progress:
        eprint(f"[1/6] Parsed cell & geom. Panel {nx}×{ny}px, edge r≈{r_edge_px:.2f}px.")
        eprint(f"      g_edge≈{g_edge:.4f} Å⁻¹ | g_enum={g_enum:.4f} | g_crowd={g_crowd:.4f}")
        a_s = sqrt(max(Gstar[0][0],1e-16)); b_s = sqrt(max(Gstar[1][1],1e-16)); c_s = sqrt(max(Gstar[2][2],1e-16))
        hmax = max(1, floor(g_enum/a_s)+2); kmax = max(1, floor(g_enum/b_s)+2); lmax = max(1, floor(g_enum/c_s)+2)
        approx = (2*hmax+1)*(2*kmax+1)*(2*lmax+1)
        eprint(f"[2/6] Enumerating HKL up to g={g_enum:.3f} (box ≈ ±{hmax}, ±{kmax}, ±{lmax} → {approx:,} triples)...")

    # Global HKL enumeration (shared read-only)
    t_enum0 = time.time()
    HKL, _ = enumerate_hkl_up_to_g(Gstar, g_enum, centering=cell["centering"])
    t_enum1 = time.time()
    if args.progress:
        eprint(f"      ...{len(HKL):,} reflections kept (elapsed {t_enum1 - t_enum0:.1f}s).")
        eprint(f"[3/6] Building unique UVW list (|u|,|v|,|w| ≤ {args.uvw_max})...")

    # Enumerate canonical zone axes
    seen=set(); dirs=[]
    rng = range(-args.uvw_max, args.uvw_max+1)
    for u in rng:
        for v in rng:
            for w in rng:
                canon = canonical_uvw(u,v,w)
                if canon and canon not in seen:
                    seen.add(canon); dirs.append(canon)

    total = len(dirs)
    if args.progress:
        eprint(f"      ...{total} unique directions.")
        eprint(f"[4/6] Scanning directions in parallel with --jobs {args.jobs}, chunksize {args.chunksize}...")

    # Prepare worker globals
    params = {
        "g_crowd": g_crowd, "tol_g": args.tol_g, "i_min_rel": args.i_min_rel,
        "r_edge_px": r_edge_px, "clen_m": geom["clen_m"], "wavelength_A": geom["wavelength_A"],
        "res_px_per_m": geom["res_px_per_m"], "ring_mult_min": args.ring_mult_min,
        "n_min": args.n_min, "m_min": args.m_min, "score_alpha": args.score_alpha, "score_beta": args.score_beta
    }

    # Parallel map across directions
    rows=[]
    if args.jobs == 1:
        step = max(1, total // max(1,args.progress_steps))
        _worker_init(HKL, Gstar, params)
        for idx,uvw in enumerate(dirs,1):
            if args.progress and (idx==1 or idx%step==0 or idx==total):
                eprint(f"      [{idx:>4}/{total}] uvw=({uvw[0]:>2} {uvw[1]:>2} {uvw[2]:>2})")
            rec = _eval_dir(uvw)
            if rec: rows.append(rec)
    else:
        import multiprocessing as mp
        # Use "fork" so big HKL list is shared CoW
        with mp.get_context("fork").Pool(processes=args.jobs, initializer=_worker_init, initargs=(HKL, Gstar, params)) as pool:
            it = pool.imap_unordered(_eval_dir, dirs, chunksize=args.chunksize)
            if args.progress:
                ping_every = max(1, total // max(1,args.progress_steps))
                seen_count = 0
                for rec in it:
                    seen_count += 1
                    if rec: rows.append(rec)
                    if (seen_count == 1) or (seen_count % ping_every == 0) or (seen_count == total):
                        eprint(f"      [{seen_count:>4}/{total}] ...")
            else:
                for rec in it:
                    if rec: rows.append(rec)

    if args.progress:
        eprint(f"[5/6] Reducing & sorting {len(rows)} results...")

    # --- Normalize N and M across surviving rows, then recompute Score (min–max to [0,1]) ---
    if rows:
        Nvals = np.array([r["N"] for r in rows], dtype=float)
        Mvals = np.array([r["M"] for r in rows], dtype=float)

        Nmin, Nmax = Nvals.min(), Nvals.max()
        Mmin, Mmax = Mvals.min(), Mvals.max()

        def _minmax(x, xmin, xmax):
            return 0.0 if xmax == xmin else (x - xmin) / (xmax - xmin)

        for r in rows:
            r["N_norm"] = _minmax(r["N"], Nmin, Nmax)
            r["M_norm"] = _minmax(r["M"], Mmin, Mmax)
            # Recompute Score using normalized values
            r["Score"] = args.score_alpha * r["N_norm"] + args.score_beta * r["M_norm"]
    # ----------------------------------------------------------------------------------------

    # Sort & print
    rows.sort(key=lambda r: (-r["Score"], -r["N"], r["r_px"], abs(r["u"])+abs(r["v"])+abs(r["w"]), r["u"], r["v"], r["w"]))

    print(f"Cell: {cell['lattice_type'].upper()} {cell['centering'].upper()} | a={cell['a']:.4f} Å b={cell['b']:.4f} Å c={cell['c']:.4f} Å "
          f"al={cell['al']:.2f}° be={cell['be']:.2f}° ga={cell['ga']:.2f}°")
    print(f"Geom: λ={geom['wavelength_A']} Å, L={geom['clen_m']} m, res={geom['res_px_per_m']} px/m | panel {nx}×{ny}px | edge r={r_edge_px:.2f}px")
    print(f"Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.3f}, g_crowd={g_crowd:.3f}, I_min_rel={args.i_min_rel}, "
          f"N_min={args.n_min}, M_min={args.m_min}, ring_mult_min={args.ring_mult_min}, α={args.score_alpha}, β={args.score_beta}, tol_g={args.tol_g}\n")

    if args.printresults:
        print("(u v w)   r_px    g_min(1/Å)  ring_mult   N    M    Score")
        for r in rows:
            print(f"{r['u']:>2} {r['v']:>2} {r['w']:>2}  {r['r_px']:7.2f}   {r['g_min_1_over_A']:.4f}    {r['ring_mult']:>3}    {r['N']:>3}  {r['M']:>4}  {r['Score']:>7.2f}")
        if args.listuvw:
            for r in rows:
                print(f"\"{r['u']:>2} {r['v']:>2} {r['w']:>2}\",")

    if args.csv:
        csv_path = args.stream.replace(".stream", "_problematic_orientations.csv")
        with open(csv_path, "w", newline="") as f:
            f.write(f"# Cell: {cell['lattice_type'].upper()} {cell['centering'].upper()} | "
                    f"a={cell['a']:.4f} Å b={cell['b']:.4f} Å c={cell['c']:.4f} Å "
                    f"al={cell['al']:.2f}° be={cell['be']:.2f}° ga={cell['ga']:.2f}°\n")
            f.write(f"# Geom: λ={geom['wavelength_A']} Å, L={geom['clen_m']} m, res={geom['res_px_per_m']} px/m | "
                    f"panel {nx}×{ny}px | edge r={r_edge_px:.2f}px\n")
            f.write(f"# Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.3f}, g_crowd={g_crowd:.3f}, "
                    f"I_min_rel={args.i_min_rel}, N_min={args.n_min}, M_min={args.m_min}, "
                    f"ring_mult_min={args.ring_mult_min}, α={args.score_alpha}, β={args.score_beta}, tol_g={args.tol_g}\n\n")
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                               ["u","v","w","ring_type","g_min_1_over_A","r_px","ring_mult","N","M","Score","inside"])
            w.writeheader()
            for r in rows: w.writerow(r)
            triplets = [f'"{r["u"]:>2} {r["v"]:>2} {r["w"]:>2}"' for r in rows]
            f.write("\n # Problematic axis triplets listed:\n")
            f.write(" ".join(triplets) + "\n")
        print(f"\nWrote CSV: {csv_path}")

    if args.progress:
        eprint(f"[6/6] Done in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()

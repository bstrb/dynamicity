#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-frame ZOLZ scoring for CrystFEL streams:
- Parses geom once; iterates all chunks
- Uses per-frame cell + astar/bstar/cstar (no det-shift)
- Infers best small-integer [u v w] from orientation
- Computes N, M and then NORMALIZES N and M before Score = α·N_norm + β·M_norm
- Outputs:
    (a) CSV with metrics (optional)
    (b) A NEW .stream file with chunks SORTED by Score and an injected line:
        'Problematic Orientation Score: <score>'
"""

import argparse, re, csv, sys, math, os, time
from math import sqrt, floor, gcd
import numpy as np
from numba import njit

# ----------------------------- Regex helpers -----------------------------
_FLOAT_RE = r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'

def eprint(*args, **kw): print(*args, file=sys.stderr, flush=True, **kw)

def read_text(path_or_dash: str) -> str:
    if path_or_dash == "-" or path_or_dash is None:
        return sys.stdin.read()
    with open(path_or_dash, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def extract_block_all(txt: str, begin_marker: str, end_marker: str):
    begin_re = re.compile(rf'^\s*{re.escape(begin_marker)}\s*$', re.MULTILINE)
    end_re   = re.compile(rf'^\s*{re.escape(end_marker)}\s*$', re.MULTILINE)
    pos = 0
    while True:
        mb = begin_re.search(txt, pos)
        if not mb: break
        me = end_re.search(txt, mb.end())
        if not me: break
        yield txt[mb.end():me.start()]
        pos = me.end()

def extract_block_once(txt: str, begin_marker: str, end_marker: str) -> str:
    blocks = list(extract_block_all(txt, begin_marker, end_marker))
    if not blocks: raise ValueError(f"Block not found: {begin_marker} .. {end_marker}")
    return blocks[0]

# Find all chunk *blocks* including markers; return (pre_header, [(begin, inner, end, fullspan)])
_CHUNK_BLOCK_RE = re.compile(
    r'(^\s*----- Begin chunk -----\s*\n)(.*?)(^\s*----- End chunk -----\s*$)',
    re.DOTALL | re.MULTILINE
)

def split_header_and_chunks(full_text: str):
    chunks = []
    last_end = 0
    for m in _CHUNK_BLOCK_RE.finditer(full_text):
        if not chunks:
            header = full_text[:m.start()]
        begin = m.group(1)
        inner = m.group(2)
        end   = m.group(3)
        chunks.append((begin, inner, end, (m.start(), m.end())))
        last_end = m.end()
    if not chunks:
        return full_text, []  # no chunks; treat all as header
    tail = full_text[last_end:]
    header = full_text[:chunks[0][3][0]]
    return header, chunks, tail

# ----------------------------- Geom parsing -----------------------------
def parse_geom_text(txt):
    def grab_float(key, unit=None, allow_bare=False):
        if unit:
            m = re.search(rf'^{re.escape(key)}\s*=\s*{_FLOAT_RE}\s*{re.escape(unit)}', txt, re.MULTILINE)
        elif allow_bare:
            m = re.search(rf'^{re.escape(key)}\s*=\s*{_FLOAT_RE}', txt, re.MULTILINE)
        else:
            m = re.search(rf'^{re.escape(key)}\s*=\s*{_FLOAT_RE}\s*$', txt, re.MULTILINE)
        return float(m.group(1)) if m else None
    def grab_int(prefix_key):
        m = re.search(rf'^p0/{re.escape(prefix_key)}\s*=\s*(\d+)', txt, re.MULTILINE)
        return int(m.group(1)) if m else None
    geom = {
        "wavelength_A": grab_float("wavelength","A"),
        "clen_m": grab_float("clen","m"),
        "res_px_per_m": grab_float("res", None, allow_bare=True),
        "min_ss": grab_int("min_ss"), "max_ss": grab_int("max_ss"),
        "min_fs": grab_int("min_fs"), "max_fs": grab_int("max_fs"),
    }
    missing = [k for k,v in geom.items() if v is None]
    if missing: raise ValueError(f"Missing geometry keys: {missing}")
    return geom

def parse_header_sections(stream_text):
    geom_txt = extract_block_once(stream_text, "----- Begin geometry file -----", "----- End geometry file -----")
    geom = parse_geom_text(geom_txt)
    return geom

# ----------------------------- Reciprocal metric -----------------------------
def deg2rad(x): return x*math.pi/180.0

def reciprocal_metric(aA,bA,cA,al_deg,be_deg,ga_deg):
    a,b,c = aA, bA, cA
    al = deg2rad(al_deg); be = deg2rad(be_deg); ga = deg2rad(ga_deg)
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    G = [[a*a, a*b*cg, a*c*cb],
         [a*b*cg, b*b, b*c*ca],
         [a*c*cb, b*c*ca, c*c]]
    detG = (G[0][0]*(G[1][1]*G[2][2]-G[1][2]*G[2][1])
           -G[0][1]*(G[1][0]*G[2][2]-G[1][2]*G[2][0])
           +G[0][2]*(G[1][0]*G[2][1]-G[1][1]*G[2][0]))
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

# ----------------------------- Lattice extinctions -----------------------------
def lattice_allowed(h,k,l, c):
    c = (c or "P").upper()
    if c == "P": return True
    if c == "I": return ((h+k+l)&1) == 0
    if c == "F": return ((h&1)==(k&1)==(l&1))
    if c == "A": return ((k+l)&1) == 0
    if c == "B": return ((h+l)&1) == 0
    if c == "C": return ((h+k)&1) == 0
    return True

# ----------------------------- HKL enumeration -----------------------------
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
    return HKL

def zone_condition_zero(h,k,l,u,v,w): return (h*u + k*v + l*w) == 0

def rel_intensity(h,k,l,Gstar, g0=0.25):
    g = g_magnitude(h,k,l,Gstar)
    return 1.0 / (1.0 + (g/g0)**2)

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

# ----------------------------- Integer uvw utils -----------------------------
def gcd3(x,y,z): return gcd(gcd(abs(int(x)),abs(int(y))),abs(int(z)))

def canonical_uvw(u,v,w):
    if u==v==w==0: return None
    g = gcd3(u,v,w)
    if g == 0: return None
    u,v,w = u//g, v//g, w//g
    if (w < 0) or (w == 0 and (v < 0 or (v == 0 and u < 0))):
        u,v,w = -u,-v,-w
    return (u,v,w)

def build_uvw_candidates(uvw_max):
    seen=set(); dirs=[]
    rng = range(-uvw_max, uvw_max+1)
    for u in rng:
        for v in rng:
            for w in rng:
                canon = canonical_uvw(u,v,w)
                if canon and canon not in seen:
                    seen.add(canon); dirs.append(canon)
    return dirs

def best_uvw_from_Astar(astar, bstar, cstar, uvw_candidates):
    """
    astar, bstar, cstar: tuples/lists (ax,ay,az) in any reciprocal unit (nm^-1 ok).
    Beam assumed along +z (CrystFEL default: fs=x, ss=y).
    Returns uvw tuple (ints) maximizing |cosine| with m = (a*_z, b*_z, c*_z).
    """
    m = np.array([astar[2], bstar[2], cstar[2]], dtype=float)  # z-components
    nm = np.linalg.norm(m)
    if not np.isfinite(nm) or nm < 1e-12:
        return None, 0.0
    m /= nm
    best=None; bestc=-1.0
    for (u,v,w) in uvw_candidates:
        vuv = np.array([u,v,w], dtype=float)
        vuv /= np.linalg.norm(vuv)
        c = abs(float(m @ vuv))
        if c > bestc:
            bestc = c; best = (u,v,w)
    return best, bestc

# ---------------------- Numba three-beam engine ----------------------
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
        if abs(h) > MAX_ABS or abs(k) > MAX_ABS or abs(l) > MAX_ABS:
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
    keys = _hkls_to_keys(hkls_int32)
    valid = np.where(keys >= 0)[0]
    keys_sorted = np.sort(keys[valid])
    n = hkls_int32.shape[0]
    count = 0
    for i in range(n):
        h1 = hkls_int32[i,0]; k1 = hkls_int32[i,1]; l1 = hkls_int32[i,2]
        for j in range(i, n):
            h2 = hkls_int32[j,0]; k2 = hkls_int32[j,1]; l2 = hkls_int32[j,2]
            hs = h1 + h2; ks = k1 + k2; ls = l1 + l2
            if (hs != 0) or (ks != 0) or (ls != 0):
                ks_key = _key_for(hs,ks,ls)
                if ks_key >= 0 and _contains(keys_sorted, ks_key):
                    count += 1
            hd = h1 - h2; kd = k1 - k2; ld = l1 - l2
            if (hd != 0) or (kd != 0) or (ld != 0):
                kd_key = _key_for(hd,kd,ld)
                if kd_key >= 0 and _contains(keys_sorted, kd_key):
                    count += 1
    return count

# ----------------------------- Core scoring -----------------------------
def rad_to_pixels(g, clen_m, wavelength_A, res_px_per_m):
    return g * clen_m * wavelength_A * res_px_per_m

def eval_frame(uvw, HKL, Gstar, params, geom):
    u,v,w = uvw
    tol_g = params["tol_g"]
    Imin  = params["i_min_rel"]
    r_edge= params["r_edge_px"]
    clen  = geom["clen_m"]
    wlA   = geom["wavelength_A"]
    respm = geom["res_px_per_m"]
    gmax  = params["g_crowd"]
    ring_mult_min = params["ring_mult_min"]
    n_min = params["n_min"]; m_min = params["m_min"]
    a = params["score_alpha"]; b = params["score_beta"]

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

    hkls_int32 = np.empty((N,3), dtype=np.int32)
    for i,(h,k,l,_) in enumerate(inzone):
        hkls_int32[i,0] = h; hkls_int32[i,1] = k; hkls_int32[i,2] = l
    M = int(three_beam_overlap_count_numba(hkls_int32))
    if M < m_min:
        return None

    # Raw score (not used for ranking when normalization enabled; kept for CSV)
    S_raw = a * N + b * M
    return {"u":u,"v":v,"w":w,"g_min_1_over_A":gmin,"r_px":rpx,"ring_mult": ring_mult,"N":N,"M":M,"Score_raw":S_raw}

# ----------------------------- Chunk parsing -----------------------------
def parse_chunk_metadata(inner):
    def gr(pat):
        m = re.search(pat, inner, re.MULTILINE)
        return m.group(1).strip() if m else None
    image = gr(r'^Image filename:\s*(.+)$')
    event = gr(r'^Event:\s*(.+)$')
    serial = gr(r'^Image serial number:\s*(\d+)$')
    return image, event, serial

def parse_chunk_cell_and_orientation(inner):
    # Cell parameters in nm and deg
    m = re.search(r'Cell parameters\s+'+_FLOAT_RE+r'\s+'+_FLOAT_RE+r'\s+'+_FLOAT_RE+r'\s+nm,\s+'
                  +_FLOAT_RE+r'\s+'+_FLOAT_RE+r'\s+'+_FLOAT_RE+r'\s+deg', inner, re.MULTILINE)
    if not m:
        return None
    a_nm, b_nm, c_nm = float(m.group(1)), float(m.group(2)), float(m.group(3))
    al, be, ga = float(m.group(4)), float(m.group(5)), float(m.group(6))
    # Convert to Å
    cell = {"a": a_nm*10.0, "b": b_nm*10.0, "c": c_nm*10.0, "al": al, "be": be, "ga": ga}

    # centering (optional; default P)
    mcent = re.search(r'^\s*centering\s*=\s*([A-Za-z])', inner, re.MULTILINE)
    centering = mcent.group(1).upper() if mcent else "P"

    # astar / bstar / cstar in nm^-1
    def vec(line_name):
        mm = re.search(rf'^{line_name}\s*=\s*{_FLOAT_RE}\s+{_FLOAT_RE}\s+{_FLOAT_RE}\s*nm\^-1', inner, re.MULTILINE)
        if not mm: return None
        return (float(mm.group(1)), float(mm.group(2)), float(mm.group(3)))
    astar = vec("astar")
    bstar = vec("bstar")
    cstar = vec("cstar")

    return cell, centering, astar, bstar, cstar

# ----------------------------- Normalization -----------------------------
def normalize_values(vals, method):
    arr = np.array(vals, dtype=float)
    if arr.size == 0:
        return arr, {}
    if method == "minmax":
        vmin = float(np.min(arr)); vmax = float(np.max(arr))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin <= 0:
            return np.zeros_like(arr), {"method":"minmax","min":vmin,"max":vmax}
        out = (arr - vmin) / (vmax - vmin)
        return out, {"method":"minmax","min":vmin,"max":vmax}
    elif method == "zscore":
        mu = float(np.mean(arr)); sd = float(np.std(arr))
        if not np.isfinite(sd) or sd <= 0:
            return np.zeros_like(arr), {"method":"zscore","mean":mu,"std":sd}
        out = (arr - mu) / sd
        return out, {"method":"zscore","mean":mu,"std":sd}
    else:  # none
        return arr, {"method":"none"}

# ----------------------------- Inject score line -----------------------------
def inject_score_line(inner, score_str):
    # Try to place after 'indexed_by' if present
    m = re.search(r'^(indexed_by\s*=\s*.*)$', inner, re.MULTILINE)
    if m:
        insert_pos = m.end()
        line_end = inner.find("\n", insert_pos)
        if line_end == -1: line_end = insert_pos
        return inner[:line_end+1] + f"Problematic Orientation Score: {score_str}\n" + inner[line_end+1:]
    # else after 'Image serial number'
    m2 = re.search(r'^(Image serial number:\s*.*)$', inner, re.MULTILINE)
    if m2:
        insert_pos = m2.end()
        line_end = inner.find("\n", insert_pos)
        if line_end == -1: line_end = insert_pos
        return inner[:line_end+1] + f"Problematic Orientation Score: {score_str}\n" + inner[line_end+1:]
    # else before '--- Begin crystal'
    m3 = re.search(r'^\s*--- Begin crystal', inner, re.MULTILINE)
    if m3:
        return inner[:m3.start()] + f"Problematic Orientation Score: {score_str}\n" + inner[m3.start():]
    # fallback: prepend
    return f"Problematic Orientation Score: {score_str}\n" + inner

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Per-frame ZOLZ scoring from CrystFEL .stream (normalizes N and M; writes scored stream).")
    ap.add_argument("--stream", required=True, help="Path to .stream, or '-' for stdin")
    ap.add_argument("--uvw-max", type=int, default=5, help="Max |u|,|v|,|w| for integer zone-axis search")
    ap.add_argument("--g-enum-scale", type=float, default=1.10, help="Enumerate HKLs up to g_enum = scale * g_edge")
    ap.add_argument("--g-max-scale", type=float, default=1.00, help="Crowding cutoff g_max = scale * g_edge")
    ap.add_argument("--i-min-rel", type=float, default=0.0)
    ap.add_argument("--ring-mult-min", type=int, default=2)
    ap.add_argument("--n-min", type=int, default=40)
    ap.add_argument("--m-min", type=int, default=8)
    ap.add_argument("--score-alpha", type=float, default=1.0)
    ap.add_argument("--score-beta",  type=float, default=0.5)
    ap.add_argument("--tol-g", type=float, default=5e-4)
    ap.add_argument("--margin-px", type=float, default=0.0)
    ap.add_argument("--norm", choices=["minmax","zscore","none"], default="minmax",
                    help="Normalization for N and M before scoring (default: minmax)")
    ap.add_argument("--score-dp", type=int, default=3, help="Decimal places when injecting score line")
    ap.add_argument("--csv", action="store_true", help="Write per-frame CSV alongside stream")
    ap.add_argument("--printresults", action="store_true", help="Print per-frame table to stdout")
    ap.add_argument("--out-stream", default=None, help="Path for scored stream output (default: <in>_scored.stream)")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    full_txt = read_text(args.stream)
    geom = parse_header_sections(full_txt)

    header_and_maybe_chunks = split_header_and_chunks(full_txt)
    if len(header_and_maybe_chunks) == 2:
        header_txt, chunk_tuples = header_and_maybe_chunks
        tail_txt = ""
    else:
        header_txt, chunk_tuples, tail_txt = header_and_maybe_chunks
    if not chunk_tuples:
        print("No chunks found; nothing to score.", file=sys.stderr)
        print(full_txt)
        return

    # Geometry-derived limits (global)
    nx = geom["max_fs"] - geom["min_fs"] + 1
    ny = geom["max_ss"] - geom["min_ss"] + 1
    r_edge_px = min(nx, ny)/2.0 - args.margin_px
    # Fix typo if any:
    try:
        pass
    except:
        pass
    r_edge_px = min(nx, ny)/2.0 - args.margin_px
    g_edge = (r_edge_px / geom["res_px_per_m"]) / (geom["clen_m"] * geom["wavelength_A"])
    g_enum = args.g_enum_scale * g_edge
    g_crowd = args.g_max_scale * g_edge
    if args.progress:
        eprint(f"Panel {nx}×{ny}px | edge r≈{r_edge_px:.2f}px; g_edge≈{g_edge:.4f} Å⁻¹ | g_enum={g_enum:.4f} | g_max={g_crowd:.4f}")
        eprint(f"Found {len(chunk_tuples)} chunks; evaluating...")

    uvw_candidates = build_uvw_candidates(args.uvw_max)
    params = {
        "tol_g": args.tol_g, "i_min_rel": args.i_min_rel,
        "r_edge_px": r_edge_px, "g_crowd": g_crowd,
        "ring_mult_min": args.ring_mult_min, "n_min": args.n_min, "m_min": args.m_min,
        "score_alpha": args.score_alpha, "score_beta": args.score_beta
    }

    # First pass: compute raw metrics per chunk
    records = []
    for idx,(begin, inner, end, span) in enumerate(chunk_tuples, 1):
        image, event, serial = parse_chunk_metadata(inner)
        parsed = parse_chunk_cell_and_orientation(inner)
        ok = False
        rec = {
            "index": idx,
            "begin": begin, "inner": inner, "end": end,
            "image": image or "", "event": event or "", "serial": int(serial) if serial else -1,
            "scored": False, "reason": ""
        }
        if parsed:
            cell, centering, astar, bstar, cstar = parsed
            if astar and bstar and cstar:
                uvw, cosim = best_uvw_from_Astar(astar, bstar, cstar, uvw_candidates)
                if uvw is not None:
                    Gstar = reciprocal_metric(cell["a"], cell["b"], cell["c"], cell["al"], cell["be"], cell["ga"])
                    HKL = enumerate_hkl_up_to_g(Gstar, g_enum, centering)
                    met = eval_frame(uvw, HKL, Gstar, params, geom)
                    if met:
                        rec.update({
                            "u":met["u"], "v":met["v"], "w":met["w"],
                            "cos_align": float(cosim),
                            "g_min": met["g_min_1_over_A"],
                            "r_px": met["r_px"], "ring_mult": met["ring_mult"],
                            "N": met["N"], "M": met["M"], "Score_raw": met["Score_raw"],
                            "scored": True
                        })
                        ok = True
        if not ok:
            rec["reason"] = "missing orientation/cell or failed criteria"
        records.append(rec)
        if args.progress and (idx % 100 == 0):
            eprint(f"  ...processed {idx} chunks")

    # Normalize N and M, compute final Score
    Ns = [r["N"] for r in records if r["scored"]]
    Ms = [r["M"] for r in records if r["scored"]]
    Nn, Nstats = normalize_values(Ns, args.norm)
    Mn, Mstats = normalize_values(Ms, args.norm)

    # Apply
    i_norm = 0
    for r in records:
        if r["scored"]:
            r["N_norm"] = float(Nn[i_norm]); r["M_norm"] = float(Mn[i_norm]); i_norm += 1
            r["Score"] = args.score_alpha * r["N_norm"] + args.score_beta * r["M_norm"]
        else:
            r["N_norm"] = float("nan"); r["M_norm"] = float("nan"); r["Score"] = float("nan")

    # Build injected chunk text and a sorted list
    scored_chunks = []
    for r in records:
        score_str = f"{r['Score']:.{args.score_dp}f}" if np.isfinite(r["Score"]) else "NA"
        inner_injected = inject_score_line(r["inner"], f"{score_str}")
        r["chunk_text"] = r["begin"] + inner_injected + r["end"] + "\n"
        scored_chunks.append(r)

    # Sort by Score (desc), NaNs last; stable for ties via original index, then serial
    def sort_key(r):
        s = r["Score"]
        key_s = (-s) if (isinstance(s, float) and np.isfinite(s)) else float("inf")
        return (key_s, r["index"])
    scored_chunks.sort(key=sort_key)

    # Assemble new stream text
    out_stream_path = args.out_stream
    if out_stream_path is None:
        base = "stdin.stream" if args.stream == "-" else os.path.basename(args.stream)
        if base.endswith(".stream"):
            out_stream_path = base[:-7] + "_scored.stream"
        else:
            out_stream_path = base + "_scored.stream"
    new_stream = [header_txt]
    new_stream.extend([r["chunk_text"] for r in scored_chunks])
    new_stream.append(tail_txt)
    new_stream_txt = "".join(new_stream)
    with open(out_stream_path, "w", encoding="utf-8") as f:
        f.write(new_stream_txt)
    print(f"Wrote scored stream (sorted, with injected scores): {out_stream_path}")

    # Optional CSV/table
    if args.csv:
        csv_path = (out_stream_path.rsplit(".stream", 1)[0]) + "_frame_scores.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "serial","image","event","u","v","w","cos_align",
                "r_px","g_min","ring_mult","N","M","N_norm","M_norm","Score","Score_raw"
            ])
            w.writeheader()
            for r in scored_chunks:
                if r["scored"]:
                    w.writerow({
                        "serial": r["serial"], "image": r["image"], "event": r["event"],
                        "u": r["u"], "v": r["v"], "w": r["w"], "cos_align": round(r["cos_align"],6),
                        "r_px": r["r_px"], "g_min": r["g_min"], "ring_mult": r["ring_mult"],
                        "N": r["N"], "M": r["M"], "N_norm": r["N_norm"], "M_norm": r["M_norm"],
                        "Score": r["Score"], "Score_raw": r["Score_raw"]
                    })
                else:
                    w.writerow({
                        "serial": r["serial"], "image": r["image"], "event": r["event"],
                        "u": "", "v": "", "w": "", "cos_align": "",
                        "r_px": "", "g_min": "", "ring_mult": "",
                        "N": "", "M": "", "N_norm": "", "M_norm": "",
                        "Score": "", "Score_raw": ""
                    })
        print(f"Wrote CSV: {csv_path}")

    if args.printresults:
        print("\nserial  (u v w)  cos  r_px    N     M   Nn    Mn   Score   image")
        for r in scored_chunks:
            if r["scored"]:
                print(f"{r['serial']:>6}  {r['u']:>2} {r['v']:>2} {r['w']:>2}  {r['cos_align']:.3f}  "
                      f"{r['r_px']:7.2f}  {r['N']:>4}  {r['M']:>4}  {r['N_norm']:.3f} {r['M_norm']:.3f}  "
                      f"{r['Score']:.3f}  {os.path.basename(r['image'])}")
            else:
                print(f"{r['serial']:>6}   -- -- --   --   --      --    --    --     --     --    {os.path.basename(r['image'])}")

    if args.progress:
        eprint(f"Done in {time.time()-t0:.1f}s. Normalization: {args.norm} "
               f"(N stats: {Nstats}; M stats: {Mstats}).")

if __name__ == "__main__":
    main()

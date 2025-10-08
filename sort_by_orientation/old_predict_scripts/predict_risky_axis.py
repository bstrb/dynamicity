#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict 'problematic' zone axes by counting ZOLZ reflections observable on the detector.

Features:
- Cell & geometry read ONLY from the stream header blocks (CrystFEL-style).
- Canonicalize [u v w] to one hemisphere (dedupe ±axes).
- Parallel evaluation over UVWs with a simple progress bar.
- First-ring must land on the detector; N = # ZOLZ reflections with |g| <= g_crowd.
- Score = N; tol_g fixed at 0.1; g_enum = 1.10*g_edge; g_crowd = g_edge.

CLI:
  --stream PATH         CrystFEL .stream file
  --uvw-max INT         max |u|,|v|,|w| (default: 10)
  --csv                 write CSV next to stream
  --nrows INT           limit printed/saved rows

Output columns:
  u,v,w, ring_type, g_min_1_over_A, r_px, ring_mult, N, Score, inside
"""

import argparse, math, re, csv, sys, os
from math import sqrt, floor, gcd
import numpy as np
import multiprocessing as mp

# ----------------------------- Utilities -----------------------------

_FLOAT_RE = r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

def fail(msg: str):
    eprint(f"[error] {msg}")
    sys.exit(1)

def _read_text(path_or_dash: str) -> str:
    if path_or_dash == "-" or path_or_dash is None:
        return sys.stdin.read()
    with open(path_or_dash, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def _extract_block(txt: str, begin_marker: str, end_marker: str) -> str:
    begin_re = re.compile(rf'^\s*{re.escape(begin_marker)}\s*$', re.MULTILINE)
    end_re   = re.compile(rf'^\s*{re.escape(end_marker)}\s*$', re.MULTILINE)
    mb = begin_re.search(txt); me = end_re.search(txt) if mb else None
    if mb and me and me.start() > mb.end():
        return txt[mb.end():me.start()]
    # fallback substring search
    bpos = txt.find(begin_marker)
    if bpos != -1:
        epos = txt.find(end_marker, bpos + len(begin_marker))
        if epos != -1 and epos > bpos:
            bend = txt.find("\n", bpos); bend = bend if bend != -1 else bpos + len(begin_marker)
            estart = txt.rfind("\n", 0, epos); estart = estart if estart != -1 else epos
            if estart > bend:
                return txt[bend:estart]
    raise ValueError(f"Could not find well-formed block: '{begin_marker}' .. '{end_marker}'")

class ProgressBar:
    def __init__(self, total, width=42, label=""):
        self.total = max(1, int(total))
        self.width = max(10, int(width))
        self.count = 0
        self.label = label
        self._render()
    def update(self, n=1):
        self.count = min(self.total, self.count + n)
        self._render()
    def _render(self):
        frac = self.count / self.total
        filled = int(self.width * frac)
        bar = "█" * filled + " " * (self.width - filled)
        eprint(f"\r{self.label}[{bar}] {self.count}/{self.total}", end="")
    def finish(self):
        self._render()
        eprint("")

# ----------------------------- Parsing -----------------------------

def parse_cell_text(txt):
    """
    Parse ONLY from the stream's 'Begin unit cell ... End unit cell' block.
    Expected keys (CrystFEL .cell style inside the block):
      a,b,c (A), al,be,ga (deg), lattice_type, centering
    """
    def grab_num(key, unit=None):
        if unit:
            m = re.search(rf'^{key}\s*=\s*{_FLOAT_RE}\s*{re.escape(unit)}', txt, re.MULTILINE)
        else:
            m = re.search(rf'^{key}\s*=\s*{_FLOAT_RE}', txt, re.MULTILINE)
        return float(m.group(1)) if m else None
    def grab_str(key):
        m = re.search(rf'^{key}\s*=\s*([A-Za-z]+)', txt, re.MULTILINE)
        return m.group(1).strip() if m else None

    cell = {
        "lattice_type": (grab_str("lattice_type") or "").strip().lower(),
        "centering":    (grab_str("centering") or "P").strip().upper(),
        "a":  grab_num("a","A"),
        "b":  grab_num("b","A"),
        "c":  grab_num("c","A"),
        "al": grab_num("al","deg"),
        "be": grab_num("be","deg"),
        "ga": grab_num("ga","deg"),
    }
    for k in ("a","b","c","al","be","ga"):
        if cell[k] is None:
            raise ValueError(f"Missing '{k}' in unit cell header")
    return cell

def parse_geom_text(txt):
    """
    Parse ONLY from the stream's 'Begin geometry file ... End geometry file' block.
    Needs: wavelength (A), clen (m), res (px/m), p*/min_fs/max_fs, p*/min_ss/max_ss
    """
    def grab_float(key, unit=None):
        if unit:
            m = re.search(rf'^{re.escape(key)}\s*=\s*{_FLOAT_RE}\s*{re.escape(unit)}', txt, re.MULTILINE)
        else:
            m = re.search(rf'^{re.escape(key)}\s*=\s*{_FLOAT_RE}\s*$', txt, re.MULTILINE)
        return float(m.group(1)) if m else None

    geom = {
        "wavelength_A": grab_float("wavelength","A"),
        "clen_m":       grab_float("clen","m"),
        "res_px_per_m": grab_float("res")
    }
    for k in ("wavelength_A","clen_m","res_px_per_m"):
        if geom[k] is None:
            raise ValueError(f"Missing '{k}' in geometry header")

    # derive nx, ny from any panels present
    fs_min, fs_max, ss_min, ss_max = None, None, None, None
    for line in txt.splitlines():
        s = line.strip()
        m = re.match(r'^p\d+/(min_fs|max_fs|min_ss|max_ss)\s*=\s*(\d+)', s)
        if not m: 
            continue
        key = m.group(1); val = int(m.group(2))
        if key == "min_fs":
            fs_min = val if fs_min is None else min(fs_min, val)
        elif key == "max_fs":
            fs_max = val if fs_max is None else max(fs_max, val)
        elif key == "min_ss":
            ss_min = val if ss_min is None else min(ss_min, val)
        elif key == "max_ss":
            ss_max = val if ss_max is None else max(ss_max, val)

    if fs_min is None or fs_max is None or ss_min is None or ss_max is None:
        raise ValueError("Missing panel extents (p*/min_fs/max_fs, p*/min_ss/max_ss) in geometry header")

    geom["nx"] = fs_max - fs_min + 1
    geom["ny"] = ss_max - ss_min + 1
    return geom

def parse_stream_headers(path_or_dash: str):
    txt = _read_text(path_or_dash)
    geom_txt = _extract_block(txt, "----- Begin geometry file -----", "----- End geometry file -----")
    cell_txt = _extract_block(txt, "----- Begin unit cell -----", "----- End unit cell -----")
    return parse_cell_text(cell_txt), parse_geom_text(geom_txt)

# ----------------------- Cell & reciprocal -----------------------

def deg2rad(x): return x*math.pi/180.0

def cell_matrices(a,b,c,al_deg,be_deg,ga_deg):
    """Direct A and reciprocal A* (no 2π; |a*|=1/a for orthogonal)."""
    al = deg2rad(al_deg); be = deg2rad(be_deg); ga = deg2rad(ga_deg)
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b*math.cos(ga), b*math.sin(ga), 0.0])
    cx = c*math.cos(be)
    cy = c*(math.cos(al) - math.cos(be)*math.cos(ga))/max(1e-12, math.sin(ga))
    cz2 = max(0.0, c*c - cx*cx - cy*cy)
    vc = np.array([cx, cy, math.sqrt(cz2)])
    A = np.column_stack([va, vb, vc])
    AinvT = np.linalg.inv(A).T
    Astar = AinvT
    return A, Astar

# ----------------------- UVW generation -----------------------

def gcd3(x,y,z): return gcd(gcd(abs(x),abs(y)),abs(z))

def canonical_uvw(u,v,w):
    if u==0 and v==0 and w==0:
        return None
    g = gcd3(u,v,w)
    if g == 0:
        return None
    u, v, w = u//g, v//g, w//g
    # Hemisphere representative: keep only one of ±[u v w]
    if (w < 0) or (w == 0 and (v < 0 or (v == 0 and u < 0))):
        u, v, w = -u, -v, -w
    return (u, v, w)

def unique_uvw_list(uvw_max: int):
    if uvw_max < 1:
        return []
    seen = set()
    out = []
    for u in range(-uvw_max, uvw_max+1):
        for v in range(-uvw_max, uvw_max+1):
            for w in range(-uvw_max, uvw_max+1):
                canon = canonical_uvw(u, v, w)
                if canon and canon not in seen:
                    seen.add(canon)
                    out.append(canon)
    return out

# ----------------------- ZOLZ enumeration -----------------------

def hkl_bounds_for_g(Astar: np.ndarray, g_enum: float):
    """Conservative per-axis HKL bounds so that |g| <= g_enum is reachable."""
    astar = Astar[:,0]; bstar = Astar[:,1]; cstar = Astar[:,2]
    H = max(1, int(math.ceil(g_enum / max(1e-12, np.linalg.norm(astar)))))
    K = max(1, int(math.ceil(g_enum / max(1e-12, np.linalg.norm(bstar)))))
    L = max(1, int(math.ceil(g_enum / max(1e-12, np.linalg.norm(cstar)))))
    return H+2, K+2, L+2

def enumerate_zolz_hkls(uvw, H, K, L):
    """Yield integer HKLs that satisfy the zone law h*u + k*v + l*w = 0 (exact integer test)."""
    u,v,w = uvw
    for h in range(-H, H+1):
        for k in range(-K, K+1):
            for l in range(-L, L+1):
                if (h==0 and k==0 and l==0): 
                    continue
                if (h*u + k*v + l*w) == 0:
                    yield (h,k,l)

def first_ring_radius_and_mult(hkls, Astar: np.ndarray, tol_g: float):
    """Find smallest non-zero |g| and multiplicity within [gmin, gmin+tol_g]."""
    if not hkls:
        return float('inf'), 0
    g_lens = []
    for h,k,l in hkls:
        g = Astar @ np.array([h,k,l], dtype=float)
        g_lens.append(float(np.linalg.norm(g)))
    g_lens = np.array([x for x in g_lens if x > 1e-12], dtype=float)
    if g_lens.size == 0:
        return float('inf'), 0
    gmin = float(np.min(g_lens))
    ring_mult = int(np.sum((g_lens >= gmin - 1e-12) & (g_lens <= gmin + tol_g)))
    return gmin, ring_mult

def rpx_from_g(g: float, clen_m: float, wavelength_A: float, res_px_per_m: float) -> float:
    # small-angle mapping: r_px = g * L * λ * (px/m)
    return float(g * clen_m * wavelength_A * res_px_per_m)

# ----------------------- Parallel worker -----------------------

# Globals populated by _worker_init so each worker avoids heavy pickling
_G = {}

def _worker_init(Astar, nx, ny, clen_m, wavelength_A, res_px_per_m, g_enum, g_crowd, tol_g):
    _G["Astar"] = Astar
    _G["nx"] = nx; _G["ny"] = ny
    _G["clen_m"] = clen_m
    _G["wavelength_A"] = wavelength_A
    _G["res_px_per_m"] = res_px_per_m
    _G["g_enum"] = g_enum
    _G["g_crowd"] = g_crowd
    _G["tol_g"] = tol_g

def _process_one_uvw(uvw):
    Astar = _G["Astar"]
    nx, ny = _G["nx"], _G["ny"]
    clen_m = _G["clen_m"]
    wavelength_A = _G["wavelength_A"]
    res_px_per_m = _G["res_px_per_m"]
    g_enum = _G["g_enum"]
    g_crowd = _G["g_crowd"]
    tol_g = _G["tol_g"]

    # Detector edge and acceptance (no margin)
    r_edge_px = min(nx, ny) / 2.0

    # index bounds and ZOLZ enumeration (cheap integer filter first)
    H, K, L = hkl_bounds_for_g(Astar, g_enum)
    hkls_zolz = list(enumerate_zolz_hkls(uvw, H, K, L))
    if not hkls_zolz:
        return None

    # First ring
    gmin, ring_mult = first_ring_radius_and_mult(hkls_zolz, Astar, tol_g)
    if not math.isfinite(gmin):
        return None

    r_first_px = rpx_from_g(gmin, clen_m, wavelength_A, res_px_per_m)
    if r_first_px > r_edge_px + 1e-9:
        return None

    # Count N inside g_crowd
    N = 0
    for h,k,l in hkls_zolz:
        gvec = Astar @ np.array([h,k,l], dtype=float)
        if np.linalg.norm(gvec) <= g_crowd + 1e-12:
            N += 1

    return {
        "u": int(uvw[0]),
        "v": int(uvw[1]),
        "w": int(uvw[2]),
        "ring_type": "ZOLZ",
        "g_min_1_over_A": float(gmin),
        "r_px": float(r_first_px),
        "ring_mult": int(ring_mult),
        "N": int(N),
        "Score": float(N),
        "inside": True
    }

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Predict problematic ZOLZ axes (Score=N, parallel).")
    ap.add_argument("--stream", required=True, help="Path to .stream file, or '-' for stdin")
    ap.add_argument("--uvw-max", type=int, default=10, help="Max |u|,|v|,|w| (default: 10)")
    ap.add_argument("--csv", action="store_true", help="Write CSV next to stream")
    ap.add_argument("--nrows", type=int, default=None, help="Max rows to print/save (default: all)")
    args = ap.parse_args()

    # Read headers
    try:
        cell, geom = parse_stream_headers(args.stream)
    except Exception as ex:
        fail(str(ex))

    # Direct & reciprocal bases
    A, Astar = cell_matrices(cell["a"], cell["b"], cell["c"], cell["al"], cell["be"], cell["ga"])

    # Geometry & reciprocal limits
    nx = geom["nx"]; ny = geom["ny"]
    r_edge_px = min(nx, ny) / 2.0
    g_edge = (r_edge_px / geom["res_px_per_m"]) / (geom["clen_m"] * geom["wavelength_A"])
    g_enum = 1.10 * g_edge
    g_crowd = g_edge
    tol_g = 0.1

    # UVW list (hemisphere canonicalized)
    dirs = unique_uvw_list(args.uvw_max)
    total = len(dirs)

    # Headers
    print(f"Cell: {cell['lattice_type'].upper()} {cell['centering'].upper()} | "
          f"a={cell['a']:.4f} Å b={cell['b']:.4f} Å c={cell['c']:.4f} Å "
          f"al={cell['al']:.2f}° be={cell['be']:.2f}° ga={cell['ga']:.2f}°")
    print(f"Geom: λ={geom['wavelength_A']} Å, L={geom['clen_m']} m, res={geom['res_px_per_m']} px/m | "
          f"panel {nx}×{ny}px | edge r={r_edge_px:.2f}px")
    print(f"Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.4f}, g_crowd={g_crowd:.4f}, tol_g={tol_g} (fixed), Score=N\n")

    # Parallel run
    eprint(f"[1/2] Scanning {total} directions in parallel...")
    pb = ProgressBar(total, label="      Progress ")

    rows = []
    # Prefer 'fork' to reduce overhead when available, else fall back
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()  # default on platform

    with ctx.Pool(processes=os.cpu_count(),
                  initializer=_worker_init,
                  initargs=(Astar, nx, ny, geom["clen_m"], geom["wavelength_A"],
                            geom["res_px_per_m"], g_enum, g_crowd, tol_g)) as pool:

        # imap_unordered so we can update the progress bar as results come in
        for rec in pool.imap_unordered(_process_one_uvw, dirs, chunksize=8):
            if rec:
                rows.append(rec)
            pb.update(1)
    pb.finish()

    if not rows:
        fail("No candidate axes survived (first ring off-panel or no ZOLZ reflections in acceptance).")
    
    # ---- Normalize Score to [0,1] from raw N (keep N as-is) ----
    Nmax = max(r["N"] for r in rows)
    if Nmax > 0:
        for r in rows:
            r["Score"] = r["N"] / Nmax
    else:
        for r in rows:
            r["Score"] = 0.0

    eprint(f"[2/2] Reducing & sorting {len(rows)} results...")

    # Sort by N desc, then r_px asc, then simpler indices
    rows.sort(key=lambda r: (-r["N"], r["r_px"], abs(r["u"])+abs(r["v"])+abs(r["w"]), r["u"], r["v"], r["w"]))

    # Truncate
    if args.nrows is not None and args.nrows > 0:
        rows = rows[:args.nrows]

    # Print table
    print("(u v w)   r_px    g_min(1/Å)  ring_mult     N    Score")
    for r in rows:
        print(f"{r['u']:>2} {r['v']:>2} {r['w']:>2}  {r['r_px']:7.2f}   {r['g_min_1_over_A']:.4f}     {r['ring_mult']:>3}   {r['N']:>5}   {r['Score']:>7.2f}")

    # CSV
    # if args.csv:
    #     out_csv = os.path.splitext(args.stream if args.stream != "-" else "stdin.stream")[0] + "_problematic_axes.csv"
    #     with open(out_csv, "w", newline="") as f:
    #         w = csv.DictWriter(f, fieldnames=["u","v","w","ring_type","g_min_1_over_A","r_px","ring_mult","N","Score","inside"])
    #         w.writeheader()
    #         for r in rows:
    #             w.writerow(r)
    #     print(f"\nWrote CSV: {out_csv}")
        # CSV (with header comments + trailing triplet list like the original)
    if args.csv:
        out_csv = os.path.splitext(args.stream if args.stream != "-" else "stdin.stream")[0] + "_problematic_axes.csv"
        with open(out_csv, "w", newline="") as f:
            # --- header comments ---
            f.write("# CrystFEL problematic zone axes prediction\n")
            f.write(f"# Stream: {args.stream}\n")
            f.write(f"# Cell: {cell['lattice_type'].upper()} {cell['centering'].upper()} | "
                    f"a={cell['a']:.4f} Å b={cell['b']:.4f} Å c={cell['c']:.4f} Å "
                    f"al={cell['al']:.2f}° be={cell['be']:.2f}° ga={cell['ga']:.2f}°\n")
            f.write(f"# Geom: λ={geom['wavelength_A']} Å, L={geom['clen_m']} m, res={geom['res_px_per_m']} px/m | "
                    f"panel {nx}×{ny}px | edge r={r_edge_px:.2f}px\n")
            f.write(f"# Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.4f}, g_crowd={g_crowd:.4f}, "
                    f"tol_g={tol_g} (fixed), Score=N\n")

            # --- table ---
            w = csv.DictWriter(
                f,
                fieldnames=["u","v","w","ring_type","g_min_1_over_A","r_px","ring_mult","N","Score","inside"]
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)

            # --- trailing triplet list (quoted, space-separated) ---
            triplets = [f"\"{r['u']:>2} {r['v']:>2} {r['w']:>2}\"" for r in rows]
            f.write("\n# Problematic axis triplets listed:\n")
            f.write("# " + " ".join(triplets) + "\n")

        print(f"\nWrote CSV: {out_csv}")


    eprint("Done.")

if __name__ == "__main__":
    main()

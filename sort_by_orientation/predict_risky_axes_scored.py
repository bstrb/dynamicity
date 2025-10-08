#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict 'problematic' zone axes by counting ZOLZ reflections observable on the detector,
with centering extinctions, excitation-error gating, low-order weighting, and area normalization.

New flags:
  --smax FLOAT          Excitation error cap S_max in 1/Å (Bragg/relrod gate)
  --thickness-nm FLOAT  Crystal thickness (nm); uses S_max ≈ 1/(2t) with t in Å
  --g0 FLOAT            Low-order weighting scale g0 in 1/Å (default: 0.40)
  --p FLOAT             Low-order weighting exponent p (default: 1.5)

Outputs add:
  N_raw   : unweighted count after extinctions + S gate (ZOLZ only)
  N_w     : weighted sum Σ w(g), w(g) = 1/(1+(g/g0)^p)
  rho_w   : N_w   / (π r_eff_px^2)
  Score   : normalized rho_w in [0,1] (used for sorting)

Notes:
- ZOLZ only (HOLZ not included here).
- Excitation error S ≈ g_perp^2 / (2k) for ZOLZ (g_z = 0). k = 1/λ with λ in Å.
- Effective radius r_eff_px = min(panel_edge_px, r_px(S_max)), where r_px(S_max) maps
  g_max = sqrt(2 k S_max) into pixels. If no S gate is provided, r_eff_px = panel_edge_px.
"""

import argparse, math, re, csv, sys, os
from math import sqrt, gcd
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

    fs_min=fs_max=ss_min=ss_max=None
    for line in txt.splitlines():
        s = line.strip()
        m = re.match(r'^p\d+/(min_fs|max_fs|min_ss|max_ss)\s*=\s*(\d+)', s)
        if not m:
            continue
        key = m.group(1); val = int(m.group(2))
        if   key=="min_fs": fs_min = val if fs_min is None else min(fs_min, val)
        elif key=="max_fs": fs_max = val if fs_max is None else max(fs_max, val)
        elif key=="min_ss": ss_min = val if ss_min is None else min(ss_min, val)
        elif key=="max_ss": ss_max = val if ss_max is None else max(ss_max, val)

    if None in (fs_min, fs_max, ss_min, ss_max):
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

# ----------------------- Lattice centering rules -----------------------

def allowed_by_centering(h: int, k: int, l: int, centering: str) -> bool:
    c = (centering or "P").upper()
    if c == "P":
        return True
    elif c == "A":
        return ((k + l) & 1) == 0
    elif c == "B":
        return ((h + l) & 1) == 0
    elif c == "C":
        return ((h + k) & 1) == 0
    elif c == "I":
        return ((h + k + l) & 1) == 0
    elif c == "F":
        hp = h & 1; kp = k & 1; lp = l & 1
        return (hp == kp == lp)
    elif c == "R":
        # hexagonal axes assumption
        return ((h - k) % 3 == 0) and ((k - l) % 3 == 0)
    else:
        return True  # permissive for unknown codes

# ----------------------- ZOLZ enumeration -----------------------

def hkl_bounds_for_g(Astar: np.ndarray, g_enum: float):
    astar = Astar[:,0]; bstar = Astar[:,1]; cstar = Astar[:,2]
    H = max(1, int(math.ceil(g_enum / max(1e-12, np.linalg.norm(astar)))))
    K = max(1, int(math.ceil(g_enum / max(1e-12, np.linalg.norm(bstar)))))
    L = max(1, int(math.ceil(g_enum / max(1e-12, np.linalg.norm(cstar)))))
    return H+2, K+2, L+2

def enumerate_zolz_hkls(uvw, H, K, L):
    u,v,w = uvw
    for h in range(-H, H+1):
        for k in range(-K, K+1):
            for l in range(-L, L+1):
                if (h==0 and k==0 and l==0):
                    continue
                if (h*u + k*v + l*w) == 0:
                    yield (h,k,l)

def first_ring_radius_and_mult(hkls, Astar: np.ndarray, tol_g: float):
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
    return float(g * clen_m * wavelength_A * res_px_per_m)

# ----------------------- Parallel worker -----------------------

_G = {}

def _worker_init(Astar, nx, ny, clen_m, wavelength_A, res_px_per_m,
                 g_enum, g_crowd, tol_g, centering,
                 use_S, Smax, k_eVperA, g0, p):
    _G["Astar"] = Astar
    _G["nx"] = nx; _G["ny"] = ny
    _G["clen_m"] = clen_m
    _G["wavelength_A"] = wavelength_A
    _G["res_px_per_m"] = res_px_per_m
    _G["g_enum"] = g_enum
    _G["g_crowd"] = g_crowd
    _G["tol_g"] = tol_g
    _G["centering"] = centering
    _G["use_S"] = use_S
    _G["Smax"] = Smax
    # _G["k"] = 1.0 / max(1e-12, wavelength_A)  # 1/Å
    _G["k_wave"] = 1.0 / max(1e-12, wavelength_A)  # 1/Å   (rename key)
    _G["g0"] = g0
    _G["p"] = p

def _process_one_uvw(uvw):
    Astar = _G["Astar"]
    nx, ny = _G["nx"], _G["ny"]
    clen_m = _G["clen_m"]
    wavelength_A = _G["wavelength_A"]
    res_px_per_m = _G["res_px_per_m"]
    g_enum = _G["g_enum"]
    g_crowd = _G["g_crowd"]
    tol_g = _G["tol_g"]
    centering = _G["centering"]
    use_S = _G["use_S"]; Smax = _G["Smax"]; 
    # k = _G["k"]
    k_wave = _G["k_wave"]   # rename local
    g0 = _G["g0"]; p = _G["p"]

    # Panel edge radius in pixels
    r_edge_px = min(nx, ny) / 2.0

    # Enumerate ZOLZ (integer zone-law)
    H, K, L = hkl_bounds_for_g(Astar, g_enum)
    hkls_zolz = list(enumerate_zolz_hkls(uvw, H, K, L))
    if not hkls_zolz:
        return None

    # Apply lattice-centering extinctions
    hkls_allowed = [(h,k,l) for (h,k,l) in hkls_zolz if allowed_by_centering(h,k,l,centering)]
    if not hkls_allowed:
        return None

    # First ring AFTER extinctions
    gmin, ring_mult = first_ring_radius_and_mult(hkls_allowed, Astar, tol_g)
    if not math.isfinite(gmin):
        return None
    r_first_px = rpx_from_g(gmin, clen_m, wavelength_A, res_px_per_m)
    if r_first_px > r_edge_px + 1e-9:
        return None

    # --- Excitation-error gating (ZOLZ: g_z ~ 0 => S ≈ g^2 / (2k)) ---
    def pass_excitation(glen: float) -> bool:
        if not use_S:
            return True
        S = (glen * glen) / (2.0 * k_wave)  # 1/Å  (use k_wave here)
        return (S <= Smax + 1e-15)

    # Effective acceptance limit from Smax (if any)
    if use_S:
        g_max = math.sqrt(max(0.0, 2.0 * k_wave * Smax))
        r_S_px = rpx_from_g(g_max, clen_m, wavelength_A, res_px_per_m)
        r_eff_px = min(r_edge_px, r_S_px)
        g_eff_max = min(g_crowd, g_max)
    else:
        r_eff_px = r_edge_px
        g_eff_max = g_crowd

    # Count within effective acceptance (post-extinction, post-S gate)
    N_raw = 0
    N_w = 0.0
    for h,kk,l in hkls_allowed:
        gvec = Astar @ np.array([h,kk,l], dtype=float)
        glen = float(np.linalg.norm(gvec))
        if glen <= g_eff_max + 1e-12 and pass_excitation(glen):
            N_raw += 1
            # low-order weight
            w = 1.0 / (1.0 + (glen / max(1e-12, g0))**p)
            N_w += w

    if N_raw == 0 and N_w <= 0.0:
        return None
    area_px = math.pi * (r_eff_px ** 2)
    rho_w   = (N_w / area_px) if area_px > 0 else 0.0

    return {
        "u": int(uvw[0]), "v": int(uvw[1]), "w": int(uvw[2]),
        "g_min_1_over_A": float(gmin),
        "r_px": float(r_first_px),
        "ring_mult": int(ring_mult),
        "N_raw": int(N_raw),
        "N_w": float(N_w),
        "rho_w": float(rho_w),     # used later for Score
        "Score": float(rho_w)      # temp; normalized later
    }

# ----------------------------- Main -----------------------------
# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Predict problematic ZOLZ axes (centering + S-gate + low-order weighting + area norm).")
    ap.add_argument("--stream", required=True, help="Path to .stream file, or '-' for stdin")
    ap.add_argument("--uvw-max", type=int, default=5, help="Max |u|,|v|,|w| (default: 5)")
    ap.add_argument("--nrows", type=int, default=None, help="Max rows to print/save (default: all)")
    ap.add_argument("--thickness-nm", type=float, default=None, help="Crystal thickness in nm (S_max ≈ 1/(2t) with t in Å)")
    ap.add_argument("--g0", type=float, default=0.40, help="Low-order weighting scale g0 (1/Å)")
    ap.add_argument("--p",  type=float, default=1.5,  help="Low-order weighting exponent p")
    args = ap.parse_args()

    # Read headers
    try:
        cell, geom = parse_stream_headers(args.stream)
    except Exception as ex:
        fail(str(ex))

    centering = (cell.get("centering") or "P").upper()

    # Direct & reciprocal bases
    A, Astar = cell_matrices(cell["a"], cell["b"], cell["c"], cell["al"], cell["be"], cell["ga"])

    # Geometry & reciprocal limits
    nx = geom["nx"]; ny = geom["ny"]
    r_edge_px = min(nx, ny) / 2.0
    g_edge = (r_edge_px / geom["res_px_per_m"]) / (geom["clen_m"] * geom["wavelength_A"])
    g_enum = 1.10 * g_edge
    g_crowd = g_edge
    tol_g = 0.1

    # Excitation-error cap Smax
    Smax = None
    if args.thickness_nm is not None:  # argparse converts '-' to '_'
        t_nm = float(args.thickness_nm)
        t_A = t_nm * 10.0
        if t_A <= 0:
            fail("thickness must be > 0")
        Smax = 1.0 / (2.0 * t_A)  # 1/Å
    use_S = (Smax is not None)
    if not use_S:
        Smax = 0.0  # placeholder

    # UVW list (hemisphere canonicalized)
    dirs = unique_uvw_list(args.uvw_max)
    total = len(dirs)

    # Headers
    print(f"Cell: {cell['lattice_type'].upper()} {centering} | "
          f"a={cell['a']:.4f} Å b={cell['b']:.4f} Å c={cell['c']:.4f} Å "
          f"al={cell['al']:.2f}° be={cell['be']:.2f}° ga={cell['ga']:.2f}°")
    print(f"Geom: λ={geom['wavelength_A']} Å, L={geom['clen_m']} m, res={geom['res_px_per_m']} px/m | "
          f"panel {nx}×{ny}px | edge r={r_edge_px:.2f}px")
    if use_S:
        k_wave_hdr = 1.0 / geom["wavelength_A"]
        g_max = math.sqrt(max(0.0, 2.0 * k_wave_hdr * Smax))
        r_S_px = rpx_from_g(g_max, geom["clen_m"], geom["wavelength_A"], geom["res_px_per_m"])
        r_eff = min(r_edge_px, r_S_px)
        print(f"Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.4f}, g_crowd={g_crowd:.4f}, tol_g={tol_g}, "
              f"centering={centering}, S_max={Smax:.5f} 1/Å -> r_eff≈{r_eff:.2f}px, g0={args.g0}, p={args.p}\n")
    else:
        print(f"Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.4f}, g_crowd={g_crowd:.4f}, tol_g={tol_g}, "
              f"centering={centering}, S_max=None, g0={args.g0}, p={args.p}\n")

    # Parallel run
    eprint(f"[1/2] Scanning {total} directions in parallel...")
    pb = ProgressBar(total, label="      Progress ")

    rows = []
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    with ctx.Pool(processes=os.cpu_count(),
                  initializer=_worker_init,
                  initargs=(Astar, nx, ny, geom["clen_m"], geom["wavelength_A"],
                            geom["res_px_per_m"], g_enum, g_crowd, tol_g, centering,
                            use_S, Smax, None, args.g0, args.p)) as pool:

        for rec in pool.imap_unordered(_process_one_uvw, dirs, chunksize=8):
            if rec:
                rows.append(rec)
            pb.update(1)
    pb.finish()

    if not rows:
        fail("No candidate axes survived (filters too strict or first ring off-panel).")

    eprint(f"[2/2] Reducing & sorting {len(rows)} results...")

    # 1) Sort by rho_w (raw density) descending first to pick the relevant set
    rows.sort(key=lambda r: (-r["rho_w"], r["r_px"],
                             abs(r["u"])+abs(r["v"])+abs(r["w"]), r["u"], r["v"], r["w"]))

    # 2) Truncate to nrows BEFORE normalization
    if args.nrows is not None and args.nrows > 0:
        rows = rows[:args.nrows]

    # 3) Min–max normalize Score *within the shown subset*
    rho_min = min(r["rho_w"] for r in rows)
    rho_max = max(r["rho_w"] for r in rows)
    den = (rho_max - rho_min)
    for r in rows:
        r["Score"] = 0.0 if den <= 0.0 else (r["rho_w"] - rho_min) / den

    # 4) Now sort by Score (desc), then tie-breakers
    rows.sort(key=lambda r: (-r["Score"], -r["rho_w"], r["r_px"],
                             abs(r["u"])+abs(r["v"])+abs(r["w"]), r["u"], r["v"], r["w"]))

    # Print table (reduced set)
    print("(u v w)   r_px   g_min(1/Å)  ring_mult   N_raw     N_w    Score")
    for r in rows:
        print(f"{r['u']:>2} {r['v']:>2} {r['w']:>2}  {r['r_px']:7.2f}   {r['g_min_1_over_A']:.4f} "
              f"    {r['ring_mult']:>3}   {r['N_raw']:>6}  {r['N_w']:>7.2f}  {r['Score']:>6.3f}")

    out_csv = os.path.splitext(args.stream if args.stream != "-" else "stdin.stream")[0] + "_problematic_axes_scored.csv"
    with open(out_csv, "w", newline="") as f:
        f.write("# CrystFEL problematic zone axes prediction (centering + S-gate + low-order weight + area norm)\n")
        f.write(f"# Stream: {args.stream}\n")
        f.write(f"# Cell: {cell['lattice_type'].upper()} {centering} | "
                f"a={cell['a']:.4f} Å b={cell['b']:.4f} Å c={cell['c']:.4f} Å "
                f"al={cell['al']:.2f}° be={cell['be']:.2f}° ga={cell['ga']:.2f}°\n")
        f.write(f"# Geom: λ={geom['wavelength_A']} Å, L={geom['clen_m']} m, res={geom['res_px_per_m']} px/m | "
                f"panel {nx}×{ny}px | edge r={r_edge_px:.2f}px\n")
        if use_S:
            f.write(f"# Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.4f}, g_crowd={g_crowd:.4f}, tol_g={tol_g}, "
                    f"centering={centering}, S_max={Smax:.5f} 1/Å, g0={args.g0}, p={args.p}\n")
        else:
            f.write(f"# Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.4f}, g_crowd={g_crowd:.4f}, tol_g={tol_g}, "
                    f"centering={centering}, S_max=None, g0={args.g0}, p={args.p}\n")

        fieldnames = ["u","v","w","g_min_1_over_A","r_px","ring_mult","N_raw","N_w","Score"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

        triplets = [f"\"{r['u']:>2} {r['v']:>2} {r['w']:>2}\"" for r in rows]
        f.write("\n# Problematic axis triplets listed (sorted by Score):\n")
        f.write("# " + " ".join(triplets) + "\n")

    print(f"\nWrote CSV: {out_csv}")
    eprint("Done.")


if __name__ == "__main__":
    main()

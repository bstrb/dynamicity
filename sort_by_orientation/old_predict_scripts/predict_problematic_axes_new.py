#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict ZOLZ-centered, visually "problematic" zone axes from a CrystFEL stream header.

Parses BOTH unit cell and geometry from the header blocks of a CrystFEL .stream:
    ----- Begin geometry file -----
    ... (CrystFEL .geom-style lines) ...
    ----- End geometry file -----
    ----- Begin unit cell -----
    ... (CrystFEL .cell-style lines) ...
    ----- End unit cell -----

Differences vs earlier version:
- Input is a single --stream (path or "-" for stdin) instead of --cell/--geom
- ZOLZ-only default (HOLZ optional)
- Canonicalization tightened (unique star representative)
- Crowdedness gate (N, M) + Score S = alpha*N + beta*M
- First-ring multiplicity gate (number of distinct reflections at |g|min)
- g_max for crowdedness derived from detector edge (unless overridden)
- 2025-08-20: optional progress/timing to stderr (use --progress).
- 2025-08-20: **robust header block extraction** (tolerates leading/trailing spaces).

Author: Buster Blomberg (with progress/timing + robust header parsing)
"""

import argparse, math, re, csv, sys, os, time
from math import sqrt, floor, gcd

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
    """
    Extract text between lines matching the begin/end markers, robust to:
    - leading/trailing whitespace
    - CRLF vs LF
    - occasional stray characters around whitespace

    Strategy:
      1) Try strict-ish regex: ^\s*<marker>\s*$
      2) Fallback to substring search: find first occurrence of <marker> and next of end_marker
    """
    # 1) tolerant line-anchored regex
    begin_re = re.compile(rf'^\s*{re.escape(begin_marker)}\s*$', re.MULTILINE)
    end_re   = re.compile(rf'^\s*{re.escape(end_marker)}\s*$', re.MULTILINE)

    mb = begin_re.search(txt)
    me = end_re.search(txt) if mb else None
    if mb and me and me.start() > mb.end():
        return txt[mb.end():me.start()]

    # 2) fallback: substring search (handles weird whitespace or minor corruption)
    bpos = txt.find(begin_marker)
    if bpos != -1:
        # find end marker AFTER begin
        epos = txt.find(end_marker, bpos + len(begin_marker))
        if epos != -1 and epos > bpos:
            # cut to line ends to avoid partial-line artifacts
            # move begin to end-of-line
            bend = txt.find("\n", bpos)
            if bend == -1: bend = bpos + len(begin_marker)
            # find start-of-line for end marker
            estart = txt.rfind("\n", 0, epos)
            if estart == -1: estart = epos
            if estart > bend:
                return txt[bend:estart]

    raise ValueError(f"Could not find well-formed block: '{begin_marker}' .. '{end_marker}'")

# ----------------------------- Parsing -----------------------------

def parse_cell_text(txt):
    cell = {
        "lattice_type": None,
        "centering": "P",
        "a": None, "b": None, "c": None,
        "al": None, "be": None, "ga": None
    }

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
    cell["a"]  = grab("a","A"); cell["b"]  = grab("b","A"); cell["c"]  = grab("c","A")
    cell["al"] = grab("al","deg"); cell["be"] = grab("be","deg"); cell["ga"] = grab("ga","deg")
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
        if geom[k] is None:
            raise ValueError(f"Missing '{k}' in geometry header")
    return geom

def parse_stream_headers(path_or_dash: str):
    txt = _read_text(path_or_dash)
    geom_txt = _extract_block(txt, "----- Begin geometry file -----", "----- End geometry file -----")
    cell_txt = _extract_block(txt, "----- Begin unit cell -----", "----- End unit cell -----")
    cell = parse_cell_text(cell_txt)
    geom = parse_geom_text(geom_txt)
    return cell, geom

# ----------------------- Metric & Extinctions ----------------------

def deg2rad(x): return x*math.pi/180.0

def reciprocal_metric(a,b,c,al_deg,be_deg,ga_deg):
    al = deg2rad(al_deg); be = deg2rad(be_deg); ga = deg2rad(ga_deg)
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    G = [
        [a*a, a*b*cg, a*c*cb],
        [a*b*cg, b*b, b*c*ca],
        [a*c*cb, b*c*ca, c*c]
    ]
    detG = (
        G[0][0]*(G[1][1]*G[2][2] - G[1][2]*G[2][1]) -
        G[0][1]*(G[1][0]*G[2][2] - G[1][2]*G[2][0]) +
        G[0][2]*(G[1][0]*G[2][1] - G[1][1]*G[2][0])
    )
    if abs(detG) < 1e-18: raise ValueError("Singular metric")
    adj = [
        [ (G[1][1]*G[2][2]-G[1][2]*G[2][1]),
         -(G[0][1]*G[2][2]-G[0][2]*G[2][1]),
          (G[0][1]*G[1][2]-G[0][2]*G[1][1]) ],
        [-(G[1][0]*G[2][2]-G[1][2]*G[2][0]),
          (G[0][0]*G[2][2]-G[0][2]*G[2][0]),
         -(G[0][0]*G[1][2]-G[0][2]*G[1][0]) ],
        [ (G[1][0]*G[2][1]-G[1][1]*G[2][0]),
         -(G[0][0]*G[2][1]-G[0][1]*G[2][0]),
          (G[0][0]*G[1][1]-G[0][1]*G[1][0]) ]
    ]
    inv = [[adj[i][j]/detG for j in range(3)] for i in range(3)]
    return inv  # G*

def g_magnitude(h,k,l,Gstar):
    v = (h,k,l)
    tmp = [Gstar[0][0]*v[0]+Gstar[0][1]*v[1]+Gstar[0][2]*v[2],
           Gstar[1][0]*v[0]+Gstar[1][1]*v[1]+Gstar[1][2]*v[2],
           Gstar[2][0]*v[0]+Gstar[2][1]*v[1]+Gstar[2][2]*v[2]]
    val = v[0]*tmp[0]+v[1]*tmp[1]+v[2]*tmp[2]
    return math.sqrt(max(val,0.0))

def lattice_allowed(h,k,l, centering):
    c = centering.upper()
    if c == "P": return True
    if c == "I": return ((h+k+l)&1) == 0
    if c == "F": return ((h&1)==(k&1)==(l&1))
    if c == "A": return ((k+l)&1) == 0
    if c == "B": return ((h+l)&1) == 0
    if c == "C": return ((h+k)&1) == 0
    return True  # fallback

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

def zone_condition_zero(h,k,l, u,v,w):
    return (h*u + k*v + l*w) == 0

def rel_intensity(h,k,l,Gstar, g0=0.25):
    g = g_magnitude(h,k,l,Gstar)
    return 1.0 / (1.0 + (g/g0)**2)

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

def three_beam_overlap_count(hkls, Gstar, gmax, tol_g=1e-4):
    S = set((h,k,l) for (h,k,l,g) in hkls)
    count = 0
    hlist = [(h,k,l) for (h,k,l,_) in hkls]
    for i,(h1,k1,l1) in enumerate(hlist):
        for j in range(i, len(hlist)):
            h2,k2,l2 = hlist[j]
            for s in (+1,-1):
                hs, ks, ls = (h1 + s*h2, k1 + s*k2, l1 + s*l2)
                if (hs,ks,ls) == (0,0,0): continue
                if (hs,ks,ls) in S:
                    count += 1
    return count

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Predict visually problematic ZOLZ-centered zone axes from a CrystFEL stream header.")
    ap.add_argument("--stream", required=True, help="Path to .stream file, or '-' to read from stdin")
    ap.add_argument("--uvw-max", type=int, default=3, help="Max |u|,|v|,|w| for zone-axis enumeration (default 3)")
    ap.add_argument("--g-enum-bound", type=float, default=None,
                    help="Max |g| (1/Å) to enumerate HKLs. Default = g_edge from geometry + 10%")
    ap.add_argument("--g-max", type=float, default=None,
                    help="|g| cutoff for crowdedness counts N/M. Default = g_edge from geometry")
    ap.add_argument("--zolz-only", action="store_true", default=True,
                    help="ZOLZ-only (default True). (Add --holz to include thin HOLZ arcs.)")
    ap.add_argument("--holz", dest="holz", action="store_true", help="Include near-zone HOLZ arcs (off by default)")
    ap.add_argument("--i-min-rel", type=float, default=0.0, help="Relative intensity gate (default 0.0)")
    ap.add_argument("--ring-mult-min", type=int, default=4, help="Min reflections on first ZOLZ ring (default 4)")
    ap.add_argument("--n-min", type=int, default=12, help="Min N (in-zone low-|g| reflections) (default 12)")
    ap.add_argument("--m-min", type=int, default=8, help="Min M (three-beam overlaps) (default 8)")
    ap.add_argument("--score-alpha", type=float, default=1.0)
    ap.add_argument("--score-beta",  type=float, default=0.5)
    ap.add_argument("--margin-px", type=float, default=0.0, help="Edge margin (pixels) (default 0)")
    ap.add_argument("--tol-g", type=float, default=5e-4, help="|g| tolerance for equality (default 5e-4)")
    ap.add_argument("--csv", action="store_true", help="Optional CSV output next to stream (default: False)")
    ap.add_argument("--listuvw", action="store_true", help="List UVW in output (for further processing)")
    ap.add_argument("--progress", action="store_true", help="Emit progress to stderr (recommended for long runs)")
    ap.add_argument("--progress-steps", type=int, default=20, help="How many progress updates to show across the UVW loop (default 20)")
    args = ap.parse_args()

    t0 = time.time()
    if args.progress:
        eprint("[0/5] Reading headers...")

    cell, geom = parse_stream_headers(args.stream)

    Gstar = reciprocal_metric(cell["a"], cell["b"], cell["c"], cell["al"], cell["be"], cell["ga"])

    nx = geom["max_fs"] - geom["min_fs"] + 1
    ny = geom["max_ss"] - geom["min_ss"] + 1
    r_edge_px = min(nx, ny)/2.0 - args.margin_px
    g_edge = (r_edge_px / geom["res_px_per_m"]) / (geom["clen_m"] * geom["wavelength_A"])
    g_enum = args.g_enum_bound if args.g_enum_bound is not None else 1.10 * g_edge
    g_crowd = args.g_max if args.g_max is not None else g_edge

    if args.progress:
        eprint(f"[1/5] Parsed cell & geom. Det panel {nx}×{ny}px, edge r≈{r_edge_px:.2f}px.")
        eprint(f"      Derived g_edge≈{g_edge:.4f} 1/Å | g_enum={g_enum:.4f} | g_crowd={g_crowd:.4f}")
        a_s = sqrt(max(Gstar[0][0],1e-16))
        b_s = sqrt(max(Gstar[1][1],1e-16))
        c_s = sqrt(max(Gstar[2][2],1e-16))
        hmax = max(1, floor(g_enum/a_s)+2)
        kmax = max(1, floor(g_enum/b_s)+2)
        lmax = max(1, floor(g_enum/c_s)+2)
        approx_triples = (2*hmax+1)*(2*kmax+1)*(2*lmax+1)
        eprint(f"[2/5] Enumerating HKL up to g={g_enum:.3f} (box ≈ ±{hmax}, ±{kmax}, ±{lmax} → {approx_triples:,} triples)...")

    t_enum0 = time.time()
    HKL, (hmax,kmax,lmax) = enumerate_hkl_up_to_g(Gstar, g_enum, centering=cell["centering"])
    t_enum1 = time.time()
    if args.progress:
        eprint(f"      ...HKL enumeration kept {len(HKL):,} reflections (elapsed {t_enum1 - t_enum0:.1f}s).")
        eprint(f"[3/5] Building unique UVW list (|u|,|v|,|w| ≤ {args.uvw_max})...")

    seen=set(); dirs=[]
    rng = range(-args.uvw_max, args.uvw_max+1)
    for u in rng:
        for v in rng:
            for w in rng:
                canon = canonical_uvw(u,v,w)
                if not canon: continue
                if canon not in seen:
                    seen.add(canon); dirs.append(canon)
    if args.progress:
        eprint(f"      ...{len(dirs)} unique directions.\n[4/5] Scanning directions...")

    rows=[]
    total = len(dirs)
    step = max(1, total // max(1,args.progress_steps))
    for idx,(u,v,w) in enumerate(dirs, start=1):
        if args.progress and (idx == 1 or idx % step == 0 or idx == total):
            eprint(f"      [{idx:>4}/{total}] uvw=({u:>2} {v:>2} {w:>2})")

        gmin, _ring_reflects, ring_hkls = first_zolz_ring(u,v,w, HKL, Gstar, tol_g=args.tol_g, I_min_rel=args.i_min_rel)
        if gmin is None:
            continue
        rpx = rad_to_pixels(gmin, geom["clen_m"], geom["wavelength_A"], geom["res_px_per_m"])
        if rpx > r_edge_px + 1e-9:
            continue

        ring_mult = len(ring_hkls)
        if ring_mult < args.ring_mult_min:
            continue

        inzone = in_zone_under_gmax(u,v,w, HKL, Gstar, g_crowd, I_min_rel=args.i_min_rel)
        N = len(inzone)
        M = three_beam_overlap_count(inzone, Gstar, g_crowd, tol_g=args.tol_g)
        if N < args.n_min or M < args.m_min:
            continue

        S = args.score_alpha * N + args.score_beta * M
        rows.append({
            "u":u,"v":v,"w":w,
            "ring_type":"ZOLZ",
            "g_min_1_over_A":gmin,
            "r_px":rpx,
            "ring_mult": ring_mult,
            "N":N, "M":M, "Score":S,
            "inside":True
        })

    rows.sort(key=lambda r: (-r["Score"], -r["N"], r["r_px"], abs(r["u"])+abs(r["v"])+abs(r["w"]), r["u"], r["v"], r["w"]))

    print(f"Cell: {cell['lattice_type'].upper()} {cell['centering'].upper()} | a={cell['a']:.4f} Å b={cell['b']:.4f} Å c={cell['c']:.4f} Å "
          f"al={cell['al']:.2f}° be={cell['be']:.2f}° ga={cell['ga']:.2f}°")
    print(f"Geom: λ={geom['wavelength_A']} Å, L={geom['clen_m']} m, res={geom['res_px_per_m']} px/m | panel {nx}×{ny}px | edge r={r_edge_px:.2f}px")
    print(f"Settings: UVW_MAX={args.uvw_max}, g_enum={g_enum:.3f}, g_crowd={g_crowd:.3f}, I_min_rel={args.i_min_rel}, "
          f"N_min={args.n_min}, M_min={args.m_min}, ring_mult_min={args.ring_mult_min}, α={args.score_alpha}, β={args.score_beta}, tol_g={args.tol_g}\n")

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

            w = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()) if rows else
                ["u","v","w","ring_type","g_min_1_over_A","r_px","ring_mult","N","M","Score","inside"]
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)

            triplets = [f'"{r["u"]:>2} {r["v"]:>2} {r["w"]:>2}"' for r in rows]
            f.write("\n # Problematic axis triplets listed:\n")
            f.write(" ".join(triplets) + "\n")

        print(f"\nWrote CSV: {csv_path}")

    if args.progress:
        eprint(f"[5/5] Done in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()

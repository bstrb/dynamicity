#!/usr/bin/env python3
"""
orientation_checker_v2.py – classify SerialED orientations

Outputs lines like
    16     ZONE [102] – heavy ring   crowd=0.65

Flags
-----
  --zmax N        highest Miller index to test as zone axis    [2]
  --dmin NM       resolution cut-off for HKL enumeration       [0.25]
  --thick NM      crystal thickness (nm)                       [150]
  --safety F      extra factor on Laue-rod half-width          [1.0]
  --ringfrac F    Gaussian width R = F·detector_radius         [0.33]
  --crowdcrit F   crowding ≥ F  ⇒ add “heavy ring”             [0.40]
"""

import argparse, itertools, math, pathlib, re, sys
import numpy as np

# ───────── geometry helpers ─────────
def read_geometry(path):
    txt = pathlib.Path(path).read_text()
    λ_nm = float(re.search(r"\bwavelength\s*=\s*([\d.eE+-]+)", txt).group(1)) * 0.1
    L_nm = float(re.search(r"\bclen\s*=\s*([\d.eE+-]+)", txt).group(1)) * 1e9
    pixel_size_nm = 1e9 / float(re.search(r"\bres\s*=\s*([\d.eE+-]+)", txt).group(1))
    cx = float(re.search(r"corner_x\s*=\s*([-\d.]+)", txt).group(1))
    cy = float(re.search(r"corner_y\s*=\s*([-\d.]+)", txt).group(1))
    r_det_nm = max(abs(cx), abs(cy)) * pixel_size_nm
    return λ_nm, L_nm, r_det_nm

# ───────── orientation loader ─────────
def load_orientations(path):
    lines = pathlib.Path(path).read_text().splitlines()

    if any(l.lstrip().startswith("Cell parameters") for l in lines):      # block format
        i = 0
        while i < len(lines):
            if not lines[i].lstrip().startswith("Cell parameters"): i += 1; continue
            nums = list(map(float, re.findall(r"[-\d.]+", lines[i])))
            a, b, c, al, be, ga = nums[:6]
            vec = [np.fromstring(lines[i+j].split("=",1)[1], sep=" ") for j in (1,2,3)]
            cent = lines[i+5].split("=",1)[1].strip()
            yield dict(tag=f"blk{i}", astar=vec[0], bstar=vec[1], cstar=vec[2],
                       cell_nm=(a,b,c,al,be,ga), centering=cent)
            i += 6
        return

    for raw in lines:                                                     # TSV / CSV
        raw = raw.strip()
        if not raw or raw.startswith("#"): continue
        row = re.split(r"[,\s]+", raw)
        if len(row) < 17: raise ValueError("Need ≥17 columns")
        tag, vecs = row[0], np.array(row[1:10], float).reshape(3,3)
        yield dict(tag=tag, astar=vecs[0], bstar=vecs[1], cstar=vecs[2],
                   cell_nm=tuple(map(float, row[10:16])), centering=row[16])

# ───────── crystallographic maths ─────────
def b_matrix(a,b,c, α,β,γ):
    α,β,γ = np.deg2rad([α,β,γ]); cosα,cosβ,cosγ=np.cos([α,β,γ]); sinγ=np.sin(γ)
    V=a*b*c*math.sqrt(1-cosα**2-cosβ**2-cosγ**2+2*cosα*cosβ*cosγ)
    a_s,b_s,c_s = (b*c*sinγ/V, a*c*math.sin(β)/V, a*b*math.sin(α)/V)
    cosαs=(cosβ*cosγ-cosα)/(math.sin(β)*sinγ); cosβs=(cosα*cosγ-cosβ)/(math.sin(α)*sinγ)
    cosγs=(cosα*cosβ-cosγ)/(math.sin(α)*math.sin(β))
    return np.array([[a_s, b_s*cosγs, c_s*cosβs],
                     [0,   b_s*sinγ,  c_s*(cosαs-cosβs*cosγs)/sinγ],
                     [0,   0,         c_s*math.sqrt(1-cosβs**2-
                           ((cosαs-cosβs*cosγs)/sinγ)**2)]])

def orientation_from_astar(astar, bstar, cstar, cell):
    UB=np.column_stack([astar,bstar,cstar])
    U0=UB@np.linalg.inv(b_matrix(*cell)); U0/=np.linalg.norm(U0,axis=0).mean()
    u,_,vt=np.linalg.svd(U0); U=u@vt
    if np.linalg.det(U)<0: U[:,-1]*=-1
    return U

# ───────── reciprocal-space helpers ─────────
def ewald_hits(k_in,g,t_nm,safety):
    return abs(np.linalg.norm(k_in+g)-np.linalg.norm(k_in)) <= safety/(2*t_nm)

def kvec_to_r(k_out,L_nm):
    u=k_out/np.linalg.norm(k_out); return math.inf if u[2]<=0 else L_nm*np.linalg.norm(u[:2])/u[2]

def hkl_range(cell,dmin):
    B=b_matrix(*cell); G=B.T@B; invd2=dmin**-2
    lim=[int(math.ceil(2*B[i,i]*dmin)) for i in range(3)]
    for h in range(-lim[0],lim[0]+1):
        for k in range(-lim[1],lim[1]+1):
            for l in range(-lim[2],lim[2]+1):
                if (h,k,l)==(0,0,0): continue
                v=np.array([h,k,l]);  yield (h,k,l) if v@G@v<invd2 else None

def zone_axis(U, cell, tol_deg, zmax):
    a, b, c = (U @ np.diag(cell[:3])).T; beam = np.array([0,0,1])
    best = (tol_deg+1, None)
    for h,k,l in itertools.product(range(zmax+1), repeat=3):
        if (h,k,l)==(0,0,0) or math.gcd(math.gcd(h,k),l)!=1: continue
        v = h*a + k*b + l*c
        ang = math.degrees(math.acos(
              max(-1, min(1, np.dot(v, beam)/np.linalg.norm(v)))))
        if ang < best[0]:
            best = (ang, f"[{h}{k}{l}]")
    return best[1] if best[0] < tol_deg else None

# ───────── new: crowding with tunable β and σ ─────────#

def crowding_score(U, cell, λ, L, rdet,
                   dmin, cent,           # no sigma / beta needed
                   t_nm, safety):
    """
    Continuous 0-1 crowd index based on the mean radial distance
    of all reflections that satisfy Ewald + detector + centering.
    """
    UB = U @ b_matrix(*cell)
    k_in = np.array([0., 0., 1/λ])

    r_sum = hits = 0.0
    for h, k, l in hkl_range(cell, dmin):
        if cent.upper() == 'I' and (h + k + l) & 1:
            continue
        g = UB @ np.array([h, k, l])
        if not ewald_hits(k_in, g, t_nm, safety):
            continue
        r_nm = kvec_to_r(k_in + g, L)
        if r_nm > rdet:
            continue

        hits += 1
        r_sum += r_nm

    if hits == 0:
        return 0.0

    mean_r = r_sum / hits
    # use RMS instead:  mean_r = math.sqrt(r2_sum / hits)
    return max(0.0, 1.0 - mean_r / rdet)

# ───────── driver ─────────
def classify(geom, orientations, args):
    λ,L,rdet = geom
    σ = args.sigfrac * rdet if args.sigfrac else args.sigma
    for o in orientations:
        U = orientation_from_astar(o['astar'], o['bstar'], o['cstar'], o['cell_nm'])
        zone = zone_axis(U, o['cell_nm'], 5, args.zmax)
        crowd = crowding_score(U, o["cell_nm"], λ, L, rdet,
                            args.dmin, o["centering"],
                            args.thick, args.safety)
        verdict = f"ZONE {zone}" if zone else "off-zone"
        if crowd >= args.crowdcrit:
            verdict += " – heavy ring"
        print(f"{o['tag']:>8}  {verdict:<25}  crowd={crowd:.2f}")

# ───────── CLI ─────────
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("geometry"); p.add_argument("orientations")
    p.add_argument("--zmax", type=int, default=2)
    p.add_argument("--dmin", type=float, default=0.25)
    p.add_argument("--thick", type=float, default=150)
    p.add_argument("--safety", type=float, default=1.0)
    # new fall-off controls
    p.add_argument("--sigfrac", type=float, default=0.33,
                   help="σ as fraction of detector radius (overrides --sigma)")
    p.add_argument("--sigma",   type=float, default=None,
                   help="explicit σ in nm (bypasses --sigfrac)")
    p.add_argument("--beta",    type=float, default=2.0,
                   help="exponent in exp[-(r/σ)^β] (β=2 is Gaussian)")
    p.add_argument("--crowdcrit", type=float, default=0.40)
    args = p.parse_args()

    geom = read_geometry(args.geometry)
    orientations = list(load_orientations(args.orientations))
    print(f"Loaded {len(orientations)} orientation(s)")
    if not orientations: sys.exit("No orientations parsed.")
    classify(geom, orientations, args)

if __name__ == "__main__":
    main()
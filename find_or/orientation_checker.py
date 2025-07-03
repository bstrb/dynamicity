#!/usr/bin/env python3
"""
orientation_checker.py – spot primary (and higher-order) zone axes and
detect “heavy-ring” diffraction patterns.

Usage
-----
    ./orientation_checker.py header.geom orientations.txt [options]

Options
-------
    --zmax N       highest Miller index to test as zone axis      [2]
    --dmin NM      resolution cutoff for HKL enumeration          [0.25]
    --thick NM     crystal thickness (nm) → Laue-rod half-width   [50]
    --safety F     extra factor on Laue-rod half-width            [1.0]
    --ringfrac F   inner-ring radius as fraction of detector      [0.33]
    --ringcrit F   heavy-ring flag if ring_score ≥ F              [0.20]
"""

import argparse, itertools, math, pathlib, re, sys
import numpy as np

# ──────────────── geometry helpers ────────────────
def read_geometry(path):
    txt = pathlib.Path(path).read_text()
    λ_nm = float(re.search(r"\bwavelength\s*=\s*([\d.eE+-]+)", txt).group(1)) * 0.1
    L_nm = float(re.search(r"\bclen\s*=\s*([\d.eE+-]+)", txt).group(1)) * 1e9
    res_match = re.search(r"\bres\s*=\s*([\d.eE+-]+)", txt)
    pixel_size_nm = 1e9 / float(res_match.group(1)) if res_match else 75_000.0
    cx = float(re.search(r"corner_x\s*=\s*([-\d.]+)", txt).group(1))
    cy = float(re.search(r"corner_y\s*=\s*([-\d.]+)", txt).group(1))
    r_det_nm = max(abs(cx), abs(cy)) * pixel_size_nm
    return λ_nm, L_nm, r_det_nm

# ─────────────── orientation file loader ───────────────
def load_orientations(path):
    lines = pathlib.Path(path).read_text().splitlines()

    if any(l.lstrip().startswith("Cell parameters") for l in lines):
        i = 0
        while i < len(lines):
            if not lines[i].lstrip().startswith("Cell parameters"):
                i += 1
                continue
            nums = list(map(float, re.findall(r"[-\d.]+", lines[i])))
            a, b, c, al, be, ga = nums[:6]
            astar = np.fromstring(lines[i + 1].split("=", 1)[1], sep=" ")
            bstar = np.fromstring(lines[i + 2].split("=", 1)[1], sep=" ")
            cstar = np.fromstring(lines[i + 3].split("=", 1)[1], sep=" ")
            cent = lines[i + 5].split("=", 1)[1].strip()
            yield dict(tag=f"blk{i}", astar=astar, bstar=bstar, cstar=cstar,
                       cell_nm=(a, b, c, al, be, ga), centering=cent)
            i += 6
        return

    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        row = re.split(r"[,\s]+", raw)
        if len(row) < 17:
            raise ValueError(f"Malformed line (needs ≥17 columns):\n{raw}")
        tag = row[0]
        vecs = np.array(row[1:10], dtype=float).reshape(3, 3)
        cell = tuple(map(float, row[10:16]))
        cent = row[16]
        yield dict(tag=tag, astar=vecs[0], bstar=vecs[1], cstar=vecs[2],
                   cell_nm=cell, centering=cent)

# ───────────── crystallographic utilities ─────────────
def b_matrix(a, b, c, α, β, γ):
    α, β, γ = np.deg2rad([α, β, γ])
    cosα, cosβ, cosγ = np.cos([α, β, γ])
    sinγ = np.sin(γ)
    V = a * b * c * math.sqrt(1 - cosα**2 - cosβ**2 - cosγ**2 +
                              2 * cosα * cosβ * cosγ)
    a_s = b * c * sinγ / V
    b_s = a * c * math.sin(β) / V
    c_s = a * b * math.sin(α) / V
    cosαs = (cosβ * cosγ - cosα) / (math.sin(β) * sinγ)
    cosβs = (cosα * cosγ - cosβ) / (math.sin(α) * sinγ)
    cosγs = (cosα * cosβ - cosγ) / (math.sin(α) * math.sin(β))
    return np.array([[a_s, b_s * cosγs, c_s * cosβs],
                     [0.0, b_s * sinγ,  c_s * (cosαs - cosβs * cosγs) / sinγ],
                     [0.0, 0.0,         c_s * math.sqrt(
                         1 - cosβs**2 - ((cosαs - cosβs * cosγs) / sinγ)**2)]])

def orientation_from_astar(astar, bstar, cstar, cell):
    UB = np.column_stack([astar, bstar, cstar])
    U0 = UB @ np.linalg.inv(b_matrix(*cell))
    U0 /= np.linalg.norm(U0, axis=0).mean()
    u, _, vt = np.linalg.svd(U0)
    U = u @ vt
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    return U, UB

# ───────────── reciprocal-space helpers ─────────────
def ewald_hits(k_in, g, t_nm, safety):
    k_len = np.linalg.norm(k_in)
    S = np.linalg.norm(k_in + g) - k_len
    return abs(S) <= safety / (2 * t_nm)

def kvec_to_r(k_out, L_nm):
    u = k_out / np.linalg.norm(k_out)
    return np.inf if u[2] <= 0 else L_nm * np.linalg.norm(u[:2]) / u[2]

def hkl_range(cell, dmin):
    B = b_matrix(*cell)
    G = B.T @ B
    limit = [int(math.ceil(2 * B[i, i] * dmin)) for i in range(3)]
    invd2_max = dmin**-2
    for h in range(-limit[0], limit[0] + 1):
        for k in range(-limit[1], limit[1] + 1):
            for l in range(-limit[2], limit[2] + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                v = np.array([h, k, l])
                if v @ G @ v < invd2_max:
                    yield h, k, l

def zone_axis(U, cell, tol_deg=5, max_index=2):
    a_vec, b_vec, c_vec = (U @ np.diag(cell[:3])).T
    beam = np.array([0, 0, 1.0])
    best_ang = tol_deg + 1
    best_label = None

    def real_dir(h, k, l):
        return h * a_vec + k * b_vec + l * c_vec

    for h, k, l in itertools.product(range(max_index + 1), repeat=3):
        if (h, k, l) == (0, 0, 0):
            continue
        if math.gcd(math.gcd(h, k), l) != 1:
            continue
        v = real_dir(h, k, l)
        ang = math.degrees(math.acos(
            np.clip(np.dot(v, beam) / (np.linalg.norm(v)), -1, 1)))
        if ang < best_ang:
            best_ang, best_label = ang, f"[{h}{k}{l}]"
    return best_label if best_ang < tol_deg else None

def ring_score(U, cell, λ, L, rdet, dmin, cent, rfrac, t_nm, safety):
    UB = U @ b_matrix(*cell)
    k_in = np.array([0, 0, 1 / λ])
    r_hi = rdet * rfrac
    hits = ring = 0
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
        ring += r_nm <= r_hi
    return 0 if hits == 0 else ring / hits

# ───────────── main evaluation loop ─────────────
def classify(geom, orientations, dmin, thick, safety,
             rfrac, ringcrit, zmax):
    λ, L, rdet = geom
    for o in orientations:
        U, _ = orientation_from_astar(o["astar"], o["bstar"],
                                      o["cstar"], o["cell_nm"])
        zone = zone_axis(U, o["cell_nm"], max_index=zmax)
        ring = ring_score(U, o["cell_nm"], λ, L, rdet,
                          dmin, o["centering"], rfrac, thick, safety)
        verdict = f"ZONE {zone}" if zone else "off-zone"
        if ring >= ringcrit:
            verdict += " – heavy ring"
        print(f"{o['tag']:>8}  {verdict:<25}  ring_score={ring:.2f}")

# ───────────── command-line interface ─────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("geometry")
    ap.add_argument("orientations")
    ap.add_argument("--zmax",      type=int,   default=2)
    ap.add_argument("--dmin",      type=float, default=0.25)
    ap.add_argument("--thick",     type=float, default=50)
    ap.add_argument("--safety",    type=float, default=1.0,
                    help="multiply rel-rod half-width by this factor")
    ap.add_argument("--ringfrac",  type=float, default=1/3)
    ap.add_argument("--ringcrit",  type=float, default=0.20)
    a = ap.parse_args()

    geom = read_geometry(a.geometry)
    orientations = list(load_orientations(a.orientations))
    print(f"Loaded {len(orientations)} orientation(s)")
    if not orientations:
        sys.exit("No usable orientations found – check file format.")

    classify(geom, orientations, a.dmin, a.thick, a.safety,
             a.ringfrac, a.ringcrit, a.zmax)

if __name__ == "__main__":
    main()

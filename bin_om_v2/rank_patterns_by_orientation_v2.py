#!/usr/bin/env python3
"""
rank_patterns_by_orientation.py  (June‑2025, parallel‑beam edition)
------------------------------------------------------------------

© 2025 Buster Blomberg – Public‑domain helper

Purpose
~~~~~~~
* Read a CrystFEL *.stream* file.
* Compute two orientation metrics for every pattern:
  1. **M** – number of reflections whose excitation error |s| lies in the
     relrod window |s| ≤ λ / (2 t).
  2. **r_min** – detector‑plane radius (mrad) of the *innermost*
     Zero‑Order Laue Zone (ZOLZ) reflection.
* Write a per‑pattern CSV; optionally write a second *.stream* sorted by M;
  optionally skip patterns whose ZOLZ ring intrudes inside a user‑defined
  radius; optionally plot either metric versus pattern index.

Usage example
~~~~~~~~~~~~~
           python rank_patterns_by_orientation_v2.py mfm300sim_0.0_0.0.stream \
                -r 0.2 -t 150 \
                --camera-length-mm 290 \
                --inner-cutoff-mrad 3.5 \
                --csv RUN_ZOLZ_filter.csv \
                --omit-inner-ring

Units & symbols
~~~~~~~~~~~~~~
    t                  crystal thickness (nm)
    λ (lambda)         electron wavelength (Å) – parsed from *.stream* header
    L                  camera length (mm)      – user must supply
    g_perp             in‑plane component of a reciprocal‑lattice vector (Å⁻¹)
    r                  detector radius (mrad)  = 1000·L(mm)·λ·g_perp

All outputs are in US English locale.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import gemmi
from tqdm import tqdm

# Optional plotting – loaded only if requested
try:
    import matplotlib.pyplot as plt  # noqa: F401
except ModuleNotFoundError:
    plt = None  # handled later

# ──────────────────────────── regex helpers ─────────────────────────────
RE_CELL = re.compile(
    r"a\s*=\s*([0-9.]+).*?b\s*=\s*([0-9.]+).*?c\s*=\s*([0-9.]+).*?"
    r"al[^=]*=\s*([0-9.]+).*?be[^=]*=\s*([0-9.]+).*?ga[^=]*=\s*([0-9.]+)",
    re.S,
)

RE_WL = re.compile(r"^\s*wavelength\s*=\s*([0-9.]+)\s*A", re.I | re.M)
RE_CHUNK = re.compile(r"^----- Begin chunk")
RE_ASTAR = re.compile(r"^\s*astar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)")
RE_BSTAR = re.compile(r"^\s*bstar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)")
RE_CSTAR = re.compile(r"^\s*cstar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)")
RE_EVENT = re.compile(r"^\s*Event:\s*(\S+)")
RE_SERIAL = re.compile(r"^\s*Image serial number:\s*(\d+)")

# ───────────────────────── lattice helpers ────────────────────────────

def recip_matrix(cell: gemmi.UnitCell) -> np.ndarray:
    """Return 3 × 3 reciprocal metric matrix (Å⁻¹)."""
    a, b, c = cell.a, cell.b, cell.c
    al, be, ga = map(math.radians, (cell.alpha, cell.beta, cell.gamma))

    ax, ay, az = a, 0, 0
    bx, by, bz = b * math.cos(ga), b * math.sin(ga), 0
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))

    aV = np.array([ax, ay, az])
    bV = np.array([bx, by, bz])
    cV = np.array([cx, cy, cz])

    V = np.dot(aV, np.cross(bV, cV))

    return np.stack([
        np.cross(bV, cV) / V,
        np.cross(cV, aV) / V,
        np.cross(aV, bV) / V,
    ])


def theta_phi_from_cstar(Brows: np.ndarray) -> Tuple[float, float]:
    """Polar (θ) and azimuthal (φ) angles of c* in degrees."""
    cx, cy, cz = Brows[2]
    r = math.sqrt(cx * cx + cy * cy + cz * cz)
    theta = math.degrees(math.acos(cz / r)) if r else 0.0
    phi = (math.degrees(math.atan2(cy, cx)) + 360.0) % 360.0
    return theta, phi

# ───────────────────────── command‑line interface ─────────────────────

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Rank patterns by dynamical‑scattering risk and ZOLZ proximity.""",
    )

    p.add_argument("stream", type=Path, help="Input CrystFEL *.stream* file")
    p.add_argument("-r", "--resolution", type=float, required=True, help="d_min (Å) for reflection list")
    p.add_argument("-t", "--thickness", type=float, required=True, help="Crystal thickness (nm)")
    p.add_argument("--camera-length-mm", type=float, required=True, help="Effective camera length L (mm)")
    p.add_argument(
        "--inner-cutoff-mrad",
        type=float,
        default=3.0,
        help="Critical radius r (mrad); patterns whose ZOLZ ring lies inside are flagged",
    )
    p.add_argument(
        "--omit-inner-ring",
        action="store_true",
        help="Skip writing patterns whose ZOLZ ring lies inside the cutoff",
    )
    p.add_argument("-o", "--csv", type=Path, help="CSV output path (default <stream>_pattern_risk.csv)")
    p.add_argument("--sorted-stream", type=Path, help="Write a second *.stream* sorted by descending M")
    p.add_argument("--plot", type=Path, help="PNG path for scatter plot (M and/or r_min)")

    return p.parse_args()

# ──────────────────────────── core routine ───────────────────────────

def main() -> None:
    args = cli()
    s_path: Path = args.stream

    # ---------- parse header: cell & wavelength ----------------------
    with s_path.open() as fh:
        header_lines = []
        for ln in fh:
            header_lines.append(ln)
            if RE_CHUNK.match(ln):
                break  # stop at first chunk header
    header = "".join(header_lines)

    m_cell = RE_CELL.search(header)
    if not m_cell:
        sys.exit("[error] Unit‑cell parameters not found in stream header.")
    cell = gemmi.UnitCell(*(float(x) for x in m_cell.groups()))

    m_wl = RE_WL.search(header)
    if not m_wl:
        sys.exit("[error] Wavelength line not found in stream header.")
    lam = float(m_wl.group(1))  # Å

    # ---------- build reflection table up to d_min ------------------
    d_min = args.resolution
    # heuristic for h,k,l range
    hmax = int(round(cell.volume ** (1.0 / 3.0) / d_min)) + 1
    hkl_table = np.array(
        [
            (h, k, l)
            for h, k, l in itertools.product(range(-hmax, hmax + 1), repeat=3)
            if (h, k, l) != (0, 0, 0) and cell.calculate_d((h, k, l)) < d_min
        ],
        dtype=int,
    )

    klen = 1.0 / lam  # |k| in Å⁻¹ for electrons (relativistic effects ignored)
    half = lam / (2.0 * args.thickness * 10.0)  # relrod half‑width, s‑units (Å⁻¹)

    L_mm = args.camera_length_mm
    inner_cut = args.inner_cutoff_mrad

    # ---------- output setup ---------------------------------------
    csv_path = args.csv or s_path.with_suffix("_pattern_risk.csv")

    sort_chunks: List[Tuple[int, str]] = []  # (M, raw_chunk_text)
    M_values: List[int] = []
    r_values: List[float] = []

    # ---------- iterate over patterns ------------------------------
    with csv_path.open("w", newline="") as csv_out, s_path.open() as fh:
        writer = csv.writer(csv_out)
        writer.writerow(
            [
                "chunk_id",
                "theta_deg",
                "phi_deg",
                "M",
                "gmin_Ainv",
                "r_mrad",
                "inner_ring",
            ]
        )

        chunk_lines: List[str] = []  # accumulate raw lines for current chunk
        chunk_id: str | None = None
        ast = bst = cst = None
        M = 0  # current pattern's M value
        gmin = float("inf")
        r_mrad = float("inf")
        inner_ring = False

        for line in tqdm(fh, unit="lines", desc="scan"):
            if RE_CHUNK.match(line):
                # flush previous chunk (if any)
                if chunk_lines:
                    if not (args.omit_inner_ring and inner_ring):
                        sort_chunks.append((M, "".join(chunk_lines)))
                    chunk_lines.clear()
                # start new chunk
                chunk_lines.append(line)
                chunk_id = None
                ast = bst = cst = None
                continue

            chunk_lines.append(line)

            if m := RE_EVENT.match(line):
                chunk_id = f"event:{m.group(1)}"
                continue
            if m := RE_SERIAL.match(line):
                if chunk_id is None:
                    chunk_id = f"serial:{m.group(1)}"
                continue
            if m := RE_ASTAR.match(line):
                ast = list(map(float, m.groups()))
                continue
            if m := RE_BSTAR.match(line):
                bst = list(map(float, m.groups()))
                continue
            if m := RE_CSTAR.match(line):
                cst = list(map(float, m.groups()))
                continue

            # process a complete basis triple
            if ast and bst and cst:
                B = np.array([ast, bst, cst])  # rows are a*, b*, c*

                # --- orientation angles for metadata
                theta, phi = theta_phi_from_cstar(B)

                # --- dynamical metric M ----------------------------------
                Rg = hkl_table @ B  # N×3 array of g‑vectors (Å⁻¹)
                g2 = (Rg ** 2).sum(axis=1)
                s = (Rg[:, 2] * klen - 0.5 * g2) / klen  # excitation error (Å⁻¹)
                mask = np.abs(s) <= half
                M = int(mask.sum())

                # --- ZOLZ inner‑ring radius ------------------------------
                g_inplane = np.linalg.norm(Rg[mask, :2], axis=1)
                if g_inplane.size:
                    gmin = g_inplane.min()
                    # detector radius in mrad: r = 1000 * L(mm) * λ * |g_perp|
                    r_mrad = 1000.0 * L_mm * 1e-3 * lam * gmin
                    inner_ring = r_mrad < inner_cut
                else:
                    gmin = float("inf")
                    r_mrad = float("inf")
                    inner_ring = False

                # write CSV row
                writer.writerow(
                    [
                        chunk_id or "unk",
                        f"{theta:.2f}",
                        f"{phi:.2f}",
                        M,
                        f"{gmin:.4f}",
                        f"{r_mrad:.2f}",
                        int(inner_ring),
                    ]
                )

                # inject helpful comment into chunk so sorted stream keeps it
                chunk_lines.append(
                    f"# M_score {M}  r_min_mrad {r_mrad:.2f} inner_ring {inner_ring}\n"
                )

                # reset basis flags for next pattern within same chunk (if any)
                ast = bst = cst = None
                M_values.append(M)
                r_values.append(r_mrad)

        # push last chunk
        if chunk_lines and not (args.omit_inner_ring and inner_ring):
            sort_chunks.append((M, "".join(chunk_lines)))

    print(f"[info] CSV written → {csv_path}")

    # ---------- optional: sorted stream ----------------------------
    if args.sorted_stream:
        with args.sorted_stream.open("w") as fout:
            fout.writelines(header_lines)  # original header
            for _M, txt in sorted(sort_chunks, key=lambda t: t[0], reverse=True):
                fout.write(txt)
        print(f"[info] sorted stream → {args.sorted_stream}")

    # ---------- optional: plot ------------------------------------
    if args.plot:
        if plt is None:
            print("[warn] matplotlib not available – cannot plot.")
        else:
            x = np.arange(1, len(M_values) + 1)
            plt.figure(figsize=(6, 4))
            plt.scatter(x, M_values, s=8, label="M (relrod count)")
            plt.scatter(x, r_values, s=8, marker="x", label="r_min (mrad)")
            plt.xlabel("Pattern index (encounter order)")
            plt.ylabel("Metric value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot, dpi=160)
            print(f"[info] plot saved → {args.plot}")


if __name__ == "__main__":
    main()

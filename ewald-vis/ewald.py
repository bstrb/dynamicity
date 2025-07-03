"""
ewald.py
========
Crystallographic math utilities shared by the GUI.

* Cell ↔ reciprocal-cell conversion
* Reciprocal-lattice generation
* Bragg-condition testing
* Relativistic electron-wavelength calculator
* Euler-angle orientation matrix
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, sin, cos, radians, asin, pi
from typing import Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Cell:
    """Direct-space unit-cell parameters (Å, °)."""
    a: float
    b: float
    c: float
    alpha: float   # degrees
    beta:  float
    gamma: float

    # ------------ direct-space basis vectors ---------------------------
    def direct_vectors(self) -> np.ndarray:
        α, β, γ = map(radians, (self.alpha, self.beta, self.gamma))

        va = np.array([self.a, 0.0, 0.0])

        vb = np.array([
            self.b * cos(γ),
            self.b * sin(γ),
            0.0,
        ])

        cx = self.c * cos(β)
        cy = self.c * (cos(α) - cos(β) * cos(γ)) / sin(γ)
        cz = sqrt(max(self.c**2 - cx**2 - cy**2, 0.0))

        vc = np.array([cx, cy, cz])

        return np.vstack([va, vb, vc])   # shape (3,3)

    # ------------ reciprocal-space basis vectors -----------------------
    def reciprocal_vectors(self) -> np.ndarray:
        a1, a2, a3 = self.direct_vectors()
        volume = np.dot(a1, np.cross(a2, a3))
        b1 = np.cross(a2, a3) / volume
        b2 = np.cross(a3, a1) / volume
        b3 = np.cross(a1, a2) / volume
        return 2.0 * pi * np.vstack([b1, b2, b3])  # 2π factor for convenience


# ----------------------------------------------------------------------
def make_reciprocal_lattice(
    cell: Cell, *, h_range: int = 5
) -> List[Tuple[int, int, int, np.ndarray]]:
    """Generate (h, k, l, vector) over −h_range…+h_range."""
    a_star, b_star, c_star = cell.reciprocal_vectors()
    pts: list[tuple[int, int, int, np.ndarray]] = []

    for h in range(-h_range, h_range + 1):
        for k in range(-h_range, h_range + 1):
            for l in range(-h_range, h_range + 1):
                if h == k == l == 0:
                    continue
                g = h * a_star + k * b_star + l * c_star
                pts.append((h, k, l, g))
    return pts


# ----------------------------------------------------------------------
def bragg_hits(
    lattice_pts: Iterable[Tuple[int, int, int, np.ndarray]],
    wavelength: float,
    tol: float = 1e-3,
) -> List[Tuple[int, int, int, np.ndarray]]:
    """
    Return reflections that satisfy |k_in + G| ≈ |k_in| = 1/λ,
    with k_in chosen along +z (lab frame).
    """
    k0 = np.array([0.0, 0.0, 1.0 / wavelength])
    radius = 1.0 / wavelength
    hits: list[tuple[int, int, int, np.ndarray]] = []

    for h, k, l, g in lattice_pts:
        kout = k0 + g
        if abs(np.linalg.norm(kout) - radius) < tol:
            hits.append((h, k, l, g))
    return hits


# ----------------------------------------------------------------------
def electron_wavelength_kv(kv: float) -> float:
    """
    Relativistically correct de Broglie wavelength of an electron (Å)
    accelerated through `kv` kilovolts.

    Formula:
        λ = h / √[ 2 m₀ e V (1 + eV / 2m₀c²) ]

    Defaults: CODATA 2018 constants.
    """
    h  = 6.62607015e-34       # J·s
    m0 = 9.1093837015e-31     # kg
    e  = 1.602176634e-19      # C
    c  = 299_792_458          # m s⁻¹
    V  = kv * 1e3             # volts
    momentum = sqrt(2 * m0 * e * V * (1 + e * V / (2 * m0 * c**2)))
    return (h / momentum) * 1e10   # meters → Å


# ----------------------------------------------------------------------
def euler_omega_chi_phi(omega: float, chi: float, phi: float) -> np.ndarray:
    """
    Return rotation matrix for ZXZ Euler angles (ω, χ, φ) in degrees,
    using the convention R = Rz(φ) · Rx(χ) · Rz(ω).
    """
    om, ch, ph = map(radians, (omega, chi, phi))

    Rz_om = np.array([[cos(om), -sin(om), 0],
                      [sin(om),  cos(om), 0],
                      [0,        0,       1]])

    Rx_ch = np.array([[1, 0,         0        ],
                      [0, cos(ch),  -sin(ch)],
                      [0, sin(ch),   cos(ch)]])

    Rz_ph = np.array([[cos(ph), -sin(ph), 0],
                      [sin(ph),  cos(ph), 0],
                      [0,        0,       1]])

    return Rz_ph @ Rx_ch @ Rz_om

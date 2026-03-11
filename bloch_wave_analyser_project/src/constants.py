"""Project-wide constants.

The forward electron scattering factors are the same approximate values used by the
original HTML/JavaScript analyser. They are used only for composition parsing and
Wilson-like scaling, not as a substitute for full scattering-factor tables.
"""

from __future__ import annotations

FE0: dict[str, float] = {
    "H": 0.529,
    "He": 0.73,
    "Li": 1.07,
    "Be": 1.34,
    "B": 1.52,
    "C": 1.69,
    "N": 1.97,
    "O": 2.26,
    "F": 2.50,
    "Ne": 2.67,
    "Na": 3.20,
    "Mg": 3.44,
    "Al": 3.66,
    "Si": 5.31,
    "P": 5.58,
    "S": 5.85,
    "Cl": 6.11,
    "Ar": 6.36,
    "K": 6.66,
    "Ca": 6.90,
    "Sc": 7.10,
    "Ti": 7.40,
    "V": 7.64,
    "Cr": 7.87,
    "Mn": 8.09,
    "Fe": 8.16,
    "Co": 8.55,
    "Ni": 8.77,
    "Cu": 8.99,
    "Zn": 8.93,
    "Ga": 9.63,
    "Ge": 9.83,
    "As": 10.02,
    "Se": 10.21,
    "Br": 9.71,
    "Kr": 10.42,
    "Rb": 10.60,
    "Sr": 10.82,
    "Y": 11.03,
    "Zr": 11.25,
    "Nb": 11.46,
    "Mo": 11.67,
    "Ag": 12.50,
    "Cd": 12.70,
    "In": 12.90,
    "Sn": 13.10,
    "Sb": 13.30,
    "I": 13.30,
    "Cs": 13.70,
    "Ba": 13.90,
    "La": 14.10,
    "Ce": 14.30,
    "Nd": 14.70,
    "Gd": 15.40,
    "W": 16.00,
    "Pt": 16.30,
    "Au": 16.50,
    "Pb": 16.50,
    "Bi": 16.70,
    "U": 17.50,
}

DEFAULT_DMIN_ANGSTROM: float = 0.6
DEFAULT_DMAX_ANGSTROM: float = 50.0
DEFAULT_EXCITATION_TOLERANCE_INV_ANGSTROM: float = 1.5e-3
EPSILON: float = 1.0e-12

"""Bloch-like structure-matrix construction and propagation."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import eigh

from .wilson import WilsonCalibration

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class StructureMatrixResult:
    """A diagonalized Bloch-like structure matrix."""

    matrix: FloatArray
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    beam_hkls: tuple[tuple[int, int, int], ...]


def extinction_distance_angstrom(
    wavelength_angstrom: float,
    cell_volume_ang3: float,
    fg_abs: float | FloatArray,
) -> float | FloatArray:
    """Compute the extinction distance proxy used by the prototype.

    The HTML analyser uses ``xi_g = pi * V / (lambda * |F_g|)``.
    """

    return np.pi * cell_volume_ang3 / (wavelength_angstrom * np.asarray(fg_abs, dtype=float))


def build_structure_matrix(
    excited_reflections: pd.DataFrame,
    calibration: WilsonCalibration,
    wavelength_angstrom: float,
    cell_volume_ang3: float,
) -> StructureMatrixResult:
    """Build and diagonalize the transmitted + diffracted beam coupling matrix.

    The matrix follows the browser prototype closely:

    - diagonal element 0 for the transmitted beam
    - diagonal ``s_g`` terms for diffracted beams
    - transmitted-diffracted coupling ``lambda * |F_g| / (2 pi V)``
    - diffracted-diffracted coupling via ``|F_{g_i - g_j}|`` using the same
      Wilson-calibrated lookup and fallback median amplitude
    """

    if excited_reflections.empty:
        raise ValueError("Cannot build a structure matrix for an empty reflection table.")

    excited = excited_reflections.reset_index(drop=True)
    n_beams = int(excited.shape[0])
    dim = n_beams + 1
    matrix = np.zeros((dim, dim), dtype=float)
    matrix[1:, 1:] = np.diag(excited["sg_invA"].to_numpy(dtype=float))

    coupling_scale = wavelength_angstrom / (2.0 * pi * cell_volume_ang3)
    fg_abs = excited["Fg_abs"].to_numpy(dtype=float)
    couplings = coupling_scale * fg_abs
    matrix[0, 1:] = couplings
    matrix[1:, 0] = couplings

    hkls = [tuple(map(int, row)) for row in excited[["h", "k", "l"]].itertuples(index=False, name=None)]
    for i, (hi, ki, li) in enumerate(hkls):
        for j in range(i + 1, n_beams):
            hj, kj, lj = hkls[j]
            delta_amp = calibration.lookup_amplitude(hi - hj, ki - kj, li - lj)
            coupling = coupling_scale * delta_amp
            matrix[i + 1, j + 1] = coupling
            matrix[j + 1, i + 1] = coupling

    eigenvalues, eigenvectors = eigh(matrix, overwrite_a=False, check_finite=True)
    return StructureMatrixResult(
        matrix=matrix,
        eigenvalues=np.asarray(eigenvalues, dtype=float),
        eigenvectors=np.asarray(eigenvectors, dtype=float),
        beam_hkls=tuple(hkls),
    )


def propagate_bloch_wave(
    eigenvalues: FloatArray,
    eigenvectors: FloatArray,
    thickness_nm: float | Sequence[float],
) -> ComplexArray:
    """Propagate amplitudes through thickness using the diagonalized basis.

    Notes
    -----
    The propagation model is chosen to be internally consistent with the proxy
    coupling matrix rather than to claim exact physical normalization. The state
    evolution is:

    ``psi(t) = V exp(2 pi i Lambda t_A) V^T psi(0)``

    where ``t_A`` is thickness in angstroms and ``psi(0)`` is a unit transmitted
    beam. For the real symmetric matrix used here, ``V^T`` is the inverse.
    """

    thickness_array_nm = np.atleast_1d(np.asarray(thickness_nm, dtype=float))
    thickness_array_angstrom = thickness_array_nm * 10.0
    psi0 = np.zeros(eigenvalues.shape[0], dtype=np.complex128)
    psi0[0] = 1.0 + 0.0j

    # Modal coefficients at t = 0. Columns of eigenvectors are eigenmodes.
    coeff0 = eigenvectors.T.astype(np.complex128) @ psi0
    phases = np.exp(2.0j * np.pi * np.outer(thickness_array_angstrom, eigenvalues))
    modal_amplitudes = phases * coeff0[np.newaxis, :]
    beam_amplitudes = modal_amplitudes @ eigenvectors.T.astype(np.complex128)
    return beam_amplitudes


def beam_intensities(beam_amplitudes: ComplexArray) -> FloatArray:
    """Return beam intensities ``|psi|^2`` from propagated amplitudes."""

    return np.abs(beam_amplitudes) ** 2

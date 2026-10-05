"""Finite-thickness kinematic reference from the pinned abTEM conventions."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from ase.io import read
from scipy import integrate
import abtem
from abtem.bloch import StructureFactor
from abtem.core.constants import kappa
from abtem.core.energy import energy2sigma, energy2wavelength

ROOT = Path(__file__).resolve().parents[3]
PY3DED = ROOT.parent / "py3DED"
OUT = ROOT / "experiments/si_symmetry_poc/kinematic_reference"
ZARR = ROOT / "experiments/si_symmetry_poc/results/20260924-121554_Si_CollCode51688_paper_cell_1x1x1000_451/bw.zarr"
CIF = ROOT / "experiments/si_symmetry_poc/Si_CollCode51688_paper_cell.cif"
ENERGY = 200000.0
G_MAX_ANALYSIS = 2.0
KINEMATIC_INTENSITY_SCALE = energy2wavelength(ENERGY) / 4.0


def authors_analysis():
    path = PY3DED / "scripts/run_py3DED_hkl+Rint.py"
    spec = importlib.util.spec_from_file_location("authors_rint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_saved():
    authors = authors_analysis()
    integrated = authors.calculate_integrated_intensities(ZARR, g_max=G_MAX_ANALYSIS)
    data = abtem.from_zarr(str(ZARR))
    sg = np.asarray(integrated["excitation_errors"])
    if sg.ndim == 3:
        sg = sg[:, 0, :]
    hkls = np.asarray(integrated["miller_indices"], dtype=int)
    alpha = np.asarray(data.ensemble_axes_metadata[0].values, dtype=float)
    thickness = np.asarray(integrated["thicknesses"], dtype=float)
    dyn_j = np.asarray(integrated["integrated_intensities"], dtype=float)
    basis = np.asarray(data.reciprocal_lattice_vectors[:, 0], dtype=float)
    return sg, hkls, alpha, thickness, dyn_j, basis


def finite_kinematic_intensity(structure_factors, hkls, basis, sg, thickness):
    """First-Born finite-thickness intensity using abTEM's A[h,0] term.

    For transmitted beam 0 and target h, abTEM has
    A_h0 = F_{-h} * prefactor * M_h * M_0,
    A_hh = 2*s_h*M_h/lambda.
    The first-order propagated intensity is |A_h0|^2 times the squared
    finite-slab integral of exp(i*pi*lambda*A_hh*z).
    """
    wavelength = energy2wavelength(ENERGY)
    k0 = 1.0 / wavelength
    prefactor = energy2sigma(ENERGY) / (kappa * wavelength * np.pi)
    g = np.einsum("aij,nj->ani", basis, hkls)
    m_h = 1.0 / np.sqrt(1.0 + g[:, :, 2] / k0)
    # StructureFactor uses F_h; the row h/column 0 matrix element uses F_{-h}.
    f_minus_h = np.asarray([structure_factors.get(tuple(-h), 0.0) for h in hkls], dtype=np.complex128)
    a_h0 = f_minus_h[None, :] * prefactor * m_h
    a_hh = 2.0 * sg * m_h / wavelength
    x = np.pi * wavelength * a_hh[:, None, :] * thickness[None, :, None]
    numerator = np.expm1(1.0j * x)
    slab_integral = np.divide(
        numerator,
        1.0j * np.pi * wavelength * a_hh[:, None, :],
        out=np.full_like(numerator, np.pi * thickness[None, :, None] * wavelength, dtype=np.complex128),
        where=np.abs(a_hh[:, None, :]) > 1e-14,
    )
    return KINEMATIC_INTENSITY_SCALE * np.abs(a_h0[:, None, :] * slab_integral) ** 2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sg, hkls, alpha, thickness, dyn_j, basis = load_saved()
    atoms = read(str(CIF))
    sf = StructureFactor(
        atoms,
        g_max=8.0,
        parametrization="lobato",
        thermal_sigma={"Si": 0.1},
        occupancy=1.0,
        centering="F",
        device="cpu",
    ).build(lazy=False)
    structure_factors = {tuple(h): value for h, value in zip(sf.hkl.astype(int), np.asarray(sf.array))}
    np.savez_compressed(OUT / "static_kinematic_structure_factors.npz", hkl=hkls, F=np.asarray([structure_factors.get(tuple(h), 0.0) for h in hkls]))
    i0 = np.abs(np.asarray([structure_factors.get(tuple(h), 0.0) for h in hkls])) ** 2
    i_alpha_z_h = finite_kinematic_intensity(structure_factors, hkls, basis, sg, thickness)
    zero_index = np.flatnonzero(np.all(hkls == 0, axis=1))
    if zero_index.size:
        i0[zero_index[0]] = np.nan
        i_alpha_z_h[:, :, zero_index[0]] = np.nan
    j_kin = np.empty((len(thickness), len(hkls)), dtype=float)
    for h_index in range(len(hkls)):
        j_kin[:, h_index] = np.abs(integrate.simpson(i_alpha_z_h[:, :, h_index], x=sg[:, h_index], axis=0))
    np.savez_compressed(OUT / "finite_thickness_kinematic.npz", i_kin_alpha_z_h=i_alpha_z_h, j_kin=j_kin, i_kin0=i0, sg=sg, hkls=hkls, alpha=alpha, thickness=thickness)
    rows = []
    for t_index, z in enumerate(thickness):
        for h_index, h in enumerate(hkls):
            rows.append({"reflection_index": h_index, "h": h[0], "k": h[1], "l": h[2], "thickness": z, "I_kin0": i0[h_index], "J_dyn": dyn_j[t_index, h_index], "J_kin": j_kin[t_index, h_index], "log_Jdyn_over_Jkin": np.log(np.divide(dyn_j[t_index, h_index], j_kin[t_index, h_index], out=np.full((), np.nan), where=j_kin[t_index, h_index] > 0))})
    pd.DataFrame(rows).to_csv(OUT / "dynamical_kinematic_comparison.csv", index=False)
    # Small validation metadata and representative values.
    validation = {"energy_eV": ENERGY, "n_hkls": len(hkls), "n_alpha": len(alpha), "n_thickness": len(thickness), "formula": "Ikin=(lambda/4)*|F_-h prefactor M_h|^2 |(exp(i*pi*lambda*A_hh*t)-1)/(i*pi*lambda*A_hh)|^2", "integration": "scipy.integrate.simpson over saved s_g(alpha)", "global_scale_factor": KINEMATIC_INTENSITY_SCALE, "000_handling": "excluded/NaN because py3DED 000 is the transmitted beam"}
    (OUT / "README.md").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print("generated", OUT)
    print("Ikin0 range", float(i0.min()), float(i0.max()))
    print("Jkin range", float(np.nanmin(j_kin)), float(np.nanmax(j_kin)))
    print("finite", np.isfinite(i_alpha_z_h).all(), np.isfinite(j_kin).all())


if __name__ == "__main__":
    main()

"""Metric calculations for proxy and thickness-aware modes."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

from .constants import EPSILON

FloatArray = NDArray[np.float64]


def two_beam_metric(sg_invA: ArrayLike, xi_angstrom: ArrayLike) -> FloatArray:
    """Compute the proxy two-beam dynamicality metric.

    ``d_2beam = 1 / (1 + (s_g * xi_g)^2)``
    """

    sg = np.asarray(sg_invA, dtype=float)
    xi = np.asarray(xi_angstrom, dtype=float)
    return 1.0 / (1.0 + np.square(sg * xi))


def orientation_sg_gradient(
    g_vectors_invA: ArrayLike,
    wavelength_angstrom: float,
    beam_direction: ArrayLike = (0.0, 0.0, 1.0),
) -> FloatArray:
    """Return ``d(s_g)/d(omega)`` for each reflection.

    The gradient is computed with respect to small rotation-vector perturbations
    ``omega`` in radians around the laboratory x/y/z axes.
    """

    g = np.asarray(g_vectors_invA, dtype=float)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError("g_vectors_invA must have shape (n, 3).")

    beam = np.asarray(beam_direction, dtype=float)
    beam_norm = np.linalg.norm(beam)
    if beam_norm <= EPSILON:
        raise ValueError("beam_direction must be non-zero.")
    beam_unit = beam / beam_norm

    k0 = beam_unit / float(wavelength_angstrom)
    shifted = g + k0[None, :]
    shifted_norm = np.linalg.norm(shifted, axis=1)

    n = np.zeros_like(shifted)
    valid = shifted_norm > EPSILON
    n[valid] = shifted[valid] / shifted_norm[valid, None]

    # d s_g = (g x n) · d(omega)
    return np.cross(g, n)


def orientation_sg_sigma(
    g_vectors_invA: ArrayLike,
    wavelength_angstrom: float,
    orientation_sigma_deg: float | tuple[float, float, float] = 0.2,
    beam_direction: ArrayLike = (0.0, 0.0, 1.0),
) -> FloatArray:
    """Return per-reflection orientation-induced ``sigma(s_g)`` in 1/angstrom.

    ``orientation_sigma_deg`` can be isotropic (single value) or anisotropic
    (sx, sy, sz) for laboratory x/y/z rotation components.
    """

    grad = orientation_sg_gradient(
        g_vectors_invA=g_vectors_invA,
        wavelength_angstrom=wavelength_angstrom,
        beam_direction=beam_direction,
    )

    sigma_arr = np.asarray(orientation_sigma_deg, dtype=float)
    if sigma_arr.size == 1:
        sigma_rad = np.deg2rad(float(sigma_arr.reshape(1)[0]))
        return np.linalg.norm(grad, axis=1) * sigma_rad

    if sigma_arr.shape != (3,):
        raise ValueError(
            "orientation_sigma_deg must be a scalar or a length-3 tuple/list (sx, sy, sz)."
        )
    sigma_rad_xyz = np.deg2rad(sigma_arr)
    return np.sqrt(np.sum(np.square(grad * sigma_rad_xyz[None, :]), axis=1))


def orientation_excitation_probability(
    sg_invA: ArrayLike,
    sg_sigma_orient_invA: ArrayLike,
    excitation_tolerance_invA: float = 0.0,
) -> FloatArray:
    """Probability that the true excitation error falls within ``±tolerance``.

    A Gaussian orientation uncertainty model is assumed:
    ``s_g,true ~ N(s_g, sigma_orient)``.
    """

    sg = np.asarray(sg_invA, dtype=float)
    sigma = np.asarray(sg_sigma_orient_invA, dtype=float)
    tol = float(max(excitation_tolerance_invA, 0.0))

    out = np.zeros_like(sg, dtype=float)
    finite_sigma = sigma > EPSILON
    if np.any(finite_sigma):
        s = sigma[finite_sigma]
        a = (tol - sg[finite_sigma]) / (sqrt(2.0) * s)
        b = (-tol - sg[finite_sigma]) / (sqrt(2.0) * s)
        out[finite_sigma] = 0.5 * (erf(a) - erf(b))

    if np.any(~finite_sigma):
        out[~finite_sigma] = (np.abs(sg[~finite_sigma]) <= tol).astype(float)

    return np.clip(out, 0.0, 1.0)


def orientation_proxy_score(
    p_excited_orient: ArrayLike,
    n_eff: ArrayLike,
    formulation: str = "log_n_eff",
) -> FloatArray:
    """Orientation-only dynamical-risk proxy from excitation probability + coupling."""

    p = np.asarray(p_excited_orient, dtype=float)
    n = np.maximum(np.asarray(n_eff, dtype=float), 0.0)
    if formulation == "log_n_eff":
        return p * (1.0 + np.log1p(n))
    if formulation == "linear_n_eff":
        return p * n
    raise ValueError(f"Unknown orientation proxy formulation: {formulation}")


def effective_coupling_multiplicity(eigenvectors: FloatArray, beam_index: int) -> float:
    """Compute the effective coupling multiplicity ``N_eff`` for one beam.

    The browser prototype defines weights as the product of beam and transmitted
    components of each eigenvector. With eigenvectors stored as columns,
    ``w_j = V[beam_index, j] * V[0, j]``.
    """

    weights = eigenvectors[beam_index, :] * eigenvectors[0, :]
    w2 = np.square(weights)
    sum_w2 = float(np.sum(w2))
    sum_w4 = float(np.sum(np.square(w2)))
    if sum_w4 <= EPSILON:
        return 1.0
    return (sum_w2 * sum_w2) / sum_w4


def combined_proxy_score(d_two_beam: ArrayLike, n_eff: ArrayLike) -> FloatArray:
    """Combine two-beam and multi-beam terms into ``S_comb``."""

    return np.asarray(d_two_beam, dtype=float) * np.asarray(n_eff, dtype=float)


def summarize_frame_proxy(frame_table: pd.DataFrame) -> dict[str, float]:
    """Summarize per-reflection proxy metrics for one frame."""

    if frame_table.empty:
        return {
            "n_excited": 0.0,
            "S_2beam": 0.0,
            "S_MB": 0.0,
            "mean_N_eff": 0.0,
            "max_N_eff": 0.0,
        }
    return {
        "n_excited": float(frame_table.shape[0]),
        "S_2beam": float(frame_table["d_2beam"].sum()),
        "S_MB": float(frame_table["S_comb"].sum()),
        "mean_N_eff": float(frame_table["N_eff"].mean()),
        "max_N_eff": float(frame_table["N_eff"].max()),
    }


def thickness_sensitivity_metrics(values: ArrayLike) -> dict[str, float]:
    """Compute simple thickness-sensitivity metrics from a value series."""

    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return {
            "thickness_std": np.nan,
            "thickness_cv": np.nan,
            "thickness_max_min_ratio": np.nan,
            "thickness_normalized_range": np.nan,
        }
    mean_value = float(np.mean(data))
    min_value = float(np.min(data))
    max_value = float(np.max(data))
    std_value = float(np.std(data))
    return {
        "thickness_std": std_value,
        "thickness_cv": std_value / (mean_value + EPSILON),
        "thickness_max_min_ratio": max_value / (min_value + EPSILON),
        "thickness_normalized_range": (max_value - min_value) / (mean_value + EPSILON),
    }


def aggregate_reflection_thickness_sensitivity(thickness_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate thickness sensitivity per reflection at fixed frame."""

    if thickness_table.empty:
        return pd.DataFrame(
            columns=[
                "frame",
                "frame_number",
                "h",
                "k",
                "l",
                "thickness_std",
                "thickness_cv",
                "thickness_max_min_ratio",
                "thickness_normalized_range",
            ]
        )

    rows: list[dict[str, Any]] = []
    group_columns = ["frame", "frame_number", "h", "k", "l"]
    for keys, group in thickness_table.groupby(group_columns, sort=False):
        metrics = thickness_sensitivity_metrics(group["intensity"].to_numpy(dtype=float))
        row = {column: value for column, value in zip(group_columns, keys, strict=True)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def aggregate_frame_thickness_sensitivity(reflection_sensitivity_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate thickness sensitivity across reflections for each frame."""

    if reflection_sensitivity_table.empty:
        return pd.DataFrame(
            columns=[
                "frame",
                "frame_number",
                "frame_thickness_mean_std",
                "frame_thickness_mean_cv",
                "frame_thickness_mean_normalized_range",
            ]
        )

    grouped = (
        reflection_sensitivity_table.groupby(["frame", "frame_number"], sort=False)
        .agg(
            frame_thickness_mean_std=("thickness_std", "mean"),
            frame_thickness_mean_cv=("thickness_cv", "mean"),
            frame_thickness_mean_normalized_range=("thickness_normalized_range", "mean"),
        )
        .reset_index()
    )
    return grouped

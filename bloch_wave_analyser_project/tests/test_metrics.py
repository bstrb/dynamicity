from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import (
    effective_coupling_multiplicity,
    orientation_excitation_probability,
    orientation_proxy_score,
    orientation_sg_sigma,
    thickness_sensitivity_metrics,
    two_beam_metric,
)


def test_two_beam_metric_sanity() -> None:
    sg = np.asarray([0.0, 0.01])
    xi = np.asarray([100.0, 100.0])
    values = two_beam_metric(sg, xi)
    assert np.isclose(values[0], 1.0)
    assert values[1] < values[0]


def test_effective_coupling_multiplicity_identity_modes() -> None:
    eigenvectors = np.eye(3)
    n_eff = effective_coupling_multiplicity(eigenvectors, beam_index=1)
    assert np.isclose(n_eff, 1.0)


def test_thickness_sensitivity_metrics() -> None:
    metrics = thickness_sensitivity_metrics(np.asarray([1.0, 2.0, 4.0]))
    assert metrics["thickness_std"] > 0.0
    assert metrics["thickness_cv"] > 0.0
    assert np.isclose(metrics["thickness_max_min_ratio"], 4.0)
    assert metrics["thickness_normalized_range"] > 0.0


def test_orientation_sigma_zero_for_beam_parallel_reflection() -> None:
    g_vectors = np.asarray([[0.0, 0.0, 0.12]], dtype=float)
    sigma = orientation_sg_sigma(g_vectors, wavelength_angstrom=0.0251, orientation_sigma_deg=0.2)
    assert np.isclose(float(sigma[0]), 0.0, atol=1e-12)


def test_orientation_sigma_positive_for_off_axis_reflection() -> None:
    g_vectors = np.asarray([[0.08, 0.01, 0.12]], dtype=float)
    sigma_iso = orientation_sg_sigma(g_vectors, wavelength_angstrom=0.0251, orientation_sigma_deg=0.2)
    sigma_aniso = orientation_sg_sigma(
        g_vectors,
        wavelength_angstrom=0.0251,
        orientation_sigma_deg=(0.1, 0.2, 0.3),
    )
    assert float(sigma_iso[0]) > 0.0
    assert float(sigma_aniso[0]) > 0.0


def test_orientation_excitation_probability_is_bounded_and_ordered() -> None:
    sg = np.asarray([0.0, 0.02], dtype=float)
    sigma = np.asarray([0.003, 0.003], dtype=float)
    p = orientation_excitation_probability(sg, sigma, excitation_tolerance_invA=0.001)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p[0] > p[1]

    p_zero_sigma = orientation_excitation_probability(
        np.asarray([0.0, 0.01], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
        excitation_tolerance_invA=0.0,
    )
    assert np.isclose(p_zero_sigma[0], 1.0)
    assert np.isclose(p_zero_sigma[1], 0.0)


def test_orientation_proxy_score_monotonic_with_probability() -> None:
    score = orientation_proxy_score(
        p_excited_orient=np.asarray([0.1, 0.8], dtype=float),
        n_eff=np.asarray([2.0, 2.0], dtype=float),
    )
    assert score[1] > score[0]

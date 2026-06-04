from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import (
    effective_coupling_multiplicity,
    empirical_amplitude_strength_proxy,
    geometry_dynamical_risk,
    orientation_excitation_probability,
    orientation_proxy_score,
    orientation_sg_sigma,
    primitive_hkl_key,
    reciprocal_strength_proxy,
    thickness_sensitivity_metrics,
    two_channel_dynamical_risk,
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


def test_geometry_dynamical_risk_is_not_target_excitation_score() -> None:
    reflections = pd.DataFrame(
        {
            "h": [2, 1],
            "k": [0, 0],
            "l": [0, 0],
            "q_invA": [0.2, 0.1],
        }
    )
    hkl_tuples = [(2, 0, 0), (1, 0, 0)]
    row_keys = [primitive_hkl_key(*hkl) for hkl in hkl_tuples]
    hkl_to_index = {hkl: index for index, hkl in enumerate(hkl_tuples)}
    strength = reciprocal_strength_proxy(reflections["q_invA"].to_numpy(dtype=float))
    g_vectors = np.asarray([[0.2, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)

    common = dict(
        candidate_reflections=reflections,
        g_vectors_invA=g_vectors,
        target_mask=np.asarray([True, False]),
        environment_mask=np.asarray([True, True]),
        zone_axis=(0, 0, 1),
        zone_axis_angle_deg=0.0,
        hkl_to_index=hkl_to_index,
        hkl_tuples=hkl_tuples,
        row_keys=row_keys,
        strength_proxy=strength,
        environment_tolerance_invA=0.01,
        environment_weight_min=0.02,
        neighbor_radius_invA=0.15,
    )
    score_a = geometry_dynamical_risk(
        sg_invA=np.asarray([0.10, 0.0], dtype=float),
        **common,
    )
    score_b = geometry_dynamical_risk(
        sg_invA=np.asarray([0.20, 0.0], dtype=float),
        **common,
    )

    assert float(score_a["S_dyn"].iloc[0]) > 0.0
    assert np.isclose(float(score_a["S_dyn"].iloc[0]), float(score_b["S_dyn"].iloc[0]))
    assert not np.isclose(float(score_a["S_dyn"].iloc[0]), 0.0)


def test_two_channel_cluster_uses_candidate_neighbor_not_target_excitation() -> None:
    reflections = pd.DataFrame(
        {
            "h": [2, 1, 0],
            "k": [0, 0, 1],
            "l": [0, 0, 0],
            "q_invA": [0.2, 0.1, 0.3],
        }
    )
    hkl_tuples = [(2, 0, 0), (1, 0, 0), (0, 1, 0)]
    hkl_to_index = {hkl: index for index, hkl in enumerate(hkl_tuples)}
    strength = reciprocal_strength_proxy(reflections["q_invA"].to_numpy(dtype=float))
    g_vectors = np.asarray(
        [
            [0.2, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.3, 0.2],
        ],
        dtype=float,
    )

    common = dict(
        candidate_reflections=reflections,
        g_vectors_invA=g_vectors,
        target_mask=np.asarray([True, False, False]),
        hkl_to_index=hkl_to_index,
        hkl_tuples=hkl_tuples,
        strength_proxy=strength,
        excitation_tolerance_invA=0.01,
        environment_weight_min=0.02,
        neighbor_radius_invA=0.15,
        zone_sigma_invA=0.03,
        row_direction_limit=0,
    )
    near_target = two_channel_dynamical_risk(
        sg_invA=np.asarray([0.0, 0.0, 0.3], dtype=float),
        **common,
    )
    far_target = two_channel_dynamical_risk(
        sg_invA=np.asarray([0.2, 0.0, 0.3], dtype=float),
        **common,
    )

    assert float(near_target["same_zone_cluster_score_geom"].iloc[0]) > 0.0
    assert np.isclose(
        float(near_target["same_zone_cluster_score_geom"].iloc[0]),
        float(far_target["same_zone_cluster_score_geom"].iloc[0]),
    )
    assert float(near_target["self_extinction_score"].iloc[0]) > float(far_target["self_extinction_score"].iloc[0])


def test_two_channel_row_score_finds_low_order_chain() -> None:
    reflections = pd.DataFrame(
        {
            "h": [2, 3, 4],
            "k": [0, 0, 1],
            "l": [0, 0, 0],
            "q_invA": [0.2, 0.3, 0.4],
        }
    )
    hkl_tuples = [(2, 0, 0), (3, 0, 0), (4, 1, 0)]
    hkl_to_index = {hkl: index for index, hkl in enumerate(hkl_tuples)}
    strength = reciprocal_strength_proxy(reflections["q_invA"].to_numpy(dtype=float))
    g_vectors = np.asarray(
        [
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.2, 0.0],
        ],
        dtype=float,
    )

    score = two_channel_dynamical_risk(
        candidate_reflections=reflections,
        g_vectors_invA=g_vectors,
        sg_invA=np.asarray([0.0, 0.0, 0.0], dtype=float),
        target_mask=np.asarray([True, False, False]),
        hkl_to_index=hkl_to_index,
        hkl_tuples=hkl_tuples,
        strength_proxy=strength,
        excitation_tolerance_invA=0.01,
        environment_weight_min=0.02,
        neighbor_radius_invA=0.05,
        row_direction_limit=1,
        row_max_steps=2,
        row_sigma_invA=0.2,
    )

    assert float(score["same_zone_cluster_score_geom"].iloc[0]) == 0.0
    assert float(score["systematic_row_cluster_score_geom"].iloc[0]) > 0.0
    assert int(score["max_row_direction_h"].iloc[0]) == 1


def test_empirical_amplitude_strength_proxy_uses_observed_family_strength() -> None:
    reflections = pd.DataFrame(
        {
            "h": [1, 2],
            "k": [0, 0],
            "l": [0, 0],
        }
    )
    observations = pd.DataFrame(
        {
            "h": [1, -1, 2],
            "k": [0, 0, 0],
            "l": [0, 0, 0],
            "I": [100.0, 100.0, 25.0],
        }
    )
    proxy = empirical_amplitude_strength_proxy(reflections, observations)

    assert np.isclose(float(proxy[0]), 1.0)
    assert 0.0 < float(proxy[1]) < float(proxy[0])

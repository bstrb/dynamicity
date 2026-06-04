"""Metric calculations for proxy and thickness-aware modes."""

from __future__ import annotations

from math import gcd, sqrt
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf
from scipy.spatial import cKDTree

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


def reciprocal_strength_proxy(
    q_invA: ArrayLike,
    *,
    power: float = 1.5,
    q_floor_invA: float = 0.05,
) -> FloatArray:
    """Return a geometry-only low-index/strong-reflection proxy.

    The proxy deliberately avoids intensities and scattering-factor refinement.
    It only says that smaller ``|g|`` reflections are usually more dangerous
    for electron-diffraction dynamical coupling. Values are normalized to
    ``[0, 1]`` within the supplied reflection set.
    """

    q = np.asarray(q_invA, dtype=float)
    floor = max(float(q_floor_invA), EPSILON)
    raw = np.power(1.0 / np.maximum(q, floor), float(power))
    max_raw = float(np.nanmax(raw)) if raw.size else 0.0
    if max_raw <= EPSILON:
        return np.zeros_like(q, dtype=float)
    return np.clip(raw / max_raw, 0.0, 1.0)


def empirical_amplitude_strength_proxy(
    candidate_reflections: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    fallback_values: ArrayLike | None = None,
) -> FloatArray:
    """Return a model-free empirical source-strength proxy for candidate HKLs.

    The proxy is based on the square root of positive observed intensities,
    merged by the same simple absolute-value family key used elsewhere in the
    prototype. Missing families fall back to ``fallback_values`` when supplied.
    """

    n_candidates = int(candidate_reflections.shape[0])
    if fallback_values is None:
        fallback = np.zeros(n_candidates, dtype=float)
    else:
        fallback = np.asarray(fallback_values, dtype=float)
        if fallback.shape[0] != n_candidates:
            raise ValueError("fallback_values must match candidate_reflections length.")

    required = {"h", "k", "l", "I"}
    if observations.empty or not required.issubset(observations.columns):
        return fallback.copy()

    obs = observations[list(required)].copy()
    obs = obs[np.isfinite(obs["I"].to_numpy(dtype=float)) & (obs["I"].to_numpy(dtype=float) > 0.0)]
    if obs.empty:
        return fallback.copy()

    obs["family_key"] = [
        tuple(sorted((abs(int(h)), abs(int(k)), abs(int(l))), reverse=True))
        for h, k, l in obs[["h", "k", "l"]].itertuples(index=False, name=None)
    ]
    family_mean = obs.groupby("family_key", sort=False)["I"].mean()
    family_amp = np.sqrt(np.maximum(family_mean.to_numpy(dtype=float), 0.0))
    amp_max = float(np.nanmax(family_amp)) if family_amp.size else 0.0
    if amp_max <= EPSILON:
        return fallback.copy()

    family_proxy = {
        key: float(value)
        for key, value in zip(family_mean.index, family_amp / amp_max, strict=True)
    }
    out = fallback.copy()
    for idx, row in enumerate(candidate_reflections[["h", "k", "l"]].itertuples(index=False, name=None)):
        key = tuple(sorted((abs(int(row[0])), abs(int(row[1])), abs(int(row[2]))), reverse=True))
        if key in family_proxy:
            out[idx] = family_proxy[key]
    return np.clip(out, 0.0, 1.0)


def primitive_hkl_key(h: int, k: int, l: int) -> tuple[int, int, int]:
    """Return a sign-normalized primitive reciprocal-row key."""

    values = (int(h), int(k), int(l))
    divisor = 0
    for value in values:
        divisor = gcd(divisor, abs(value))
    if divisor == 0:
        return (0, 0, 0)
    key = tuple(value // divisor for value in values)
    for value in key:
        if value != 0:
            return tuple(-item for item in key) if value < 0 else key
    return key


def _low_order_row_directions(limit: int) -> list[tuple[int, int, int]]:
    row_limit = max(int(limit), 0)
    if row_limit == 0:
        return []

    seen: set[tuple[int, int, int]] = set()
    directions: list[tuple[int, int, int]] = []
    for h in range(-row_limit, row_limit + 1):
        for k in range(-row_limit, row_limit + 1):
            for l in range(-row_limit, row_limit + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                key = primitive_hkl_key(h, k, l)
                if key == (0, 0, 0) or key in seen:
                    continue
                seen.add(key)
                directions.append(key)
    return sorted(directions, key=lambda item: (item[0] * item[0] + item[1] * item[1] + item[2] * item[2], item))


def _coupling_proxy_from_q(
    q_invA: ArrayLike,
    *,
    q0_invA: float,
    power: float,
    q_floor_invA: float,
) -> FloatArray:
    q = np.maximum(np.asarray(q_invA, dtype=float), max(float(q_floor_invA), EPSILON))
    q0 = max(float(q0_invA), EPSILON)
    return 1.0 / (1.0 + np.power(q / q0, float(power)))


def two_channel_dynamical_risk(
    *,
    candidate_reflections: pd.DataFrame,
    g_vectors_invA: ArrayLike,
    sg_invA: ArrayLike,
    target_mask: ArrayLike,
    hkl_to_index: dict[tuple[int, int, int], int],
    hkl_tuples: Sequence[tuple[int, int, int]],
    strength_proxy: ArrayLike,
    empirical_strength_proxy: ArrayLike | None = None,
    environment_mask: ArrayLike | None = None,
    beam_direction: ArrayLike = (0.0, 0.0, 1.0),
    excitation_tolerance_invA: float = 1.0e-2,
    environment_weight_min: float = 2.0e-2,
    zone_sigma_invA: float = 0.06,
    neighbor_radius_invA: float = 0.12,
    neighbor_sigma_invA: float | None = None,
    row_direction_limit: int = 2,
    row_max_steps: int = 5,
    row_sigma_invA: float = 0.25,
    coupling_q0_invA: float = 0.25,
    coupling_power: float = 2.0,
    coupling_q_floor_invA: float = 0.02,
    weight_self: float = 1.0,
    weight_zone: float = 1.0,
    weight_row: float = 1.0,
    sigma_alpha: float = 1.0,
) -> pd.DataFrame:
    """Two-channel reflection-level dynamical-risk proxy.

    Targets are selected by ``target_mask``. Neighbor beams are taken from the
    full candidate set subject only to excitation and ``environment_mask``; they
    are not required to be observed or integrated.
    """

    g_vectors = np.asarray(g_vectors_invA, dtype=float)
    sg = np.asarray(sg_invA, dtype=float)
    targets = np.asarray(target_mask, dtype=bool)
    strength = np.asarray(strength_proxy, dtype=float)
    empirical = strength if empirical_strength_proxy is None else np.asarray(empirical_strength_proxy, dtype=float)
    if environment_mask is None:
        env_base = np.ones(targets.shape[0], dtype=bool)
    else:
        env_base = np.asarray(environment_mask, dtype=bool)

    if g_vectors.ndim != 2 or g_vectors.shape[1] != 3:
        raise ValueError("g_vectors_invA must have shape (n, 3).")
    n_reflections = g_vectors.shape[0]
    if sg.shape[0] != n_reflections or targets.shape[0] != n_reflections:
        raise ValueError("sg_invA and target_mask must match g_vectors length.")
    if env_base.shape[0] != n_reflections:
        raise ValueError("environment_mask must match g_vectors length.")
    if strength.shape[0] != n_reflections or empirical.shape[0] != n_reflections:
        raise ValueError("strength proxies must match g_vectors length.")

    columns = [
        "excitation_weight_target",
        "beam_parallel_zeta_invA",
        "direct_coupling_proxy",
        "self_extinction_score",
        "attenuation_risk",
        "same_zone_cluster_score_geom",
        "same_zone_cluster_score_iw",
        "systematic_row_cluster_score_geom",
        "systematic_row_cluster_score_iw",
        "cluster_score_geom",
        "cluster_score_iw",
        "cluster_risk_geom",
        "cluster_risk_iw",
        "total_dynamical_risk_geom",
        "total_dynamical_risk_iw",
        "sigma_dyn_rel",
        "n_zone_neighbors",
        "n_row_neighbors",
        "max_row_direction_h",
        "max_row_direction_k",
        "max_row_direction_l",
    ]

    target_indices = np.flatnonzero(targets)
    if target_indices.size == 0:
        return pd.DataFrame(columns=columns)

    beam = np.asarray(beam_direction, dtype=float)
    beam_norm = float(np.linalg.norm(beam))
    if beam_norm <= EPSILON:
        raise ValueError("beam_direction must be non-zero.")
    beam_unit = beam / beam_norm
    zeta = g_vectors @ beam_unit

    excitation_sigma = max(float(excitation_tolerance_invA), EPSILON)
    excitation_weight = np.exp(-0.5 * np.square(sg / excitation_sigma))
    env_indices = np.flatnonzero(
        env_base
        & np.isfinite(excitation_weight)
        & (excitation_weight >= float(environment_weight_min))
    )

    direct_coupling_all = _coupling_proxy_from_q(
        candidate_reflections["q_invA"].to_numpy(dtype=float),
        q0_invA=coupling_q0_invA,
        power=coupling_power,
        q_floor_invA=coupling_q_floor_invA,
    )

    tree = cKDTree(g_vectors[env_indices]) if env_indices.size else None
    neighbor_radius = max(float(neighbor_radius_invA), EPSILON)
    neighbor_sigma = max(float(neighbor_sigma_invA) if neighbor_sigma_invA is not None else 0.5 * neighbor_radius, EPSILON)
    zone_sigma = max(float(zone_sigma_invA), EPSILON)
    row_sigma = max(float(row_sigma_invA), EPSILON)
    row_directions = _low_order_row_directions(int(row_direction_limit))
    max_steps = max(int(row_max_steps), 0)

    rows: list[dict[str, float | int]] = []
    for target_index_raw in target_indices:
        target_index = int(target_index_raw)
        target_g = g_vectors[target_index]
        target_zeta = float(zeta[target_index])
        e_target = float(excitation_weight[target_index])
        direct_coupling = float(direct_coupling_all[target_index])
        self_score = e_target * direct_coupling
        attenuation_risk = float(1.0 - np.exp(-float(weight_self) * self_score))

        zone_geom = 0.0
        zone_iw = 0.0
        n_zone_neighbors = 0
        if tree is not None:
            local_neighbor_indices = tree.query_ball_point(target_g, r=neighbor_radius)
            for local_index in local_neighbor_indices:
                neighbor_index = int(env_indices[int(local_index)])
                if neighbor_index == target_index:
                    continue
                diff = target_g - g_vectors[neighbor_index]
                diff_norm = float(np.linalg.norm(diff))
                if diff_norm <= EPSILON:
                    continue
                coupling = float(
                    _coupling_proxy_from_q(
                        np.asarray([diff_norm], dtype=float),
                        q0_invA=coupling_q0_invA,
                        power=coupling_power,
                        q_floor_invA=coupling_q_floor_invA,
                    )[0]
                )
                zone_compat = float(np.exp(-0.5 * np.square((target_zeta - float(zeta[neighbor_index])) / zone_sigma)))
                local_window = float(np.exp(-0.5 * np.square(diff_norm / neighbor_sigma)))
                base = float(excitation_weight[neighbor_index]) * coupling * zone_compat * local_window
                zone_geom += base * float(strength[neighbor_index])
                zone_iw += base * float(empirical[neighbor_index])
                n_zone_neighbors += 1

        target_hkl = hkl_tuples[target_index]
        best_row_geom = 0.0
        best_row_iw = 0.0
        best_row_count = 0
        best_direction = (0, 0, 0)
        for direction in row_directions:
            row_geom = 0.0
            row_iw = 0.0
            row_count = 0
            for step in range(-max_steps, max_steps + 1):
                if step == 0:
                    continue
                neighbor_hkl = (
                    target_hkl[0] + step * direction[0],
                    target_hkl[1] + step * direction[1],
                    target_hkl[2] + step * direction[2],
                )
                neighbor_index = hkl_to_index.get(neighbor_hkl)
                if neighbor_index is None or neighbor_index == target_index:
                    continue
                neighbor_excitation = float(excitation_weight[neighbor_index])
                if (
                    not bool(env_base[neighbor_index])
                    or not np.isfinite(neighbor_excitation)
                    or neighbor_excitation < float(environment_weight_min)
                ):
                    continue
                diff_norm = float(np.linalg.norm(target_g - g_vectors[neighbor_index]))
                if diff_norm <= EPSILON:
                    continue
                coupling = float(
                    _coupling_proxy_from_q(
                        np.asarray([diff_norm], dtype=float),
                        q0_invA=coupling_q0_invA,
                        power=coupling_power,
                        q_floor_invA=coupling_q_floor_invA,
                    )[0]
                )
                row_window = float(np.exp(-0.5 * np.square(diff_norm / row_sigma)))
                base = neighbor_excitation * coupling * row_window
                row_geom += base * float(strength[neighbor_index])
                row_iw += base * float(empirical[neighbor_index])
                row_count += 1
            if row_geom > best_row_geom:
                best_row_geom = row_geom
                best_row_iw = row_iw
                best_row_count = row_count
                best_direction = direction

        cluster_score_geom = float(weight_zone) * zone_geom + float(weight_row) * best_row_geom
        cluster_score_iw = float(weight_zone) * zone_iw + float(weight_row) * best_row_iw
        cluster_risk_geom = float(1.0 - np.exp(-cluster_score_geom))
        cluster_risk_iw = float(1.0 - np.exp(-cluster_score_iw))
        total_risk_geom = float(1.0 - np.exp(-(float(weight_self) * self_score + cluster_score_geom)))
        total_risk_iw = float(1.0 - np.exp(-(float(weight_self) * self_score + cluster_score_iw)))

        rows.append(
            {
                "excitation_weight_target": e_target,
                "beam_parallel_zeta_invA": target_zeta,
                "direct_coupling_proxy": direct_coupling,
                "self_extinction_score": self_score,
                "attenuation_risk": attenuation_risk,
                "same_zone_cluster_score_geom": float(zone_geom),
                "same_zone_cluster_score_iw": float(zone_iw),
                "systematic_row_cluster_score_geom": float(best_row_geom),
                "systematic_row_cluster_score_iw": float(best_row_iw),
                "cluster_score_geom": cluster_score_geom,
                "cluster_score_iw": cluster_score_iw,
                "cluster_risk_geom": cluster_risk_geom,
                "cluster_risk_iw": cluster_risk_iw,
                "total_dynamical_risk_geom": total_risk_geom,
                "total_dynamical_risk_iw": total_risk_iw,
                "sigma_dyn_rel": float(1.0 + float(sigma_alpha) * cluster_risk_geom),
                "n_zone_neighbors": int(n_zone_neighbors),
                "n_row_neighbors": int(best_row_count),
                "max_row_direction_h": int(best_direction[0]),
                "max_row_direction_k": int(best_direction[1]),
                "max_row_direction_l": int(best_direction[2]),
            }
        )

    return pd.DataFrame.from_records(rows, columns=columns)


def _low_index_axis_weight(axis: tuple[int, int, int]) -> float:
    norm = sqrt(float(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]))
    if norm <= EPSILON:
        return 0.0
    return 1.0 / norm


def _axis_zone_risk(
    axis: tuple[int, int, int],
    angle_deg: float,
    sigma_deg: float,
) -> float:
    sigma = max(float(sigma_deg), EPSILON)
    angular_weight = float(np.exp(-0.5 * np.square(float(angle_deg) / sigma)))
    return _low_index_axis_weight(axis) * angular_weight


def geometry_dynamical_risk(
    *,
    candidate_reflections: pd.DataFrame,
    g_vectors_invA: ArrayLike,
    sg_invA: ArrayLike,
    target_mask: ArrayLike,
    environment_mask: ArrayLike,
    zone_axis: tuple[int, int, int],
    zone_axis_angle_deg: float,
    hkl_to_index: dict[tuple[int, int, int], int],
    hkl_tuples: Sequence[tuple[int, int, int]],
    row_keys: Sequence[tuple[int, int, int]],
    strength_proxy: ArrayLike,
    environment_tolerance_invA: float = 1.0e-2,
    environment_weight_min: float = 2.0e-2,
    neighbor_radius_invA: float = 0.12,
    zone_axis_sigma_deg: float = 3.0,
    component_weights: tuple[float, float, float, float, float] = (2.0, 1.0, 1.0, 1.5, 0.5),
) -> pd.DataFrame:
    """Score orientation-determined dynamical risk from coupling geometry.

    ``target_mask`` is the visibility/relevance gate for reflections being
    scored. The target reflection's own excitation error is not used as the
    risk score. Instead, risk is built from the simultaneously excited
    low-index environment around the target:

    - ZOLZ proximity to the nearest low-index zone axis
    - weighted nearby excited-neighbor density
    - excited beams along the target's reciprocal row
    - two-step pathways ``0 -> h -> g`` where ``g-h`` is plausible/strong
    - local crowding of excited beams around ``g``
    """

    g_vectors = np.asarray(g_vectors_invA, dtype=float)
    sg = np.asarray(sg_invA, dtype=float)
    targets = np.asarray(target_mask, dtype=bool)
    env_base = np.asarray(environment_mask, dtype=bool)
    strength = np.asarray(strength_proxy, dtype=float)
    if g_vectors.ndim != 2 or g_vectors.shape[1] != 3:
        raise ValueError("g_vectors_invA must have shape (n, 3).")
    n_reflections = g_vectors.shape[0]
    if sg.shape[0] != n_reflections or targets.shape[0] != n_reflections:
        raise ValueError("sg_invA and target_mask must match g_vectors length.")
    if env_base.shape[0] != n_reflections or strength.shape[0] != n_reflections:
        raise ValueError("environment_mask and strength_proxy must match g_vectors length.")

    target_indices = np.flatnonzero(targets)
    columns = [
        "visibility_gate",
        "strength_proxy",
        "zone_order",
        "is_zolz",
        "zone_axis_risk",
        "ZOLZ_zone_axis_risk",
        "neighbor_density",
        "row_coupling",
        "pathway_risk",
        "local_crowding",
        "S_dyn",
    ]
    if target_indices.size == 0:
        return pd.DataFrame(columns=columns)

    env_tol = max(float(environment_tolerance_invA), EPSILON)
    excitation_weight = np.exp(-0.5 * np.square(sg / env_tol))
    env_indices = np.flatnonzero(
        env_base
        & np.isfinite(excitation_weight)
        & (excitation_weight >= float(environment_weight_min))
    )
    env_score = strength[env_indices] * excitation_weight[env_indices]
    env_index_to_score = {
        int(index): float(score)
        for index, score in zip(env_indices, env_score, strict=True)
    }

    row_sums: dict[tuple[int, int, int], float] = {}
    for index, score in env_index_to_score.items():
        key = row_keys[index]
        row_sums[key] = row_sums.get(key, 0.0) + score

    if env_indices.size:
        tree = cKDTree(g_vectors[env_indices])
    else:
        tree = None

    axis = tuple(int(value) for value in zone_axis)
    axis_risk = _axis_zone_risk(axis, float(zone_axis_angle_deg), float(zone_axis_sigma_deg))
    zone_order_all = (
        candidate_reflections[["h", "k", "l"]].to_numpy(dtype=int)
        @ np.asarray(axis, dtype=int)
    )
    is_zolz_all = zone_order_all == 0

    zone_weight, neighbor_weight, row_weight, pathway_weight, crowding_weight = component_weights
    neighbor_radius = max(float(neighbor_radius_invA), EPSILON)

    rows: list[dict[str, float | int | bool]] = []
    for target_index in target_indices:
        target_index_int = int(target_index)
        neighbors: list[int] = []
        if tree is not None:
            local_neighbor_indices = tree.query_ball_point(g_vectors[target_index_int], r=neighbor_radius)
            neighbors = [int(env_indices[local_index]) for local_index in local_neighbor_indices]

        neighbor_density = 0.0
        local_crowding = 0.0
        for neighbor_index in neighbors:
            if neighbor_index == target_index_int:
                continue
            score = env_index_to_score.get(neighbor_index, 0.0)
            neighbor_density += score
            local_crowding += excitation_weight[neighbor_index]

        target_row_key = row_keys[target_index_int]
        row_coupling = row_sums.get(target_row_key, 0.0) - env_index_to_score.get(target_index_int, 0.0)
        row_coupling = max(row_coupling, 0.0)

        target_hkl = hkl_tuples[target_index_int]
        pathway_risk = 0.0
        for env_index in env_indices:
            env_index_int = int(env_index)
            if env_index_int == target_index_int:
                continue
            env_hkl = hkl_tuples[env_index_int]
            delta = (
                target_hkl[0] - env_hkl[0],
                target_hkl[1] - env_hkl[1],
                target_hkl[2] - env_hkl[2],
            )
            delta_index = hkl_to_index.get(delta)
            if delta_index is None or delta_index == target_index_int:
                continue
            pathway_risk += (
                float(excitation_weight[env_index_int])
                * float(strength[env_index_int])
                * float(strength[delta_index])
            )

        zolz_zone_axis_risk = axis_risk if bool(is_zolz_all[target_index_int]) else 0.0
        environment_risk = (
            float(zone_weight) * zolz_zone_axis_risk
            + float(neighbor_weight) * np.log1p(neighbor_density)
            + float(row_weight) * np.log1p(row_coupling)
            + float(pathway_weight) * np.log1p(pathway_risk)
            + float(crowding_weight) * np.log1p(local_crowding)
        )
        s_dyn = float(strength[target_index_int]) * float(environment_risk)

        rows.append(
            {
                "visibility_gate": 1.0,
                "strength_proxy": float(strength[target_index_int]),
                "zone_order": int(zone_order_all[target_index_int]),
                "is_zolz": bool(is_zolz_all[target_index_int]),
                "zone_axis_risk": float(axis_risk),
                "ZOLZ_zone_axis_risk": float(zolz_zone_axis_risk),
                "neighbor_density": float(neighbor_density),
                "row_coupling": float(row_coupling),
                "pathway_risk": float(pathway_risk),
                "local_crowding": float(local_crowding),
                "S_dyn": s_dyn,
            }
        )

    return pd.DataFrame.from_records(rows, columns=columns)


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

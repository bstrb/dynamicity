"""Observation-level dynamical uncertainty screening (intensity-independent).

This module estimates a per-observation dynamical uncertainty contribution using
only indexed geometry/orientation metadata and a simplified forward response.
Observed intensities are not used as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from .geometry import RotationSeriesOrientationModel
from .parsers import (
    GXPARMData,
    IntegrateData,
    crystfel_stream_to_analysis_inputs,
    parse_crystfel_stream,
    parse_gxparm,
    parse_integrate_hkl,
)

FloatArray = np.ndarray


@dataclass(frozen=True)
class DynamicalUncertaintyConfig:
    """Configuration for observation-level orientation/thickness screening."""

    orientation_axes: tuple[str, ...] = ("x", "y", "z")
    orientation_step_deg: float = 0.05
    orientation_n_steps: int = 1
    thickness_min_nm: float = 20.0
    thickness_max_nm: float = 300.0
    n_thickness_steps: int = 15
    orientation_thickness_ref_nm: float | None = None
    fg_q0_invA: float = 0.25
    fg_scale: float = 1.0
    zone_axis_layer_band_invA: float = 0.06
    zone_axis_zolz_relative_width: float = 0.08
    zone_axis_boost_weight: float = 4.0
    risk_weight_orientation: float = 1.0
    risk_weight_thickness: float = 1.0
    risk_normalization_quantile: float = 0.95
    dyn_sigma_form: Literal["linear", "exp"] = "linear"
    dyn_sigma_alpha: float = 1.0
    include_sigma_dyn: bool = False
    beam_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    eps: float = 1e-12
    max_min_ratio_cap: float = 1e6

    def validate(self) -> None:
        if self.orientation_step_deg <= 0.0:
            raise ValueError("orientation_step_deg must be > 0.")
        if self.orientation_n_steps < 1:
            raise ValueError("orientation_n_steps must be >= 1.")
        if not self.orientation_axes:
            raise ValueError("orientation_axes must not be empty.")
        valid_axes = {"x", "y", "z"}
        bad_axes = [axis for axis in self.orientation_axes if axis not in valid_axes]
        if bad_axes:
            raise ValueError(f"Unsupported orientation axes: {bad_axes}")
        if self.thickness_min_nm <= 0.0 or self.thickness_max_nm <= 0.0:
            raise ValueError("thickness_min_nm and thickness_max_nm must be > 0.")
        if self.thickness_max_nm < self.thickness_min_nm:
            raise ValueError("thickness_max_nm must be >= thickness_min_nm.")
        if self.n_thickness_steps < 2:
            raise ValueError("n_thickness_steps must be >= 2.")
        if self.zone_axis_layer_band_invA <= 0.0:
            raise ValueError("zone_axis_layer_band_invA must be > 0.")
        if self.zone_axis_zolz_relative_width <= 0.0:
            raise ValueError("zone_axis_zolz_relative_width must be > 0.")
        if not (0.0 < self.risk_normalization_quantile <= 1.0):
            raise ValueError("risk_normalization_quantile must be in (0, 1].")
        if self.dyn_sigma_form not in {"linear", "exp"}:
            raise ValueError("dyn_sigma_form must be 'linear' or 'exp'.")
        if np.linalg.norm(np.asarray(self.beam_direction, dtype=float)) <= self.eps:
            raise ValueError("beam_direction must be non-zero.")

    def thickness_grid_nm(self) -> FloatArray:
        return np.linspace(
            float(self.thickness_min_nm),
            float(self.thickness_max_nm),
            int(self.n_thickness_steps),
            dtype=float,
        )

    def reference_thickness_nm(self) -> float:
        if self.orientation_thickness_ref_nm is not None:
            return float(self.orientation_thickness_ref_nm)
        return 0.5 * (float(self.thickness_min_nm) + float(self.thickness_max_nm))


@dataclass(frozen=True)
class CanonicalObservationData:
    """Canonical observation-level representation for uncertainty screening."""

    source: str
    dataset_id: str
    gxparm: GXPARMData
    observations: pd.DataFrame


@dataclass(frozen=True)
class DynamicalUncertaintyResult:
    """Output tables from the observation-level dynamical uncertainty pipeline."""

    config: DynamicalUncertaintyConfig
    canonical: CanonicalObservationData
    uncertainty_table: pd.DataFrame


@dataclass(frozen=True)
class _OrientationNeighborhood:
    """Precomputed local orientation perturbation scheme."""

    rotations: FloatArray  # (n_pert, 3, 3)
    deltas_rad: FloatArray  # (n_pert, 3)
    radial_magnitude_rad: FloatArray  # (n_pert,)
    axis_levels_to_index: dict[str, dict[int, int]]


def _rotation_matrix_x_deg(angle_deg: float) -> FloatArray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rotation_matrix_y_deg(angle_deg: float) -> FloatArray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rotation_matrix_z_deg(angle_deg: float) -> FloatArray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _axis_rotation(axis: str, angle_deg: float) -> FloatArray:
    if axis == "x":
        return _rotation_matrix_x_deg(angle_deg)
    if axis == "y":
        return _rotation_matrix_y_deg(angle_deg)
    if axis == "z":
        return _rotation_matrix_z_deg(angle_deg)
    raise ValueError(f"Unsupported axis {axis!r}")


def _orientation_neighborhood(config: DynamicalUncertaintyConfig) -> _OrientationNeighborhood:
    rotations: list[FloatArray] = [np.eye(3, dtype=float)]
    deltas: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    axis_levels_to_index: dict[str, dict[int, int]] = {axis: {0: 0} for axis in config.orientation_axes}
    axis_to_idx = {"x": 0, "y": 1, "z": 2}

    for axis in config.orientation_axes:
        for level in range(1, int(config.orientation_n_steps) + 1):
            for signed_level in (-level, level):
                angle_deg = float(signed_level) * float(config.orientation_step_deg)
                rotation = _axis_rotation(axis, angle_deg)
                delta_vec = [0.0, 0.0, 0.0]
                delta_vec[axis_to_idx[axis]] = np.deg2rad(angle_deg)
                index = len(rotations)
                rotations.append(rotation)
                deltas.append((delta_vec[0], delta_vec[1], delta_vec[2]))
                axis_levels_to_index[axis][signed_level] = index

    deltas_arr = np.asarray(deltas, dtype=float)
    radial = np.linalg.norm(deltas_arr, axis=1)
    return _OrientationNeighborhood(
        rotations=np.asarray(rotations, dtype=float),
        deltas_rad=deltas_arr,
        radial_magnitude_rad=radial,
        axis_levels_to_index=axis_levels_to_index,
    )


def _cell_volume_ang3(gxparm: GXPARMData) -> float:
    cell = gxparm.unit_cell
    alpha = np.deg2rad(float(cell.alpha))
    beta = np.deg2rad(float(cell.beta))
    gamma = np.deg2rad(float(cell.gamma))
    return float(
        cell.a
        * cell.b
        * cell.c
        * np.sqrt(
            max(
                1.0
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma),
                0.0,
            )
        )
    )


def _excitation_error(
    g_vectors: FloatArray,
    wavelength_angstrom: float,
    beam_direction: ArrayLike = (0.0, 0.0, 1.0),
) -> FloatArray:
    beam = np.asarray(beam_direction, dtype=float)
    beam_norm = np.linalg.norm(beam)
    if beam_norm == 0.0:
        raise ValueError("beam_direction must be non-zero.")
    beam_unit = beam / beam_norm
    k0 = beam_unit / float(wavelength_angstrom)
    shifted = g_vectors + k0[None, :]
    return np.linalg.norm(shifted, axis=1) - np.linalg.norm(k0)


def _fg_proxy_from_q(q_invA: FloatArray, config: DynamicalUncertaintyConfig) -> FloatArray:
    q0 = max(float(config.fg_q0_invA), float(config.eps))
    scale = float(config.fg_scale)
    return scale / (1.0 + np.square(q_invA / q0))


def _response_proxy(
    sg_invA: FloatArray,
    q_invA: FloatArray,
    wavelength_angstrom: float,
    cell_volume_ang3: float,
    thickness_nm: float | FloatArray,
    config: DynamicalUncertaintyConfig,
) -> FloatArray:
    """Simplified dynamical response proxy (intensity-independent)."""

    fg = np.maximum(_fg_proxy_from_q(q_invA, config), float(config.eps))
    xi = pi * float(cell_volume_ang3) / (float(wavelength_angstrom) * fg)
    sg_xi = sg_invA * xi
    d2 = 1.0 / (1.0 + np.square(sg_xi))
    xi_eff = xi * np.sqrt(1.0 + np.square(sg_xi))

    thickness_array_nm = np.asarray(thickness_nm, dtype=float)
    thickness_ang = thickness_array_nm * 10.0
    phase = np.pi * thickness_ang / (xi_eff[..., None] if thickness_array_nm.ndim > 0 else xi_eff)
    oscillation = np.sin(phase) ** 2
    return np.clip(d2[..., None] * oscillation if thickness_array_nm.ndim > 0 else d2 * oscillation, 0.0, 1.0)


def _resolve_frame_index(
    frame_input: int,
    reciprocal_by_frame: Mapping[int, FloatArray],
) -> int:
    if frame_input in reciprocal_by_frame:
        return int(frame_input)
    if (frame_input - 1) in reciprocal_by_frame:
        return int(frame_input - 1)
    if (frame_input + 1) in reciprocal_by_frame:
        return int(frame_input + 1)
    if not reciprocal_by_frame:
        raise ValueError("reciprocal_by_frame is empty.")
    available = np.asarray(sorted(int(key) for key in reciprocal_by_frame), dtype=int)
    nearest = int(available[np.argmin(np.abs(available - int(frame_input)))])
    return nearest


def _canonical_from_core_inputs(
    gxparm: GXPARMData,
    integrate: IntegrateData,
    reciprocal_by_frame: Mapping[int, FloatArray],
    source: str,
    dataset_id: str,
) -> CanonicalObservationData:
    required = {"h", "k", "l", "frame_est"}
    missing = sorted(required.difference(integrate.observations.columns))
    if missing:
        raise ValueError(f"IntegrateData is missing required columns: {missing}")

    obs = integrate.observations.copy().reset_index(drop=True)
    obs["frame_input"] = obs["frame_est"].astype(int)
    unique_frames = sorted(obs["frame_input"].unique())
    frame_map = {
        int(frame_input): _resolve_frame_index(int(frame_input), reciprocal_by_frame)
        for frame_input in unique_frames
    }
    obs["frame_index"] = obs["frame_input"].map(frame_map).astype(int)
    obs["frame_number"] = obs["frame_index"] + 1

    ub_lookup: dict[int, FloatArray] = {
        int(frame_idx): np.asarray(matrix, dtype=float).reshape(3, 3)
        for frame_idx, matrix in reciprocal_by_frame.items()
    }
    if not ub_lookup:
        raise ValueError("No per-frame reciprocal matrices available.")

    ub_flat = np.vstack([ub_lookup[int(frame_idx)].reshape(9) for frame_idx in obs["frame_index"]])
    canonical = pd.DataFrame(
        {
            "obs_id": np.arange(1, obs.shape[0] + 1, dtype=int),
            "source": str(source),
            "dataset_id": str(dataset_id),
            "frame_input": obs["frame_input"].astype(int),
            "frame_index": obs["frame_index"].astype(int),
            "frame_number": obs["frame_number"].astype(int),
            "event_id": [f"{dataset_id}:frame{int(frame_number):06d}" for frame_number in obs["frame_number"]],
            "h": obs["h"].astype(int),
            "k": obs["k"].astype(int),
            "l": obs["l"].astype(int),
            "z_cal": obs["z_cal"].astype(float) if "z_cal" in obs.columns else obs["frame_input"].astype(float),
            "sigma_exp": obs["sigma"].astype(float) if "sigma" in obs.columns else np.nan,
            "wavelength_angstrom": float(gxparm.wavelength_angstrom),
            "beam_dir_x": 0.0,
            "beam_dir_y": 0.0,
            "beam_dir_z": 1.0,
            "a_angstrom": float(gxparm.unit_cell.a),
            "b_angstrom": float(gxparm.unit_cell.b),
            "c_angstrom": float(gxparm.unit_cell.c),
            "alpha_deg": float(gxparm.unit_cell.alpha),
            "beta_deg": float(gxparm.unit_cell.beta),
            "gamma_deg": float(gxparm.unit_cell.gamma),
            "detector_nx": int(gxparm.detector_nx),
            "detector_ny": int(gxparm.detector_ny),
            "pixel_x_mm": float(gxparm.pixel_x_mm),
            "pixel_y_mm": float(gxparm.pixel_y_mm),
            "distance_mm": float(gxparm.distance_mm),
        }
    )
    for col_idx, col_name in enumerate(
        ("UB11", "UB12", "UB13", "UB21", "UB22", "UB23", "UB31", "UB32", "UB33")
    ):
        canonical[col_name] = ub_flat[:, col_idx]

    return CanonicalObservationData(
        source=str(source),
        dataset_id=str(dataset_id),
        gxparm=gxparm,
        observations=canonical,
    )


def canonical_observations_from_xds(
    gxparm_path: str | Path,
    integrate_path: str | Path,
    dataset_id: str | None = None,
) -> CanonicalObservationData:
    """Build canonical observations from XDS-style inputs."""

    gxparm = parse_gxparm(gxparm_path)
    integrate = parse_integrate_hkl(integrate_path)

    orienter = RotationSeriesOrientationModel(gxparm)
    reciprocal_by_frame: dict[int, FloatArray] = {}
    for frame in range(int(integrate.estimated_n_frames)):
        rotation = orienter.rotation_matrix(frame, offset=0.0)
        reciprocal_by_frame[frame] = rotation @ gxparm.reciprocal_reference

    dataset = dataset_id or Path(gxparm_path).resolve().parent.name
    return _canonical_from_core_inputs(
        gxparm=gxparm,
        integrate=integrate,
        reciprocal_by_frame=reciprocal_by_frame,
        source="xds",
        dataset_id=str(dataset),
    )


def canonical_observations_from_stream(
    stream_path: str | Path,
    dataset_id: str | None = None,
) -> CanonicalObservationData:
    """Build canonical observations from CrystFEL stream input."""

    stream_data = parse_crystfel_stream(stream_path)
    gxparm, integrate, reciprocal_by_frame = crystfel_stream_to_analysis_inputs(stream_data)
    dataset = dataset_id or Path(stream_path).stem
    return _canonical_from_core_inputs(
        gxparm=gxparm,
        integrate=integrate,
        reciprocal_by_frame=reciprocal_by_frame,
        source="stream",
        dataset_id=str(dataset),
    )


def _ori_thick_metrics_for_frame(
    hkls: FloatArray,
    ub_frame: FloatArray,
    wavelength_angstrom: float,
    cell_volume_ang3: float,
    neighborhood: _OrientationNeighborhood,
    config: DynamicalUncertaintyConfig,
    thickness_grid_nm: FloatArray,
) -> dict[str, FloatArray]:
    n_obs = hkls.shape[0]
    n_pert = neighborhood.rotations.shape[0]
    orientation_responses = np.zeros((n_obs, n_pert), dtype=float)
    t_ref_nm = float(config.reference_thickness_nm())

    for p_idx in range(n_pert):
        ub_pert = neighborhood.rotations[p_idx] @ ub_frame
        g = hkls @ ub_pert.T
        sg = _excitation_error(g, wavelength_angstrom, beam_direction=config.beam_direction)
        q = np.linalg.norm(g, axis=1)
        orientation_responses[:, p_idx] = _response_proxy(
            sg_invA=sg,
            q_invA=q,
            wavelength_angstrom=wavelength_angstrom,
            cell_volume_ang3=cell_volume_ang3,
            thickness_nm=t_ref_nm,
            config=config,
        )

    ub_center = ub_frame
    g_center = hkls @ ub_center.T
    sg_center = _excitation_error(g_center, wavelength_angstrom, beam_direction=config.beam_direction)
    q_center = np.linalg.norm(g_center, axis=1)
    thickness_responses = _response_proxy(
        sg_invA=sg_center,
        q_invA=q_center,
        wavelength_angstrom=wavelength_angstrom,
        cell_volume_ang3=cell_volume_ang3,
        thickness_nm=thickness_grid_nm,
        config=config,
    )
    if thickness_responses.ndim != 2:
        raise ValueError("thickness response array must be 2D.")

    eps = float(config.eps)
    center = orientation_responses[:, 0]
    ori_mean = np.mean(orientation_responses, axis=1)
    ori_std = np.std(orientation_responses, axis=1)
    ori_range = np.max(orientation_responses, axis=1) - np.min(orientation_responses, axis=1)
    ori_cv = ori_std / (np.abs(ori_mean) + eps)

    radial = neighborhood.radial_magnitude_rad
    grad_numerators = orientation_responses[:, 1:] - center[:, None]
    grad_denominator = radial[1:][None, :] + eps
    ori_grad_rms = np.sqrt(np.mean(np.square(grad_numerators / grad_denominator), axis=1))

    step_rad = np.deg2rad(float(config.orientation_step_deg))
    curvatures: list[FloatArray] = []
    multipeaks: list[FloatArray] = []
    for axis in config.orientation_axes:
        level_map = neighborhood.axis_levels_to_index[axis]
        if -1 in level_map and 1 in level_map:
            idx_m = level_map[-1]
            idx_p = level_map[1]
            curvatures.append((orientation_responses[:, idx_p] - 2.0 * center + orientation_responses[:, idx_m]) / (step_rad * step_rad + eps))

        axis_levels = list(range(-int(config.orientation_n_steps), int(config.orientation_n_steps) + 1))
        if all(level in level_map for level in axis_levels):
            axis_indices = [level_map[level] for level in axis_levels]
            axis_values = orientation_responses[:, axis_indices]
            deriv = np.diff(axis_values, axis=1)
            signs = np.sign(deriv)
            sign_change_count = np.sum((signs[:, 1:] * signs[:, :-1]) < 0.0, axis=1)
            denominator = max(axis_values.shape[1] - 2, 1)
            multipeaks.append(sign_change_count.astype(float) / float(denominator))

    if curvatures:
        ori_curvature = np.sqrt(np.mean(np.square(np.stack(curvatures, axis=1)), axis=1))
    else:
        ori_curvature = np.zeros(n_obs, dtype=float)
    if multipeaks:
        ori_multipeak_score = np.mean(np.stack(multipeaks, axis=1), axis=1)
    else:
        ori_multipeak_score = np.zeros(n_obs, dtype=float)

    thick_mean = np.mean(thickness_responses, axis=1)
    thick_std = np.std(thickness_responses, axis=1)
    thick_range = np.max(thickness_responses, axis=1) - np.min(thickness_responses, axis=1)
    thick_cv = thick_std / (np.abs(thick_mean) + eps)

    thickness_step = np.diff(thickness_grid_nm)
    thick_derivative = np.diff(thickness_responses, axis=1) / (thickness_step[None, :] + eps)
    thick_derivative_rms = np.sqrt(np.mean(np.square(thick_derivative), axis=1))

    thick_min = np.min(thickness_responses, axis=1)
    thick_max = np.max(thickness_responses, axis=1)
    thick_max_min_ratio = thick_max / (thick_min + eps)
    thick_max_min_ratio = np.minimum(thick_max_min_ratio, float(config.max_min_ratio_cap))

    beam = np.asarray(config.beam_direction, dtype=float)
    beam_unit = beam / np.linalg.norm(beam)
    gz_center = g_center @ beam_unit
    zolz_width = float(config.zone_axis_zolz_relative_width)
    zolz_den = zolz_width * np.maximum(q_center, eps) + eps
    zone_axis_proximity = np.exp(-np.square(np.abs(gz_center) / zolz_den))

    band = float(config.zone_axis_layer_band_invA)
    diff = (gz_center[:, None] - gz_center[None, :]) / (band + eps)
    layer_kernel = np.exp(-np.square(diff))
    zone_axis_layer_density = (np.sum(layer_kernel, axis=1) - 1.0) / max(n_obs - 1, 1)
    zone_axis_score = zone_axis_proximity * zone_axis_layer_density

    ori_range_rel = ori_range / (np.abs(ori_mean) + eps)
    ori_grad_scaled = ori_grad_rms * step_rad
    ori_curvature_scaled = ori_curvature * (step_rad ** 2)
    zone_axis_scaled = float(config.zone_axis_boost_weight) * zone_axis_score
    risk_orientation = np.sqrt(
        np.square(ori_cv)
        + np.square(ori_range_rel)
        + np.square(ori_grad_scaled)
        + np.square(ori_curvature_scaled)
        + np.square(ori_multipeak_score)
        + np.square(zone_axis_scaled)
    )

    thickness_delta = float(np.mean(thickness_step)) if thickness_step.size else 0.0
    thick_range_rel = thick_range / (np.abs(thick_mean) + eps)
    thick_derivative_scaled = thick_derivative_rms * thickness_delta
    thick_ratio_term = np.log(np.maximum(thick_max_min_ratio, 1.0))
    risk_thickness = np.sqrt(
        np.square(thick_cv)
        + np.square(thick_range_rel)
        + np.square(thick_derivative_scaled)
        + np.square(thick_ratio_term)
    )

    return {
        "ori_mean": ori_mean,
        "ori_std": ori_std,
        "ori_cv": ori_cv,
        "ori_range": ori_range,
        "ori_grad_rms": ori_grad_rms,
        "ori_curvature": ori_curvature,
        "ori_multipeak_score": ori_multipeak_score,
        "zone_axis_proximity": zone_axis_proximity,
        "zone_axis_layer_density": zone_axis_layer_density,
        "zone_axis_score": zone_axis_score,
        "thick_mean": thick_mean,
        "thick_std": thick_std,
        "thick_cv": thick_cv,
        "thick_range": thick_range,
        "thick_derivative_rms": thick_derivative_rms,
        "thick_max_min_ratio": thick_max_min_ratio,
        "risk_orientation": risk_orientation,
        "risk_thickness": risk_thickness,
    }


def run_dynamical_uncertainty_pipeline(
    canonical: CanonicalObservationData,
    config: DynamicalUncertaintyConfig | None = None,
) -> DynamicalUncertaintyResult:
    """Run observation-level orientation+thickness dynamical uncertainty screening."""

    cfg = config or DynamicalUncertaintyConfig()
    cfg.validate()

    observations = canonical.observations.copy().reset_index(drop=True)
    required = {"obs_id", "frame_index", "h", "k", "l", "UB11", "UB12", "UB13", "UB21", "UB22", "UB23", "UB31", "UB32", "UB33"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Canonical observation table is missing columns: {missing}")

    neighborhood = _orientation_neighborhood(cfg)
    thickness_grid_nm = cfg.thickness_grid_nm()
    cell_volume_ang3 = _cell_volume_ang3(canonical.gxparm)
    wavelength = float(canonical.gxparm.wavelength_angstrom)

    output = observations[
        [
            "obs_id",
            "source",
            "dataset_id",
            "event_id",
            "frame_input",
            "frame_index",
            "frame_number",
            "h",
            "k",
            "l",
            "wavelength_angstrom",
        ]
    ].copy()

    all_metric_columns = [
        "ori_mean",
        "ori_std",
        "ori_cv",
        "ori_range",
        "ori_grad_rms",
        "ori_curvature",
        "ori_multipeak_score",
        "zone_axis_proximity",
        "zone_axis_layer_density",
        "zone_axis_score",
        "thick_mean",
        "thick_std",
        "thick_cv",
        "thick_range",
        "thick_derivative_rms",
        "thick_max_min_ratio",
        "risk_orientation",
        "risk_thickness",
    ]
    for col in all_metric_columns:
        output[col] = np.nan

    for frame_index, frame_group in observations.groupby("frame_index", sort=False):
        idx = frame_group.index.to_numpy(dtype=int)
        ub_row = frame_group.iloc[0][["UB11", "UB12", "UB13", "UB21", "UB22", "UB23", "UB31", "UB32", "UB33"]]
        ub = ub_row.to_numpy(dtype=float).reshape(3, 3)
        hkls = frame_group[["h", "k", "l"]].to_numpy(dtype=float)

        metrics = _ori_thick_metrics_for_frame(
            hkls=hkls,
            ub_frame=ub,
            wavelength_angstrom=wavelength,
            cell_volume_ang3=cell_volume_ang3,
            neighborhood=neighborhood,
            config=cfg,
            thickness_grid_nm=thickness_grid_nm,
        )
        for col in all_metric_columns:
            output.loc[idx, col] = metrics[col]

    risk_ori = output["risk_orientation"].to_numpy(dtype=float)
    risk_thick = output["risk_thickness"].to_numpy(dtype=float)
    output["risk_total"] = np.sqrt(
        np.square(float(cfg.risk_weight_orientation) * risk_ori)
        + np.square(float(cfg.risk_weight_thickness) * risk_thick)
    )

    risk_total = output["risk_total"].to_numpy(dtype=float)
    finite = np.isfinite(risk_total)
    if np.any(finite):
        scale = float(np.quantile(risk_total[finite], float(cfg.risk_normalization_quantile)))
        if not np.isfinite(scale) or scale <= float(cfg.eps):
            scale = float(np.max(risk_total[finite]))
    else:
        scale = 1.0
    scale = max(scale, float(cfg.eps))
    output["risk_total_norm"] = output["risk_total"] / scale

    if cfg.dyn_sigma_form == "linear":
        output["dyn_sigma_rel"] = 1.0 + float(cfg.dyn_sigma_alpha) * output["risk_total_norm"]
    else:
        output["dyn_sigma_rel"] = np.exp(float(cfg.dyn_sigma_alpha) * output["risk_total_norm"])
    output["dyn_uncertainty_rel"] = output["dyn_sigma_rel"]

    if cfg.include_sigma_dyn and "sigma_exp" in observations.columns:
        sigma_exp = observations["sigma_exp"].to_numpy(dtype=float)
        output["sigma_exp"] = sigma_exp
        output["sigma_dyn"] = sigma_exp * output["dyn_sigma_rel"].to_numpy(dtype=float)

    output = output.sort_values("obs_id").reset_index(drop=True)
    return DynamicalUncertaintyResult(
        config=cfg,
        canonical=canonical,
        uncertainty_table=output,
    )


def run_dynamical_uncertainty_from_xds(
    gxparm_path: str | Path,
    integrate_path: str | Path,
    dataset_id: str | None = None,
    config: DynamicalUncertaintyConfig | None = None,
) -> DynamicalUncertaintyResult:
    """Run uncertainty screening from XDS-style inputs."""

    canonical = canonical_observations_from_xds(
        gxparm_path=gxparm_path,
        integrate_path=integrate_path,
        dataset_id=dataset_id,
    )
    return run_dynamical_uncertainty_pipeline(canonical, config=config)


def run_dynamical_uncertainty_from_stream(
    stream_path: str | Path,
    dataset_id: str | None = None,
    config: DynamicalUncertaintyConfig | None = None,
) -> DynamicalUncertaintyResult:
    """Run uncertainty screening from CrystFEL stream input."""

    canonical = canonical_observations_from_stream(
        stream_path=stream_path,
        dataset_id=dataset_id,
    )
    return run_dynamical_uncertainty_pipeline(canonical, config=config)

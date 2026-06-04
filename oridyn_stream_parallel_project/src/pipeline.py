"""High-level analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .bloch import (
    beam_intensities,
    build_structure_matrix,
    extinction_distance_angstrom,
    propagate_bloch_wave,
)
from .constants import (
    DEFAULT_DMAX_ANGSTROM,
    DEFAULT_DMIN_ANGSTROM,
    DEFAULT_EXCITATION_TOLERANCE_INV_ANGSTROM,
)
from .geometry import (
    OrientationModel,
    ReciprocalMatrixOrientationModel,
    RotationSeriesOrientationModel,
    build_zone_axes,
    cell_volume,
    excitation_error,
    generate_candidate_reflections,
    inside_detector,
    mark_untrusted_rectangles,
    nearest_zone_axis,
    nearest_zone_axis_from_reciprocal_matrix,
    project_to_detector,
    rotate_reference_vectors,
)
from .metrics import (
    aggregate_frame_thickness_sensitivity,
    aggregate_reflection_thickness_sensitivity,
    combined_proxy_score,
    effective_coupling_multiplicity,
    empirical_amplitude_strength_proxy,
    geometry_dynamical_risk,
    orientation_excitation_probability,
    orientation_proxy_score,
    orientation_sg_sigma,
    primitive_hkl_key,
    reciprocal_strength_proxy,
    summarize_frame_proxy,
    two_channel_dynamical_risk,
    two_beam_metric,
)
from .pets2 import load_pets_model, pets_model_to_analysis_inputs
from .parsers import (
    CompositionResult,
    GXPARMData,
    IntegrateData,
    XDSInputData,
    crystfel_stream_to_analysis_inputs,
    load_optional_xds_inp,
    parse_composition,
    parse_crystfel_stream,
    parse_gxparm,
    parse_integrate_hkl,
)
from .wilson import WilsonCalibration, wilson_calibrate


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the Bloch-wave dynamicality pipeline."""

    dmin_angstrom: float = DEFAULT_DMIN_ANGSTROM
    dmax_angstrom: float = DEFAULT_DMAX_ANGSTROM
    excitation_tolerance_invA: float = DEFAULT_EXCITATION_TOLERANCE_INV_ANGSTROM
    mode: Literal["proxy", "thickness"] = "proxy"
    thickness_nm: float | Sequence[float] | None = None
    zone_axis_limit: int = 5
    filter_untrusted: bool = False
    orientation_only: bool = False
    orientation_sigma_deg: float | tuple[float, float, float] = 0.2
    orientation_sigma_alpha: float = 0.5
    orientation_score_formulation: Literal["log_n_eff", "linear_n_eff"] = "log_n_eff"
    detector_xy_swapped: bool = False
    orientation_beam_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    dynamical_environment_tolerance_invA: float = 1.0e-2
    dynamical_environment_weight_min: float = 2.0e-2
    dynamical_neighbor_radius_invA: float = 0.12
    dynamical_zone_axis_sigma_deg: float = 3.0
    dynamical_zone_sigma_invA: float = 0.06
    dynamical_neighbor_sigma_invA: float | None = None
    dynamical_row_direction_limit: int = 2
    dynamical_row_max_steps: int = 5
    dynamical_row_sigma_invA: float = 0.25
    dynamical_coupling_q0_invA: float = 0.25
    dynamical_coupling_power: float = 2.0
    dynamical_weight_self: float = 1.0
    dynamical_weight_zone: float = 1.0
    dynamical_weight_row: float = 1.0
    dynamical_cluster_sigma_alpha: float = 1.0
    stream_detector_shift_sign: float = 1.0
    stream_mirror_x_axis: bool = True
    frame_numbers: Sequence[int] | None = None
    progress_every: int = 0

    def thickness_array_nm(self) -> np.ndarray | None:
        """Return thickness values as a 1D array in nanometers."""

        if self.mode == "proxy":
            return None
        if self.thickness_nm is None:
            return np.asarray([100.0], dtype=float)
        return np.atleast_1d(np.asarray(self.thickness_nm, dtype=float))


@dataclass(frozen=True)
class AnalysisResult:
    """Container for pipeline outputs."""

    config: AnalysisConfig
    gxparm: GXPARMData
    integrate: IntegrateData
    xds_input: XDSInputData | None
    composition: CompositionResult
    wilson: WilsonCalibration | None
    candidate_reflections: pd.DataFrame
    reflections_long: pd.DataFrame
    frame_summary: pd.DataFrame
    thickness_long: pd.DataFrame | None
    reflection_sensitivity: pd.DataFrame | None

    def frame_table(self, frame: int) -> pd.DataFrame:
        """Return the per-reflection table for one frame."""

        return self.reflections_long[self.reflections_long["frame"] == frame].copy()


def _frame_count_from_inputs(integrate: IntegrateData, xds_input: XDSInputData | None) -> int:
    if xds_input is None or xds_input.data_range is None:
        return int(integrate.estimated_n_frames)
    start, end = xds_input.data_range
    if start == 1:
        # Preserve the usual HTML behavior when DATA_RANGE starts at 1.
        return int(end)
    return int(end - start + 1)


def run_analysis(
    gxparm: GXPARMData,
    integrate: IntegrateData,
    composition: CompositionResult,
    xds_input: XDSInputData | None = None,
    config: AnalysisConfig | None = None,
    orientation_model: OrientationModel | None = None,
) -> AnalysisResult:
    """Run the full analysis pipeline.

    Parameters
    ----------
    gxparm, integrate, composition, xds_input:
        Parsed inputs.
    config:
        Analysis parameters.
    orientation_model:
        Optional override for future frame-specific orientation models.
    """

    cfg = config or AnalysisConfig()
    orienter = orientation_model or RotationSeriesOrientationModel(gxparm)
    use_direct_reciprocal_vectors = (
        isinstance(orienter, ReciprocalMatrixOrientationModel) and orienter.use_direct_reciprocal_vectors
    )
    n_frames = _frame_count_from_inputs(integrate, xds_input)
    if cfg.orientation_only and cfg.mode == "thickness":
        raise ValueError("orientation_only mode does not support thickness propagation.")
    progress_every = max(int(cfg.progress_every), 0)
    if progress_every:
        print("Preparing analysis inputs...", flush=True)

    calibration: WilsonCalibration | None = None
    candidate_reflections = generate_candidate_reflections(
        gxparm,
        dmin_angstrom=cfg.dmin_angstrom,
        dmax_angstrom=cfg.dmax_angstrom,
    ).copy()
    candidate_reflections["strength_proxy"] = reciprocal_strength_proxy(
        candidate_reflections["q_invA"].to_numpy(dtype=float)
    )
    candidate_reflections["empirical_strength_proxy"] = empirical_amplitude_strength_proxy(
        candidate_reflections,
        integrate.observations,
        fallback_values=candidate_reflections["strength_proxy"].to_numpy(dtype=float),
    )
    candidate_hkl_tuples = [
        tuple(map(int, hkl))
        for hkl in candidate_reflections[["h", "k", "l"]].itertuples(index=False, name=None)
    ]
    candidate_row_keys = [primitive_hkl_key(*hkl) for hkl in candidate_hkl_tuples]
    candidate_hkl_to_index = {hkl: index for index, hkl in enumerate(candidate_hkl_tuples)}
    volume_ang3 = cell_volume(gxparm.unit_cell)

    if cfg.orientation_only:
        candidate_reflections["Fg_abs"] = np.nan
        candidate_reflections["xi_angstrom"] = np.nan
        candidate_reflections["xi_nm"] = np.nan
    else:
        calibration = wilson_calibrate(integrate.observations, composition.sum_fj2)
        candidate_reflections["Fg_abs"] = [
            calibration.lookup_amplitude(int(h), int(k), int(l))
            for h, k, l in candidate_reflections[["h", "k", "l"]].itertuples(index=False, name=None)
        ]
        candidate_reflections["xi_angstrom"] = extinction_distance_angstrom(
            gxparm.wavelength_angstrom,
            volume_ang3,
            candidate_reflections["Fg_abs"].to_numpy(dtype=float),
        )
        candidate_reflections["xi_nm"] = candidate_reflections["xi_angstrom"] / 10.0

    zone_axes = build_zone_axes(limit=cfg.zone_axis_limit)
    reference_vectors = candidate_reflections[["gx_ref", "gy_ref", "gz_ref"]].to_numpy(dtype=float)
    thickness_array_nm = cfg.thickness_array_nm()
    observed_by_frame: dict[int, pd.DataFrame] = {}
    observed_required = {"frame_est", "h", "k", "l", "I", "sigma"}
    if observed_required.issubset(integrate.observations.columns):
        observed_agg = (
            integrate.observations.groupby(["frame_est", "h", "k", "l"], as_index=False)
            .agg(
                I_obs=("I", "median"),
                sigma_obs=("sigma", "median"),
                n_observations=("I", "size"),
            )
        )
        observed_by_frame = {
            int(frame): group.drop(columns=["frame_est"]).reset_index(drop=True)
            for frame, group in observed_agg.groupby("frame_est", sort=False)
        }

    reflection_tables: list[pd.DataFrame] = []
    frame_summary_rows: list[dict[str, float | int | str]] = []
    thickness_rows: list[dict[str, float | int]] = []
    rectangles = xds_input.untrusted_rectangles if xds_input is not None else []

    frames_to_run: list[int] = list(range(n_frames))
    if cfg.frame_numbers:
        requested = sorted({int(value) for value in cfg.frame_numbers})
        out_of_range = [frame for frame in requested if frame < 1 or frame > n_frames]
        if out_of_range:
            raise ValueError(
                "Requested analysis frames are out of range: "
                f"{out_of_range}. Valid range is 1..{n_frames}."
            )
        frames_to_run = [frame_number - 1 for frame_number in requested]

    progress_start = time.monotonic()
    total_frames = len(frames_to_run)
    if progress_every:
        print(f"Analyzing {total_frames} frame(s)...", flush=True)

    def report_progress(processed_count: int, frame_number: int, n_excited: int) -> None:
        if not progress_every:
            return
        if processed_count != total_frames and processed_count % progress_every != 0:
            return
        elapsed = max(time.monotonic() - progress_start, 1.0e-9)
        frames_per_second = processed_count / elapsed
        remaining = total_frames - processed_count
        eta_seconds = remaining / frames_per_second if frames_per_second > 0.0 else float("inf")
        eta_minutes = eta_seconds / 60.0 if np.isfinite(eta_seconds) else float("inf")
        print(
            "Progress: "
            f"{processed_count}/{total_frames} frames "
            f"(frame {frame_number}, n_excited={n_excited}, "
            f"{frames_per_second:.2f} frames/s, eta={eta_minutes:.1f} min)",
            flush=True,
        )

    for processed_count, frame in enumerate(frames_to_run, start=1):
        frame_gxparm = gxparm
        if use_direct_reciprocal_vectors:
            reciprocal_frame = orienter.reciprocal_matrix(frame)
            hkl_array = candidate_reflections[["h", "k", "l"]].to_numpy(dtype=float)
            g_mid = hkl_array @ reciprocal_frame.T
            g_start = g_mid
            g_end = g_mid
            sg_mid = excitation_error(
                g_mid,
                gxparm.wavelength_angstrom,
                beam_direction=cfg.orientation_beam_direction,
            )
            sg_start = sg_mid
            sg_end = sg_mid

            orgx_px = gxparm.orgx_px
            orgy_px = gxparm.orgy_px
            if orienter.detector_center_by_frame is not None and frame in orienter.detector_center_by_frame:
                orgx_px, orgy_px = orienter.detector_center_by_frame[frame]
            distance_mm = gxparm.distance_mm
            if orienter.distance_by_frame is not None and frame in orienter.distance_by_frame:
                distance_mm = orienter.distance_by_frame[frame]
            frame_gxparm = replace(
                gxparm,
                orgx_px=float(orgx_px),
                orgy_px=float(orgy_px),
                distance_mm=float(distance_mm),
            )
            x_px, y_px, positive_sz = project_to_detector(
                g_mid,
                frame_gxparm,
                beam_direction=cfg.orientation_beam_direction,
            )
            if cfg.stream_mirror_x_axis:
                y_px = 2.0 * float(frame_gxparm.orgy_px) - y_px
            zone_axis = nearest_zone_axis_from_reciprocal_matrix(
                zone_axes,
                reciprocal_frame,
                beam_direction=cfg.orientation_beam_direction,
            )
            rotation_mid = np.eye(3, dtype=float)
        else:
            rotation_start = orienter.rotation_matrix(frame, offset=0.0)
            rotation_mid = orienter.rotation_matrix(frame, offset=0.5)
            rotation_end = orienter.rotation_matrix(frame, offset=1.0)

            g_start = rotate_reference_vectors(rotation_start, reference_vectors)
            g_mid = rotate_reference_vectors(rotation_mid, reference_vectors)
            g_end = rotate_reference_vectors(rotation_end, reference_vectors)

            sg_start = excitation_error(
                g_start,
                gxparm.wavelength_angstrom,
                beam_direction=cfg.orientation_beam_direction,
            )
            sg_mid = excitation_error(
                g_mid,
                gxparm.wavelength_angstrom,
                beam_direction=cfg.orientation_beam_direction,
            )
            sg_end = excitation_error(
                g_end,
                gxparm.wavelength_angstrom,
                beam_direction=cfg.orientation_beam_direction,
            )

            x_px, y_px, positive_sz = project_to_detector(
                g_mid,
                frame_gxparm,
                beam_direction=cfg.orientation_beam_direction,
            )
            zone_axis = nearest_zone_axis(zone_axes, gxparm.real_space_reference, rotation_mid)

        if cfg.detector_xy_swapped:
            x_px, y_px = y_px.copy(), x_px.copy()
            within_detector = (
                (x_px >= 0.0)
                & (x_px < frame_gxparm.detector_ny)
                & (y_px >= 0.0)
                & (y_px < frame_gxparm.detector_nx)
            )
        else:
            within_detector = inside_detector(x_px, y_px, frame_gxparm)
        in_untrusted = mark_untrusted_rectangles(x_px, y_px, rectangles)

        excited_mask = (
            ((sg_start * sg_end) <= 0.0) | (np.abs(sg_mid) < cfg.excitation_tolerance_invA)
        ) & positive_sz & within_detector
        if cfg.filter_untrusted:
            excited_mask &= ~in_untrusted

        frame_table = candidate_reflections.loc[excited_mask].copy().reset_index(drop=True)
        frame_number = frame + 1
        phi_deg = gxparm.phi0_deg + gxparm.dphi_deg * frame

        if frame_table.empty:
            frame_summary_rows.append(
                {
                    "frame": frame,
                    "frame_number": frame_number,
                    "phi_deg": phi_deg,
                    "zone_axis": zone_axis.label,
                    "zone_axis_angle_deg": zone_axis.angle_deg,
                    "n_excited": 0,
                    "S_2beam": 0.0,
                    "S_MB": 0.0,
                    "S_dyn": 0.0,
                    "S_orient": 0.0,
                    "mean_N_eff": 0.0,
                    "max_N_eff": 0.0,
                    "mean_orientation_sigma_sg_invA": 0.0,
                    "mean_orientation_p_excited": 0.0,
                    "eigenvalue_spread_invA": 0.0,
                    "sum_self_extinction_score": 0.0,
                    "mean_attenuation_risk": 0.0,
                    "sum_cluster_score_geom": 0.0,
                    "mean_cluster_risk_geom": 0.0,
                    "p95_cluster_risk_geom": 0.0,
                    "sum_cluster_score_iw": 0.0,
                    "mean_cluster_risk_iw": 0.0,
                    "p95_cluster_risk_iw": 0.0,
                    "mean_sigma_dyn_rel": 1.0,
                    "n_observed_targets": 0,
                }
            )
            report_progress(processed_count, frame_number, 0)
            continue

        frame_table["frame"] = frame
        frame_table["frame_number"] = frame_number
        frame_table["phi_deg"] = phi_deg
        frame_table["sg_invA"] = sg_mid[excited_mask]
        frame_table["x_px"] = x_px[excited_mask]
        frame_table["y_px"] = y_px[excited_mask]
        frame_table["in_untrusted_region"] = in_untrusted[excited_mask]

        dynamical_risk_table = geometry_dynamical_risk(
            candidate_reflections=candidate_reflections,
            g_vectors_invA=g_mid,
            sg_invA=sg_mid,
            target_mask=excited_mask,
            environment_mask=positive_sz,
            zone_axis=zone_axis.axis,
            zone_axis_angle_deg=zone_axis.angle_deg,
            hkl_to_index=candidate_hkl_to_index,
            hkl_tuples=candidate_hkl_tuples,
            row_keys=candidate_row_keys,
            strength_proxy=candidate_reflections["strength_proxy"].to_numpy(dtype=float),
            environment_tolerance_invA=cfg.dynamical_environment_tolerance_invA,
            environment_weight_min=cfg.dynamical_environment_weight_min,
            neighbor_radius_invA=cfg.dynamical_neighbor_radius_invA,
            zone_axis_sigma_deg=cfg.dynamical_zone_axis_sigma_deg,
        )
        for column in dynamical_risk_table.columns:
            frame_table[column] = dynamical_risk_table[column].to_numpy()
        frame_table["legacy_S_dyn"] = frame_table["S_dyn"].to_numpy(dtype=float)

        two_channel_table = two_channel_dynamical_risk(
            candidate_reflections=candidate_reflections,
            g_vectors_invA=g_mid,
            sg_invA=sg_mid,
            target_mask=excited_mask,
            hkl_to_index=candidate_hkl_to_index,
            hkl_tuples=candidate_hkl_tuples,
            strength_proxy=candidate_reflections["strength_proxy"].to_numpy(dtype=float),
            empirical_strength_proxy=candidate_reflections["empirical_strength_proxy"].to_numpy(dtype=float),
            beam_direction=cfg.orientation_beam_direction,
            excitation_tolerance_invA=cfg.dynamical_environment_tolerance_invA,
            environment_weight_min=cfg.dynamical_environment_weight_min,
            zone_sigma_invA=cfg.dynamical_zone_sigma_invA,
            neighbor_radius_invA=cfg.dynamical_neighbor_radius_invA,
            neighbor_sigma_invA=cfg.dynamical_neighbor_sigma_invA,
            row_direction_limit=cfg.dynamical_row_direction_limit,
            row_max_steps=cfg.dynamical_row_max_steps,
            row_sigma_invA=cfg.dynamical_row_sigma_invA,
            coupling_q0_invA=cfg.dynamical_coupling_q0_invA,
            coupling_power=cfg.dynamical_coupling_power,
            weight_self=cfg.dynamical_weight_self,
            weight_zone=cfg.dynamical_weight_zone,
            weight_row=cfg.dynamical_weight_row,
            sigma_alpha=cfg.dynamical_cluster_sigma_alpha,
        )
        for column in two_channel_table.columns:
            frame_table[column] = two_channel_table[column].to_numpy()

        observed_frame = observed_by_frame.get(frame)
        if observed_frame is not None and not observed_frame.empty:
            frame_table = frame_table.merge(observed_frame, on=["h", "k", "l"], how="left")
        else:
            frame_table["I_obs"] = np.nan
            frame_table["sigma_obs"] = np.nan
            frame_table["n_observations"] = 0
        frame_table["n_observations"] = frame_table["n_observations"].fillna(0).astype(int)
        frame_table["is_observed_target"] = frame_table["n_observations"] > 0
        sigma_obs = frame_table["sigma_obs"].to_numpy(dtype=float)
        sigma_dyn_rel = frame_table["sigma_dyn_rel"].to_numpy(dtype=float)
        frame_table["sigma_new"] = sigma_obs * sigma_dyn_rel
        sigma_new = frame_table["sigma_new"].to_numpy(dtype=float)
        frame_table["weight_new"] = np.where(
            np.isfinite(sigma_new) & (sigma_new > 0.0),
            1.0 / np.square(sigma_new),
            np.nan,
        )

        structure = None
        if cfg.orientation_only:
            d2beam_values = np.zeros(frame_table.shape[0], dtype=float)
            n_eff_values = np.zeros(frame_table.shape[0], dtype=float)
            s_comb_values = np.zeros(frame_table.shape[0], dtype=float)
        else:
            structure = build_structure_matrix(
                frame_table,
                calibration=calibration,
                wavelength_angstrom=gxparm.wavelength_angstrom,
                cell_volume_ang3=volume_ang3,
            )
            d2beam_values = two_beam_metric(
                frame_table["sg_invA"].to_numpy(dtype=float),
                frame_table["xi_angstrom"].to_numpy(dtype=float),
            )
            n_eff_values = np.asarray(
                [
                    effective_coupling_multiplicity(structure.eigenvectors, beam_index=i + 1)
                    for i in range(frame_table.shape[0])
                ],
                dtype=float,
            )
            s_comb_values = combined_proxy_score(d2beam_values, n_eff_values)
        orientation_sigma_values = orientation_sg_sigma(
            g_vectors_invA=g_mid[excited_mask],
            wavelength_angstrom=gxparm.wavelength_angstrom,
            orientation_sigma_deg=cfg.orientation_sigma_deg,
            beam_direction=cfg.orientation_beam_direction,
        )
        orientation_p_excited_values = orientation_excitation_probability(
            sg_invA=frame_table["sg_invA"].to_numpy(dtype=float),
            sg_sigma_orient_invA=orientation_sigma_values,
            excitation_tolerance_invA=cfg.excitation_tolerance_invA,
        )
        if cfg.orientation_only:
            s_orient_values = frame_table["S_dyn"].to_numpy(dtype=float)
        else:
            s_orient_values = orientation_proxy_score(
                p_excited_orient=orientation_p_excited_values,
                n_eff=n_eff_values,
                formulation=cfg.orientation_score_formulation,
            )
        frame_table["d_2beam"] = d2beam_values
        frame_table["N_eff"] = n_eff_values
        frame_table["S_comb"] = s_comb_values
        frame_table["orientation_sigma_sg_invA"] = orientation_sigma_values
        frame_table["orientation_p_excited"] = orientation_p_excited_values
        frame_table["S_orient"] = s_orient_values
        frame_table["sigma_orient_scale"] = 1.0 + float(cfg.orientation_sigma_alpha) * s_orient_values

        if cfg.mode == "thickness" and thickness_array_nm is not None and structure is not None:
            amplitudes = propagate_bloch_wave(
                structure.eigenvalues,
                structure.eigenvectors,
                thickness_array_nm,
            )
            intensities = beam_intensities(amplitudes)
            diffracted_amplitudes = amplitudes[:, 1:]
            diffracted_intensities = intensities[:, 1:]

            if thickness_array_nm.size == 1:
                frame_table["thickness_nm"] = float(thickness_array_nm[0])
                frame_table["amplitude_abs"] = np.abs(diffracted_amplitudes[0, :])
                frame_table["intensity"] = diffracted_intensities[0, :]

            for thickness_index, thickness_nm in enumerate(thickness_array_nm):
                for beam_index, reflection in frame_table.iterrows():
                    thickness_rows.append(
                        {
                            "frame": frame,
                            "frame_number": frame_number,
                            "phi_deg": phi_deg,
                            "h": int(reflection["h"]),
                            "k": int(reflection["k"]),
                            "l": int(reflection["l"]),
                            "reflection_index": int(reflection["reflection_index"]),
                            "thickness_nm": float(thickness_nm),
                            "sg_invA": float(reflection["sg_invA"]),
                            "xi_angstrom": float(reflection["xi_angstrom"]),
                            "d_2beam": float(reflection["d_2beam"]),
                            "N_eff": float(reflection["N_eff"]),
                            "S_comb": float(reflection["S_comb"]),
                            "amplitude_abs": float(np.abs(diffracted_amplitudes[thickness_index, beam_index])),
                            "intensity": float(diffracted_intensities[thickness_index, beam_index]),
                        }
                    )

        reflection_tables.append(frame_table)
        frame_metrics = summarize_frame_proxy(frame_table)
        frame_summary_rows.append(
            {
                "frame": frame,
                "frame_number": frame_number,
                "phi_deg": phi_deg,
                "zone_axis": zone_axis.label,
                "zone_axis_angle_deg": zone_axis.angle_deg,
                "n_excited": int(frame_metrics["n_excited"]),
                "S_2beam": float(frame_metrics["S_2beam"]),
                "S_MB": float(frame_metrics["S_MB"]),
                "S_dyn": float(frame_table["S_dyn"].sum()),
                "S_orient": float(np.sum(s_orient_values)),
                "mean_N_eff": float(frame_metrics["mean_N_eff"]),
                "max_N_eff": float(frame_metrics["max_N_eff"]),
                "mean_orientation_sigma_sg_invA": float(np.mean(orientation_sigma_values)),
                "mean_orientation_p_excited": float(np.mean(orientation_p_excited_values)),
                "eigenvalue_spread_invA": (
                    0.0
                    if structure is None
                    else float(structure.eigenvalues.max() - structure.eigenvalues.min())
                ),
                "sum_self_extinction_score": float(frame_table["self_extinction_score"].sum()),
                "mean_attenuation_risk": float(frame_table["attenuation_risk"].mean()),
                "sum_cluster_score_geom": float(frame_table["cluster_score_geom"].sum()),
                "mean_cluster_risk_geom": float(frame_table["cluster_risk_geom"].mean()),
                "p95_cluster_risk_geom": float(frame_table["cluster_risk_geom"].quantile(0.95)),
                "sum_cluster_score_iw": float(frame_table["cluster_score_iw"].sum()),
                "mean_cluster_risk_iw": float(frame_table["cluster_risk_iw"].mean()),
                "p95_cluster_risk_iw": float(frame_table["cluster_risk_iw"].quantile(0.95)),
                "mean_sigma_dyn_rel": float(frame_table["sigma_dyn_rel"].mean()),
                "n_observed_targets": int(frame_table["is_observed_target"].sum()),
            }
        )
        report_progress(processed_count, frame_number, int(frame_metrics["n_excited"]))

    reflections_long = (
        pd.concat(reflection_tables, ignore_index=True)
        if reflection_tables
        else pd.DataFrame()
    )
    frame_summary = pd.DataFrame.from_records(frame_summary_rows).sort_values("frame").reset_index(drop=True)
    thickness_long = pd.DataFrame.from_records(thickness_rows) if thickness_rows else None
    reflection_sensitivity: pd.DataFrame | None = None

    if thickness_long is not None and not thickness_long.empty and thickness_long["thickness_nm"].nunique() > 1:
        reflection_sensitivity = aggregate_reflection_thickness_sensitivity(thickness_long)
        frame_thickness = aggregate_frame_thickness_sensitivity(reflection_sensitivity)
        frame_summary = frame_summary.merge(frame_thickness, on=["frame", "frame_number"], how="left")

    return AnalysisResult(
        config=cfg,
        gxparm=gxparm,
        integrate=integrate,
        xds_input=xds_input,
        composition=composition,
        wilson=calibration,
        candidate_reflections=candidate_reflections,
        reflections_long=reflections_long,
        frame_summary=frame_summary,
        thickness_long=thickness_long,
        reflection_sensitivity=reflection_sensitivity,
    )


def run_analysis_from_paths(
    gxparm_path: str | Path,
    integrate_path: str | Path,
    composition_text: str,
    xdsinp_path: str | Path | None = None,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Parse inputs from disk and run the pipeline."""

    gxparm = parse_gxparm(gxparm_path)
    integrate = parse_integrate_hkl(integrate_path)
    composition = parse_composition(composition_text)
    xds_input = load_optional_xds_inp(xdsinp_path)
    return run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        xds_input=xds_input,
        config=config,
    )


def run_analysis_from_stream_path(
    stream_path: str | Path,
    composition_text: str,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Parse a CrystFEL ``.stream`` file and run snapshot-oriented analysis."""

    cfg = config or AnalysisConfig()
    stream_data = parse_crystfel_stream(stream_path)
    gxparm, integrate, reciprocal_by_frame = crystfel_stream_to_analysis_inputs(stream_data)
    crystal_table = stream_data.crystal_table
    detector_center_by_frame = {
        int(row["frame"]): (
            float(gxparm.orgx_px) + float(cfg.stream_detector_shift_sign) * float(row["det_shift_x_mm"]) / float(gxparm.pixel_x_mm),
            float(gxparm.orgy_px) + float(cfg.stream_detector_shift_sign) * float(row["det_shift_y_mm"]) / float(gxparm.pixel_y_mm),
        )
        for _, row in crystal_table.iterrows()
    }
    distance_by_frame = {
        int(row["frame"]): float(row["distance_mm"])
        for _, row in crystal_table.iterrows()
        if np.isfinite(float(row["distance_mm"]))
    }
    orientation_model = ReciprocalMatrixOrientationModel(
        reciprocal_by_frame=reciprocal_by_frame,
        reciprocal_reference=gxparm.reciprocal_reference,
        use_direct_reciprocal_vectors=True,
        detector_center_by_frame=detector_center_by_frame,
        distance_by_frame=distance_by_frame,
    )
    composition = parse_composition(composition_text)
    return run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        xds_input=None,
        config=cfg,
        orientation_model=orientation_model,
    )


def run_analysis_from_pets_project_path(
    pets_project_path: str | Path,
    composition_text: str,
    config: AnalysisConfig | None = None,
    *,
    pts2_path: str | Path | None = None,
    ptsopt_path: str | Path | None = None,
    rprofall_path: str | Path | None = None,
    ub_convention: str = "columns",
    orientation_mode: str = "pets_ab_xy",
    angle_reference: str = "absolute",
    include_domega_in_lattice: bool = False,
    invert_rotation: bool = False,
    use_only_for_calc: bool = False,
    detector_nx: int | None = None,
    detector_ny: int | None = None,
    alignment_rotation: np.ndarray | None = None,
    reindex_matrix: np.ndarray | None = None,
) -> AnalysisResult:
    """Parse PETS project files and run analysis."""

    model = load_pets_model(
        pets_project_path,
        pts2_path=pts2_path,
        ptsopt_path=ptsopt_path,
        rprofall_path=rprofall_path,
        detector_nx=detector_nx,
        detector_ny=detector_ny,
    )
    gxparm, integrate, reciprocal_by_frame, _ = pets_model_to_analysis_inputs(
        model,
        ub_convention=ub_convention,
        orientation_mode=orientation_mode,
        angle_reference=angle_reference,
        include_domega_in_lattice=include_domega_in_lattice,
        invert_rotation=invert_rotation,
        use_only_for_calc=use_only_for_calc,
        alignment_rotation=alignment_rotation,
        reindex_matrix=reindex_matrix,
    )
    orientation_model_obj = ReciprocalMatrixOrientationModel(
        reciprocal_by_frame=reciprocal_by_frame,
        reciprocal_reference=gxparm.reciprocal_reference,
    )
    composition = parse_composition(composition_text)
    return run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        xds_input=None,
        config=config,
        orientation_model=orientation_model_obj,
    )

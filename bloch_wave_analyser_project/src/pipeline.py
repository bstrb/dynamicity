"""High-level analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    RotationSeriesOrientationModel,
    build_zone_axes,
    cell_volume,
    excitation_error,
    generate_candidate_reflections,
    inside_detector,
    mark_untrusted_rectangles,
    nearest_zone_axis,
    project_to_detector,
    rotate_reference_vectors,
)
from .metrics import (
    aggregate_frame_thickness_sensitivity,
    aggregate_reflection_thickness_sensitivity,
    combined_proxy_score,
    effective_coupling_multiplicity,
    summarize_frame_proxy,
    two_beam_metric,
)
from .parsers import (
    CompositionResult,
    GXPARMData,
    IntegrateData,
    XDSInputData,
    load_optional_xds_inp,
    parse_composition,
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
    wilson: WilsonCalibration
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
    n_frames = _frame_count_from_inputs(integrate, xds_input)

    calibration = wilson_calibrate(integrate.observations, composition.sum_fj2)
    candidate_reflections = generate_candidate_reflections(
        gxparm,
        dmin_angstrom=cfg.dmin_angstrom,
        dmax_angstrom=cfg.dmax_angstrom,
    ).copy()

    candidate_reflections["Fg_abs"] = [
        calibration.lookup_amplitude(int(h), int(k), int(l))
        for h, k, l in candidate_reflections[["h", "k", "l"]].itertuples(index=False, name=None)
    ]
    volume_ang3 = cell_volume(gxparm.unit_cell)
    candidate_reflections["xi_angstrom"] = extinction_distance_angstrom(
        gxparm.wavelength_angstrom,
        volume_ang3,
        candidate_reflections["Fg_abs"].to_numpy(dtype=float),
    )
    candidate_reflections["xi_nm"] = candidate_reflections["xi_angstrom"] / 10.0

    zone_axes = build_zone_axes(limit=cfg.zone_axis_limit)
    reference_vectors = candidate_reflections[["gx_ref", "gy_ref", "gz_ref"]].to_numpy(dtype=float)
    thickness_array_nm = cfg.thickness_array_nm()

    reflection_tables: list[pd.DataFrame] = []
    frame_summary_rows: list[dict[str, float | int | str]] = []
    thickness_rows: list[dict[str, float | int]] = []
    rectangles = xds_input.untrusted_rectangles if xds_input is not None else []

    for frame in range(n_frames):
        rotation_start = orienter.rotation_matrix(frame, offset=0.0)
        rotation_mid = orienter.rotation_matrix(frame, offset=0.5)
        rotation_end = orienter.rotation_matrix(frame, offset=1.0)

        g_start = rotate_reference_vectors(rotation_start, reference_vectors)
        g_mid = rotate_reference_vectors(rotation_mid, reference_vectors)
        g_end = rotate_reference_vectors(rotation_end, reference_vectors)

        sg_start = excitation_error(g_start, gxparm.wavelength_angstrom)
        sg_mid = excitation_error(g_mid, gxparm.wavelength_angstrom)
        sg_end = excitation_error(g_end, gxparm.wavelength_angstrom)

        x_px, y_px, positive_sz = project_to_detector(g_mid, gxparm)
        within_detector = inside_detector(x_px, y_px, gxparm)
        in_untrusted = mark_untrusted_rectangles(x_px, y_px, rectangles)

        excited_mask = (
            ((sg_start * sg_end) <= 0.0) | (np.abs(sg_mid) < cfg.excitation_tolerance_invA)
        ) & positive_sz & within_detector
        if cfg.filter_untrusted:
            excited_mask &= ~in_untrusted

        frame_table = candidate_reflections.loc[excited_mask].copy().reset_index(drop=True)
        frame_number = frame + 1
        phi_deg = gxparm.phi0_deg + gxparm.dphi_deg * frame
        zone_axis = nearest_zone_axis(zone_axes, gxparm.real_space_reference, rotation_mid)

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
                    "mean_N_eff": 0.0,
                    "max_N_eff": 0.0,
                    "eigenvalue_spread_invA": 0.0,
                }
            )
            continue

        frame_table["frame"] = frame
        frame_table["frame_number"] = frame_number
        frame_table["phi_deg"] = phi_deg
        frame_table["sg_invA"] = sg_mid[excited_mask]
        frame_table["x_px"] = x_px[excited_mask]
        frame_table["y_px"] = y_px[excited_mask]
        frame_table["in_untrusted_region"] = in_untrusted[excited_mask]

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
        frame_table["d_2beam"] = d2beam_values
        frame_table["N_eff"] = n_eff_values
        frame_table["S_comb"] = s_comb_values

        if cfg.mode == "thickness" and thickness_array_nm is not None:
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
                "mean_N_eff": float(frame_metrics["mean_N_eff"]),
                "max_N_eff": float(frame_metrics["max_N_eff"]),
                "eigenvalue_spread_invA": float(structure.eigenvalues.max() - structure.eigenvalues.min()),
            }
        )

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

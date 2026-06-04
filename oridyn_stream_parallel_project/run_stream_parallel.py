#!/usr/bin/env python3
"""Parallel CrystFEL stream dynamical-risk analyser.

This project is intentionally stream-only. It reuses the parser/geometry/metric
code copied from ``bloch_wave_analyser_project/src`` and parallelizes over
indexed stream frames. Workers write chunk CSVs directly; the parent process
then concatenates them into the usual summary files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from dataclasses import replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import (
    ReciprocalMatrixOrientationModel,
    build_zone_axes,
    excitation_error,
    generate_candidate_reflections,
    inside_detector,
    nearest_zone_axis_from_reciprocal_matrix,
    project_to_detector,
)
from src.metrics import (
    empirical_amplitude_strength_proxy,
    geometry_dynamical_risk,
    orientation_excitation_probability,
    orientation_sg_sigma,
    primitive_hkl_key,
    reciprocal_strength_proxy,
    summarize_frame_proxy,
    two_channel_dynamical_risk,
)
from src.parsers import (
    crystfel_stream_to_analysis_inputs,
    parse_crystfel_stream,
)


LEGACY_DYN_COLUMNS = [
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

WORKER: dict[str, Any] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="Input CrystFEL .stream file")
    parser.add_argument(
        "--composition",
        default=None,
        help="Accepted for compatibility; geometry-only stream risk does not use composition",
    )
    parser.add_argument("--dmin", type=float, default=0.4, help="Minimum d-spacing in Angstrom")
    parser.add_argument("--dmax", type=float, default=20.0, help="Maximum d-spacing in Angstrom")
    parser.add_argument(
        "--excitation-tolerance",
        type=float,
        default=0.01,
        help="Target reflection excitation gate in 1/A",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--analysis-frame",
        dest="analysis_frames",
        action="append",
        type=int,
        default=None,
        help="One-based frame number to analyze; repeatable",
    )
    parser.add_argument(
        "--analysis-frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="One-based inclusive frame range to analyze",
    )
    parser.add_argument(
        "--analysis-frame-step",
        type=int,
        default=1,
        help="Step size for --analysis-frame-range",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max((os.cpu_count() or 2) - 1, 1),
        help="Worker processes. Use 0 for all logical CPUs; use 1 for serial debugging.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of frames per worker task/checkpoint CSV",
    )
    parser.add_argument(
        "--mp-start-method",
        choices=mp.get_all_start_methods(),
        default=("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()),
        help="Multiprocessing start method. fork is fastest/memory-friendliest on Linux.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress after this many completed chunks; use 0 to disable",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    parser.add_argument(
        "--remove-chunks",
        action="store_true",
        help="Remove chunk CSVs after final concatenation",
    )
    parser.add_argument(
        "--skip-legacy-s-dyn",
        action="store_true",
        help="Skip the older geometry_dynamical_risk/S_dyn pathway score for speed",
    )
    parser.add_argument(
        "--stream-det-shift-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="Convert stream det_shift_mm to beam-center shift as sign * shift_mm / pixel_mm",
    )
    parser.add_argument(
        "--stream-mirror-x-axis",
        dest="stream_mirror_x_axis",
        action="store_true",
        default=True,
        help="Mirror stream detector projections about the detector x-axis",
    )
    parser.add_argument(
        "--no-stream-mirror-x-axis",
        dest="stream_mirror_x_axis",
        action="store_false",
        help="Disable stream detector x-axis mirroring",
    )
    parser.add_argument(
        "--detector-xy-swapped",
        action="store_true",
        help="Swap detector x/y after projection",
    )
    parser.add_argument(
        "--beam-direction",
        choices=["plus_z", "minus_z"],
        default="plus_z",
        help="Beam direction used for excitation/projection",
    )
    parser.add_argument(
        "--orientation-sigma-deg",
        type=float,
        default=0.2,
        help="Isotropic orientation uncertainty in degrees",
    )
    parser.add_argument(
        "--dynamical-environment-tolerance",
        type=float,
        default=0.01,
        help="Broad excitation window in 1/A for neighboring beams",
    )
    parser.add_argument(
        "--dynamical-environment-weight-min",
        type=float,
        default=2.0e-2,
        help="Minimum Gaussian excitation weight for environment beams",
    )
    parser.add_argument(
        "--dynamical-neighbor-radius",
        type=float,
        default=0.12,
        help="Reciprocal-space radius in 1/A for local excited-neighbor density",
    )
    parser.add_argument(
        "--dynamical-zone-axis-sigma",
        type=float,
        default=2.0,
        help="Angular scale in degrees for low-index zone-axis ZOLZ boost",
    )
    parser.add_argument(
        "--dynamical-zone-sigma",
        type=float,
        default=0.06,
        help="Beam-parallel reciprocal-coordinate sigma in 1/A for same-Laue-zone scoring",
    )
    parser.add_argument(
        "--dynamical-neighbor-sigma",
        type=float,
        default=None,
        help="Optional Gaussian sigma in 1/A for local neighbor window",
    )
    parser.add_argument(
        "--dynamical-row-direction-limit",
        type=int,
        default=2,
        help="Low-order HKL direction limit for systematic-row scoring",
    )
    parser.add_argument(
        "--dynamical-row-max-steps",
        type=int,
        default=5,
        help="Maximum +/- HKL steps along each low-order row direction",
    )
    parser.add_argument(
        "--dynamical-row-sigma",
        type=float,
        default=0.25,
        help="Gaussian reciprocal-distance sigma in 1/A for row-neighbor scoring",
    )
    parser.add_argument(
        "--dynamical-coupling-q0",
        type=float,
        default=0.25,
        help="q0 in 1/A for model-free difference-vector coupling proxy",
    )
    parser.add_argument(
        "--dynamical-coupling-power",
        type=float,
        default=2.0,
        help="Power for model-free difference-vector coupling proxy",
    )
    parser.add_argument("--dynamical-weight-self", type=float, default=1.0)
    parser.add_argument("--dynamical-weight-zone", type=float, default=1.0)
    parser.add_argument("--dynamical-weight-row", type=float, default=1.0)
    parser.add_argument(
        "--dynamical-cluster-sigma-alpha",
        type=float,
        default=1.0,
        help="Cluster-only sigma scale: sigma_dyn_rel = 1 + alpha * cluster_risk_geom",
    )
    return parser


def parse_analysis_frames(args: argparse.Namespace, n_frames: int) -> list[int]:
    frames: list[int] = []
    if args.analysis_frames:
        frames.extend(int(value) for value in args.analysis_frames if value is not None)
    if args.analysis_frame_range is not None:
        start, end = (int(value) for value in args.analysis_frame_range)
        step = max(int(args.analysis_frame_step), 1)
        if end < start:
            start, end = end, start
        frames.extend(range(start, end + 1, step))
    if not frames:
        frames = list(range(1, n_frames + 1))
    unique = sorted(set(frames))
    out_of_range = [frame for frame in unique if frame < 1 or frame > n_frames]
    if out_of_range:
        raise ValueError(f"Requested frames are out of range: {out_of_range}. Valid range is 1..{n_frames}.")
    return [frame_number - 1 for frame_number in unique]


def batched(values: list[int], size: int) -> Iterable[list[int]]:
    size = max(int(size), 1)
    for index in range(0, len(values), size):
        yield values[index : index + size]


def init_worker(payload: dict[str, Any]) -> None:
    global WORKER
    WORKER = payload


def empty_summary_row(frame: int, frame_number: int, zone_axis: Any) -> dict[str, float | int | str]:
    return {
        "frame": frame,
        "frame_number": frame_number,
        "phi_deg": 0.0,
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


def legacy_placeholder(frame_table: pd.DataFrame, zone_axis: tuple[int, int, int]) -> pd.DataFrame:
    rows = pd.DataFrame(index=frame_table.index)
    rows["visibility_gate"] = 1.0
    rows["strength_proxy"] = frame_table["strength_proxy"].to_numpy(dtype=float)
    rows["zone_order"] = (
        frame_table[["h", "k", "l"]].to_numpy(dtype=int) @ np.asarray(zone_axis, dtype=int)
    )
    rows["is_zolz"] = rows["zone_order"].to_numpy(dtype=int) == 0
    rows["zone_axis_risk"] = 0.0
    rows["ZOLZ_zone_axis_risk"] = 0.0
    rows["neighbor_density"] = 0.0
    rows["row_coupling"] = 0.0
    rows["pathway_risk"] = 0.0
    rows["local_crowding"] = 0.0
    rows["S_dyn"] = 0.0
    return rows.loc[:, LEGACY_DYN_COLUMNS]


def process_one_frame(frame: int) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    gxparm = WORKER["gxparm"]
    orienter = WORKER["orientation_model"]
    cfg = WORKER["config"]
    candidate_reflections = WORKER["candidate_reflections"]
    hkl_array = WORKER["hkl_array"]
    hkl_tuples = WORKER["hkl_tuples"]
    candidate_hkl_to_index = WORKER["candidate_hkl_to_index"]
    candidate_row_keys = WORKER["candidate_row_keys"]
    strength = WORKER["strength"]
    empirical_strength = WORKER["empirical_strength"]
    zone_axes = WORKER["zone_axes"]
    observed_by_frame = WORKER["observed_by_frame"]
    beam_direction = WORKER["beam_direction"]

    reciprocal_frame = orienter.reciprocal_matrix(frame)
    g_mid = hkl_array @ reciprocal_frame.T
    sg_mid = excitation_error(g_mid, gxparm.wavelength_angstrom, beam_direction=beam_direction)

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

    x_px, y_px, positive_sz = project_to_detector(g_mid, frame_gxparm, beam_direction=beam_direction)
    if cfg["stream_mirror_x_axis"]:
        y_px = 2.0 * float(frame_gxparm.orgy_px) - y_px

    if cfg["detector_xy_swapped"]:
        x_px, y_px = y_px.copy(), x_px.copy()
        within_detector = (
            (x_px >= 0.0)
            & (x_px < frame_gxparm.detector_ny)
            & (y_px >= 0.0)
            & (y_px < frame_gxparm.detector_nx)
        )
    else:
        within_detector = inside_detector(x_px, y_px, frame_gxparm)

    excited_mask = (np.abs(sg_mid) < cfg["excitation_tolerance_invA"]) & positive_sz & within_detector
    frame_table = candidate_reflections.loc[excited_mask].copy().reset_index(drop=True)
    frame_number = frame + 1
    zone_axis = nearest_zone_axis_from_reciprocal_matrix(
        zone_axes,
        reciprocal_frame,
        beam_direction=beam_direction,
    )
    if frame_table.empty:
        return empty_summary_row(frame, frame_number, zone_axis), frame_table

    frame_table["frame"] = frame
    frame_table["frame_number"] = frame_number
    frame_table["phi_deg"] = 0.0
    frame_table["sg_invA"] = sg_mid[excited_mask]
    frame_table["x_px"] = x_px[excited_mask]
    frame_table["y_px"] = y_px[excited_mask]
    frame_table["in_untrusted_region"] = False

    if cfg["include_legacy_s_dyn"]:
        legacy_table = geometry_dynamical_risk(
            candidate_reflections=candidate_reflections,
            g_vectors_invA=g_mid,
            sg_invA=sg_mid,
            target_mask=excited_mask,
            environment_mask=positive_sz,
            zone_axis=zone_axis.axis,
            zone_axis_angle_deg=zone_axis.angle_deg,
            hkl_to_index=candidate_hkl_to_index,
            hkl_tuples=hkl_tuples,
            row_keys=candidate_row_keys,
            strength_proxy=strength,
            environment_tolerance_invA=cfg["dynamical_environment_tolerance_invA"],
            environment_weight_min=cfg["dynamical_environment_weight_min"],
            neighbor_radius_invA=cfg["dynamical_neighbor_radius_invA"],
            zone_axis_sigma_deg=cfg["dynamical_zone_axis_sigma_deg"],
        )
    else:
        legacy_table = legacy_placeholder(frame_table, zone_axis.axis)
    for column in legacy_table.columns:
        frame_table[column] = legacy_table[column].to_numpy()
    frame_table["legacy_S_dyn"] = frame_table["S_dyn"].to_numpy(dtype=float)

    two_channel_table = two_channel_dynamical_risk(
        candidate_reflections=candidate_reflections,
        g_vectors_invA=g_mid,
        sg_invA=sg_mid,
        target_mask=excited_mask,
        hkl_to_index=candidate_hkl_to_index,
        hkl_tuples=hkl_tuples,
        strength_proxy=strength,
        empirical_strength_proxy=empirical_strength,
        beam_direction=beam_direction,
        excitation_tolerance_invA=cfg["dynamical_environment_tolerance_invA"],
        environment_weight_min=cfg["dynamical_environment_weight_min"],
        zone_sigma_invA=cfg["dynamical_zone_sigma_invA"],
        neighbor_radius_invA=cfg["dynamical_neighbor_radius_invA"],
        neighbor_sigma_invA=cfg["dynamical_neighbor_sigma_invA"],
        row_direction_limit=cfg["dynamical_row_direction_limit"],
        row_max_steps=cfg["dynamical_row_max_steps"],
        row_sigma_invA=cfg["dynamical_row_sigma_invA"],
        coupling_q0_invA=cfg["dynamical_coupling_q0_invA"],
        coupling_power=cfg["dynamical_coupling_power"],
        weight_self=cfg["dynamical_weight_self"],
        weight_zone=cfg["dynamical_weight_zone"],
        weight_row=cfg["dynamical_weight_row"],
        sigma_alpha=cfg["dynamical_cluster_sigma_alpha"],
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

    frame_table["d_2beam"] = 0.0
    frame_table["N_eff"] = 0.0
    frame_table["S_comb"] = 0.0
    orientation_sigma_values = orientation_sg_sigma(
        g_vectors_invA=g_mid[excited_mask],
        wavelength_angstrom=gxparm.wavelength_angstrom,
        orientation_sigma_deg=cfg["orientation_sigma_deg"],
        beam_direction=beam_direction,
    )
    orientation_p_excited_values = orientation_excitation_probability(
        sg_invA=frame_table["sg_invA"].to_numpy(dtype=float),
        sg_sigma_orient_invA=orientation_sigma_values,
        excitation_tolerance_invA=cfg["excitation_tolerance_invA"],
    )
    s_orient_values = frame_table["S_dyn"].to_numpy(dtype=float)
    frame_table["orientation_sigma_sg_invA"] = orientation_sigma_values
    frame_table["orientation_p_excited"] = orientation_p_excited_values
    frame_table["S_orient"] = s_orient_values
    frame_table["sigma_orient_scale"] = 1.0 + 0.5 * s_orient_values

    frame_metrics = summarize_frame_proxy(frame_table)
    summary_row = {
        "frame": frame,
        "frame_number": frame_number,
        "phi_deg": 0.0,
        "zone_axis": zone_axis.label,
        "zone_axis_angle_deg": zone_axis.angle_deg,
        "n_excited": int(frame_metrics["n_excited"]),
        "S_2beam": 0.0,
        "S_MB": 0.0,
        "S_dyn": float(frame_table["S_dyn"].sum()),
        "S_orient": float(np.sum(s_orient_values)),
        "mean_N_eff": 0.0,
        "max_N_eff": 0.0,
        "mean_orientation_sigma_sg_invA": float(np.mean(orientation_sigma_values)),
        "mean_orientation_p_excited": float(np.mean(orientation_p_excited_values)),
        "eigenvalue_spread_invA": 0.0,
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
    return summary_row, frame_table


def process_chunk(task: tuple[int, list[int]]) -> dict[str, Any]:
    part_index, frames = task
    start = time.monotonic()
    summary_rows: list[dict[str, float | int | str]] = []
    reflection_tables: list[pd.DataFrame] = []
    for frame in frames:
        summary_row, frame_table = process_one_frame(frame)
        summary_rows.append(summary_row)
        if not frame_table.empty:
            reflection_tables.append(frame_table)

    chunk_dir = Path(WORKER["chunk_dir"])
    summary_path = chunk_dir / f"frame_summary_part_{part_index:06d}.csv"
    reflections_path = chunk_dir / f"reflections_long_part_{part_index:06d}.csv"
    pd.DataFrame.from_records(summary_rows).sort_values("frame").to_csv(summary_path, index=False)
    if reflection_tables:
        reflections = pd.concat(reflection_tables, ignore_index=True)
    else:
        reflections = pd.DataFrame()
    reflections.to_csv(reflections_path, index=False)
    return {
        "part_index": part_index,
        "frames": len(frames),
        "first_frame": int(frames[0]) + 1,
        "last_frame": int(frames[-1]) + 1,
        "n_reflection_rows": int(reflections.shape[0]),
        "n_excited": int(sum(int(row["n_excited"]) for row in summary_rows)),
        "elapsed_s": float(time.monotonic() - start),
        "summary_path": str(summary_path),
        "reflections_path": str(reflections_path),
    }


def concatenate_csv_files(paths: list[Path], output_path: Path) -> None:
    wrote_header = False
    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                header = handle.readline()
                if not header:
                    continue
                if not wrote_header:
                    out.write(header)
                    wrote_header = True
                for line in handle:
                    out.write(line)


def write_final_outputs(output_dir: Path, manifest: pd.DataFrame) -> None:
    manifest = manifest.sort_values("part_index").reset_index(drop=True)
    summary_paths = [Path(value) for value in manifest["summary_path"]]
    reflection_paths = [Path(value) for value in manifest["reflections_path"]]

    frame_summary = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    frame_summary = frame_summary.sort_values("frame").reset_index(drop=True)
    frame_summary.to_csv(output_dir / "frame_summary.csv", index=False)
    concatenate_csv_files(reflection_paths, output_dir / "reflections_long.csv")
    manifest.to_csv(output_dir / "chunk_manifest.csv", index=False)

    compact_cols = [
        "frame_number",
        "h",
        "k",
        "l",
        "sg_invA",
        "q_invA",
        "self_extinction_score",
        "attenuation_risk",
        "same_zone_cluster_score_geom",
        "systematic_row_cluster_score_geom",
        "cluster_risk_geom",
        "cluster_risk_iw",
        "sigma_dyn_rel",
        "I_obs",
        "sigma_obs",
        "sigma_new",
        "weight_new",
        "is_observed_target",
    ]
    top_tables: dict[str, list[pd.DataFrame]] = {
        "attenuation_risk": [],
        "cluster_risk_geom": [],
        "cluster_risk_iw": [],
    }
    observed_top_tables: dict[str, list[pd.DataFrame]] = {
        "attenuation_risk": [],
        "cluster_risk_geom": [],
    }
    summary_arrays: dict[str, list[np.ndarray]] = {
        "attenuation_risk": [],
        "cluster_risk_geom": [],
        "cluster_risk_iw": [],
        "sigma_dyn_rel": [],
    }
    n_rows = 0
    n_observed_targets = 0
    for path in reflection_paths:
        table = pd.read_csv(path)
        if table.empty:
            continue
        n_rows += int(table.shape[0])
        if "is_observed_target" in table.columns:
            n_observed_targets += int(table["is_observed_target"].sum())
        available_cols = [col for col in compact_cols if col in table.columns]
        for column in top_tables:
            if column in table.columns:
                top_tables[column].append(table.nlargest(200, column).loc[:, available_cols])
                summary_arrays[column].append(table[column].to_numpy(dtype=float))
        if "sigma_dyn_rel" in table.columns:
            summary_arrays["sigma_dyn_rel"].append(table["sigma_dyn_rel"].to_numpy(dtype=float))
        if "is_observed_target" in table.columns:
            obs = table[table["is_observed_target"]].copy()
            if not obs.empty:
                for column in observed_top_tables:
                    if column in obs.columns:
                        observed_top_tables[column].append(obs.nlargest(200, column).loc[:, available_cols])

    top_output_names = {
        "attenuation_risk": "top_self_extinction_risk.csv",
        "cluster_risk_geom": "top_cluster_risk_geom.csv",
        "cluster_risk_iw": "top_cluster_risk_intensity_weighted.csv",
    }
    for column, tables in top_tables.items():
        if tables:
            pd.concat(tables, ignore_index=True).nlargest(200, column).to_csv(
                output_dir / top_output_names[column],
                index=False,
            )
    observed_output_names = {
        "attenuation_risk": "top_observed_self_extinction_risk.csv",
        "cluster_risk_geom": "top_observed_cluster_risk_geom.csv",
    }
    for column, tables in observed_top_tables.items():
        if tables:
            pd.concat(tables, ignore_index=True).nlargest(200, column).to_csv(
                output_dir / observed_output_names[column],
                index=False,
            )

    summary_lines = [
        f"n_reflection_rows: {n_rows}",
        f"n_observed_targets: {n_observed_targets}",
    ]
    for column in ("attenuation_risk", "cluster_risk_geom", "cluster_risk_iw", "sigma_dyn_rel"):
        arrays = summary_arrays[column]
        if not arrays:
            continue
        values = np.concatenate(arrays)
        summary_lines.extend(
            [
                f"mean_{column}: {float(np.mean(values)):.6g}",
                f"p95_{column}: {float(np.quantile(values, 0.95)):.6g}",
            ]
        )
    (output_dir / "two_channel_summary.txt").write_text("\n".join(summary_lines) + "\n")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is not empty: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    return chunk_dir


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    chunk_dir = prepare_output_dir(output_dir, bool(args.overwrite))

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1
    chunk_size = max(int(args.chunk_size), 1)
    beam_direction = (0.0, 0.0, -1.0) if args.beam_direction == "minus_z" else (0.0, 0.0, 1.0)

    run_metadata = {
        "argv": list(sys.argv),
        "args": vars(args),
        "resolved_workers": workers,
        "project": "bloch_stream_parallel_project",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, ensure_ascii=True)
    )

    parse_start = time.monotonic()
    print(f"Parsing stream: {args.stream}", flush=True)
    stream_data = parse_crystfel_stream(args.stream)
    gxparm, integrate, reciprocal_by_frame = crystfel_stream_to_analysis_inputs(stream_data)
    print(
        f"Parsed {stream_data.crystal_table.shape[0]} indexed crystals and "
        f"{integrate.observations.shape[0]} observations in {time.monotonic() - parse_start:.1f}s",
        flush=True,
    )

    prep_start = time.monotonic()
    print("Preparing candidate reflections and per-frame observation lookup...", flush=True)
    candidate_reflections = generate_candidate_reflections(
        gxparm,
        dmin_angstrom=float(args.dmin),
        dmax_angstrom=float(args.dmax),
    ).copy()
    candidate_reflections["strength_proxy"] = reciprocal_strength_proxy(
        candidate_reflections["q_invA"].to_numpy(dtype=float)
    )
    candidate_reflections["empirical_strength_proxy"] = empirical_amplitude_strength_proxy(
        candidate_reflections,
        integrate.observations,
        fallback_values=candidate_reflections["strength_proxy"].to_numpy(dtype=float),
    )
    candidate_reflections["Fg_abs"] = np.nan
    candidate_reflections["xi_angstrom"] = np.nan
    candidate_reflections["xi_nm"] = np.nan

    crystal_table = stream_data.crystal_table
    detector_center_by_frame = {
        int(row["frame"]): (
            float(gxparm.orgx_px)
            + float(args.stream_det_shift_sign) * float(row["det_shift_x_mm"]) / float(gxparm.pixel_x_mm),
            float(gxparm.orgy_px)
            + float(args.stream_det_shift_sign) * float(row["det_shift_y_mm"]) / float(gxparm.pixel_y_mm),
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

    n_frames = int(integrate.estimated_n_frames)
    frames_to_run = parse_analysis_frames(args, n_frames)
    tasks = [(part_index, chunk) for part_index, chunk in enumerate(batched(frames_to_run, chunk_size), start=1)]
    candidate_hkl_tuples = [
        tuple(map(int, hkl))
        for hkl in candidate_reflections[["h", "k", "l"]].itertuples(index=False, name=None)
    ]

    payload = {
        "gxparm": gxparm,
        "orientation_model": orientation_model,
        "candidate_reflections": candidate_reflections,
        "hkl_array": candidate_reflections[["h", "k", "l"]].to_numpy(dtype=float),
        "hkl_tuples": candidate_hkl_tuples,
        "candidate_hkl_to_index": {hkl: index for index, hkl in enumerate(candidate_hkl_tuples)},
        "candidate_row_keys": [primitive_hkl_key(*hkl) for hkl in candidate_hkl_tuples],
        "strength": candidate_reflections["strength_proxy"].to_numpy(dtype=float),
        "empirical_strength": candidate_reflections["empirical_strength_proxy"].to_numpy(dtype=float),
        "zone_axes": build_zone_axes(limit=5),
        "observed_by_frame": observed_by_frame,
        "beam_direction": beam_direction,
        "chunk_dir": str(chunk_dir),
        "config": {
            "excitation_tolerance_invA": float(args.excitation_tolerance),
            "stream_mirror_x_axis": bool(args.stream_mirror_x_axis),
            "detector_xy_swapped": bool(args.detector_xy_swapped),
            "orientation_sigma_deg": float(args.orientation_sigma_deg),
            "include_legacy_s_dyn": not bool(args.skip_legacy_s_dyn),
            "dynamical_environment_tolerance_invA": float(args.dynamical_environment_tolerance),
            "dynamical_environment_weight_min": float(args.dynamical_environment_weight_min),
            "dynamical_neighbor_radius_invA": float(args.dynamical_neighbor_radius),
            "dynamical_zone_axis_sigma_deg": float(args.dynamical_zone_axis_sigma),
            "dynamical_zone_sigma_invA": float(args.dynamical_zone_sigma),
            "dynamical_neighbor_sigma_invA": (
                None if args.dynamical_neighbor_sigma is None else float(args.dynamical_neighbor_sigma)
            ),
            "dynamical_row_direction_limit": int(args.dynamical_row_direction_limit),
            "dynamical_row_max_steps": int(args.dynamical_row_max_steps),
            "dynamical_row_sigma_invA": float(args.dynamical_row_sigma),
            "dynamical_coupling_q0_invA": float(args.dynamical_coupling_q0),
            "dynamical_coupling_power": float(args.dynamical_coupling_power),
            "dynamical_weight_self": float(args.dynamical_weight_self),
            "dynamical_weight_zone": float(args.dynamical_weight_zone),
            "dynamical_weight_row": float(args.dynamical_weight_row),
            "dynamical_cluster_sigma_alpha": float(args.dynamical_cluster_sigma_alpha),
        },
    }
    print(
        f"Prepared {candidate_reflections.shape[0]} candidate reflections and "
        f"{len(tasks)} chunk(s) in {time.monotonic() - prep_start:.1f}s",
        flush=True,
    )
    print(
        f"Analyzing {len(frames_to_run)} frame(s) with {workers} worker(s), "
        f"chunk_size={chunk_size}, start_method={args.mp_start_method}",
        flush=True,
    )

    analysis_start = time.monotonic()
    results: list[dict[str, Any]] = []
    completed_chunks = 0
    completed_frames = 0
    completed_rows = 0
    progress_every = max(int(args.progress_every), 0)

    if workers == 1:
        init_worker(payload)
        for task in tasks:
            result = process_chunk(task)
            results.append(result)
            completed_chunks += 1
            completed_frames += int(result["frames"])
            completed_rows += int(result["n_reflection_rows"])
            if progress_every and (completed_chunks % progress_every == 0 or completed_chunks == len(tasks)):
                print_progress(completed_chunks, len(tasks), completed_frames, len(frames_to_run), completed_rows, analysis_start)
    else:
        context = mp.get_context(args.mp_start_method)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=init_worker,
            initargs=(payload,),
        ) as executor:
            future_map = {executor.submit(process_chunk, task): task for task in tasks}
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                completed_chunks += 1
                completed_frames += int(result["frames"])
                completed_rows += int(result["n_reflection_rows"])
                if progress_every and (completed_chunks % progress_every == 0 or completed_chunks == len(tasks)):
                    print_progress(
                        completed_chunks,
                        len(tasks),
                        completed_frames,
                        len(frames_to_run),
                        completed_rows,
                        analysis_start,
                    )

    manifest = pd.DataFrame.from_records(results).sort_values("part_index").reset_index(drop=True)
    print("Writing final CSV outputs...", flush=True)
    write_final_outputs(output_dir, manifest)
    if args.remove_chunks:
        shutil.rmtree(chunk_dir)
    elapsed = time.monotonic() - analysis_start
    print(
        f"Done: {completed_frames} frames, {completed_rows} reflection rows, "
        f"analysis_time={elapsed:.1f}s, output={output_dir.resolve()}",
        flush=True,
    )


def print_progress(
    completed_chunks: int,
    total_chunks: int,
    completed_frames: int,
    total_frames: int,
    completed_rows: int,
    start_time: float,
) -> None:
    elapsed = max(time.monotonic() - start_time, 1.0e-9)
    frames_per_second = completed_frames / elapsed
    remaining = total_frames - completed_frames
    eta_seconds = remaining / frames_per_second if frames_per_second > 0.0 else float("inf")
    eta_minutes = eta_seconds / 60.0 if np.isfinite(eta_seconds) else float("inf")
    print(
        "Progress: "
        f"{completed_chunks}/{total_chunks} chunks, "
        f"{completed_frames}/{total_frames} frames, "
        f"{completed_rows} reflection rows, "
        f"{frames_per_second:.2f} frames/s, eta={eta_minutes:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()

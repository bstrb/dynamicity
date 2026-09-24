#!/usr/bin/env python
"""Executable launcher for the OriDyn cRED/XDS validation workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "matplotlib").mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

import pandas as pd
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xds_cred.config import collect_software_versions, load_config, write_json
from xds_cred.geometry import (
    audit_geometry,
    calculate_observation_risk,
    choose_geometry,
    model_from_geometry,
    parse_integrate_lp_geometry,
    parse_xparm,
)
from xds_cred.reporting import (
    Progress,
    log_and_flush,
    prepare_output_dir,
    setup_logger,
    write_geometry_summary,
    write_plots,
    write_summary,
)
from xds_cred.risk import enumerate_neighbor_offsets, score_integrate_file
from xds_cred.scale import parse_correct_scales, parse_integrate_scales
from xds_cred.statistics import (
    add_correct_status,
    add_leave_one_out_disagreement,
    add_resolution_shells,
    correlation_tables,
    finalize_group_table,
    resolution_summary,
    risk_distribution,
)
from xds_cred.symmetry import assign_symmetry_ids, require_symmetry_backend
from xds_cred.xds_parser import (
    count_integrate_records,
    dump_input_manifest,
    parse_integrate_header,
    parse_resolution_shells,
    parse_xds_ascii_matches,
    parse_xds_inp,
    read_integrate_subset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to the JSON workflow configuration.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    prepare_output_dir(config.output_dir, config.allow_overwrite)
    logger = setup_logger(config.output_dir)
    try:
        backend = require_symmetry_backend()
        log_and_flush(logger, f"Using crystallographic symmetry backend: {backend}")
        parameters = config.to_resolved_dict()
        parameters["config_path"] = str(args.config)
        write_json(config.output_dir / "parameters.json", parameters)
        dump_input_manifest(config.output_dir / "input_files.json", config.input_dir)
        write_json(config.output_dir / "software_versions.json", collect_software_versions(PROJECT_ROOT))

        input_paths = {name: config.input_dir / name for name in (
            "XDS.INP",
            "XPARM.XDS",
            "GXPARM.XDS",
            "INTEGRATE.HKL",
            "INTEGRATE.LP",
            "CORRECT.LP",
            "XDS_ASCII.HKL",
        )}
        for name, path in input_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Required XDS input is missing: {name} at {path}")

        integrate_header = parse_integrate_header(input_paths["INTEGRATE.HKL"])
        xds_inp = parse_xds_inp(input_paths["XDS.INP"])
        include_range = _include_resolution_range(integrate_header.metadata, xds_inp)
        shells = parse_resolution_shells(input_paths["CORRECT.LP"], include_range=include_range)

        geometries = {
            "XPARM.XDS": model_from_geometry(
                parse_xparm(input_paths["XPARM.XDS"], "XPARM.XDS"),
                "XPARM.XDS",
                angle_offset_frames=config.zcal_angle_offset_frames,
            ),
            "GXPARM.XDS": model_from_geometry(
                parse_xparm(input_paths["GXPARM.XDS"], "GXPARM.XDS"),
                "GXPARM.XDS",
                angle_offset_frames=config.zcal_angle_offset_frames,
            ),
        }
        if config.use_integrate_lp_batches:
            geometries["INTEGRATE.LP_BATCH"] = parse_integrate_lp_geometry(
                input_paths["INTEGRATE.LP"],
                geometries["XPARM.XDS"].base,
                angle_offset_frames=config.zcal_angle_offset_frames,
            )
        audit_subset = read_integrate_subset(
            input_paths["INTEGRATE.HKL"],
            integrate_header.columns,
            config.geometry_audit.max_observations,
        )
        if audit_subset.empty:
            raise RuntimeError("No finite INTEGRATE.HKL observations were available for geometry audit.")
        geometry_audits = []
        geometry_summaries = []
        for geometry in geometries.values():
            reference_geometry = geometry.base
            audit_offsets, _distances, _coupling = enumerate_neighbor_offsets(
                reference_geometry.reciprocal_matrix_zero,
                config.score.r_cut,
                config.score.sigma_c,
            )
            audit, summary = audit_geometry(
                geometry,
                audit_subset,
                prediction_window_frames=config.geometry_audit.prediction_window_frames,
                score=config.score,
                neighbor_offsets=audit_offsets,
            )
            geometry_audits.append(audit)
            geometry_summaries.append(summary)
        geometry_audit = pd.concat(geometry_audits, ignore_index=True)
        selected_source, geometry_reason = choose_geometry(
            geometry_summaries,
            config.geometry_audit.max_median_abs_z_residual_frames,
            config.geometry_audit.max_median_abs_excitation_invA,
        )
        selected_geometry = _geometry_by_source(geometries, selected_source)
        geometry_audit.to_csv(config.output_dir / "geometry.csv.gz", index=False)
        write_geometry_summary(config.output_dir / "geometry_summary.txt", geometry_summaries, selected_source, geometry_reason)
        log_and_flush(logger, geometry_reason)

        total = count_integrate_records(input_paths["INTEGRATE.HKL"], config.max_observations)
        progress = Progress(logger, "risk scoring", total)
        risk_path = config.output_dir / "risk_chunks.csv"
        excluded_parse, risk_metadata = score_integrate_file(
            input_paths["INTEGRATE.HKL"],
            integrate_header.columns,
            selected_geometry,
            config.score,
            config.chunk_size,
            config.resolved_worker_count(),
            risk_path,
            progress=progress.update,
            max_observations=config.max_observations,
        )
        progress.update(0, force=True)
        parameters["risk_scoring"] = risk_metadata
        parameters["geometry_source"] = selected_source
        parameters["geometry_selection_reason"] = geometry_reason
        parameters["geometry_candidates"] = geometry_summaries
        parameters["space_group_number"] = selected_geometry.base.space_group_number
        parameters["resolution_shell_source"] = "CORRECT.LP" if not shells.empty else "not parsed"
        write_json(config.output_dir / "parameters.json", parameters)

        observations = pd.read_csv(risk_path)
        matches, match_metadata = parse_xds_ascii_matches(input_paths["XDS_ASCII.HKL"])
        parameters["correct_status_matching"] = match_metadata
        observations = add_correct_status(observations, matches)
        observations, groups, symmetry_metadata = assign_symmetry_ids(observations, selected_geometry.base.space_group_number)
        parameters["symmetry"] = symmetry_metadata
        observations = add_resolution_shells(observations, shells)
        observations, excluded_primary = add_leave_one_out_disagreement(
            observations,
            config.min_symmetry_multiplicity,
        )
        groups = finalize_group_table(observations, groups)

        risk_dist = risk_distribution(observations)
        correlations = correlation_tables(observations)
        resolution = resolution_summary(observations, shells)
        scale_integrate = parse_integrate_scales(input_paths["INTEGRATE.LP"])
        scale_correct = parse_correct_scales(input_paths["CORRECT.LP"], config.input_dir)
        audit_offsets, _distances, _coupling = enumerate_neighbor_offsets(
            selected_geometry.base.reciprocal_matrix_zero,
            config.score.r_cut,
            config.score.sigma_c,
        )
        orientation_audit, neighbor_audit = _build_orientation_audit(
            observations,
            selected_geometry,
            config.score,
            audit_offsets,
            config.orientation_audit_count,
            config.strongest_neighbors,
        )

        observations.to_csv(config.output_dir / "observations.csv.gz", index=False)
        groups.to_csv(config.output_dir / "groups.csv.gz", index=False)
        excluded_all = _combine_exclusions(excluded_parse, excluded_primary)
        excluded_all.to_csv(config.output_dir / "excluded.csv.gz", index=False)
        risk_dist.to_csv(config.output_dir / "risk_distribution.csv", index=False)
        correlations.to_csv(config.output_dir / "correlations.csv", index=False)
        resolution.to_csv(config.output_dir / "resolution.csv", index=False)
        scale_integrate.to_csv(config.output_dir / "scales_integrate.csv", index=False)
        scale_correct.to_csv(config.output_dir / "scales_correct.csv", index=False)
        orientation_audit.to_csv(config.output_dir / "orientation_audit.csv", index=False)
        neighbor_audit.to_csv(config.output_dir / "neighbor_audit.csv.gz", index=False)
        write_plots(config.output_dir, observations, geometry_audit)
        write_summary(
            config.output_dir / "summary.txt",
            config.dataset_name,
            observations,
            groups,
            correlations,
            risk_dist,
            resolution,
            geometry_reason,
            scale_integrate,
            scale_correct,
            parameters,
        )
        write_json(config.output_dir / "parameters.json", parameters)
        log_and_flush(logger, "Workflow complete. Full production analysis is performed only when the user runs the production config.")
        return 0
    except Exception as exc:
        log_and_flush(logger, f"ERROR: {exc}")
        raise


def _include_resolution_range(header_metadata: dict[str, object], xds_inp: dict[str, object]) -> tuple[float, float] | None:
    value = header_metadata.get("INCLUDE_RESOLUTION_RANGE") or xds_inp.get("INCLUDE_RESOLUTION_RANGE")
    if isinstance(value, list) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


def _combine_exclusions(excluded_parse: pd.DataFrame, excluded_primary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if excluded_parse is not None and not excluded_parse.empty:
        frames.append(excluded_parse)
    if excluded_primary is not None and not excluded_primary.empty:
        frames.append(excluded_primary)
    if not frames:
        return pd.DataFrame(columns=["observation_id", "h", "k", "l", "symmetry_id", "exclusion_reason"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _geometry_by_source(geometries: dict[str, object], source: str):
    """Return the geometry model whose reported source was selected."""

    for geometry in geometries.values():
        if getattr(geometry, "source", None) == source:
            return geometry
    available = ", ".join(str(getattr(geometry, "source", key)) for key, geometry in geometries.items())
    raise KeyError(f"Selected geometry source {source!r} was not found. Available sources: {available}")


def _build_orientation_audit(
    observations: pd.DataFrame,
    geometry,
    score,
    offsets,
    audit_count: int,
    strongest_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = _select_orientation_audit_rows(observations, audit_count)
    target_records = []
    neighbor_records = []
    for row in selected.itertuples(index=False):
        result = calculate_observation_risk(
            (int(row.h), int(row.k), int(row.l)),
            float(row.ZCAL),
            geometry,
            score,
            offsets,
            strongest_count=strongest_count,
        )
        strongest = ";".join(
            f"{item.h},{item.k},{item.l}:{item.contribution:.6g}"
            for item in result.strongest_neighbors
        )
        target_records.append(
            {
                "observation_id": int(row.observation_id),
                "h": result.h,
                "k": result.k,
                "l": result.l,
                "ZCAL": result.zcal,
                "rotation_angle_deg": result.rotation_angle,
                "geometry_source": result.geometry_source,
                "geometry_batch": result.geometry_batch,
                "q_target_x": result.q_target[0],
                "q_target_y": result.q_target[1],
                "q_target_z": result.q_target[2],
                "s_target": result.s_target,
                "E_target": result.E_target,
                "neighbor_count": result.neighbor_count,
                "R_env": result.R_env,
                "S_risk": result.S_risk,
                "z_pred": result.z_pred,
                "z_residual_frames": result.z_residual_frames,
                "z_residual_degrees": result.z_residual_degrees,
                "strongest_neighbor_contributions": strongest,
            }
        )
        for neighbor in result.neighbors:
            neighbor_records.append(
                {
                    "observation_id": int(row.observation_id),
                    "target_h": result.h,
                    "target_k": result.k,
                    "target_l": result.l,
                    "ZCAL": result.zcal,
                    "neighbor_h": neighbor.h,
                    "neighbor_k": neighbor.k,
                    "neighbor_l": neighbor.l,
                    "d_gh": neighbor.d_gh,
                    "s_neighbor": neighbor.s_neighbor,
                    "E_neighbor": neighbor.E_neighbor,
                    "C_gh": neighbor.C_gh,
                    "contribution": neighbor.contribution,
                    "R_env": result.R_env,
                    "S_risk": result.S_risk,
                }
            )
    return pd.DataFrame.from_records(target_records), pd.DataFrame.from_records(neighbor_records)


def _select_orientation_audit_rows(observations: pd.DataFrame, audit_count: int) -> pd.DataFrame:
    if observations.empty:
        return observations.copy()
    indices: list[int] = []
    h330 = observations[(observations["h"] == 3) & (observations["k"] == 3) & (observations["l"] == 0)]
    if not h330.empty:
        indices.append(int(h330.index[0]))
    count = max(int(audit_count), 1)
    spread = [0] if len(observations) == 1 else [int(round(value)) for value in np.linspace(0, len(observations) - 1, count)]
    spread = sorted(set(spread))
    indices.extend(int(observations.index[position]) for position in spread if 0 <= position < len(observations))
    unique_indices = list(dict.fromkeys(indices))[:count]
    return observations.loc[unique_indices].copy()


if __name__ == "__main__":
    raise SystemExit(main())

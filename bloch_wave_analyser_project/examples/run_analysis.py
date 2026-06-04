"""Command-line entry point for the Bloch-wave dynamicality analyser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import ReciprocalMatrixOrientationModel, RotationSeriesOrientationModel
from src.pets2 import (
    estimate_pets_alignment,
    load_pets_model,
    pets_model_to_analysis_inputs,
    predict_pets_detector_spots,
)
from src.parsers import (
    crystfel_stream_to_analysis_inputs,
    load_optional_xds_inp,
    parse_composition,
    parse_crystfel_stream,
    parse_gxparm,
    parse_integrate_hkl,
)
from src.pipeline import AnalysisConfig, run_analysis
from src.visualization import plot_detector_frame, plot_frame_summary, plot_thickness_scan
from src.wilson import wilson_calibrate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default=None, help="Optional path to CrystFEL .stream input")
    parser.add_argument(
        "--pets-project",
        default=None,
        help="Optional PETS project folder (e.g. root or *_petsdata) or PETS file path",
    )
    parser.add_argument("--pets-pts2", default=None, help="Optional explicit PETS .pts2/.pts2.backup path")
    parser.add_argument("--pets-ptsopt", default=None, help="Optional explicit PETS .ptsopt path")
    parser.add_argument("--pets-rprofall", default=None, help="Optional explicit PETS .rprofall path")
    parser.add_argument("--pets-hkl", default=None, help="Optional SHELX .hkl file with reflection observations for PETS")
    parser.add_argument(
        "--pets-align-gxparm",
        default=None,
        help="Optional GXPARM/XPARM used to align PETS UB (reindex + rotation)",
    )
    parser.add_argument("--gxparm", default=None, help="Path to GXPARM.XDS or XPARM.XDS")
    parser.add_argument("--integrate", default=None, help="Path to INTEGRATE.HKL")
    parser.add_argument("--xdsinp", default=None, help="Optional path to XDS.INP")
    parser.add_argument(
        "--orientation-gxparm",
        default=None,
        help="Optional GXPARM/XPARM used only for frame orientation progression override",
    )
    parser.add_argument(
        "--pets-ub-convention",
        choices=["columns", "rows"],
        default="columns",
        help="Interpretation of PETS UB matrix layout",
    )
    parser.add_argument(
        "--pets-orientation-mode",
        choices=["pets_ab_xy", "pets_ab_yx", "fixed_x_alpha", "euler_yxz", "euler_xyz", "axis_alpha_legacy", "none"],
        default="pets_ab_xy",
        help="PETS frame-orientation model",
    )
    parser.add_argument(
        "--pets-angle-reference",
        choices=["absolute", "first_frame", "zero"],
        default="absolute",
        help="How PETS frame angles are interpreted",
    )
    parser.add_argument(
        "--pets-include-domega-in-lattice",
        action="store_true",
        help="Include PETS domega as in-lattice Rz rotation",
    )
    parser.add_argument(
        "--pets-invert-rotation",
        action="store_true",
        help="Use inverse PETS frame rotation convention",
    )
    parser.add_argument(
        "--pets-use-only-for-calc",
        action="store_true",
        help="Use only PETS frames with useforcalc > 0",
    )
    parser.add_argument(
        "--pets-detector-nx",
        type=int,
        default=0,
        help="Optional PETS detector width override",
    )
    parser.add_argument(
        "--pets-detector-ny",
        type=int,
        default=0,
        help="Optional PETS detector height override",
    )
    parser.add_argument(
        "--pets-detector-projection",
        choices=["full", "paraxial"],
        default="full",
        help="PETS detector projection mode",
    )
    parser.add_argument(
        "--pets-detector-omega-map-mode",
        choices=["frame_absolute", "global_plus_frame", "global", "frame_only", "none"],
        default="frame_absolute",
        help="How PETS omega is applied in detector mapping",
    )
    parser.add_argument(
        "--pets-detector-omega-sign",
        type=float,
        default=1.0,
        help="Sign applied to PETS omega in detector mapping",
    )
    parser.add_argument(
        "--pets-detector-omega-offset-deg",
        type=float,
        default=0.0,
        help="Constant PETS detector in-plane offset angle in degrees",
    )
    parser.add_argument(
        "--pets-detector-swap-xy",
        action="store_true",
        help="Swap PETS detector x/y in geometry plotting",
    )
    parser.add_argument(
        "--pets-detector-flip-x",
        action="store_true",
        help="Flip PETS detector x axis in geometry plotting",
    )
    parser.add_argument(
        "--pets-detector-flip-y",
        action="store_true",
        help="Flip PETS detector y axis in geometry plotting",
    )
    parser.add_argument("--composition", required=True, help='Composition string, e.g. "24 Si, 48 O"')
    parser.add_argument("--dmin", type=float, default=0.6, help="Minimum d-spacing in angstrom")
    parser.add_argument("--dmax", type=float, default=50.0, help="Maximum d-spacing in angstrom")
    parser.add_argument("--excitation-tolerance", type=float, default=1.5e-3, help="Excitation tolerance in 1/A")
    parser.add_argument("--mode", choices=["proxy", "thickness"], default="proxy", help="Analysis mode")
    parser.add_argument("--thickness", type=float, default=None, help="Single thickness in nm")
    parser.add_argument("--thickness-start", type=float, default=None, help="Thickness scan start in nm")
    parser.add_argument("--thickness-stop", type=float, default=None, help="Thickness scan stop in nm")
    parser.add_argument("--thickness-step", type=float, default=None, help="Thickness scan step in nm")
    parser.add_argument("--output-dir", default="analysis_output", help="Directory for outputs")
    parser.add_argument(
        "--filter-untrusted",
        action="store_true",
        help="Exclude reflections in XDS untrusted rectangles",
    )
    parser.add_argument(
        "--show-untrusted-rectangles",
        action="store_true",
        help="Draw XDS untrusted rectangles on detector plots",
    )
    parser.add_argument(
        "--detector-xy-swapped",
        action="store_true",
        help="Swap detector x/y after projection",
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
        help="Mirror stream detector projections about the detector x-axis to match CrystFEL view",
    )
    parser.add_argument(
        "--no-stream-mirror-x-axis",
        dest="stream_mirror_x_axis",
        action="store_false",
        help="Disable stream detector x-axis mirroring",
    )
    parser.add_argument(
        "--detector-frame",
        dest="detector_frames",
        action="append",
        type=int,
        default=None,
        help="One-based frame number to export as detector plot (repeatable)",
    )
    parser.add_argument(
        "--analysis-frame",
        dest="analysis_frames",
        action="append",
        type=int,
        default=None,
        help="One-based frame number to analyze (repeatable)",
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
        help="Step size for --analysis-frame-range (default: 1)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress after this many analyzed frames; use 0 to disable",
    )
    parser.add_argument(
        "--orientation-only",
        action="store_true",
        help="Skip Wilson/coupling terms and score reflections using orientation uncertainty only",
    )
    parser.add_argument(
        "--orientation-sigma-deg",
        type=float,
        default=0.2,
        help="Isotropic orientation uncertainty in degrees",
    )
    parser.add_argument(
        "--orientation-sigma-axis-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Optional anisotropic orientation uncertainty in degrees around lab x/y/z axes",
    )
    parser.add_argument(
        "--orientation-sigma-alpha",
        type=float,
        default=0.5,
        help="Linear scaling factor for sigma_orient_scale = 1 + alpha * S_orient",
    )
    parser.add_argument(
        "--orientation-score-formulation",
        choices=["log_n_eff", "linear_n_eff"],
        default="log_n_eff",
        help="How to combine orientation excitation probability with coupling multiplicity",
    )
    parser.add_argument(
        "--dynamical-environment-tolerance",
        type=float,
        default=1.0e-2,
        help="Broad excitation window in 1/A for neighboring beams in geometry-only dynamical risk",
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
        default=3.0,
        help="Angular scale in degrees for low-index zone-axis ZOLZ boost",
    )
    parser.add_argument(
        "--dynamical-zone-sigma",
        type=float,
        default=0.06,
        help="Beam-parallel reciprocal-coordinate sigma in 1/A for same-Laue-zone cluster scoring",
    )
    parser.add_argument(
        "--dynamical-neighbor-sigma",
        type=float,
        default=None,
        help="Optional Gaussian sigma in 1/A for local neighbor window; default is half the neighbor radius",
    )
    parser.add_argument(
        "--dynamical-row-direction-limit",
        type=int,
        default=2,
        help="Low-order HKL direction limit for systematic-row cluster scoring",
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
    parser.add_argument("--dynamical-weight-self", type=float, default=1.0, help="Weight for self-extinction risk")
    parser.add_argument("--dynamical-weight-zone", type=float, default=1.0, help="Weight for same-zone cluster risk")
    parser.add_argument("--dynamical-weight-row", type=float, default=1.0, help="Weight for systematic-row cluster risk")
    parser.add_argument(
        "--dynamical-cluster-sigma-alpha",
        type=float,
        default=1.0,
        help="Cluster-only sigma scale: sigma_dyn_rel = 1 + alpha * cluster_risk_geom",
    )
    parser.add_argument(
        "--beam-direction",
        choices=["plus_z", "minus_z"],
        default=None,
        help="Beam direction used for excitation/projection (default: +z for XDS/stream, -z for PETS)",
    )
    return parser


def parse_thickness_arguments(args: argparse.Namespace) -> float | np.ndarray | None:
    if args.mode == "proxy":
        return None
    if args.thickness is not None:
        return float(args.thickness)
    if None not in (args.thickness_start, args.thickness_stop, args.thickness_step):
        stop_inclusive = float(args.thickness_stop) + 0.5 * float(args.thickness_step)
        return np.arange(float(args.thickness_start), stop_inclusive, float(args.thickness_step))
    return 100.0


def parse_analysis_frames(args: argparse.Namespace) -> list[int] | None:
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
        return None
    return sorted(set(frames))


def resolve_detector_frames(frame_summary, requested_frame_numbers, summary_column: str) -> list[tuple[int, int]]:
    if frame_summary is None or getattr(frame_summary, "empty", True):
        return []
    selected: list[tuple[int, int]] = []
    seen: set[int] = set()

    best_row = frame_summary.sort_values(summary_column, ascending=False).iloc[0]
    best_frame = int(best_row["frame"])
    best_frame_number = int(best_row.get("frame_number", best_frame + 1))
    selected.append((best_frame, best_frame_number))
    seen.add(best_frame)

    if not requested_frame_numbers:
        return selected

    indexed = frame_summary.copy()
    indexed["frame_number"] = indexed["frame_number"].astype(int)
    indexed = indexed.drop_duplicates("frame_number", keep="last").set_index("frame_number")
    for frame_number in requested_frame_numbers:
        if frame_number not in indexed.index:
            print(f"Skipping detector plot for missing frame {frame_number}.")
            continue
        frame = int(indexed.loc[frame_number, "frame"])
        if frame in seen:
            continue
        selected.append((frame, frame_number))
        seen.add(frame)
    return selected


def map_detector_point(
    x: float,
    y: float,
    detector_nx: int,
    detector_ny: int,
    *,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
) -> tuple[float, float]:
    if swap_xy:
        x, y = y, x
        detector_nx, detector_ny = detector_ny, detector_nx
    if flip_x:
        x = float(detector_nx - 1) - x
    if flip_y:
        y = float(detector_ny - 1) - y
    return float(x), float(y)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_count = sum(value is not None for value in (args.stream, args.gxparm, args.pets_project))
    if mode_count != 1:
        raise SystemExit("Provide exactly one of --stream, --pets-project, or --gxparm.")
    if args.stream is not None and any(v is not None for v in (args.integrate, args.xdsinp)):
        raise SystemExit("With --stream, do not provide --integrate or --xdsinp.")
    if args.stream is not None and args.pets_project is not None:
        raise SystemExit("With --stream, do not provide --pets-project.")
    if args.pets_project is not None and any(v is not None for v in (args.gxparm, args.integrate, args.xdsinp)):
        raise SystemExit("With --pets-project, do not provide --gxparm/--integrate/--xdsinp.")
    if args.gxparm is not None and args.integrate is None:
        raise SystemExit("With --gxparm, provide --integrate.")

    run_metadata = {
        "argv": list(sys.argv),
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, ensure_ascii=True)
    )

    orientation_model = None
    xds_input = None
    pets_mode = False
    pets_frame_geometry = None
    pets_model = None
    if args.stream is not None:
        stream_data = parse_crystfel_stream(args.stream)
        gxparm, integrate, reciprocal_by_frame = crystfel_stream_to_analysis_inputs(stream_data)
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
        print(f"Parsed {stream_data.crystal_table.shape[0]} indexed crystals from stream")
    elif args.pets_project is not None:
        pets_mode = True
        pets_model = load_pets_model(
            args.pets_project,
            pts2_path=args.pets_pts2,
            ptsopt_path=args.pets_ptsopt,
            rprofall_path=args.pets_rprofall,
            detector_nx=(None if int(args.pets_detector_nx) <= 0 else int(args.pets_detector_nx)),
            detector_ny=(None if int(args.pets_detector_ny) <= 0 else int(args.pets_detector_ny)),
        )
        alignment_rotation = None
        reindex_matrix = None
        if args.pets_align_gxparm is not None:
            align_gxparm = parse_gxparm(args.pets_align_gxparm)
            pets_reciprocal = (
                pets_model.ub_matrix.T
                if args.pets_ub_convention == "rows"
                else pets_model.ub_matrix
            )
            alignment = estimate_pets_alignment(pets_reciprocal, align_gxparm.reciprocal_reference)
            alignment_rotation = alignment.rotation_matrix
            reindex_matrix = alignment.reindex_matrix
            print(
                "PETS alignment to "
                f"{Path(args.pets_align_gxparm).resolve()} "
                f"residual={alignment.residual:.4e}\n"
                f"reindex=\n{alignment.reindex_matrix}"
            )
        gxparm, integrate, reciprocal_by_frame, pets_frame_geometry = pets_model_to_analysis_inputs(
            pets_model,
            ub_convention=args.pets_ub_convention,
            orientation_mode=args.pets_orientation_mode,
            angle_reference=args.pets_angle_reference,
            include_domega_in_lattice=bool(args.pets_include_domega_in_lattice),
            invert_rotation=bool(args.pets_invert_rotation),
            use_only_for_calc=bool(args.pets_use_only_for_calc),
            hkl_path=args.pets_hkl,
            alignment_rotation=alignment_rotation,
            reindex_matrix=reindex_matrix,
        )
        orientation_model = ReciprocalMatrixOrientationModel(
            reciprocal_by_frame=reciprocal_by_frame,
            reciprocal_reference=gxparm.reciprocal_reference,
        )
        print(
            "Parsed PETS project "
            f"{Path(args.pets_project).resolve()} with {len(pets_model.frames)} frames "
            f"and {integrate.observations.shape[0]} observations from {pets_model.rprofall_path.name}"
        )
        print(
            "PETS orientation model: "
            f"ub={args.pets_ub_convention}, mode={args.pets_orientation_mode}, "
            f"angle_ref={args.pets_angle_reference}, "
            f"include_domega={bool(args.pets_include_domega_in_lattice)}, "
            f"invert_rotation={bool(args.pets_invert_rotation)}"
        )
    else:
        gxparm = parse_gxparm(args.gxparm)
        integrate = parse_integrate_hkl(args.integrate)
        xds_input = load_optional_xds_inp(args.xdsinp)

    if args.orientation_gxparm is not None:
        orientation_model = RotationSeriesOrientationModel(parse_gxparm(args.orientation_gxparm))
        print(f"Using orientation model from: {args.orientation_gxparm}")

    composition = parse_composition(args.composition)
    thickness_nm = parse_thickness_arguments(args)
    selected_frames = parse_analysis_frames(args)
    if args.orientation_only and args.mode == "thickness":
        raise SystemExit("--orientation-only is only supported with --mode proxy.")
    if args.beam_direction is None:
        beam_direction = (0.0, 0.0, -1.0) if pets_mode else (0.0, 0.0, 1.0)
    else:
        beam_direction = (0.0, 0.0, -1.0) if args.beam_direction == "minus_z" else (0.0, 0.0, 1.0)

    config = AnalysisConfig(
        dmin_angstrom=args.dmin,
        dmax_angstrom=args.dmax,
        excitation_tolerance_invA=args.excitation_tolerance,
        mode=args.mode,
        thickness_nm=thickness_nm,
        filter_untrusted=args.filter_untrusted,
        orientation_only=bool(args.orientation_only),
        orientation_sigma_deg=(
            tuple(float(v) for v in args.orientation_sigma_axis_deg)
            if args.orientation_sigma_axis_deg is not None
            else float(args.orientation_sigma_deg)
        ),
        orientation_sigma_alpha=float(args.orientation_sigma_alpha),
        orientation_score_formulation=args.orientation_score_formulation,
        detector_xy_swapped=bool(args.detector_xy_swapped),
        orientation_beam_direction=beam_direction,
        dynamical_environment_tolerance_invA=float(args.dynamical_environment_tolerance),
        dynamical_neighbor_radius_invA=float(args.dynamical_neighbor_radius),
        dynamical_zone_axis_sigma_deg=float(args.dynamical_zone_axis_sigma),
        dynamical_zone_sigma_invA=float(args.dynamical_zone_sigma),
        dynamical_neighbor_sigma_invA=(
            None if args.dynamical_neighbor_sigma is None else float(args.dynamical_neighbor_sigma)
        ),
        dynamical_row_direction_limit=int(args.dynamical_row_direction_limit),
        dynamical_row_max_steps=int(args.dynamical_row_max_steps),
        dynamical_row_sigma_invA=float(args.dynamical_row_sigma),
        dynamical_coupling_q0_invA=float(args.dynamical_coupling_q0),
        dynamical_weight_self=float(args.dynamical_weight_self),
        dynamical_weight_zone=float(args.dynamical_weight_zone),
        dynamical_weight_row=float(args.dynamical_weight_row),
        dynamical_cluster_sigma_alpha=float(args.dynamical_cluster_sigma_alpha),
        stream_detector_shift_sign=float(args.stream_det_shift_sign),
        stream_mirror_x_axis=bool(args.stream_mirror_x_axis),
        frame_numbers=selected_frames,
        progress_every=int(args.progress_every),
    )

    print(f"Parsed {integrate.observations.shape[0]} observations")
    if config.orientation_only:
        print("Orientation-only mode: skipping Wilson scaling and coupling-derived scores.")
    else:
        calibration = wilson_calibrate(integrate.observations, composition.sum_fj2)
        print(f"Wilson-like scale factor K = {calibration.scale_factor:.5f}")
        print(f"Merged reflections = {calibration.merged_table.shape[0]}")

    result = run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        xds_input=xds_input,
        config=config,
        orientation_model=orientation_model,
    )

    result.frame_summary.to_csv(output_dir / "frame_summary.csv", index=False)
    result.reflections_long.to_csv(output_dir / "reflections_long.csv", index=False)
    if result.wilson is not None:
        result.wilson.merged_table.to_csv(output_dir / "wilson_merged.csv", index=False)
    if result.thickness_long is not None:
        result.thickness_long.to_csv(output_dir / "thickness_long.csv", index=False)
    if result.reflection_sensitivity is not None:
        result.reflection_sensitivity.to_csv(output_dir / "reflection_sensitivity.csv", index=False)

    if not result.reflections_long.empty and "cluster_risk_geom" in result.reflections_long.columns:
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
        ]
        compact_cols = [col for col in compact_cols if col in result.reflections_long.columns]
        result.reflections_long.sort_values("attenuation_risk", ascending=False).head(200).loc[:, compact_cols].to_csv(
            output_dir / "top_self_extinction_risk.csv",
            index=False,
        )
        result.reflections_long.sort_values("cluster_risk_geom", ascending=False).head(200).loc[:, compact_cols].to_csv(
            output_dir / "top_cluster_risk_geom.csv",
            index=False,
        )
        result.reflections_long.sort_values("cluster_risk_iw", ascending=False).head(200).loc[:, compact_cols].to_csv(
            output_dir / "top_cluster_risk_intensity_weighted.csv",
            index=False,
        )
        if "is_observed_target" in result.reflections_long.columns:
            observed_targets = result.reflections_long[result.reflections_long["is_observed_target"]].copy()
            if not observed_targets.empty:
                observed_targets.sort_values("cluster_risk_geom", ascending=False).head(200).loc[:, compact_cols].to_csv(
                    output_dir / "top_observed_cluster_risk_geom.csv",
                    index=False,
                )
                observed_targets.sort_values("attenuation_risk", ascending=False).head(200).loc[:, compact_cols].to_csv(
                    output_dir / "top_observed_self_extinction_risk.csv",
                    index=False,
                )
        n_observed_targets = (
            int(result.reflections_long["is_observed_target"].sum())
            if "is_observed_target" in result.reflections_long.columns
            else 0
        )
        summary_lines = [
            f"n_reflection_rows: {int(result.reflections_long.shape[0])}",
            f"n_observed_targets: {n_observed_targets}",
            f"mean_attenuation_risk: {float(result.reflections_long['attenuation_risk'].mean()):.6g}",
            f"p95_attenuation_risk: {float(result.reflections_long['attenuation_risk'].quantile(0.95)):.6g}",
            f"mean_cluster_risk_geom: {float(result.reflections_long['cluster_risk_geom'].mean()):.6g}",
            f"p95_cluster_risk_geom: {float(result.reflections_long['cluster_risk_geom'].quantile(0.95)):.6g}",
            f"mean_cluster_risk_iw: {float(result.reflections_long['cluster_risk_iw'].mean()):.6g}",
            f"p95_cluster_risk_iw: {float(result.reflections_long['cluster_risk_iw'].quantile(0.95)):.6g}",
            f"mean_sigma_dyn_rel: {float(result.reflections_long['sigma_dyn_rel'].mean()):.6g}",
        ]
        (output_dir / "two_channel_summary.txt").write_text("\n".join(summary_lines) + "\n")

    summary_column = "S_orient" if config.orientation_only else "S_MB"
    ax = plot_frame_summary(result.frame_summary, y=summary_column, title=f"{summary_column} across frames")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / f"frame_summary_{summary_column}.png", dpi=200)
    plt.close(ax.figure)

    for extra_summary_column in (
        "mean_cluster_risk_geom",
        "p95_cluster_risk_geom",
        "mean_attenuation_risk",
        "mean_sigma_dyn_rel",
    ):
        if extra_summary_column in result.frame_summary.columns and not result.frame_summary.empty:
            ax = plot_frame_summary(
                result.frame_summary,
                y=extra_summary_column,
                title=f"{extra_summary_column} across frames",
            )
            ax.figure.tight_layout()
            ax.figure.savefig(output_dir / f"frame_summary_{extra_summary_column}.png", dpi=200)
            plt.close(ax.figure)

    if not result.reflections_long.empty and {"attenuation_risk", "cluster_risk_geom"} <= set(result.reflections_long.columns):
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        ax.scatter(
            result.reflections_long["attenuation_risk"],
            result.reflections_long["cluster_risk_geom"],
            s=8,
            alpha=0.35,
            edgecolors="none",
        )
        ax.set_xlabel("attenuation_risk")
        ax.set_ylabel("cluster_risk_geom")
        ax.set_title("Self-extinction vs cluster-leakage risk")
        fig.tight_layout()
        fig.savefig(output_dir / "self_vs_cluster_risk.png", dpi=200)
        plt.close(fig)

    if not result.frame_summary.empty:
        detector_score_column = "S_orient" if config.orientation_only else "S_comb"
        detector_frames = resolve_detector_frames(result.frame_summary, args.detector_frames, summary_column)
        for detector_frame, detector_frame_number in detector_frames:
            center_x_px = None
            center_y_px = None
            if pets_mode and pets_frame_geometry is not None and not pets_frame_geometry.empty:
                row = pets_frame_geometry[pets_frame_geometry["frame"] == int(detector_frame)]
                if not row.empty:
                    info = row.iloc[-1]
                    center_x_px, center_y_px = map_detector_point(
                        x=float(info["xcenter"]),
                        y=float(info["ycenter"]),
                        detector_nx=int(gxparm.detector_nx),
                        detector_ny=int(gxparm.detector_ny),
                        swap_xy=bool(args.pets_detector_swap_xy),
                        flip_x=bool(args.pets_detector_flip_x),
                        flip_y=bool(args.pets_detector_flip_y),
                    )
                    print(
                        f"PETS frame {detector_frame_number}: "
                        f"alpha={float(info['alpha']):.6f}, beta={float(info['beta']):.6f}, "
                        f"domega={float(info['domega']):.6f}, "
                        f"xcenter={float(info['xcenter']):.3f}, ycenter={float(info['ycenter']):.3f}, "
                        f"plot_center=({center_x_px:.3f}, {center_y_px:.3f})"
                    )
            frame_table = result.frame_table(detector_frame)
            plot_table = frame_table
            if pets_mode and pets_model is not None:
                hkls = result.candidate_reflections[["h", "k", "l"]].to_numpy(dtype=int)
                beam_choice = "minus_z" if beam_direction[2] < 0.0 else "plus_z"
                geom_table = predict_pets_detector_spots(
                    pets_model,
                    frame_number=int(detector_frame_number),
                    hkls=hkls,
                    ub_convention=args.pets_ub_convention,
                    orientation_mode=args.pets_orientation_mode,
                    angle_reference=args.pets_angle_reference,
                    include_domega_in_lattice=bool(args.pets_include_domega_in_lattice),
                    invert_rotation=bool(args.pets_invert_rotation),
                    excitation_tolerance=float(args.excitation_tolerance),
                    projection=args.pets_detector_projection,
                    beam_direction=beam_choice,
                    omega_map_mode=args.pets_detector_omega_map_mode,
                    omega_sign=float(args.pets_detector_omega_sign),
                    omega_offset_deg=float(args.pets_detector_omega_offset_deg),
                    swap_xy=bool(args.pets_detector_swap_xy),
                    flip_x=bool(args.pets_detector_flip_x),
                    flip_y=bool(args.pets_detector_flip_y),
                    detector_nx=int(gxparm.detector_nx),
                    detector_ny=int(gxparm.detector_ny),
                    alignment_rotation=alignment_rotation,
                    reindex_matrix=reindex_matrix,
                )
                if not geom_table.empty:
                    score_lookup = (
                        frame_table[["h", "k", "l", detector_score_column]]
                        .drop_duplicates(subset=["h", "k", "l"], keep="last")
                    )
                    plot_table = geom_table.merge(score_lookup, on=["h", "k", "l"], how="left")
                    if plot_table[detector_score_column].isna().all():
                        if "sg_invA" in plot_table.columns:
                            tol = max(float(args.excitation_tolerance), 1.0e-9)
                            plot_table[detector_score_column] = np.exp(
                                -np.square(plot_table["sg_invA"].to_numpy(dtype=float) / tol)
                            )
                        else:
                            plot_table[detector_score_column] = 0.0
                    else:
                        if "sg_invA" in plot_table.columns:
                            tol = max(float(args.excitation_tolerance), 1.0e-9)
                            fallback = np.exp(-np.square(plot_table["sg_invA"].to_numpy(dtype=float) / tol))
                            values = plot_table[detector_score_column].to_numpy(dtype=float).copy()
                            missing = ~np.isfinite(values)
                            values[missing] = fallback[missing]
                            plot_table[detector_score_column] = values
                        else:
                            plot_table[detector_score_column] = plot_table[detector_score_column].fillna(0.0)
            score_vmin = 0.0 if (config.orientation_only and detector_score_column == "S_orient") else None
            score_vmax = 1.0 if (config.orientation_only and detector_score_column == "S_orient") else None
            ax = plot_detector_frame(
                plot_table,
                gxparm,
                rectangles=(
                    xds_input.untrusted_rectangles
                    if (bool(args.show_untrusted_rectangles) and xds_input is not None)
                    else None
                ),
                score_column=detector_score_column,
                score_vmin=score_vmin,
                score_vmax=score_vmax,
                title=f"Detector plot for frame {detector_frame_number}",
                center_x_px=center_x_px,
                center_y_px=center_y_px,
            )
            ax.figure.tight_layout()
            ax.figure.savefig(output_dir / f"detector_frame_{detector_frame_number:04d}.png", dpi=200)
            plt.close(ax.figure)
            for extra_score_column in ("cluster_risk_geom", "attenuation_risk", "cluster_risk_iw"):
                if extra_score_column not in plot_table.columns:
                    continue
                ax = plot_detector_frame(
                    plot_table,
                    gxparm,
                    rectangles=(
                        xds_input.untrusted_rectangles
                        if (bool(args.show_untrusted_rectangles) and xds_input is not None)
                        else None
                    ),
                    score_column=extra_score_column,
                    score_vmin=0.0,
                    score_vmax=1.0,
                    title=f"{extra_score_column} for frame {detector_frame_number}",
                    center_x_px=center_x_px,
                    center_y_px=center_y_px,
                )
                ax.figure.tight_layout()
                ax.figure.savefig(output_dir / f"detector_frame_{detector_frame_number:04d}_{extra_score_column}.png", dpi=200)
                plt.close(ax.figure)

        if result.thickness_long is not None and not result.thickness_long.empty:
            best_frame, best_frame_number = detector_frames[0]
            frame_table = result.frame_table(best_frame)
            top_reflections = (
                frame_table.sort_values("S_comb", ascending=False)[["h", "k", "l"]]
                .head(5)
                .itertuples(index=False, name=None)
            )
            selected = [tuple(map(int, hkl)) for hkl in top_reflections]
            if selected:
                ax = plot_thickness_scan(
                    result.thickness_long,
                    frame=best_frame,
                    reflections=selected,
                    metric="intensity",
                    title=f"Thickness scan at frame {best_frame_number}",
                )
                ax.figure.tight_layout()
                ax.figure.savefig(output_dir / f"thickness_scan_frame_{best_frame_number:04d}.png", dpi=200)
                plt.close(ax.figure)

    print(f"Wrote results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

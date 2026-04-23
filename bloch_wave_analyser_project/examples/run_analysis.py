"""Command-line entry point for the Bloch-wave dynamicality analyser."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers import (
    crystfel_stream_to_analysis_inputs,
    load_optional_xds_inp,
    parse_composition,
    parse_crystfel_stream,
    parse_gxparm,
    parse_integrate_hkl,
    parse_pets_project,
    parse_rprofall,
    pets_project_to_analysis_inputs,
    rprofall_to_integrate_data,
)
from src.geometry import ReciprocalMatrixOrientationModel, RotationSeriesOrientationModel
from src.pipeline import AnalysisConfig, run_analysis
from src.visualization import plot_detector_frame, plot_frame_summary, plot_thickness_scan
from src.wilson import wilson_calibrate


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default=None, help="Optional path to CrystFEL .stream snapshot indexing output")
    parser.add_argument(
        "--pets-project",
        default=None,
        help="Optional PETS project folder or .pts2(.backup) file (PETS-only mode)",
    )
    parser.add_argument(
        "--pets-rprofall",
        default=None,
        help="Optional PETS .rprofall override path (used only with --pets-project)",
    )
    parser.add_argument("--gxparm", default=None, help="Path to GXPARM.XDS or XPARM.XDS")
    parser.add_argument("--integrate", default=None, help="Path to INTEGRATE.HKL")
    parser.add_argument("--rprofall", default=None, help="Optional PETS2 .rprofall path (alternative to --integrate)")
    parser.add_argument("--xdsinp", default=None, help="Optional path to XDS.INP")
    parser.add_argument(
        "--orientation-gxparm",
        default=None,
        help="Optional GXPARM/XPARM used only for frame orientation progression (can override PETS orientation model)",
    )
    parser.add_argument(
        "--geometry-gxparm",
        default=None,
        help=(
            "Optional GXPARM/XPARM used for detector/cell/reference geometry "
            "(useful with PETS observations when frame geometry is known from XDS)"
        ),
    )
    parser.add_argument("--composition", required=True, help='Composition string, e.g. "24 Si, 48 O"')
    parser.add_argument("--dmin", type=float, default=0.6, help="Minimum d-spacing in angstrom")
    parser.add_argument("--dmax", type=float, default=50.0, help="Maximum d-spacing in angstrom")
    parser.add_argument(
        "--excitation-tolerance",
        type=float,
        default=1.5e-3,
        help="Excitation tolerance in inverse angstrom",
    )
    parser.add_argument(
        "--mode",
        choices=["proxy", "thickness"],
        default="proxy",
        help="Analysis mode",
    )
    parser.add_argument("--thickness", type=float, default=None, help="Single thickness in nm")
    parser.add_argument("--thickness-start", type=float, default=None, help="Thickness scan start in nm")
    parser.add_argument("--thickness-stop", type=float, default=None, help="Thickness scan stop in nm")
    parser.add_argument("--thickness-step", type=float, default=None, help="Thickness scan step in nm")
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Directory for CSV and plot outputs",
    )
    parser.add_argument(
        "--filter-untrusted",
        action="store_true",
        help="Exclude reflections whose detector coordinates fall in XDS untrusted rectangles",
    )
    parser.add_argument(
        "--detector-xy-swapped",
        action="store_true",
        help="Swap detector x/y after projection (opt-in for datasets with transposed detector axes)",
    )
    parser.add_argument(
        "--detector-frame",
        dest="detector_frames",
        action="append",
        type=int,
        default=None,
        help="One-based frame number to export as a detector plot (repeatable)",
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
        help="Isotropic orientation uncertainty in degrees for per-reflection orientation sigma",
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
    return parser


def parse_thickness_arguments(args: argparse.Namespace) -> float | np.ndarray | None:
    """Resolve thickness CLI arguments."""

    if args.mode == "proxy":
        return None
    if args.thickness is not None:
        return float(args.thickness)
    if None not in (args.thickness_start, args.thickness_stop, args.thickness_step):
        stop_inclusive = float(args.thickness_stop) + 0.5 * float(args.thickness_step)
        return np.arange(float(args.thickness_start), stop_inclusive, float(args.thickness_step))
    return 100.0


def resolve_detector_frames(
    frame_summary: np.ndarray | list[dict[str, object]] | object,
    requested_frame_numbers: list[int] | None,
    summary_column: str,
) -> list[tuple[int, int]]:
    """Resolve detector frames to export as ``(frame, frame_number)`` pairs."""

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


def pets_detector_table(
    pets_data: object,
    frame_number: int,
) -> pd.DataFrame:
    """Build a detector scatter table from PETS ``.dyntmp`` coordinates."""

    dyntmp = getattr(pets_data, "dyntmp", None)
    if dyntmp is None or dyntmp.empty:
        return pd.DataFrame()

    frame_rows = dyntmp[pd.to_numeric(dyntmp["frame"], errors="coerce") == float(frame_number)].copy()
    if frame_rows.empty:
        return pd.DataFrame()

    x_series = pd.to_numeric(frame_rows["xobs"], errors="coerce")
    y_series = pd.to_numeric(frame_rows["yobs"], errors="coerce")
    if not (x_series.notna().any() and y_series.notna().any()):
        x_series = pd.to_numeric(frame_rows["xcalc"], errors="coerce")
        y_series = pd.to_numeric(frame_rows["ycalc"], errors="coerce")
    else:
        missing_mask = x_series.isna() | y_series.isna()
        if missing_mask.any():
            x_series.loc[missing_mask] = pd.to_numeric(frame_rows.loc[missing_mask, "xcalc"], errors="coerce")
            y_series.loc[missing_mask] = pd.to_numeric(frame_rows.loc[missing_mask, "ycalc"], errors="coerce")

    table = pd.DataFrame(
        {
            "x_px": x_series,
            "y_px": y_series,
            "spot_weight": np.maximum(pd.to_numeric(frame_rows["iobs"], errors="coerce").fillna(0.0), 0.0),
        }
    ).dropna(subset=["x_px", "y_px"])
    return table.reset_index(drop=True)


def main() -> None:
    """Run the CLI workflow."""

    args = build_parser().parse_args()

    orientation_override_gxparm = None
    if args.orientation_gxparm is not None:
        orientation_override_gxparm = parse_gxparm(args.orientation_gxparm)

    geometry_override_gxparm = None
    if args.geometry_gxparm is not None:
        geometry_override_gxparm = parse_gxparm(args.geometry_gxparm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orientation_model = None
    reciprocal_by_frame: dict[int, np.ndarray] | None = None
    xds_input = None
    pets_data = None
    if args.stream is not None:
        if any(
            value is not None
            for value in (
                args.pets_project,
                args.pets_rprofall,
                args.gxparm,
                args.integrate,
                args.rprofall,
                args.xdsinp,
            )
        ):
            raise SystemExit(
                "When --stream is provided, do not also provide --pets-project/--pets-rprofall/"
                "--gxparm/--integrate/--rprofall/--xdsinp."
            )
        stream_data = parse_crystfel_stream(args.stream)
        gxparm, integrate, reciprocal_by_frame = crystfel_stream_to_analysis_inputs(stream_data)
        orientation_model = ReciprocalMatrixOrientationModel(
            reciprocal_by_frame=reciprocal_by_frame,
            reciprocal_reference=gxparm.reciprocal_reference,
        )
        print(f"Parsed {stream_data.crystal_table.shape[0]} indexed crystals from stream")
    elif args.pets_project is not None:
        if any(value is not None for value in (args.gxparm, args.integrate, args.rprofall, args.xdsinp)):
            raise SystemExit(
                "When --pets-project is provided, do not also provide --gxparm/--integrate/--rprofall/--xdsinp."
            )
        pets_data = parse_pets_project(args.pets_project, rprofall_path=args.pets_rprofall)
        gxparm, integrate, reciprocal_by_frame = pets_project_to_analysis_inputs(pets_data)
        orientation_model = ReciprocalMatrixOrientationModel(
            reciprocal_by_frame=reciprocal_by_frame,
            reciprocal_reference=gxparm.reciprocal_reference,
        )
        print(
            "Parsed PETS project "
            f"{pets_data.pts_path.name} with {pets_data.imagelist.shape[0]} frames "
            f"and {pets_data.rprofall.rows.shape[0]} .rprofall rows"
        )
        for note in pets_data.metadata_notes:
            print(f"PETS note: {note}")
    else:
        if args.pets_rprofall is not None:
            raise SystemExit("--pets-rprofall can only be used together with --pets-project.")
        if args.gxparm is None:
            raise SystemExit("Provide --gxparm (or use --stream or --pets-project).")
        if bool(args.integrate) == bool(args.rprofall):
            raise SystemExit("Provide exactly one of --integrate or --rprofall.")
        gxparm = parse_gxparm(args.gxparm)
        if args.integrate is not None:
            integrate = parse_integrate_hkl(args.integrate)
        else:
            rprofall = parse_rprofall(args.rprofall)
            integrate = rprofall_to_integrate_data(rprofall)
        xds_input = load_optional_xds_inp(args.xdsinp)

    if geometry_override_gxparm is not None:
        gxparm = geometry_override_gxparm
        print(f"Using geometry from: {args.geometry_gxparm}")
        if args.pets_project is not None and orientation_override_gxparm is None:
            orientation_model = RotationSeriesOrientationModel(geometry_override_gxparm)
            print("Using orientation model from geometry override for PETS input.")

    if orientation_override_gxparm is not None:
        orientation_model = RotationSeriesOrientationModel(orientation_override_gxparm)
        print(f"Using orientation model from: {args.orientation_gxparm}")
    composition = parse_composition(args.composition)

    thickness_nm = parse_thickness_arguments(args)
    if args.orientation_only and args.mode == "thickness":
        raise SystemExit("--orientation-only is only supported with --mode proxy.")
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

    # Summary plot across frames.
    summary_column = "S_orient" if config.orientation_only else "S_MB"
    ax = plot_frame_summary(
        result.frame_summary,
        y=summary_column,
        title=f"{summary_column} across frames",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / f"frame_summary_{summary_column}.png", dpi=200)
    plt.close(ax.figure)

    if not result.frame_summary.empty:
        detector_score_column = "S_orient" if config.orientation_only else "S_comb"
        detector_frames = resolve_detector_frames(result.frame_summary, args.detector_frames, summary_column)
        for detector_frame, detector_frame_number in detector_frames:
            if pets_data is not None:
                details: list[str] = []
                frame_geometry = pets_data.frame_geometry
                if not frame_geometry.empty and "frame" in frame_geometry.columns:
                    geom_row = frame_geometry[frame_geometry["frame"].astype(int) == int(detector_frame)]
                    if not geom_row.empty:
                        row = geom_row.iloc[-1]
                        for col in ("alpha", "beta", "domega", "xcenter", "ycenter"):
                            if col in row.index and pd.notna(row[col]):
                                details.append(f"{col}={float(row[col]):.6f}")
                if reciprocal_by_frame is not None and detector_frame in reciprocal_by_frame:
                    details.append(f"detUB={float(np.linalg.det(reciprocal_by_frame[detector_frame])):.6e}")
                if details:
                    print(f"PETS frame {detector_frame_number}: " + ", ".join(details))

            frame_table = result.frame_table(detector_frame)
            output_name = output_dir / f"detector_frame_{detector_frame_number:04d}.png"
            native_pets_table = pets_detector_table(pets_data, detector_frame_number) if pets_data is not None else pd.DataFrame()
            plot_table = frame_table
            plot_score_column = detector_score_column
            plot_title = f"Detector plot for frame {detector_frame_number}"
            if not native_pets_table.empty:
                plot_table = native_pets_table
                plot_score_column = "spot_weight"
                plot_title = f"PETS detector plot for frame {detector_frame_number}"
                predicted_name = output_dir / f"detector_frame_{detector_frame_number:04d}_predicted.png"
                ax = plot_detector_frame(
                    frame_table,
                    gxparm,
                    rectangles=(xds_input.untrusted_rectangles if xds_input is not None else None),
                    score_column=detector_score_column,
                    title=f"Predicted detector plot for frame {detector_frame_number}",
                )
                ax.figure.tight_layout()
                ax.figure.savefig(predicted_name, dpi=200)
                plt.close(ax.figure)

            ax = plot_detector_frame(
                plot_table,
                gxparm,
                rectangles=(xds_input.untrusted_rectangles if xds_input is not None else None),
                score_column=plot_score_column,
                title=plot_title,
            )
            ax.figure.tight_layout()
            ax.figure.savefig(output_name, dpi=200)
            plt.close(ax.figure)

        if result.thickness_long is not None and not result.thickness_long.empty:
            best_frame = detector_frames[0][0]
            best_frame_number = detector_frames[0][1]
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

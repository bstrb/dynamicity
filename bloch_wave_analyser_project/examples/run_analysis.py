"""Command-line entry point for the Bloch-wave dynamicality analyser."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

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
from src.geometry import ReciprocalMatrixOrientationModel
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


def main() -> None:
    """Run the CLI workflow."""

    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orientation_model = None
    xds_input = None
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
        best_frame = int(result.frame_summary.sort_values(summary_column, ascending=False).iloc[0]["frame"])
        frame_table = result.frame_table(best_frame)
        detector_score_column = "S_orient" if config.orientation_only else "S_comb"
        ax = plot_detector_frame(
            frame_table,
            gxparm,
            rectangles=(xds_input.untrusted_rectangles if xds_input is not None else None),
            score_column=detector_score_column,
            title=f"Detector plot for frame {best_frame + 1}",
        )
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / f"detector_frame_{best_frame + 1:04d}.png", dpi=200)
        plt.close(ax.figure)

        if result.thickness_long is not None and not result.thickness_long.empty:
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
                    title=f"Thickness scan at frame {best_frame + 1}",
                )
                ax.figure.tight_layout()
                ax.figure.savefig(output_dir / f"thickness_scan_frame_{best_frame + 1:04d}.png", dpi=200)
                plt.close(ax.figure)

    print(f"Wrote results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

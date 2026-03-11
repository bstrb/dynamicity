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

from src.parsers import parse_composition, parse_gxparm, parse_integrate_hkl, load_optional_xds_inp
from src.pipeline import AnalysisConfig, run_analysis
from src.visualization import plot_detector_frame, plot_frame_summary, plot_thickness_scan
from src.wilson import wilson_calibrate


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gxparm", required=True, help="Path to GXPARM.XDS or XPARM.XDS")
    parser.add_argument("--integrate", required=True, help="Path to INTEGRATE.HKL")
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

    gxparm = parse_gxparm(args.gxparm)
    integrate = parse_integrate_hkl(args.integrate)
    xds_input = load_optional_xds_inp(args.xdsinp)
    composition = parse_composition(args.composition)

    thickness_nm = parse_thickness_arguments(args)
    config = AnalysisConfig(
        dmin_angstrom=args.dmin,
        dmax_angstrom=args.dmax,
        excitation_tolerance_invA=args.excitation_tolerance,
        mode=args.mode,
        thickness_nm=thickness_nm,
        filter_untrusted=args.filter_untrusted,
    )

    calibration = wilson_calibrate(integrate.observations, composition.sum_fj2)
    print(f"Parsed {integrate.observations.shape[0]} observations")
    print(f"Wilson-like scale factor K = {calibration.scale_factor:.5f}")
    print(f"Merged reflections = {calibration.merged_table.shape[0]}")

    result = run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        xds_input=xds_input,
        config=config,
    )

    result.frame_summary.to_csv(output_dir / "frame_summary.csv", index=False)
    result.reflections_long.to_csv(output_dir / "reflections_long.csv", index=False)
    result.wilson.merged_table.to_csv(output_dir / "wilson_merged.csv", index=False)
    if result.thickness_long is not None:
        result.thickness_long.to_csv(output_dir / "thickness_long.csv", index=False)
    if result.reflection_sensitivity is not None:
        result.reflection_sensitivity.to_csv(output_dir / "reflection_sensitivity.csv", index=False)

    # Summary plot across frames.
    ax = plot_frame_summary(result.frame_summary, y="S_MB", title="S_MB across frames")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "frame_summary_S_MB.png", dpi=200)
    plt.close(ax.figure)

    if not result.frame_summary.empty:
        best_frame = int(result.frame_summary.sort_values("S_MB", ascending=False).iloc[0]["frame"])
        frame_table = result.frame_table(best_frame)
        ax = plot_detector_frame(
            frame_table,
            gxparm,
            rectangles=(xds_input.untrusted_rectangles if xds_input is not None else None),
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

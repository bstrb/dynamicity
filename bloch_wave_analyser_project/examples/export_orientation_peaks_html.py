"""Export an interactive HTML visualization of orientation-dependent dynamical peaks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.html_visualization import export_orientation_peaks_html
from src.parsers import load_optional_xds_inp, parse_composition, parse_gxparm, parse_integrate_hkl
from src.pipeline import AnalysisConfig, run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gxparm", default="data/GXPARM.XDS", help="Path to GXPARM.XDS")
    parser.add_argument("--integrate", default="data/INTEGRATE.HKL", help="Path to INTEGRATE.HKL")
    parser.add_argument("--xdsinp", default="data/XDS.INP", help="Optional path to XDS.INP")
    parser.add_argument("--composition", default="24 Si, 48 O", help="Composition string")

    parser.add_argument("--mode", choices=["proxy", "thickness"], default="proxy", help="Pipeline mode")
    parser.add_argument("--thickness", type=float, default=100.0, help="Thickness in nm for thickness mode")

    parser.add_argument("--dmin", type=float, default=0.6, help="Minimum d-spacing (angstrom)")
    parser.add_argument("--dmax", type=float, default=50.0, help="Maximum d-spacing (angstrom)")
    parser.add_argument(
        "--excitation-tolerance",
        type=float,
        default=1.5e-3,
        help="Excitation tolerance (1/angstrom)",
    )
    parser.add_argument("--filter-untrusted", action="store_true", help="Exclude untrusted detector regions")

    parser.add_argument("--score-column", default="S_comb", help="Reflection score column for coloring/sizing")
    parser.add_argument("--frame-metric", default="S_MB", help="Frame summary metric on the left panel")
    parser.add_argument("--max-points-per-frame", type=int, default=700, help="Maximum reflections shown per frame")
    parser.add_argument("--surface-grid-size", type=int, default=130, help="Grid resolution for 3D gaussian surface")
    parser.add_argument("--min-sigma-px", type=float, default=8.0, help="Minimum gaussian width in pixels")
    parser.add_argument("--max-sigma-px", type=float, default=40.0, help="Maximum gaussian width in pixels")
    parser.add_argument(
        "--overlap-mode",
        choices=["max", "sum"],
        default="max",
        help="How overlapping gaussians are combined in the 3D surface",
    )
    parser.add_argument("--surface-opacity", type=float, default=0.78, help="Opacity of 3D gaussian surface")
    parser.add_argument(
        "--color-locality-gamma",
        type=float,
        default=1.8,
        help="Higher values concentrate color near peak tops",
    )
    parser.add_argument(
        "--tail-clip-fraction",
        type=float,
        default=0.03,
        help="Hide tails below this fraction of max surface height",
    )
    parser.add_argument(
        "--output-html",
        default="analysis_output/orientation_dynamical_peaks.html",
        help="Output HTML file path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    gxparm_path = Path(args.gxparm)
    integrate_path = Path(args.integrate)
    xdsinp_path = Path(args.xdsinp) if args.xdsinp else None

    gxparm = parse_gxparm(gxparm_path)
    integrate = parse_integrate_hkl(integrate_path)
    xds_input = load_optional_xds_inp(xdsinp_path if xdsinp_path is not None and xdsinp_path.exists() else None)
    composition = parse_composition(args.composition)

    thickness_nm = args.thickness if args.mode == "thickness" else None
    config = AnalysisConfig(
        dmin_angstrom=args.dmin,
        dmax_angstrom=args.dmax,
        excitation_tolerance_invA=args.excitation_tolerance,
        mode=args.mode,
        thickness_nm=thickness_nm,
        filter_untrusted=args.filter_untrusted,
    )

    result = run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        xds_input=xds_input,
        config=config,
    )

    output_path = export_orientation_peaks_html(
        result,
        output_html=args.output_html,
        score_column=args.score_column,
        frame_metric=args.frame_metric,
        max_points_per_frame=args.max_points_per_frame,
        surface_grid_size=args.surface_grid_size,
        min_sigma_px=args.min_sigma_px,
        max_sigma_px=args.max_sigma_px,
        overlap_mode=args.overlap_mode,
        surface_opacity=args.surface_opacity,
        color_locality_gamma=args.color_locality_gamma,
        tail_clip_fraction=args.tail_clip_fraction,
    )

    print(f"Wrote interactive report to: {output_path}")


if __name__ == "__main__":
    main()

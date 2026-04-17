"""High-level pipeline and command-line interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .gaussian_fit import GaussianFitConfig
from .geometry import choose_rotation_sign_from_integrate, detector_geometry_from_xds
from .image_io import ImageResolver
from .parsers import load_optional_xds_inp, parse_gxparm, parse_integrate_hkl, parse_spot_xds
from .plotting import plot_detector_track, plot_rocking_curve
from .prediction import RelevanceConfig
from .rocking_curve import RockingCurveConfig, build_rocking_curve, save_curve_outputs


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for analyzing one reflection in one dataset."""

    dataset_name: str
    thickness_nm: float | None
    hkl: tuple[int, int, int]
    relevance_mode: str = "sg"
    sg_threshold: float = 0.02
    window_half_width: int = 3
    patch_half_size: int = 7
    max_center_shift_px: float = 3.0
    initial_sigma_px: float = 1.5
    min_sigma_px: float = 0.5
    max_sigma_px: float = 6.0
    auto_choose_rotation_sign: bool = True
    rotation_sign: float = 1.0


@dataclass(frozen=True)
class AnalysisResults:
    """Returned tables and metadata."""

    curve: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict[str, object]


def analyze_single_reflection_dataset(
    gxparm_path: str | Path,
    image_glob: str | None,
    image_template: str | None,
    config: AnalysisConfig,
    output_dir: str | Path,
    xds_inp_path: str | Path | None = None,
    spot_xds_path: str | Path | None = None,
    integrate_hkl_path: str | Path | None = None,
) -> AnalysisResults:
    """Run the single-reflection rocking-curve pipeline."""

    gxparm = parse_gxparm(gxparm_path)
    xds_inp = load_optional_xds_inp(xds_inp_path)
    detector = detector_geometry_from_xds(gxparm, xds_inp)

    integrate_data = parse_integrate_hkl(integrate_hkl_path) if integrate_hkl_path is not None else None
    if config.auto_choose_rotation_sign and integrate_data is not None:
        rotation_calibration = choose_rotation_sign_from_integrate(
            gxparm=gxparm,
            detector=detector,
            integrate_observations=integrate_data.observations,
        )
        rotation_sign = rotation_calibration.rotation_sign
    else:
        rotation_calibration = None
        rotation_sign = config.rotation_sign

    if image_glob is None and image_template is None and xds_inp is not None:
        image_template = xds_inp.name_template
    image_resolver = ImageResolver(
        image_glob=image_glob,
        image_template=image_template,
        starting_frame=gxparm.starting_frame,
        frame_range=None if xds_inp is None else xds_inp.data_range,
    )

    spot_data = parse_spot_xds(spot_xds_path) if spot_xds_path is not None else None
    curve_cfg = RockingCurveConfig(
        patch_half_size=config.patch_half_size,
        relevance=RelevanceConfig(
            mode=config.relevance_mode,
            sg_threshold=config.sg_threshold,
            window_half_width=config.window_half_width,
        ),
        gaussian_fit=GaussianFitConfig(
            max_center_shift_px=config.max_center_shift_px,
            initial_sigma_px=config.initial_sigma_px,
            min_sigma_px=config.min_sigma_px,
            max_sigma_px=config.max_sigma_px,
        ),
    )
    curve, predictions = build_rocking_curve(
        dataset_name=config.dataset_name,
        thickness_nm=config.thickness_nm,
        gxparm=gxparm,
        detector=detector,
        image_resolver=image_resolver,
        hkl=config.hkl,
        config=curve_cfg,
        rotation_sign=rotation_sign,
        spot_data=spot_data,
    )

    outputs = save_curve_outputs(curve, predictions, output_dir)
    plot_rocking_curve(curve, Path(output_dir) / "rocking_curve.png", normalized=False)
    plot_rocking_curve(curve, Path(output_dir) / "rocking_curve_normalized.png", normalized=True)
    plot_detector_track(predictions, curve, Path(output_dir) / "detector_track.png")

    metadata = {
        "dataset_name": config.dataset_name,
        "thickness_nm": config.thickness_nm,
        "hkl": list(config.hkl),
        "gxparm_path": str(gxparm_path),
        "xds_inp_path": str(xds_inp_path) if xds_inp_path is not None else None,
        "spot_xds_path": str(spot_xds_path) if spot_xds_path is not None else None,
        "integrate_hkl_path": str(integrate_hkl_path) if integrate_hkl_path is not None else None,
        "image_glob": image_glob,
        "image_template": image_template,
        "rotation_sign": rotation_sign,
        "rotation_sign_median_pixel_error": None if rotation_calibration is None else rotation_calibration.median_pixel_error,
        "n_available_frames": len(image_resolver.available_frames),
        "n_relevant_frames": int(predictions["is_relevant"].sum()),
        "n_successful_fits": int(curve["fit_success"].sum()) if not curve.empty else 0,
        "config": asdict(config),
        "output_files": {key: str(value) for key, value in outputs.items()},
    }
    metadata_path = Path(output_dir) / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    metadata["output_files"]["metadata"] = str(metadata_path)
    return AnalysisResults(curve=curve, predictions=predictions, metadata=metadata)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface parser."""

    parser = argparse.ArgumentParser(description="Reconstruct a local rocking curve from TIFF images and XDS geometry")
    parser.add_argument("--gxparm", required=True, help="Path to GXPARM.XDS or XPARM.XDS")
    parser.add_argument("--xds-inp", default=None, help="Optional path to XDS.INP")
    parser.add_argument("--spot-xds", default=None, help="Optional path to SPOT.XDS used only for validation")
    parser.add_argument("--integrate-hkl", default=None, help="Optional path to INTEGRATE.HKL for rotation-sign validation")
    parser.add_argument("--image-glob", default=None, help="Glob matching TIFF image files in frame order")
    parser.add_argument("--image-template", default=None, help="Optional XDS-style template containing ? placeholders")
    parser.add_argument("--dataset-name", default="dataset", help="Dataset identifier written to outputs")
    parser.add_argument("--thickness-nm", type=float, default=None, help="Thickness label stored in outputs")
    parser.add_argument("--hkl", nargs=3, type=int, required=True, metavar=("H", "K", "L"), help="Target reflection")
    parser.add_argument("--relevance-mode", choices=["sg", "window"], default="sg")
    parser.add_argument("--sg-threshold", type=float, default=0.02, help="Maximum |sg| for relevance when mode=sg")
    parser.add_argument("--window-half-width", type=int, default=3, help="Half width around the predicted crossing frame when mode=window")
    parser.add_argument("--patch-half-size", type=int, default=7, help="Half-size of the extracted square patch")
    parser.add_argument("--max-center-shift-px", type=float, default=3.0, help="Maximum fit center refinement from the predicted center")
    parser.add_argument("--initial-sigma-px", type=float, default=1.5)
    parser.add_argument("--min-sigma-px", type=float, default=0.5)
    parser.add_argument("--max-sigma-px", type=float, default=6.0)
    parser.add_argument("--rotation-sign", type=float, default=1.0, choices=[-1.0, 1.0], help="Manual rotation sign if auto validation is disabled")
    parser.add_argument("--no-auto-rotation-sign", action="store_true", help="Disable automatic rotation-sign selection from INTEGRATE.HKL")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser


def run_from_cli(argv: list[str] | None = None) -> AnalysisResults:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = AnalysisConfig(
        dataset_name=args.dataset_name,
        thickness_nm=args.thickness_nm,
        hkl=(args.hkl[0], args.hkl[1], args.hkl[2]),
        relevance_mode=args.relevance_mode,
        sg_threshold=args.sg_threshold,
        window_half_width=args.window_half_width,
        patch_half_size=args.patch_half_size,
        max_center_shift_px=args.max_center_shift_px,
        initial_sigma_px=args.initial_sigma_px,
        min_sigma_px=args.min_sigma_px,
        max_sigma_px=args.max_sigma_px,
        auto_choose_rotation_sign=not args.no_auto_rotation_sign,
        rotation_sign=args.rotation_sign,
    )
    results = analyze_single_reflection_dataset(
        gxparm_path=args.gxparm,
        image_glob=args.image_glob,
        image_template=args.image_template,
        config=cfg,
        output_dir=args.output,
        xds_inp_path=args.xds_inp,
        spot_xds_path=args.spot_xds,
        integrate_hkl_path=args.integrate_hkl,
    )
    print(f"Wrote outputs to: {args.output}")
    print(f"Relevant frames: {results.metadata['n_relevant_frames']}")
    print(f"Successful fits: {results.metadata['n_successful_fits']}")
    return results


if __name__ == "__main__":
    run_from_cli()

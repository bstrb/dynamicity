"""CLI for observation-level dynamical uncertainty screening."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamical_uncertainty import (
    DynamicalUncertaintyConfig,
    run_dynamical_uncertainty_from_stream,
    run_dynamical_uncertainty_from_xds,
)


def _parse_axes(text: str) -> tuple[str, ...]:
    normalized = text.replace(",", "").replace(" ", "").lower()
    if not normalized:
        raise ValueError("orientation axes string is empty.")
    axes = tuple(ch for ch in normalized if ch in {"x", "y", "z"})
    if not axes:
        raise ValueError("Could not parse orientation axes from input.")
    # Preserve order but drop duplicates.
    seen: set[str] = set()
    deduped: list[str] = []
    for axis in axes:
        if axis not in seen:
            seen.add(axis)
            deduped.append(axis)
    return tuple(deduped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default=None, help="Path to CrystFEL .stream input")
    parser.add_argument("--gxparm", default=None, help="Path to GXPARM/XPARM file for XDS-style input")
    parser.add_argument("--integrate", default=None, help="Path to INTEGRATE.HKL for XDS-style input")
    parser.add_argument("--dataset-id", default=None, help="Optional dataset id in output tables")

    parser.add_argument("--orientation-axes", default="xyz", help="Orientation perturbation axes, e.g. xyz or xy")
    parser.add_argument("--orientation-step-deg", type=float, default=0.05, help="Angular step (degrees)")
    parser.add_argument("--orientation-n-steps", type=int, default=1, help="Number of ± steps per axis")
    parser.add_argument("--thickness-min-nm", type=float, default=20.0, help="Minimum thickness (nm)")
    parser.add_argument("--thickness-max-nm", type=float, default=300.0, help="Maximum thickness (nm)")
    parser.add_argument("--n-thickness-steps", type=int, default=15, help="Number of thickness grid points")
    parser.add_argument(
        "--orientation-thickness-ref-nm",
        type=float,
        default=None,
        help="Optional thickness (nm) for orientation-neighborhood screening; default midpoint",
    )
    parser.add_argument("--fg-q0-invA", type=float, default=0.25, help="q0 in Fg proxy")
    parser.add_argument("--fg-scale", type=float, default=1.0, help="Scale factor in Fg proxy")
    parser.add_argument(
        "--zone-axis-layer-band-invA",
        type=float,
        default=0.06,
        help="Gaussian bandwidth in g_z for Laue-layer crowding density",
    )
    parser.add_argument(
        "--zone-axis-zolz-relative-width",
        type=float,
        default=0.08,
        help="Relative width for ZOLZ proximity exp(-( |g_z| / (w*q) )^2)",
    )
    parser.add_argument(
        "--zone-axis-boost-weight",
        type=float,
        default=4.0,
        help="Weight for zone-axis/layer-crowding boost in orientation risk",
    )
    parser.add_argument("--risk-weight-orientation", type=float, default=1.0, help="Weight for orientation risk")
    parser.add_argument("--risk-weight-thickness", type=float, default=1.0, help="Weight for thickness risk")
    parser.add_argument(
        "--risk-normalization-quantile",
        type=float,
        default=0.95,
        help="Quantile used to normalize risk_total",
    )
    parser.add_argument(
        "--dyn-sigma-form",
        choices=["linear", "exp"],
        default="linear",
        help="Mapping from normalized risk to dyn_sigma_rel",
    )
    parser.add_argument("--dyn-sigma-alpha", type=float, default=1.0, help="Scale factor in dyn_sigma_rel mapping")
    parser.add_argument(
        "--include-sigma-dyn",
        action="store_true",
        help="Also output sigma_dyn = sigma_exp * dyn_sigma_rel when sigma_exp is available",
    )
    parser.add_argument(
        "--beam-direction",
        choices=["plus_z", "minus_z"],
        default="plus_z",
        help="Beam direction used for excitation and Laue-layer coordinates",
    )
    parser.add_argument(
        "--top-observations-n",
        type=int,
        default=2000,
        help="Number of highest-risk observations exported in compact summaries",
    )
    parser.add_argument("--output-dir", default="analysis_output_uncertainty", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    mode_count = sum(
        value is not None
        for value in (args.stream, args.gxparm)
    )
    if mode_count != 1:
        raise SystemExit("Provide exactly one of --stream or --gxparm.")
    if args.gxparm is not None and args.integrate is None:
        raise SystemExit("With --gxparm, provide --integrate.")

    config = DynamicalUncertaintyConfig(
        orientation_axes=_parse_axes(args.orientation_axes),
        orientation_step_deg=float(args.orientation_step_deg),
        orientation_n_steps=int(args.orientation_n_steps),
        thickness_min_nm=float(args.thickness_min_nm),
        thickness_max_nm=float(args.thickness_max_nm),
        n_thickness_steps=int(args.n_thickness_steps),
        orientation_thickness_ref_nm=(
            None if args.orientation_thickness_ref_nm is None else float(args.orientation_thickness_ref_nm)
        ),
        fg_q0_invA=float(args.fg_q0_invA),
        fg_scale=float(args.fg_scale),
        zone_axis_layer_band_invA=float(args.zone_axis_layer_band_invA),
        zone_axis_zolz_relative_width=float(args.zone_axis_zolz_relative_width),
        zone_axis_boost_weight=float(args.zone_axis_boost_weight),
        risk_weight_orientation=float(args.risk_weight_orientation),
        risk_weight_thickness=float(args.risk_weight_thickness),
        risk_normalization_quantile=float(args.risk_normalization_quantile),
        dyn_sigma_form=args.dyn_sigma_form,
        dyn_sigma_alpha=float(args.dyn_sigma_alpha),
        include_sigma_dyn=bool(args.include_sigma_dyn),
        beam_direction=(0.0, 0.0, -1.0) if args.beam_direction == "minus_z" else (0.0, 0.0, 1.0),
    )

    if args.stream is not None:
        result = run_dynamical_uncertainty_from_stream(
            stream_path=args.stream,
            dataset_id=args.dataset_id,
            config=config,
        )
    else:
        result = run_dynamical_uncertainty_from_xds(
            gxparm_path=args.gxparm,
            integrate_path=args.integrate,
            dataset_id=args.dataset_id,
            config=config,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.canonical.observations.to_csv(output_dir / "canonical_observations.csv", index=False)
    result.uncertainty_table.to_csv(output_dir / "observation_dynamical_uncertainty.csv", index=False)

    compact_dir = output_dir / "compact_summaries"
    compact_dir.mkdir(parents=True, exist_ok=True)
    table = result.uncertainty_table
    frame_summary = (
        table.groupby("frame_number", as_index=False)
        .agg(
            n_obs=("obs_id", "size"),
            mean_risk_total=("risk_total", "mean"),
            p95_risk_total=("risk_total", lambda s: s.quantile(0.95)),
            mean_dyn_sigma_rel=("dyn_sigma_rel", "mean"),
            p95_dyn_sigma_rel=("dyn_sigma_rel", lambda s: s.quantile(0.95)),
            mean_zone_axis_score=("zone_axis_score", "mean"),
            p95_zone_axis_score=("zone_axis_score", lambda s: s.quantile(0.95)),
        )
        .sort_values("mean_risk_total", ascending=False)
        .reset_index(drop=True)
    )
    frame_summary.to_csv(compact_dir / "frame_risk_summary.csv", index=False)

    hkl_summary = (
        table.groupby(["h", "k", "l"], as_index=False)
        .agg(
            n_obs=("obs_id", "size"),
            mean_risk_total=("risk_total", "mean"),
            p95_risk_total=("risk_total", lambda s: s.quantile(0.95)),
            mean_dyn_sigma_rel=("dyn_sigma_rel", "mean"),
            p95_dyn_sigma_rel=("dyn_sigma_rel", lambda s: s.quantile(0.95)),
            mean_zone_axis_score=("zone_axis_score", "mean"),
            p95_zone_axis_score=("zone_axis_score", lambda s: s.quantile(0.95)),
        )
        .sort_values(["mean_risk_total", "n_obs"], ascending=[False, False])
        .reset_index(drop=True)
    )
    hkl_summary.to_csv(compact_dir / "hkl_risk_summary.csv", index=False)

    top_n = max(int(args.top_observations_n), 1)
    table.sort_values("risk_total", ascending=False).head(top_n).to_csv(
        compact_dir / f"top{top_n}_observations_by_risk.csv",
        index=False,
    )

    frame_zoneaxis_summary = (
        table.assign(
            is_top10pct_zone_axis=lambda df: df["zone_axis_proximity"] >= df.groupby("frame_number")["zone_axis_proximity"].transform(lambda s: s.quantile(0.90))
        )
        .groupby("frame_number", as_index=False)
        .agg(
            n_obs=("obs_id", "size"),
            n_top10pct_zone_axis=("is_top10pct_zone_axis", "sum"),
            mean_risk_all=("risk_total", "mean"),
            mean_risk_top10pct_zone_axis=("risk_total", lambda s: float("nan")),  # placeholder overwritten below
            mean_dyn_sigma_all=("dyn_sigma_rel", "mean"),
            mean_dyn_sigma_top10pct_zone_axis=("dyn_sigma_rel", lambda s: float("nan")),  # placeholder overwritten below
        )
    )
    # Fill top-10%-zone-axis means with explicit masked groupby.
    zone_axis_means = (
        table.assign(
            is_top10pct_zone_axis=lambda df: df["zone_axis_proximity"] >= df.groupby("frame_number")["zone_axis_proximity"].transform(lambda s: s.quantile(0.90))
        )
        .loc[lambda df: df["is_top10pct_zone_axis"]]
        .groupby("frame_number", as_index=False)
        .agg(
            mean_risk_top10pct_zone_axis=("risk_total", "mean"),
            mean_dyn_sigma_top10pct_zone_axis=("dyn_sigma_rel", "mean"),
        )
    )
    frame_zoneaxis_summary = frame_zoneaxis_summary.drop(
        columns=["mean_risk_top10pct_zone_axis", "mean_dyn_sigma_top10pct_zone_axis"]
    ).merge(zone_axis_means, on="frame_number", how="left")
    frame_zoneaxis_summary = frame_zoneaxis_summary.sort_values("mean_risk_all", ascending=False).reset_index(drop=True)
    frame_zoneaxis_summary.to_csv(compact_dir / "frame_zone_axis_focus_summary.csv", index=False)

    summary = {
        "n_observations": int(result.uncertainty_table.shape[0]),
        "mean_risk_total": float(result.uncertainty_table["risk_total"].mean()),
        "p95_risk_total": float(result.uncertainty_table["risk_total"].quantile(0.95)),
        "p99_risk_total": float(result.uncertainty_table["risk_total"].quantile(0.99)),
        "mean_dyn_sigma_rel": float(result.uncertainty_table["dyn_sigma_rel"].mean()),
        "p95_dyn_sigma_rel": float(result.uncertainty_table["dyn_sigma_rel"].quantile(0.95)),
        "p99_dyn_sigma_rel": float(result.uncertainty_table["dyn_sigma_rel"].quantile(0.99)),
        "max_dyn_sigma_rel": float(result.uncertainty_table["dyn_sigma_rel"].max()),
    }
    (output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n"
    )

    print(f"Source: {result.canonical.source}")
    print(f"Dataset: {result.canonical.dataset_id}")
    print(f"Observations screened: {summary['n_observations']}")
    print(f"Mean dyn_sigma_rel: {summary['mean_dyn_sigma_rel']:.4f}")
    print(f"Wrote compact summaries to {compact_dir.resolve()}")
    print(f"Wrote results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

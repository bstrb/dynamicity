"""Command-line interface for OriDyn."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import OridynConfig, ScoreWeights
from .hkl_plots import make_hkl_trace_plots
from .hkl_trace import run_hkl_trace
from .pipeline import run_axes, run_pipeline
from .plots import make_standard_plots, plot_residuals
from .reweight import cli_weight_preset, load_weight_presets, reweight_scores_file
from .stream_rewrite import rewrite_stream_sigmas
from .summaries import make_information_summaries, write_information_summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Geometry-only orientation-aware dynamical-risk scoring.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the complete observed-reflection scoring pipeline.")
    _add_common_config(run)
    run.add_argument("--stream", required=True)
    run.add_argument("--output", required=True)

    trace = sub.add_parser(
        "trace-hkls",
        help="Score a selected HKL list across every indexed frame for interactive score exploration.",
    )
    _add_common_config(trace)
    trace.add_argument("--stream", required=True)
    trace.add_argument("--hkls", required=True, help="CSV/TSV with h,k,l and optional label/hkl_label columns.")
    trace.add_argument("--output", required=True)

    axes = sub.add_parser("axes", help="Predict problematic axes from the stream cell.")
    _add_common_config(axes)
    axes.add_argument("--stream", required=True)
    axes.add_argument("--output", required=True)

    frames = sub.add_parser("frames", help="Compute frame scores; currently runs the complete first-version path.")
    _add_common_config(frames)
    frames.add_argument("--stream", required=True)
    frames.add_argument("--axes")
    frames.add_argument("--output", required=True)

    score = sub.add_parser("score-reflections", help="Score reflections; currently runs the complete first-version path.")
    _add_common_config(score)
    score.add_argument("--stream", required=True)
    score.add_argument("--axes")
    score.add_argument("--output", required=True)

    reweight = sub.add_parser(
        "reweight",
        help="Recombine existing normalized score terms with new weights without rerunning geometry.",
    )
    reweight.add_argument("--scores", required=True, help="reflection_scores.csv or hkl_frame_trajectories.csv")
    reweight.add_argument("--output", required=True)
    reweight.add_argument("--weights", help="JSON file with one flat weight dict or named weight presets.")
    reweight.add_argument("--alpha", type=float, default=1.0)
    reweight.add_argument("--sigma-map", choices=["exponential", "linear"], default="exponential")
    reweight.add_argument(
        "--sigma-tail-quantile",
        type=float,
        default=0.0,
        help="Use 0 for no percentile gate; otherwise inflate only scores above this quantile.",
    )
    reweight.add_argument("--preset-name", default="cli")
    reweight.add_argument("--w-self", type=float, default=1.0)
    reweight.add_argument("--w-graph", type=float, default=0.50)
    reweight.add_argument("--w-zone", type=float, default=0.5)
    reweight.add_argument("--w-row", type=float, default=0.5)
    reweight.add_argument("--w-frame", type=float, default=1.0)
    reweight.add_argument("--w-interaction", type=float, default=0.25)
    reweight.add_argument(
        "--no-overwrite-single",
        action="store_true",
        help="For a single preset, do not overwrite canonical S_dyn_geom/sigma_dyn_rel columns.",
    )
    reweight.add_argument(
        "--no-score-rescale-by-weights",
        dest="score_rescale_by_weights",
        action="store_false",
        default=True,
        help="Do not divide recombined scores by the active weight sum.",
    )

    plot = sub.add_parser("plot", help="Create standard plots from existing score tables.")
    plot.add_argument("--scores", required=True)
    plot.add_argument("--frames", required=True)
    plot.add_argument("--output", required=True)

    hkl_plot = sub.add_parser("plot-hkl-traces", help="Plot selected-HKL trajectories and component traces.")
    hkl_plot.add_argument("--scores", required=True, help="hkl_frame_trajectories.csv or a reweighted version.")
    hkl_plot.add_argument("--output", required=True)
    hkl_plot.add_argument("--score-columns", nargs="*", default=None)
    hkl_plot.add_argument("--max-hkls", type=int, default=50)

    residual = sub.add_parser("plot-residuals", help="Plot external residuals versus S_dyn_geom.")
    residual.add_argument("--scores", required=True)
    residual.add_argument("--residuals", required=True)
    residual.add_argument("--output", required=True)

    summarize = sub.add_parser("summarize", help="Create information summaries from existing output tables.")
    summarize.add_argument("--scores", required=True)
    summarize.add_argument("--frames", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--axis-sigma-deg", type=float, default=2.0)
    summarize.add_argument("--high-frame-quantile", type=float, default=0.90)
    summarize.add_argument("--top-reflections-per-frame", type=int, default=20)
    summarize.add_argument("--top-frames", type=int, default=50)

    # Kept for backwards compatibility with the generated project. It is not part
    # of the recommended first exploration workflow.
    rewrite = sub.add_parser("rewrite-sigmas", help="Experimental: write a stream copy with sigmas replaced by OriDyn sigma_dyn.")
    rewrite.add_argument("--stream", required=True)
    rewrite.add_argument("--scores", required=True)
    rewrite.add_argument("--output", required=True)
    rewrite.add_argument("--sigma-column", default="sigma_dyn")
    rewrite.add_argument("--sigma-dyn-rel-cap", type=float, default=None)
    rewrite.add_argument("--backup", action="store_true")
    return parser


def _add_common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dmin", type=float, default=0.5)
    parser.add_argument("--dmax", type=float, default=20.0)
    parser.add_argument("--hkl-limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=200_000)
    parser.add_argument("--uvw-max", type=int, default=5)
    parser.add_argument(
        "--max-problematic-axes",
        type=int,
        default=100,
        help="Only the top N ranked problematic axes are allowed to drive scoring; use 0 for no cap.",
    )
    parser.add_argument("--axis-score-min", type=float, default=0.0)
    parser.add_argument("--axis-sigma-deg", type=float, default=2.0)
    parser.add_argument("--sg0", type=float, default=0.01)
    parser.add_argument("--excitation-kernel", choices=["gaussian", "lorentzian"], default="gaussian")
    parser.add_argument("--neighbor-excitation-min", type=float, default=0.05)
    parser.add_argument("--neighbor-hkl-radius", type=int, default=3)
    parser.add_argument("--max-neighbors-per-reflection", type=int, default=64)
    parser.add_argument("--max-excited-nodes-per-frame", type=int, default=2000)
    parser.add_argument("--row-direction-limit", type=int, default=5)
    parser.add_argument("--row-max-steps", type=int, default=12)
    normalization_choices = ["global_minmax", "median_mad", "percentile", "rank", "none"]
    parser.add_argument("--normalization", choices=normalization_choices, default="global_minmax")
    parser.add_argument(
        "--frame-normalization",
        choices=["inherit", *normalization_choices],
        default="inherit",
        help="Frame-axis normalization only; 'inherit' uses --normalization.",
    )
    parser.add_argument("--normalization-clip", type=float, default=10.0)
    parser.add_argument("--resolution-shells", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--sigma-map", choices=["exponential", "linear"], default="exponential")
    parser.add_argument(
        "--sigma-tail-quantile",
        type=float,
        default=0.0,
        help="Use 0 for no percentile gate; otherwise inflate only scores above this quantile.",
    )
    parser.add_argument(
        "--no-score-rescale-by-weights",
        dest="score_rescale_by_weights",
        action="store_false",
        default=True,
        help="Do not divide S_dyn_geom by the active weight sum.",
    )
    parser.add_argument("--w-self", type=float, default=1.0)
    parser.add_argument("--w-graph", type=float, default=0.5)
    parser.add_argument("--w-zone", type=float, default=0.5)
    parser.add_argument("--w-row", type=float, default=0.5)
    parser.add_argument("--w-frame", type=float, default=1.0)
    parser.add_argument("--w-interaction", type=float, default=0.25)
    parser.add_argument("--export-candidates", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")


def config_from_args(args: argparse.Namespace) -> OridynConfig:
    weights = ScoreWeights(
        self=args.w_self,
        graph=args.w_graph,
        zone=args.w_zone,
        row=args.w_row,
        frame=args.w_frame,
        interaction=args.w_interaction,
    )
    return OridynConfig(
        dmin=args.dmin,
        dmax=args.dmax,
        hkl_limit=args.hkl_limit,
        max_candidates=args.max_candidates,
        uvw_max=args.uvw_max,
        max_problematic_axes=None if args.max_problematic_axes <= 0 else args.max_problematic_axes,
        axis_score_min=args.axis_score_min,
        axis_sigma_deg=args.axis_sigma_deg,
        sg0=args.sg0,
        excitation_kernel=args.excitation_kernel,
        neighbor_excitation_min=args.neighbor_excitation_min,
        neighbor_hkl_radius=args.neighbor_hkl_radius,
        max_neighbors_per_reflection=args.max_neighbors_per_reflection,
        max_excited_nodes_per_frame=args.max_excited_nodes_per_frame,
        row_direction_limit=args.row_direction_limit,
        row_max_steps=args.row_max_steps,
        normalization=args.normalization,
        frame_normalization=None if args.frame_normalization == "inherit" else args.frame_normalization,
        normalization_clip=args.normalization_clip,
        resolution_shells=args.resolution_shells,
        alpha=args.alpha,
        sigma_map=args.sigma_map,
        sigma_tail_quantile=args.sigma_tail_quantile,
        score_rescale_by_weights=args.score_rescale_by_weights,
        weights=weights,
        export_candidates=args.export_candidates,
        workers=args.workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
        progress=not args.quiet,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plot":
        make_standard_plots(pd.read_csv(args.scores), pd.read_csv(args.frames), Path(args.output))
        return
    if args.command == "plot-hkl-traces":
        make_hkl_trace_plots(
            pd.read_csv(args.scores, dtype={"hkl_label": "string"}),
            Path(args.output),
            score_columns=args.score_columns,
            max_hkls=args.max_hkls,
        )
        return
    if args.command == "plot-residuals":
        plot_residuals(args.scores, args.residuals, args.output)
        return
    if args.command == "reweight":
        if args.weights:
            presets = load_weight_presets(args.weights)
        else:
            presets = cli_weight_preset(
                args.w_self,
                args.w_graph,
                args.w_zone,
                args.w_row,
                args.w_frame,
                args.w_interaction,
                name=args.preset_name,
            )
        out = reweight_scores_file(
            args.scores,
            args.output,
            presets,
            alpha=args.alpha,
            sigma_map=args.sigma_map,
            sigma_tail_quantile=args.sigma_tail_quantile,
            score_rescale_by_weights=args.score_rescale_by_weights,
            overwrite_single=not args.no_overwrite_single,
        )
        print(f"wrote {len(out)} reweighted rows to {args.output}")
        return
    if args.command == "summarize":
        summaries = make_information_summaries(
            pd.read_csv(args.frames),
            pd.read_csv(args.scores),
            axis_sigma_deg=args.axis_sigma_deg,
            high_frame_quantile=args.high_frame_quantile,
            top_reflections_per_frame=args.top_reflections_per_frame,
            top_frames=args.top_frames,
        )
        write_information_summaries(args.output, summaries)
        return
    if args.command == "rewrite-sigmas":
        stats = rewrite_stream_sigmas(
            args.stream,
            args.scores,
            args.output,
            sigma_column=args.sigma_column,
            sigma_dyn_rel_cap=args.sigma_dyn_rel_cap,
            make_backup=args.backup,
        )
        print(
            "rewrote "
            f"{stats.reflection_rows_rewritten}/{stats.reflection_rows_seen} reflection sigmas "
            f"({stats.missing_score_rows} missing score rows, {stats.duplicate_score_keys} duplicate score keys)"
        )
        if stats.backup_path:
            print(f"backup: {stats.backup_path}")
        return
    config = config_from_args(args)
    if args.command == "axes":
        run_axes(args.stream, args.output, config)
        return
    if args.command == "trace-hkls":
        run_hkl_trace(args.stream, args.hkls, args.output, config)
        return
    # The first version keeps the complete scoring path as the reliable unit of
    # work. The frames and score-reflections aliases preserve the requested CLI.
    run_pipeline(args.stream, args.output, config)


if __name__ == "__main__":
    main()

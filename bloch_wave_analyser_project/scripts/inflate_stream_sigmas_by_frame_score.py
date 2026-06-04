#!/usr/bin/env python3
"""Copy a CrystFEL stream while scaling sigma(I) by a frame-level score."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="Input CrystFEL .stream file")
    parser.add_argument("--frame-summary", required=True, help="frame_summary.csv with frame-level scores")
    parser.add_argument("--output", required=True, help="Output .stream file")
    parser.add_argument("--score-column", default="S_orient", help="Frame score column used for sigma scaling")
    parser.add_argument(
        "--scale-mode",
        choices=["score", "one-plus-score", "max-one-score"],
        default="score",
        help="Use sigma*score, sigma*(1+score), or sigma*max(1, score)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional text summary path; default is output path with .summary.txt suffix",
    )
    return parser


def _load_scores(frame_summary_path: Path, score_column: str) -> dict[int, float]:
    frame_summary = pd.read_csv(frame_summary_path)
    required = {"frame", score_column}
    missing = sorted(required.difference(frame_summary.columns))
    if missing:
        raise ValueError(f"Frame summary is missing required columns: {missing}")

    table = frame_summary[["frame", score_column]].copy()
    table = table[np.isfinite(table[score_column].to_numpy(dtype=float))]
    return {int(row.frame): float(getattr(row, score_column)) for row in table.itertuples(index=False)}


def _scale_from_score(score: float, scale_mode: str) -> float:
    if scale_mode == "one-plus-score":
        return 1.0 + float(score)
    if scale_mode == "max-one-score":
        return max(1.0, float(score))
    return float(score)


def _format_reflection_row(parts: list[str], sigma_new: float) -> str:
    h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
    intensity = float(parts[3])
    peak = float(parts[5])
    background = float(parts[6])
    fs_px = float(parts[7])
    ss_px = float(parts[8])
    panel = parts[9] if len(parts) > 9 else "p0"
    return (
        f"{h:4d}{k:5d}{l:5d}"
        f"{intensity:11.2f}{sigma_new:11.2f}{peak:11.2f}{background:11.2f}"
        f"{fs_px:7.1f}{ss_px:7.1f} {panel}\n"
    )


def inflate_stream_sigmas(
    *,
    stream_path: Path,
    frame_summary_path: Path,
    output_path: Path,
    score_column: str,
    scale_mode: str,
    summary_path: Path,
) -> None:
    scores = _load_scores(frame_summary_path, score_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    crystal_index = -1
    current_score: float | None = None
    current_scale: float | None = None
    in_reflections = False

    n_crystals = 0
    n_crystals_matched = 0
    n_reflection_rows = 0
    n_scaled = 0
    n_unmatched_crystal_rows = 0
    n_unparsed_reflection_rows = 0
    sigma_sum_before = 0.0
    sigma_sum_after = 0.0
    scores_used: list[float] = []
    scales_used: list[float] = []

    with stream_path.open("r", errors="replace") as handle_in, output_path.open("w") as handle_out:
        for raw_line in handle_in:
            if raw_line.startswith("--- Begin crystal"):
                crystal_index += 1
                n_crystals += 1
                current_score = scores.get(crystal_index)
                current_scale = None if current_score is None else _scale_from_score(current_score, scale_mode)
                if current_score is not None:
                    n_crystals_matched += 1
                    scores_used.append(float(current_score))
                    scales_used.append(float(current_scale))
                handle_out.write(raw_line)
                continue

            if "Reflections measured after indexing" in raw_line:
                in_reflections = True
                handle_out.write(raw_line)
                continue

            if in_reflections and "End of reflections" in raw_line:
                in_reflections = False
                handle_out.write(raw_line)
                continue

            if in_reflections:
                parts = raw_line.split()
                if len(parts) >= 9:
                    try:
                        sigma = float(parts[4])
                    except ValueError:
                        handle_out.write(raw_line)
                        continue

                    n_reflection_rows += 1
                    if current_scale is None:
                        n_unmatched_crystal_rows += 1
                        handle_out.write(raw_line)
                        continue

                    try:
                        sigma_new = sigma * float(current_scale)
                        handle_out.write(_format_reflection_row(parts, sigma_new))
                    except ValueError:
                        n_unparsed_reflection_rows += 1
                        handle_out.write(raw_line)
                        continue

                    n_scaled += 1
                    sigma_sum_before += sigma
                    sigma_sum_after += sigma_new
                    continue

            handle_out.write(raw_line)

    score_array = np.asarray(scores_used, dtype=float)
    scale_array = np.asarray(scales_used, dtype=float)
    summary_lines = [
        f"input_stream: {stream_path}",
        f"frame_summary: {frame_summary_path}",
        f"output_stream: {output_path}",
        f"score_column: {score_column}",
        f"scale_mode: {scale_mode}",
        f"n_crystals_seen: {n_crystals}",
        f"n_crystals_matched: {n_crystals_matched}",
        f"n_reflection_rows_seen: {n_reflection_rows}",
        f"n_reflection_rows_scaled: {n_scaled}",
        f"n_unmatched_crystal_rows: {n_unmatched_crystal_rows}",
        f"n_unparsed_reflection_rows: {n_unparsed_reflection_rows}",
        f"mean_score_matched_crystals: {float(score_array.mean()) if score_array.size else float('nan')}",
        f"p95_score_matched_crystals: {float(np.quantile(score_array, 0.95)) if score_array.size else float('nan')}",
        f"max_score_matched_crystals: {float(score_array.max()) if score_array.size else float('nan')}",
        f"mean_sigma_scale_matched_crystals: {float(scale_array.mean()) if scale_array.size else float('nan')}",
        f"min_sigma_scale_matched_crystals: {float(scale_array.min()) if scale_array.size else float('nan')}",
        f"max_sigma_scale_matched_crystals: {float(scale_array.max()) if scale_array.size else float('nan')}",
        f"sum_sigma_scale_scaled_rows: {float(sigma_sum_after / sigma_sum_before) if sigma_sum_before > 0 else float('nan')}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(output_path.suffix + ".summary.txt")
    inflate_stream_sigmas(
        stream_path=Path(args.stream),
        frame_summary_path=Path(args.frame_summary),
        output_path=output_path,
        score_column=str(args.score_column),
        scale_mode=str(args.scale_mode),
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write an INTEGRATE.HKL copy with sigma scaled by a frame-level score."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrate", required=True, help="Input INTEGRATE.HKL")
    parser.add_argument("--frame-summary", required=True, help="frame_summary.csv with frame-level scores")
    parser.add_argument("--output", required=True, help="Output INTEGRATE.HKL copy")
    parser.add_argument("--score-column", default="S_orient", help="Frame-level score column used as sigma scale")
    parser.add_argument(
        "--frame-coordinate",
        choices=["zobs", "zcal"],
        default="zobs",
        help="INTEGRATE.HKL coordinate used to choose the frame",
    )
    parser.add_argument(
        "--frame-rounding",
        choices=["floor", "round"],
        default="floor",
        help="How to convert ZOBS/ZCAL to integer frame index",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["score", "one-plus-score"],
        default="score",
        help="Use sigma*score or sigma*(1+score)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional text summary path; default is output path with .summary.txt suffix",
    )
    return parser


def _frame_index(value: float, rounding: str) -> int:
    if rounding == "round":
        return int(np.rint(float(value)))
    return int(np.floor(float(value)))


def _load_frame_lookup(frame_summary_path: Path, score_column: str) -> dict[int, float]:
    frame_summary = pd.read_csv(frame_summary_path)
    required = {"frame", score_column}
    missing = sorted(required.difference(frame_summary.columns))
    if missing:
        raise ValueError(f"Frame summary is missing required columns: {missing}")

    table = frame_summary[["frame", score_column]].copy()
    table = table[np.isfinite(table[score_column].to_numpy(dtype=float))]
    return {int(row.frame): float(getattr(row, score_column)) for row in table.itertuples(index=False)}


def inflate_integrate_sigma_by_frame_score(
    *,
    integrate_path: Path,
    frame_summary_path: Path,
    output_path: Path,
    score_column: str,
    frame_coordinate: str,
    frame_rounding: str,
    scale_mode: str,
    summary_path: Path,
) -> None:
    lookup = _load_frame_lookup(frame_summary_path, score_column)
    z_index = 14 if frame_coordinate == "zobs" else 7

    n_data = 0
    n_updated = 0
    n_unmatched = 0
    original_sigma_sum = 0.0
    new_sigma_sum = 0.0
    score_values: list[float] = []
    scale_values: list[float] = []
    output_lines: list[str] = []

    for raw_line in integrate_path.read_text().splitlines():
        if raw_line.startswith("!") or not raw_line.strip():
            output_lines.append(raw_line)
            continue

        parts = raw_line.split()
        if len(parts) < 21:
            output_lines.append(raw_line)
            continue

        n_data += 1
        try:
            sigma = float(parts[4])
            z_value = float(parts[z_index])
        except ValueError:
            output_lines.append(raw_line)
            continue

        frame = _frame_index(z_value, frame_rounding)
        score = lookup.get(frame)
        if score is None:
            n_unmatched += 1
            output_lines.append(raw_line)
            continue

        scale = float(score) if scale_mode == "score" else 1.0 + float(score)
        sigma_new = sigma * scale
        parts[4] = f"{sigma_new:.6E}"
        output_lines.append(" ".join(parts))

        n_updated += 1
        original_sigma_sum += sigma
        new_sigma_sum += sigma_new
        score_values.append(float(score))
        scale_values.append(float(scale))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n")

    score_array = np.asarray(score_values, dtype=float)
    scale_array = np.asarray(scale_values, dtype=float)
    summary_lines = [
        f"input_integrate: {integrate_path}",
        f"frame_summary: {frame_summary_path}",
        f"output_integrate: {output_path}",
        f"score_column: {score_column}",
        f"frame_coordinate: {frame_coordinate}",
        f"frame_rounding: {frame_rounding}",
        f"scale_mode: {scale_mode}",
        f"n_data_rows: {n_data}",
        f"n_updated: {n_updated}",
        f"n_unmatched: {n_unmatched}",
        f"mean_score_updated: {float(np.mean(score_array)) if score_array.size else float('nan')}",
        f"p05_score_updated: {float(np.quantile(score_array, 0.05)) if score_array.size else float('nan')}",
        f"p95_score_updated: {float(np.quantile(score_array, 0.95)) if score_array.size else float('nan')}",
        f"mean_sigma_scale_updated: {float(np.mean(scale_array)) if scale_array.size else float('nan')}",
        f"min_sigma_scale_updated: {float(np.min(scale_array)) if scale_array.size else float('nan')}",
        f"max_sigma_scale_updated: {float(np.max(scale_array)) if scale_array.size else float('nan')}",
        f"sum_sigma_scale_updated: {float(new_sigma_sum / original_sigma_sum) if original_sigma_sum > 0 else float('nan')}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary is not None else output_path.with_suffix(output_path.suffix + ".summary.txt")
    inflate_integrate_sigma_by_frame_score(
        integrate_path=Path(args.integrate),
        frame_summary_path=Path(args.frame_summary),
        output_path=output_path,
        score_column=str(args.score_column),
        frame_coordinate=str(args.frame_coordinate),
        frame_rounding=str(args.frame_rounding),
        scale_mode=str(args.scale_mode),
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()

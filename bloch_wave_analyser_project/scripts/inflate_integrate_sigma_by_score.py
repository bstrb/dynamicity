#!/usr/bin/env python3
"""Write an INTEGRATE.HKL copy with sigma inflated by reflection scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrate", required=True, help="Input INTEGRATE.HKL")
    parser.add_argument("--scores", required=True, help="reflections_long.csv with frame/h/k/l score columns")
    parser.add_argument("--output", required=True, help="Output INTEGRATE.HKL copy")
    parser.add_argument("--score-column", default="S_dyn", help="Score column used for sigma inflation")
    parser.add_argument(
        "--frame-coordinate",
        choices=["zobs", "zcal"],
        default="zobs",
        help="INTEGRATE.HKL coordinate used to choose the observed frame",
    )
    parser.add_argument(
        "--frame-rounding",
        choices=["floor", "round"],
        default="floor",
        help="How to convert ZOBS/ZCAL to integer frame index",
    )
    parser.add_argument(
        "--score-aggregation",
        choices=["max", "mean"],
        default="max",
        help="How to collapse duplicate score rows for the same frame/h/k/l",
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


def _load_score_lookup(
    scores_path: Path,
    score_column: str,
    aggregation: str,
) -> dict[tuple[int, int, int, int], float]:
    scores = pd.read_csv(scores_path)
    required = {"frame", "h", "k", "l", score_column}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Score table is missing required columns: {missing}")

    table = scores[["frame", "h", "k", "l", score_column]].copy()
    table = table[np.isfinite(table[score_column].to_numpy(dtype=float))]
    if aggregation == "mean":
        grouped = table.groupby(["frame", "h", "k", "l"], as_index=False)[score_column].mean()
    else:
        grouped = table.groupby(["frame", "h", "k", "l"], as_index=False)[score_column].max()

    return {
        (int(row.frame), int(row.h), int(row.k), int(row.l)): float(getattr(row, score_column))
        for row in grouped.itertuples(index=False)
    }


def inflate_integrate_sigma(
    *,
    integrate_path: Path,
    scores_path: Path,
    output_path: Path,
    score_column: str,
    frame_coordinate: str,
    frame_rounding: str,
    score_aggregation: str,
    summary_path: Path,
) -> None:
    lookup = _load_score_lookup(scores_path, score_column, score_aggregation)
    z_index = 14 if frame_coordinate == "zobs" else 7

    n_data = 0
    n_updated = 0
    n_unmatched = 0
    original_sigma_sum = 0.0
    new_sigma_sum = 0.0
    score_values: list[float] = []
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
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
            sigma = float(parts[4])
            z_value = float(parts[z_index])
        except ValueError:
            output_lines.append(raw_line)
            continue

        frame = _frame_index(z_value, frame_rounding)
        score = lookup.get((frame, h, k, l))
        if score is None:
            n_unmatched += 1
            output_lines.append(raw_line)
            continue

        sigma_new = sigma * (1.0 + float(score))
        parts[4] = f"{sigma_new:.6E}"
        output_lines.append(" ".join(parts))

        n_updated += 1
        original_sigma_sum += sigma
        new_sigma_sum += sigma_new
        score_values.append(float(score))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n")

    score_array = np.asarray(score_values, dtype=float)
    summary_lines = [
        f"input_integrate: {integrate_path}",
        f"scores: {scores_path}",
        f"output_integrate: {output_path}",
        f"score_column: {score_column}",
        f"frame_coordinate: {frame_coordinate}",
        f"frame_rounding: {frame_rounding}",
        f"score_aggregation: {score_aggregation}",
        f"n_data_rows: {n_data}",
        f"n_updated: {n_updated}",
        f"n_unmatched: {n_unmatched}",
        f"mean_score_updated: {float(np.mean(score_array)) if score_array.size else float('nan')}",
        f"p95_score_updated: {float(np.quantile(score_array, 0.95)) if score_array.size else float('nan')}",
        f"mean_sigma_scale_updated: {float(new_sigma_sum / original_sigma_sum) if original_sigma_sum > 0 else float('nan')}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary is not None else output_path.with_suffix(output_path.suffix + ".summary.txt")
    inflate_integrate_sigma(
        integrate_path=Path(args.integrate),
        scores_path=Path(args.scores),
        output_path=output_path,
        score_column=str(args.score_column),
        frame_coordinate=str(args.frame_coordinate),
        frame_rounding=str(args.frame_rounding),
        score_aggregation=str(args.score_aggregation),
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()

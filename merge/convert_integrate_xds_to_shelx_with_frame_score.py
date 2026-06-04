#!/usr/bin/env python3
"""Convert XDS INTEGRATE.HKL to HKLF4-like output with a frame score column."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert XDS INTEGRATE.HKL to SHELX HKLF4-like columns and append "
            "a frame-level score, such as S_orient, selected from ZOBS/ZCAL."
        )
    )
    parser.add_argument("input_file", help="Path to the INTEGRATE.HKL file")
    parser.add_argument("--frame-summary", required=True, help="frame_summary.csv with frame-level scores")
    parser.add_argument("-o", "--output", dest="output_file", default=None, help="Path to output HKL file")
    parser.add_argument("--score-column", default="S_orient", help="Frame score column to append")
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
        "--sort",
        dest="sort_by",
        choices=["d_spacing", "original"],
        default="d_spacing",
        help="Sorting method",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary path; default is output path with .summary.txt suffix",
    )
    return parser


def _frame_index(value: float, rounding: str) -> int:
    if rounding == "round":
        return int(np.rint(float(value)))
    return int(np.floor(float(value)))


def _load_frame_scores(frame_summary_file: str | os.PathLike[str], score_column: str) -> dict[int, float]:
    frame_summary = pd.read_csv(frame_summary_file)
    required = {"frame", score_column}
    missing = sorted(required.difference(frame_summary.columns))
    if missing:
        raise ValueError(f"Frame summary is missing required columns: {missing}")

    table = frame_summary[["frame", score_column]].copy()
    table = table[np.isfinite(table[score_column].to_numpy(dtype=float))]
    return {int(row.frame): float(getattr(row, score_column)) for row in table.itertuples(index=False)}


def _sort_by_d_spacing(df: pd.DataFrame, unit_cell: list[float] | None) -> pd.DataFrame:
    if unit_cell is None:
        print("No unit cell found in header; keeping original order.")
        return df

    a, b, c, alpha, beta, gamma = unit_cell
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    metric = np.array(
        [
            [a**2, a * b * np.cos(gamma_rad), a * c * np.cos(beta_rad)],
            [a * b * np.cos(gamma_rad), b**2, b * c * np.cos(alpha_rad)],
            [a * c * np.cos(beta_rad), b * c * np.cos(alpha_rad), c**2],
        ]
    )
    metric_inv = np.linalg.inv(metric)

    hkl = df[["H", "K", "L"]].to_numpy(dtype=float)
    inv_d2 = np.sum((hkl @ metric_inv) * hkl, axis=1)
    return df.assign(inv_d2=inv_d2).sort_values("inv_d2", ascending=True).drop(columns="inv_d2")


def convert_integrate_xds_to_shelx_with_frame_score(
    input_file: str | os.PathLike[str],
    frame_summary_file: str | os.PathLike[str],
    output_file: str | os.PathLike[str] | None = None,
    *,
    score_column: str = "S_orient",
    frame_coordinate: str = "zobs",
    frame_rounding: str = "floor",
    sort_by: str = "d_spacing",
    summary_file: str | os.PathLike[str] | None = None,
) -> None:
    input_path = Path(input_file)
    if output_file is None:
        output_path = input_path.with_name(f"{input_path.stem}_{score_column}_HKLF4.HKL")
    else:
        output_path = Path(output_file)

    summary_path = (
        Path(summary_file)
        if summary_file is not None
        else output_path.with_suffix(output_path.suffix + ".summary.txt")
    )

    frame_scores = _load_frame_scores(frame_summary_file, score_column)
    z_index = 14 if frame_coordinate == "zobs" else 7

    print(f"Reading {input_path}...")
    data: list[list[float | int]] = []
    unit_cell: list[float] | None = None
    n_data = 0
    n_matched = 0
    n_unmatched = 0
    scores_used: list[float] = []
    frames_used: list[int] = []

    with input_path.open("r") as handle:
        for line in handle:
            if line.startswith("!UNIT_CELL_CONSTANTS="):
                unit_cell = [float(value) for value in line.split("=")[1].split()]
            if line.startswith("!"):
                continue

            parts = line.split()
            if len(parts) < 15:
                continue

            try:
                h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                intensity = float(parts[3])
                sigma = float(parts[4])
                z_value = float(parts[z_index])
            except ValueError:
                continue

            n_data += 1
            frame = _frame_index(z_value, frame_rounding)
            score = frame_scores.get(frame)
            if score is None:
                n_unmatched += 1
                continue

            data.append([h, k, l, intensity, sigma, frame, score])
            n_matched += 1
            frames_used.append(frame)
            scores_used.append(score)

    if n_unmatched:
        raise ValueError(
            f"{n_unmatched} reflections did not match a frame in {frame_summary_file}; "
            "no output was written."
        )

    df = pd.DataFrame(data, columns=["H", "K", "L", "I", "SIGMA", "FRAME", score_column])
    if sort_by == "d_spacing":
        print("Sorting by d-spacing (ascending)...")
        df = _sort_by_d_spacing(df, unit_cell)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_path}...")
    with output_path.open("w") as handle:
        for _, row in df.iterrows():
            handle.write(
                f"{int(row['H']):4d}{int(row['K']):4d}{int(row['L']):4d}"
                f"{row['I']:8.2f}{row['SIGMA']:8.2f}{row[score_column]:12.6f}\n"
            )
        handle.write("   0   0   0    0.00    0.00    0.000000\n")

    score_array = np.asarray(scores_used, dtype=float)
    frame_array = np.asarray(frames_used, dtype=int)
    summary_lines = [
        f"input_integrate: {input_path}",
        f"frame_summary: {frame_summary_file}",
        f"output_hkl: {output_path}",
        f"score_column: {score_column}",
        f"frame_coordinate: {frame_coordinate}",
        f"frame_rounding: {frame_rounding}",
        f"sort_by: {sort_by}",
        f"n_data_rows: {n_data}",
        f"n_written: {n_matched}",
        f"n_unmatched: {n_unmatched}",
        f"frame_min_zero_based: {int(frame_array.min()) if frame_array.size else 'nan'}",
        f"frame_max_zero_based: {int(frame_array.max()) if frame_array.size else 'nan'}",
        f"score_mean: {float(score_array.mean()) if score_array.size else float('nan')}",
        f"score_min: {float(score_array.min()) if score_array.size else float('nan')}",
        f"score_max: {float(score_array.max()) if score_array.size else float('nan')}",
        f"score_p05: {float(np.quantile(score_array, 0.05)) if score_array.size else float('nan')}",
        f"score_p95: {float(np.quantile(score_array, 0.95)) if score_array.size else float('nan')}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Writing {summary_path}...")


if __name__ == "__main__":
    args = build_parser().parse_args()

    if not os.path.isfile(args.input_file):
        raise SystemExit(f"[ERROR] Input file not found: {args.input_file}")

    convert_integrate_xds_to_shelx_with_frame_score(
        args.input_file,
        args.frame_summary,
        args.output_file,
        score_column=args.score_column,
        frame_coordinate=args.frame_coordinate,
        frame_rounding=args.frame_rounding,
        sort_by=args.sort_by,
        summary_file=args.summary,
    )

#!/usr/bin/env python3
"""Summarize OriDyn sigma inflation by HKL and resolution shell."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_row_self_exp_notail/reflection_scores.csv"
)
DEFAULT_HKL_OUTPUT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_row_self_exp_notail/per_hkl_sigma_dyn_rel_spread_mmm.csv"
)
DEFAULT_RESOLUTION_OUTPUT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_row_self_exp_notail/per_resolution_shell_sigma_dyn_rel_spread.csv"
)

HKL_COLUMNS = ("h", "k", "l")
SCORE_COLUMN = "S_dyn_geom"
SIGMA_COLUMN = "sigma_dyn_rel"
FRAME_COLUMNS = ("frame", "frame_number")
D_SPACING_CANDIDATES = (
    "d",
    "d_angstrom",
    "d_spacing",
    "d_spacing_angstrom",
    "resolution",
    "resolution_angstrom",
)
INVERSE_D_CANDIDATES = ("q_invA", "inv_d", "inv_d_angstrom", "resolution_invA")
SIGMA_THRESHOLDS = (2.0, 5.0, 10.0, 20.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether high OriDyn sigma_dyn_rel observations are "
            "HKL-wide, orientation-specific, or resolution-shell-specific."
        )
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--hkl-output", type=Path, default=DEFAULT_HKL_OUTPUT)
    parser.add_argument("--resolution-output", type=Path, default=DEFAULT_RESOLUTION_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--resolution-shells", type=int, default=12)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--min-ranked-observations",
        type=int,
        default=20,
        help="Minimum n_obs for printed ranked HKL tables.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print detected columns and exit.",
    )
    return parser.parse_args()


def read_columns(scores_path: Path) -> list[str]:
    return list(pd.read_csv(scores_path, nrows=0).columns)


def first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def detect_columns(columns: list[str]) -> dict[str, object]:
    missing = [column for column in (*HKL_COLUMNS, SCORE_COLUMN, SIGMA_COLUMN) if column not in columns]
    if missing:
        raise SystemExit(f"Missing required column(s): {missing}")

    d_column = first_present(columns, D_SPACING_CANDIDATES)
    inverse_d_column = None if d_column is not None else first_present(columns, INVERSE_D_CANDIDATES)

    return {
        "hkl": list(HKL_COLUMNS),
        "score": SCORE_COLUMN,
        "sigma": SIGMA_COLUMN,
        "frame": [column for column in FRAME_COLUMNS if column in columns],
        "d_spacing": d_column,
        "inverse_d": inverse_d_column,
    }


def print_column_inspection(columns: list[str], detected: dict[str, object]) -> None:
    print("Available columns:")
    print(", ".join(columns))
    print()
    print("Detected analysis columns:")
    print(f"  HKL: {', '.join(detected['hkl'])}")
    print(f"  score: {detected['score']}")
    print(f"  sigma inflation: {detected['sigma']}")
    frame_columns = detected["frame"]
    print(f"  frame identifier(s): {', '.join(frame_columns) if frame_columns else 'none found'}")
    if detected["d_spacing"]:
        print(f"  d-spacing/resolution: {detected['d_spacing']}")
    elif detected["inverse_d"]:
        print(f"  d-spacing/resolution: derived as 1 / {detected['inverse_d']}")
    else:
        print("  d-spacing/resolution: none found")
    print()


def load_slim_scores(scores_path: Path, detected: dict[str, object], chunksize: int) -> pd.DataFrame:
    usecols = [*HKL_COLUMNS, SCORE_COLUMN, SIGMA_COLUMN]
    d_column = detected["d_spacing"]
    inverse_d_column = detected["inverse_d"]
    if d_column:
        usecols.append(str(d_column))
    elif inverse_d_column:
        usecols.append(str(inverse_d_column))

    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_kept = 0
    print(f"Reading slim score table in chunks of {chunksize:,} row(s)...")

    for i, chunk in enumerate(pd.read_csv(scores_path, usecols=usecols, chunksize=chunksize), start=1):
        rows_read += len(chunk)
        slim = pd.DataFrame(
            {
                "H": pd.to_numeric(chunk["h"], errors="coerce").abs(),
                "K": pd.to_numeric(chunk["k"], errors="coerce").abs(),
                "L": pd.to_numeric(chunk["l"], errors="coerce").abs(),
                "S_dyn_geom": pd.to_numeric(chunk[SCORE_COLUMN], errors="coerce"),
                "sigma_dyn_rel": pd.to_numeric(chunk[SIGMA_COLUMN], errors="coerce"),
            }
        )
        if d_column:
            slim["d_angstrom"] = pd.to_numeric(chunk[str(d_column)], errors="coerce")
        elif inverse_d_column:
            inv_d = pd.to_numeric(chunk[str(inverse_d_column)], errors="coerce")
            slim["d_angstrom"] = np.where(inv_d > 0.0, 1.0 / inv_d, np.nan)

        slim = slim.dropna(subset=["H", "K", "L", "S_dyn_geom", "sigma_dyn_rel"])
        slim = slim.astype({"H": "int32", "K": "int32", "L": "int32"})
        chunks.append(slim)
        rows_kept += len(slim)

        if i == 1 or i % 5 == 0:
            print(f"  chunks read: {i:>3d}, rows read: {rows_read:,}, rows kept: {rows_kept:,}")

    if not chunks:
        raise SystemExit("No usable score rows were found.")

    out = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(out):,} usable row(s) with {len(out.columns)} slim column(s).")
    print()
    return out


def quantile_agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, sort=False, observed=True)
    summary = grouped.agg(
        n_obs=("sigma_dyn_rel", "size"),
        S_median=("S_dyn_geom", "median"),
        S_p90=("S_dyn_geom", lambda x: x.quantile(0.90)),
        S_p95=("S_dyn_geom", lambda x: x.quantile(0.95)),
        S_p99=("S_dyn_geom", lambda x: x.quantile(0.99)),
        S_max=("S_dyn_geom", "max"),
        sigma_median=("sigma_dyn_rel", "median"),
        sigma_p90=("sigma_dyn_rel", lambda x: x.quantile(0.90)),
        sigma_p95=("sigma_dyn_rel", lambda x: x.quantile(0.95)),
        sigma_p99=("sigma_dyn_rel", lambda x: x.quantile(0.99)),
        sigma_max=("sigma_dyn_rel", "max"),
    ).reset_index()

    if "d_angstrom" in df:
        d_summary = grouped.agg(
            d_mean=("d_angstrom", "mean"),
            d_median=("d_angstrom", "median"),
            d_min=("d_angstrom", "min"),
            d_max=("d_angstrom", "max"),
        ).reset_index()
        summary = summary.merge(d_summary, on=group_cols, how="left")

    for threshold in SIGMA_THRESHOLDS:
        frac = grouped["sigma_dyn_rel"].agg(
            lambda x, t=threshold: float((x >= t).mean())
        ).reset_index(name=f"frac_sigma_ge_{int(threshold)}")
        summary = summary.merge(frac, on=group_cols, how="left")

    add_spread_columns(summary)
    return summary


def add_spread_columns(summary: pd.DataFrame) -> None:
    median = summary["sigma_median"].to_numpy(dtype=float)
    p95 = summary["sigma_p95"].to_numpy(dtype=float)
    max_value = summary["sigma_max"].to_numpy(dtype=float)
    summary["sigma_p95_over_median"] = np.divide(
        p95,
        median,
        out=np.full_like(p95, np.nan, dtype=float),
        where=median > 0.0,
    )
    summary["sigma_max_over_median"] = np.divide(
        max_value,
        median,
        out=np.full_like(max_value, np.nan, dtype=float),
        where=median > 0.0,
    )
    summary["sigma_p95_minus_median"] = summary["sigma_p95"] - summary["sigma_median"]
    summary["sigma_max_minus_median"] = summary["sigma_max"] - summary["sigma_median"]


def write_hkl_summary(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = quantile_agg(df, ["H", "K", "L"])
    sort_cols = ["sigma_p95", "S_p95", "n_obs"]
    summary = summary.sort_values(sort_cols, ascending=[False, False, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Wrote per-HKL summary: {output_path}")
    print()
    return summary


def write_resolution_summary(df: pd.DataFrame, output_path: Path, n_shells: int) -> pd.DataFrame | None:
    if "d_angstrom" not in df:
        print("No d-spacing/resolution column was found, so resolution-shell analysis was skipped.")
        print()
        return None

    with_d = df.dropna(subset=["d_angstrom"]).copy()
    with_d = with_d.loc[with_d["d_angstrom"] > 0.0].copy()
    if with_d.empty:
        print("D-spacing/resolution values were present but not usable, so resolution-shell analysis was skipped.")
        print()
        return None

    shell_count = max(int(n_shells), 1)
    with_d["resolution_shell"] = pd.qcut(
        with_d["d_angstrom"],
        q=shell_count,
        duplicates="drop",
    )
    summary = quantile_agg(with_d, ["resolution_shell"])
    summary["resolution_shell"] = summary["resolution_shell"].astype(str)
    summary = summary.sort_values("d_median", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Wrote resolution-shell summary: {output_path}")
    print()
    return summary


def ranked_columns(summary: pd.DataFrame) -> list[str]:
    cols = ["H", "K", "L", "n_obs"]
    for optional in ("d_mean", "d_median"):
        if optional in summary:
            cols.append(optional)
    cols.extend(
        [
            "sigma_median",
            "sigma_p95",
            "sigma_p99",
            "sigma_max",
            "sigma_p95_over_median",
            "frac_sigma_ge_5",
            "frac_sigma_ge_10",
            "S_median",
            "S_p95",
        ]
    )
    return cols


def print_ranked(title: str, table: pd.DataFrame, columns: list[str], top: int) -> None:
    print(title)
    if table.empty:
        print("  no rows")
        print()
        return
    print(table.loc[:, columns].head(top).to_string(index=False))
    print()


def print_hkl_rankings(summary: pd.DataFrame, top: int, min_obs: int) -> None:
    ranked = summary.loc[summary["n_obs"] >= min_obs].copy()
    cols = ranked_columns(summary)
    print(f"Printed HKL rankings are restricted to n_obs >= {min_obs}.")
    print()
    print_ranked(
        "HKLs with highest sigma_p95:",
        ranked.sort_values(["sigma_p95", "n_obs"], ascending=[False, False]),
        cols,
        top,
    )
    print_ranked(
        "HKLs with largest sigma_p95 / sigma_median:",
        ranked.sort_values(["sigma_p95_over_median", "sigma_p95"], ascending=[False, False]),
        cols,
        top,
    )
    print_ranked(
        "Orientation-specific candidates: sigma_median < 1.5 but sigma_p95 >= 5:",
        ranked.loc[(ranked["sigma_median"] < 1.5) & (ranked["sigma_p95"] >= 5.0)].sort_values(
            ["sigma_p95_over_median", "sigma_p95"], ascending=[False, False]
        ),
        cols,
        top,
    )
    print_ranked(
        "Globally high candidates: high sigma_median:",
        ranked.sort_values(["sigma_median", "sigma_p95"], ascending=[False, False]),
        cols,
        top,
    )


def print_resolution_ranking(summary: pd.DataFrame | None, top: int) -> None:
    if summary is None:
        return
    columns = [
        "resolution_shell",
        "n_obs",
        "d_min",
        "d_median",
        "d_max",
        "sigma_median",
        "sigma_p95",
        "sigma_p99",
        "sigma_max",
        "frac_sigma_ge_5",
        "frac_sigma_ge_10",
        "S_median",
        "S_p95",
    ]
    print_ranked(
        "Resolution shells ranked by sigma_p95:",
        summary.sort_values("sigma_p95", ascending=False),
        columns,
        top,
    )


def main() -> None:
    args = parse_args()
    columns = read_columns(args.scores)
    detected = detect_columns(columns)
    print_column_inspection(columns, detected)

    if args.inspect_only:
        return

    df = load_slim_scores(args.scores, detected, args.chunksize)
    hkl_summary = write_hkl_summary(df, args.hkl_output)
    resolution_summary = write_resolution_summary(df, args.resolution_output, args.resolution_shells)
    print_hkl_rankings(hkl_summary, args.top, args.min_ranked_observations)
    print_resolution_ranking(resolution_summary, args.top)
    print("Interpretation:")
    print("  high sigma_median: the HKL is usually risky across orientations.")
    print("  low sigma_median with high sigma_p95/max: only a subset of orientations is risky.")
    print("  high frac_sigma_ge_N: many observations cross that sigma inflation threshold.")
    print("  large sigma_p95_over_median: strong orientation-dependent spread for that HKL.")


if __name__ == "__main__":
    main()

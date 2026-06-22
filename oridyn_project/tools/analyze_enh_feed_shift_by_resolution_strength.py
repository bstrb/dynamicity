#!/usr/bin/env python3
"""Analyze enhancement-feed shifts by resolution-local reflection strength.

This diagnostic enriches signed-HKL enhancement-feed shift rows with d-spacing,
resolution bins, and strength tertiles computed within each resolution bin.
It does not modify streams, HKL files, or intensities.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


HKL_COLUMNS = ["h", "k", "l"]
RESOLUTION_BINS = [
    (20.0, 4.0, "20-4A"),
    (4.0, 3.0, "4-3A"),
    (3.0, 2.0, "3-2A"),
    (2.0, 1.5, "2-1.5A"),
    (1.5, 1.09, "1.5-1.09A"),
    (1.09, 0.86, "1.09-0.86A"),
    (0.86, 0.75, "0.86-0.75A"),
    (0.75, 0.68, "0.75-0.68A"),
    (0.68, 0.60, "0.68-0.60A"),
    (0.60, 0.50, "0.60-0.50A"),
    (0.50, 0.40, "0.50-0.40A"),
]
BIN_ORDER = [label for _d_high, _d_low, label in RESOLUTION_BINS] + ["outside"]
STRENGTH_ORDER = ["weak", "middle", "strong", "too_few"]
TOP_ROWS = 100
SUMMARY_TOP_ROWS = 20
MIN_HKLS_FOR_TERTILES = 9

STREAM_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
STREAM_CELL_RE = re.compile(
    rf"^\s*Cell parameters\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+nm,"
    rf"\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+deg"
)
STREAM_UNITCELL_LENGTH_RE = re.compile(rf"^\s*([abc])\s*=\s*({STREAM_FLOAT})\s*A")
STREAM_UNITCELL_ANGLE_RE = re.compile(rf"^\s*(al|be|ga|alpha|beta|gamma)\s*=\s*({STREAM_FLOAT})\s*deg")

REQUIRED_SHIFT_COLUMNS = [
    *HKL_COLUMNS,
    "Iweight_low_median",
    "Iweight_high_median",
    "relative_shift_Iweight",
    "relative_shift_I",
    "relative_shift_weight",
    "n_low",
    "n_high",
]
ENRICHED_EXTRA_COLUMNS = [
    "d_angstrom",
    "inv_nm",
    "resolution_bin",
    "local_strength_in_bin",
]
SUMMARY_COLUMNS = [
    "resolution_bin",
    "local_strength_in_bin",
    "n_hkl",
    "median_Iweight_low",
    "median_relative_shift_Iweight",
    "mean_relative_shift_Iweight",
    "fraction_positive_Iweight",
    "p25_relative_shift_Iweight",
    "p75_relative_shift_Iweight",
    "median_relative_shift_I",
    "mean_relative_shift_I",
    "fraction_positive_I",
    "median_relative_shift_weight",
    "fraction_positive_weight",
    "median_n_low",
    "median_n_high",
]
PIVOT_COLUMNS = [
    "resolution_bin",
    "weak_median_shift_Iweight",
    "middle_median_shift_Iweight",
    "strong_median_shift_Iweight",
    "weak_fraction_positive_Iweight",
    "middle_fraction_positive_Iweight",
    "strong_fraction_positive_Iweight",
    "weak_n_hkl",
    "middle_n_hkl",
    "strong_n_hkl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shift-by-hkl-csv",
        required=True,
        type=Path,
        help="Path to enh_feed_shift_by_signed_hkl_with_strength_class.csv.",
    )
    parser.add_argument(
        "--stream",
        required=True,
        type=Path,
        help="Original CrystFEL stream. Used only to parse the unit cell.",
    )
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory for diagnostic tables.")
    args = parser.parse_args()
    if not args.shift_by_hkl_csv.exists():
        raise SystemExit(f"--shift-by-hkl-csv not found: {args.shift_by_hkl_csv}")
    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {missing}")


def load_shift_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    require_columns(table, REQUIRED_SHIFT_COLUMNS, "shift-by-HKL CSV")
    out = table.copy()

    if "local_strength_class" in out.columns:
        if "global_strength_class" not in out.columns:
            out["global_strength_class"] = out["local_strength_class"]
        out = out.drop(columns=["local_strength_class"])

    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    bad_hkl = out[HKL_COLUMNS].isna().any(axis=1)
    if bool(bad_hkl.any()):
        log(f"Dropping {int(bad_hkl.sum()):,} rows with non-numeric HKLs")
    out = out.loc[~bad_hkl].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")

    for column in [
        "Iweight_low_median",
        "Iweight_high_median",
        "relative_shift_Iweight",
        "relative_shift_I",
        "relative_shift_weight",
        "n_low",
        "n_high",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def load_unit_cell_from_stream(stream_path: Path) -> dict[str, float]:
    """Parse a stream-level or first-crystal unit cell in Angstrom/degrees."""
    cell: dict[str, float] = {}
    in_unit_cell = False
    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if "Begin unit cell" in raw_line:
                in_unit_cell = True
                continue
            if in_unit_cell and "End unit cell" in raw_line:
                in_unit_cell = False
                if has_complete_cell(cell):
                    return cell
                continue

            if match := STREAM_CELL_RE.match(raw_line):
                a_nm, b_nm, c_nm, alpha, beta, gamma = (float(match.group(i)) for i in range(1, 7))
                return {
                    "a": 10.0 * a_nm,
                    "b": 10.0 * b_nm,
                    "c": 10.0 * c_nm,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                }

            if in_unit_cell or not line.startswith("----- Begin chunk -----"):
                if match := STREAM_UNITCELL_LENGTH_RE.match(raw_line):
                    cell[match.group(1).lower()] = float(match.group(2))
                    if has_complete_cell(cell):
                        return cell
                    continue
                if match := STREAM_UNITCELL_ANGLE_RE.match(raw_line):
                    key = {"al": "alpha", "be": "beta", "ga": "gamma"}.get(
                        match.group(1).lower(),
                        match.group(1).lower(),
                    )
                    cell[key] = float(match.group(2))
                    if has_complete_cell(cell):
                        return cell
                    continue

    missing = [key for key in ["a", "b", "c", "alpha", "beta", "gamma"] if key not in cell]
    raise SystemExit(f"Could not parse unit cell from stream; missing {missing}")


def has_complete_cell(cell: dict[str, float]) -> bool:
    return all(key in cell for key in ["a", "b", "c", "alpha", "beta", "gamma"])


def reciprocal_basis_from_cell(cell: dict[str, float]) -> np.ndarray:
    """Return reciprocal basis matrix with columns a*, b*, c* in 1/Angstrom."""
    a = float(cell["a"])
    b = float(cell["b"])
    c = float(cell["c"])
    alpha = math.radians(float(cell["alpha"]))
    beta = math.radians(float(cell["beta"]))
    gamma = math.radians(float(cell["gamma"]))

    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    if abs(sin_g) < 1e-12:
        raise SystemExit("Invalid unit cell: sin(gamma) is too small")

    a_vec = np.array([a, 0.0, 0.0], dtype=float)
    b_vec = np.array([b * cos_g, b * sin_g, 0.0], dtype=float)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0.0:
        raise SystemExit("Invalid unit cell: computed c_z^2 <= 0")
    c_vec = np.array([cx, cy, math.sqrt(cz_sq)], dtype=float)
    volume = float(np.dot(a_vec, np.cross(b_vec, c_vec)))
    if abs(volume) < 1e-15:
        raise SystemExit("Invalid unit cell: near-zero volume")

    a_star = np.cross(b_vec, c_vec) / volume
    b_star = np.cross(c_vec, a_vec) / volume
    c_star = np.cross(a_vec, b_vec) / volume
    return np.column_stack([a_star, b_star, c_star])


def d_spacing_angstrom(h: int, k: int, l: int, reciprocal_basis: np.ndarray) -> float:
    hkl = np.array([int(h), int(k), int(l)], dtype=float)
    g_vec = np.asarray(reciprocal_basis, dtype=float) @ hkl
    g_norm = float(np.linalg.norm(g_vec))
    if not np.isfinite(g_norm) or g_norm <= 0.0:
        return np.nan
    return 1.0 / g_norm


def resolution_bin_for_d(d_angstrom: float) -> str:
    if not np.isfinite(d_angstrom):
        return "outside"
    for idx, (d_high, d_low, label) in enumerate(RESOLUTION_BINS):
        if idx == 0:
            if d_low <= d_angstrom <= d_high:
                return label
        elif idx == len(RESOLUTION_BINS) - 1:
            if d_low <= d_angstrom < d_high:
                return label
        elif d_low <= d_angstrom < d_high:
            return label
    return "outside"


def enrich_with_resolution(table: pd.DataFrame, cell: dict[str, float]) -> pd.DataFrame:
    out = table.copy()
    basis = reciprocal_basis_from_cell(cell)
    d_values = [
        d_spacing_angstrom(int(row.h), int(row.k), int(row.l), basis)
        for row in out[HKL_COLUMNS].itertuples(index=False)
    ]
    out["d_angstrom"] = d_values
    d_numeric = pd.to_numeric(out["d_angstrom"], errors="coerce").to_numpy(dtype=float)
    out["inv_nm"] = np.where(d_numeric > 0.0, 10.0 / d_numeric, np.nan)
    out["resolution_bin"] = [resolution_bin_for_d(float(value)) for value in out["d_angstrom"]]
    return out


def classify_strength_within_resolution_bins(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["local_strength_in_bin"] = "too_few"
    for resolution_bin, group in out.groupby("resolution_bin", sort=False):
        finite = pd.to_numeric(group["Iweight_low_median"], errors="coerce").map(np.isfinite)
        finite_index = group.loc[finite].index
        if len(finite_index) < MIN_HKLS_FOR_TERTILES:
            out.loc[group.index, "local_strength_in_bin"] = "too_few"
            continue
        ordered = out.loc[finite_index, "Iweight_low_median"].sort_values(kind="mergesort").index
        class_ids = np.minimum((np.arange(len(ordered)) * 3) // len(ordered), 2)
        labels = [STRENGTH_ORDER[int(class_id)] for class_id in class_ids]
        out.loc[ordered, "local_strength_in_bin"] = labels
        nonfinite_index = group.index.difference(finite_index)
        out.loc[nonfinite_index, "local_strength_in_bin"] = "too_few"
    return out


def median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def quantile_or_nan(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def fraction_positive(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0.0).mean()) if not values.empty else np.nan


def summarize_resolution_strength(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (resolution_bin, strength), group in enriched.groupby(["resolution_bin", "local_strength_in_bin"], sort=False):
        rows.append(
            {
                "resolution_bin": str(resolution_bin),
                "local_strength_in_bin": str(strength),
                "n_hkl": int(len(group)),
                "median_Iweight_low": median_or_nan(group["Iweight_low_median"]),
                "median_relative_shift_Iweight": median_or_nan(group["relative_shift_Iweight"]),
                "mean_relative_shift_Iweight": mean_or_nan(group["relative_shift_Iweight"]),
                "fraction_positive_Iweight": fraction_positive(group["relative_shift_Iweight"]),
                "p25_relative_shift_Iweight": quantile_or_nan(group["relative_shift_Iweight"], 0.25),
                "p75_relative_shift_Iweight": quantile_or_nan(group["relative_shift_Iweight"], 0.75),
                "median_relative_shift_I": median_or_nan(group["relative_shift_I"]),
                "mean_relative_shift_I": mean_or_nan(group["relative_shift_I"]),
                "fraction_positive_I": fraction_positive(group["relative_shift_I"]),
                "median_relative_shift_weight": median_or_nan(group["relative_shift_weight"]),
                "fraction_positive_weight": fraction_positive(group["relative_shift_weight"]),
                "median_n_low": median_or_nan(group["n_low"]),
                "median_n_high": median_or_nan(group["n_high"]),
            }
        )
    summary = pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)
    return sort_resolution_strength(summary)


def sort_resolution_strength(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["_bin_order"] = out["resolution_bin"].map({label: idx for idx, label in enumerate(BIN_ORDER)}).fillna(len(BIN_ORDER))
    out["_strength_order"] = out["local_strength_in_bin"].map({label: idx for idx, label in enumerate(STRENGTH_ORDER)}).fillna(len(STRENGTH_ORDER))
    out = out.sort_values(["_bin_order", "_strength_order"]).drop(columns=["_bin_order", "_strength_order"])
    return out.reset_index(drop=True)


def make_pivot(summary: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    present_bins = [label for label in BIN_ORDER if label in set(enriched["resolution_bin"].astype(str))]
    for resolution_bin in present_bins:
        row: dict[str, Any] = {"resolution_bin": resolution_bin}
        for strength in ["weak", "middle", "strong"]:
            group = summary.loc[
                (summary["resolution_bin"] == resolution_bin)
                & (summary["local_strength_in_bin"] == strength)
            ]
            prefix = strength
            if group.empty:
                row[f"{prefix}_median_shift_Iweight"] = np.nan
                row[f"{prefix}_fraction_positive_Iweight"] = np.nan
                row[f"{prefix}_n_hkl"] = 0
            else:
                first = group.iloc[0]
                row[f"{prefix}_median_shift_Iweight"] = float(first["median_relative_shift_Iweight"])
                row[f"{prefix}_fraction_positive_Iweight"] = float(first["fraction_positive_Iweight"])
                row[f"{prefix}_n_hkl"] = int(first["n_hkl"])
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=PIVOT_COLUMNS)


def top_examples(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weak_positive = (
        enriched.loc[enriched["local_strength_in_bin"] == "weak"]
        .sort_values("relative_shift_Iweight", ascending=False)
        .head(TOP_ROWS)
        .copy()
    )
    strong_negative = (
        enriched.loc[enriched["local_strength_in_bin"] == "strong"]
        .sort_values("relative_shift_Iweight", ascending=True)
        .head(TOP_ROWS)
        .copy()
    )
    return weak_positive, strong_negative


def markdown_table(table: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 20) -> str:
    if table is None or table.empty:
        return "_No rows._"
    if columns is None:
        columns = table.columns.tolist()
    view = table.loc[:, [column for column in columns if column in table.columns]].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def interpretation_from_pivot(pivot: pd.DataFrame) -> dict[str, list[str]]:
    if pivot.empty:
        return {
            "median_pattern_bins": [],
            "fraction_supported_bins": [],
            "pattern_not_hold_bins": [],
        }

    classified = pivot.loc[(pivot["weak_n_hkl"] > 0) & (pivot["strong_n_hkl"] > 0)].copy()
    median_pattern = classified.loc[
        (classified["weak_median_shift_Iweight"] > 0.0)
        & (classified["strong_median_shift_Iweight"] < 0.0),
        "resolution_bin",
    ].astype(str).tolist()
    fraction_supported = classified.loc[
        (classified["weak_fraction_positive_Iweight"] > 0.5)
        & (classified["strong_fraction_positive_Iweight"] < 0.5),
        "resolution_bin",
    ].astype(str).tolist()
    pattern_not_hold = classified.loc[
        ~(
            (classified["weak_median_shift_Iweight"] > 0.0)
            & (classified["strong_median_shift_Iweight"] < 0.0)
        ),
        "resolution_bin",
    ].astype(str).tolist()
    return {
        "median_pattern_bins": median_pattern,
        "fraction_supported_bins": fraction_supported,
        "pattern_not_hold_bins": pattern_not_hold,
    }


def write_summary(
    outdir: Path,
    args: argparse.Namespace,
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
    pivot: pd.DataFrame,
    weak_positive: pd.DataFrame,
    strong_negative: pd.DataFrame,
) -> None:
    interp = interpretation_from_pivot(pivot)
    inside = int((enriched["resolution_bin"] != "outside").sum())
    outside = int((enriched["resolution_bin"] == "outside").sum())
    top_columns = [
        "h",
        "k",
        "l",
        "resolution_bin",
        "local_strength_in_bin",
        "d_angstrom",
        "inv_nm",
        "relative_shift_Iweight",
        "relative_shift_I",
        "relative_shift_weight",
        "Iweight_low_median",
        "Iweight_high_median",
        "n_low",
        "n_high",
    ]
    if "global_strength_class" in enriched.columns:
        top_columns.append("global_strength_class")

    lines = [
        "# Enhancement-Feed Shift By Resolution-Local Strength",
        "",
        "## Inputs",
        "",
        f"- Shift-by-HKL CSV: `{args.shift_by_hkl_csv}`",
        f"- Stream: `{args.stream}`",
        f"- Output directory: `{args.outdir}`",
        "",
        "## Counts",
        "",
        f"- Total signed HKLs: {len(enriched)}",
        f"- HKLs inside listed resolution bins: {inside}",
        f"- HKLs outside listed resolution bins: {outside}",
        "",
        "## Method",
        "",
        "- `d_angstrom` and `inv_nm = 10 / d_angstrom` are computed from the stream unit cell and signed HKL.",
        "- Strength is classified within each resolution bin using tertiles of `Iweight_low_median`.",
        "- This avoids global weak/strong labels being confounded with resolution.",
        f"- Bins with fewer than {MIN_HKLS_FOR_TERTILES} finite `Iweight_low_median` HKLs are labeled `too_few`.",
        "- Existing global `local_strength_class`, if present, is retained only as `global_strength_class`.",
        "",
        "## Resolution-Strength Summary",
        "",
        markdown_table(summary, SUMMARY_COLUMNS, max_rows=len(summary)),
        "",
        "## Pivot Summary",
        "",
        markdown_table(pivot, PIVOT_COLUMNS, max_rows=len(pivot)),
        "",
        "## Top Locally Weak Positive Shifts",
        "",
        markdown_table(weak_positive, top_columns, max_rows=SUMMARY_TOP_ROWS),
        "",
        "## Top Locally Strong Negative Shifts",
        "",
        markdown_table(strong_negative, top_columns, max_rows=SUMMARY_TOP_ROWS),
        "",
        "## Automatic Interpretation",
        "",
        "- Bins where weak median `relative_shift_Iweight` is positive and strong median is negative: "
        + format_list(interp["median_pattern_bins"]),
        "- Bins where the weak-positive / strong-negative pattern is also supported by positive-fraction contrast "
        "`weak_fraction_positive_Iweight > 0.5` and `strong_fraction_positive_Iweight < 0.5`: "
        + format_list(interp["fraction_supported_bins"]),
        "- Bins where the median weak-positive / strong-negative pattern does not hold: "
        + format_list(interp["pattern_not_hold_bins"]),
        "",
        "## Output Files",
        "",
        "- `enh_feed_shift_by_hkl_resolution_strength.csv`",
        "- `enh_feed_shift_resolution_strength_summary.csv`",
        "- `enh_feed_shift_resolution_strength_pivot.csv`",
        "- `top_resolution_local_weak_positive_shifts.csv`",
        "- `top_resolution_local_strong_negative_shifts.csv`",
        "- `summary.md`",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    log("Loading signed-HKL shift table")
    shifts = load_shift_table(args.shift_by_hkl_csv)
    log(f"Loaded {len(shifts):,} signed HKLs")

    log("Parsing stream unit cell")
    cell = load_unit_cell_from_stream(args.stream)
    log(
        "Parsed unit cell: "
        f"a={cell['a']:.6g} A, b={cell['b']:.6g} A, c={cell['c']:.6g} A, "
        f"alpha={cell['alpha']:.6g}, beta={cell['beta']:.6g}, gamma={cell['gamma']:.6g}"
    )

    log("Computing d-spacings and resolution bins")
    enriched = enrich_with_resolution(shifts, cell)
    log("Classifying local strength within resolution bins")
    enriched = classify_strength_within_resolution_bins(enriched)

    log("Building summary tables")
    summary = summarize_resolution_strength(enriched)
    pivot = make_pivot(summary, enriched)
    weak_positive, strong_negative = top_examples(enriched)

    enriched.to_csv(args.outdir / "enh_feed_shift_by_hkl_resolution_strength.csv", index=False)
    summary.to_csv(args.outdir / "enh_feed_shift_resolution_strength_summary.csv", index=False)
    pivot.to_csv(args.outdir / "enh_feed_shift_resolution_strength_pivot.csv", index=False)
    weak_positive.to_csv(args.outdir / "top_resolution_local_weak_positive_shifts.csv", index=False)
    strong_negative.to_csv(args.outdir / "top_resolution_local_strong_negative_shifts.csv", index=False)
    write_summary(args.outdir, args, enriched, summary, pivot, weak_positive, strong_negative)

    print("Resolution-strength pivot:")
    print(markdown_table(pivot, PIVOT_COLUMNS, max_rows=len(pivot)))
    print(f"Wrote: {args.outdir / 'enh_feed_shift_by_hkl_resolution_strength.csv'}")
    print(f"Wrote: {args.outdir / 'enh_feed_shift_resolution_strength_summary.csv'}")
    print(f"Wrote: {args.outdir / 'enh_feed_shift_resolution_strength_pivot.csv'}")
    print(f"Wrote: {args.outdir / 'top_resolution_local_weak_positive_shifts.csv'}")
    print(f"Wrote: {args.outdir / 'top_resolution_local_strong_negative_shifts.csv'}")
    print(f"Wrote: {args.outdir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

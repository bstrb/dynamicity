#!/usr/bin/env python3
"""Diagnose v5 excitation-imbalance tendency within selected signed HKLs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_same_hkl_low_high_orientation_examples import (  # noqa: E402
    KEY_COLUMNS,
    HKL_COLUMNS,
    load_stream_orientation_for_selected,
    normalize_key_columns,
)
from oridyn.axis_prediction import unique_zone_axes  # noqa: E402
from oridyn.geometry import axis_angle_deg, beam_in_direct_coordinates, triplet_label  # noqa: E402
from oridyn.stream_parser import STREAM_MATRIX_COLUMNS, reciprocal_matrix_from_row  # noqa: E402


BASE = Path("/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524")
DEFAULT_STREAM = BASE / "MFM300-VIII_cut_20-0_3.stream"
DEFAULT_V5_SCORES = (
    BASE
    / "oridyn_v5_nonself_local_excitation_raw_20_0p3_20260705"
    / "geometry_coupling_v5_nonself_local_excitation_raw_scores.csv"
)
DEFAULT_ACCEPTED = (
    BASE
    / "oridyn_v4_local_crowding_raw_20_0p3_20260704"
    / "partialator_survivor_mask"
    / "p1_iter1_20260705T1214"
    / "v4_p1_iter1_partialator_survivors_only_scores.csv"
)
DEFAULT_STRENGTH_TABLE = BASE / "MFM300-VIII_cut_20-0_3_partialator_results_20260705T1214" / "crystfel.hkl"
DEFAULT_OUT_DIR = BASE / "oridyn_v5_excitation_imbalance_orientation_diagnostic_20_0p3_20260712"
DEFAULT_SCORE_COLUMN = "nonself_local_excitation_raw"
DEFAULT_COUPLING_COLUMN = "nonself_neighbor_count_effective"
DEFAULT_TARGET_COLUMN = "target_excitation_Eg"
DEFAULT_SG_COLUMN = "sg_target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--v5-scores", type=Path, default=DEFAULT_V5_SCORES)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--strength-table", type=Path, default=DEFAULT_STRENGTH_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--coupling-column", default=DEFAULT_COUPLING_COLUMN)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--sg-column", default=DEFAULT_SG_COLUMN)
    parser.add_argument("--min-accepted-obs", type=int, default=50)
    parser.add_argument("--hkls-per-class", type=int, default=4)
    parser.add_argument("--weak-quantile", type=float, default=0.25)
    parser.add_argument("--strong-quantile", type=float, default=0.75)
    parser.add_argument("--medium-half-width", type=float, default=0.05)
    parser.add_argument("--high-target-quantile", type=float, default=0.75)
    parser.add_argument("--near-zero-abs-quantile", type=float, default=0.20)
    parser.add_argument("--uvw-max", type=int, default=5)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    for label, path in [
        ("--stream", args.stream),
        ("--v5-scores", args.v5_scores),
        ("--accepted", args.accepted),
        ("--strength-table", args.strength_table),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if int(args.min_accepted_obs) < 1:
        raise SystemExit("--min-accepted-obs must be >= 1")
    if int(args.hkls_per_class) < 1:
        raise SystemExit("--hkls-per-class must be >= 1")
    if int(args.uvw_max) < 1:
        raise SystemExit("--uvw-max must be >= 1")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if not (0.0 < float(args.weak_quantile) < 1.0):
        raise SystemExit("--weak-quantile must satisfy 0 < q < 1")
    if not (0.0 < float(args.strong_quantile) < 1.0):
        raise SystemExit("--strong-quantile must satisfy 0 < q < 1")
    if float(args.weak_quantile) >= float(args.strong_quantile):
        raise SystemExit("--weak-quantile must be less than --strong-quantile")
    if not (0.0 < float(args.high_target_quantile) < 1.0):
        raise SystemExit("--high-target-quantile must satisfy 0 < q < 1")
    if not (0.0 < float(args.near_zero_abs_quantile) < 1.0):
        raise SystemExit("--near-zero-abs-quantile must satisfy 0 < q < 1")
    if not (0.0 < float(args.medium_half_width) < 0.5):
        raise SystemExit("--medium-half-width must satisfy 0 < value < 0.5")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def require_columns(header: list[str], required: list[str], label: str) -> None:
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def hkl_slug(h: int, k: int, l: int) -> str:
    def one(value: int) -> str:
        return f"m{abs(int(value))}" if int(value) < 0 else str(int(value))

    return f"hkl_{one(h)}_{one(k)}_{one(l)}"


def hkl_mask(table: pd.DataFrame, hkls: list[tuple[int, int, int]]) -> np.ndarray:
    wanted = pd.MultiIndex.from_tuples(hkls, names=HKL_COLUMNS)
    return pd.MultiIndex.from_frame(table.loc[:, HKL_COLUMNS]).isin(wanted)


def read_crystfel_hkl_strength(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("CrystFEL") or line.startswith("Symmetry:"):
                continue
            if line.startswith("End of reflections"):
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                intensity = float(parts[3])
                sigma = float(parts[5]) if parts[4] == "-" else float(parts[4])
                nmeas = int(parts[6]) if len(parts) > 6 and parts[4] == "-" else int(parts[5])
            except ValueError:
                continue
            rows.append(
                {
                    "h": h,
                    "k": k,
                    "l": l,
                    "merged_intensity_or_Fobs": intensity,
                    "merged_sigma": sigma,
                    "merged_nmeas": nmeas,
                    "strength_source_column": "CrystFEL_I",
                }
            )
    if not rows:
        raise SystemExit(f"No reflection rows could be parsed from strength table: {path}")
    return pd.DataFrame.from_records(rows).drop_duplicates(HKL_COLUMNS, keep="first")


def load_accepted_counts(path: Path, chunksize: int) -> pd.DataFrame:
    header = read_header(path)
    require_columns(header, KEY_COLUMNS, "accepted observation table")
    counts: dict[tuple[int, int, int], int] = {}
    rows_read = 0
    for idx, chunk in enumerate(pd.read_csv(path, usecols=KEY_COLUMNS, chunksize=int(chunksize)), start=1):
        rows_read += int(len(chunk))
        work = normalize_key_columns(chunk)
        grouped = work.groupby(HKL_COLUMNS, sort=False).size().reset_index(name="n")
        for row in grouped.itertuples(index=False):
            key = (int(row.h), int(row.k), int(row.l))
            counts[key] = counts.get(key, 0) + int(row.n)
        if idx == 1 or idx % 10 == 0:
            log(f"Accepted count scan: chunks={idx:,}, rows_read={rows_read:,}, signed_hkls={len(counts):,}")
    return pd.DataFrame.from_records(
        [{"h": h, "k": k, "l": l, "n_accepted": n} for (h, k, l), n in counts.items()]
    )


def choose_selected_hkls(strength: pd.DataFrame, counts: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = strength.merge(counts, on=HKL_COLUMNS, how="inner", validate="one_to_one")
    eligible = eligible.loc[
        (eligible["n_accepted"].to_numpy(dtype=np.int64) >= int(args.min_accepted_obs))
        & np.isfinite(pd.to_numeric(eligible["merged_intensity_or_Fobs"], errors="coerce").to_numpy(dtype=float))
        & (pd.to_numeric(eligible["merged_intensity_or_Fobs"], errors="coerce").to_numpy(dtype=float) > 0.0)
    ].copy()
    if eligible.empty:
        raise SystemExit("No positive merged-intensity signed HKLs passed --min-accepted-obs")
    eligible["strength_percentile"] = eligible["merged_intensity_or_Fobs"].rank(method="average", pct=True)

    n = int(args.hkls_per_class)
    weak_q = float(args.weak_quantile)
    strong_q = float(args.strong_quantile)
    medium_low = 0.5 - float(args.medium_half_width)
    medium_high = 0.5 + float(args.medium_half_width)

    weak = (
        eligible.loc[eligible["strength_percentile"] <= weak_q]
        .sort_values(["n_accepted", "merged_intensity_or_Fobs"], ascending=[False, True], kind="mergesort")
        .head(n)
        .copy()
    )
    medium_pool = eligible.loc[(eligible["strength_percentile"] >= medium_low) & (eligible["strength_percentile"] <= medium_high)].copy()
    medium_pool["_distance_to_median_percentile"] = (medium_pool["strength_percentile"] - 0.5).abs()
    medium = (
        medium_pool.sort_values(["_distance_to_median_percentile", "n_accepted"], ascending=[True, False], kind="mergesort")
        .head(n)
        .drop(columns=["_distance_to_median_percentile"], errors="ignore")
        .copy()
    )
    strong = (
        eligible.loc[eligible["strength_percentile"] >= strong_q]
        .sort_values(["n_accepted", "merged_intensity_or_Fobs"], ascending=[False, False], kind="mergesort")
        .head(n)
        .copy()
    )

    selected_parts = []
    for label, table in [("weak", weak), ("medium", medium), ("strong", strong)]:
        table = table.copy()
        table["reflection_strength_class"] = label
        selected_parts.append(table)
    selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates(HKL_COLUMNS, keep="first")
    if selected.empty:
        raise SystemExit("Representative HKL selection produced no rows")
    selected["selection_rank_within_class"] = selected.groupby("reflection_strength_class", sort=False).cumcount() + 1
    selected = selected.sort_values(["reflection_strength_class", "selection_rank_within_class"], kind="mergesort").reset_index(drop=True)

    meta = {
        "strength_source": str(args.strength_table),
        "strength_metric": "positive full-data merged CrystFEL intensity I; HKLs with I <= 0 excluded from strength quantiles",
        "min_accepted_obs": int(args.min_accepted_obs),
        "weak_quantile_max": weak_q,
        "medium_quantile_range": [medium_low, medium_high],
        "strong_quantile_min": strong_q,
        "eligible_positive_hkls": int(len(eligible)),
        "selected_hkls": selected.loc[:, [*HKL_COLUMNS, "reflection_strength_class", "merged_intensity_or_Fobs", "n_accepted"]].to_dict("records"),
    }
    return selected, meta


def load_selected_accepted(path: Path, hkls: list[tuple[int, int, int]], chunksize: int) -> pd.DataFrame:
    header = read_header(path)
    require_columns(header, KEY_COLUMNS, "accepted observation table")
    usecols = [*KEY_COLUMNS, *[column for column in ["partiality", "I_unmerged"] if column in header]]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        work = normalize_key_columns(chunk)
        work = work.loc[hkl_mask(work, hkls)].copy()
        if work.empty:
            continue
        for column in ["partiality", "I_unmerged"]:
            if column not in work.columns:
                work[column] = np.nan
            work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        chunks.append(work.loc[:, [*KEY_COLUMNS, "partiality", "I_unmerged"]].copy())
    return (
        pd.concat(chunks, ignore_index=True).drop_duplicates(KEY_COLUMNS, keep="first")
        if chunks
        else pd.DataFrame(columns=[*KEY_COLUMNS, "partiality", "I_unmerged"])
    )


def load_selected_v5(path: Path, hkls: list[tuple[int, int, int]], args: argparse.Namespace) -> pd.DataFrame:
    header = read_header(path)
    required = [*KEY_COLUMNS, args.score_column, args.coupling_column, args.target_column, args.sg_column]
    require_columns(header, required, "v5 score table")
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=required, chunksize=int(args.chunksize)):
        work = normalize_key_columns(chunk)
        work = work.loc[hkl_mask(work, hkls)].copy()
        if work.empty:
            continue
        for column in [args.score_column, args.coupling_column, args.target_column, args.sg_column]:
            work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        chunks.append(work.loc[:, required].copy())
    return (
        pd.concat(chunks, ignore_index=True).drop_duplicates(KEY_COLUMNS, keep="first")
        if chunks
        else pd.DataFrame(columns=required)
    )


def add_dexc_columns(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = table.copy()
    out["nonself_local_excitation_raw"] = pd.to_numeric(out[args.score_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["coupling_sum_raw"] = pd.to_numeric(out[args.coupling_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["target_excitation_Eg"] = pd.to_numeric(out[args.target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["sg_target"] = pd.to_numeric(out[args.sg_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["coupled_excitation_imbalance_raw"] = out["nonself_local_excitation_raw"] - out["target_excitation_Eg"] * out["coupling_sum_raw"]
    coupling = out["coupling_sum_raw"].to_numpy(dtype=float)
    raw = out["nonself_local_excitation_raw"].to_numpy(dtype=float)
    eg = out["target_excitation_Eg"].to_numpy(dtype=float)
    out["coupled_excitation_imbalance_norm"] = np.divide(
        raw,
        coupling,
        out=np.zeros_like(raw, dtype=float),
        where=np.isfinite(coupling) & (coupling != 0.0),
    ) - np.where(np.isfinite(coupling) & (coupling != 0.0), eg, 0.0)
    out.loc[~np.isfinite(out["coupled_excitation_imbalance_norm"]), "coupled_excitation_imbalance_norm"] = np.nan
    return out


def add_relative_intensity(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    med = out.groupby(HKL_COLUMNS, sort=False)["I_unmerged"].transform("median")
    out["hkl_median_I_unmerged"] = med
    denom = med.to_numpy(dtype=float)
    values = pd.to_numeric(out["I_unmerged"], errors="coerce").to_numpy(dtype=float)
    out["I_over_hkl_median"] = np.divide(
        values,
        denom,
        out=np.full_like(values, np.nan, dtype=float),
        where=np.isfinite(denom) & (np.abs(denom) > 1e-12),
    )
    return out


def continuous_uvw_text(values: np.ndarray) -> str:
    return f"[{values[0]:.6g} {values[1]:.6g} {values[2]:.6g}]"


def add_orientation_columns(table: pd.DataFrame, stream: Path, uvw_max: int) -> pd.DataFrame:
    log(f"Recovering exact orientation/UVW for {len(table):,} selected accepted observations")
    orientation = load_stream_orientation_for_selected(stream, table.loc[:, KEY_COLUMNS])
    if len(orientation) != len(table):
        found = set(tuple(row) for row in orientation.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))
        missing = [tuple(row) for row in table.loc[:, KEY_COLUMNS].itertuples(index=False, name=None) if tuple(row) not in found]
        raise SystemExit(f"Stream orientation lookup missed {len(missing)} exact key(s); first missing: {missing[:3]}")
    joined = table.merge(
        orientation.loc[:, [*KEY_COLUMNS, *STREAM_MATRIX_COLUMNS]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    axes = unique_zone_axes(int(uvw_max))
    continuous_xyz: list[np.ndarray] = []
    closest: list[str] = []
    angles: list[float] = []
    for _, row in joined.iterrows():
        reciprocal = reciprocal_matrix_from_row(row)
        beam_uvw = beam_in_direct_coordinates(reciprocal)
        best_axis = min(axes, key=lambda axis: axis_angle_deg(reciprocal, axis))
        continuous_xyz.append(beam_uvw)
        closest.append(triplet_label(best_axis))
        angles.append(float(axis_angle_deg(reciprocal, best_axis)))
    xyz = np.vstack(continuous_xyz) if continuous_xyz else np.empty((0, 3))
    joined["continuous_uvw_x"] = xyz[:, 0] if len(xyz) else []
    joined["continuous_uvw_y"] = xyz[:, 1] if len(xyz) else []
    joined["continuous_uvw_z"] = xyz[:, 2] if len(xyz) else []
    joined["continuous_uvw"] = [continuous_uvw_text(values) for values in continuous_xyz]
    joined["closest_uvw"] = closest
    joined["closest_uvw_angle_deg"] = angles
    return joined


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    b = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 3:
        return np.nan
    ar = a.loc[mask].rank(method="average")
    br = b.loc[mask].rank(method="average")
    return float(ar.corr(br))


def median_or_nan(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.median()) if len(finite) else np.nan


def compute_per_hkl_stats(table: pd.DataFrame, high_target_quantile: float, near_zero_abs_quantile: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hkl, group in table.groupby(HKL_COLUMNS, sort=False):
        h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        high_threshold = float(group["target_excitation_Eg"].quantile(float(high_target_quantile)))
        high = group.loc[group["target_excitation_Eg"] >= high_threshold].copy()
        abs_norm = group["coupled_excitation_imbalance_norm"].abs()
        near_threshold = float(abs_norm.quantile(float(near_zero_abs_quantile))) if abs_norm.notna().any() else np.nan
        if not np.isfinite(near_threshold):
            near_threshold = 0.0
        negative = group.loc[group["coupled_excitation_imbalance_norm"] < -near_threshold]
        near_zero = group.loc[group["coupled_excitation_imbalance_norm"].abs() <= near_threshold]
        positive = group.loc[group["coupled_excitation_imbalance_norm"] > near_threshold]
        rows.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "reflection_strength_class": str(group["reflection_strength_class"].iloc[0]),
                "merged_intensity_or_Fobs": float(group["merged_intensity_or_Fobs"].iloc[0]),
                "n_observations": int(len(group)),
                "high_target_quantile": float(high_target_quantile),
                "high_target_threshold": high_threshold,
                "n_high_target": int(len(high)),
                "near_zero_abs_dexc_norm_quantile": float(near_zero_abs_quantile),
                "near_zero_abs_dexc_norm_threshold": near_threshold,
                "spearman_Irel_vs_Dexc_raw": spearman_corr(group["I_over_hkl_median"], group["coupled_excitation_imbalance_raw"]),
                "spearman_Irel_vs_Dexc_norm": spearman_corr(group["I_over_hkl_median"], group["coupled_excitation_imbalance_norm"]),
                "spearman_Irel_vs_Dexc_raw_high_target": spearman_corr(
                    high["I_over_hkl_median"], high["coupled_excitation_imbalance_raw"]
                ),
                "spearman_Irel_vs_Dexc_norm_high_target": spearman_corr(
                    high["I_over_hkl_median"], high["coupled_excitation_imbalance_norm"]
                ),
                "n_negative_Dexc_norm": int(len(negative)),
                "n_near_zero_Dexc_norm": int(len(near_zero)),
                "n_positive_Dexc_norm": int(len(positive)),
                "median_Irel_negative_Dexc_norm": median_or_nan(negative["I_over_hkl_median"]),
                "median_Irel_near_zero_Dexc_norm": median_or_nan(near_zero["I_over_hkl_median"]),
                "median_Irel_positive_Dexc_norm": median_or_nan(positive["I_over_hkl_median"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_orientation_extremes(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _hkl, group in table.groupby(HKL_COLUMNS, sort=False):
        candidates = [
            ("most_negative_Dexc_norm", group["coupled_excitation_imbalance_norm"].idxmin()),
            ("near_zero_Dexc_norm", group["coupled_excitation_imbalance_norm"].abs().idxmin()),
            ("most_positive_Dexc_norm", group["coupled_excitation_imbalance_norm"].idxmax()),
        ]
        for label, idx in candidates:
            row = group.loc[idx].copy()
            row["orientation_case"] = label
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def plot_per_hkl(table: pd.DataFrame, plot_dir: Path) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    sign_colors = {"negative": "#2f6fbb", "zero": "#7a7a7a", "positive": "#c53b3b"}
    for (h, k, l), group in table.groupby(HKL_COLUMNS, sort=False):
        strength_class = str(group["reflection_strength_class"].iloc[0])
        stem = f"{strength_class}_{hkl_slug(int(h), int(k), int(l))}"

        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        sc = ax.scatter(
            group["coupled_excitation_imbalance_norm"],
            group["I_over_hkl_median"],
            c=group["target_excitation_Eg"],
            s=14,
            alpha=0.75,
            cmap="viridis",
            edgecolors="none",
        )
        ax.axvline(0.0, color="0.55", lw=0.8)
        ax.axhline(1.0, color="0.55", lw=0.8)
        ax.set_xlabel("coupled_excitation_imbalance_norm")
        ax.set_ylabel("I_unmerged / median(I_unmerged for signed HKL)")
        ax.set_title(f"{strength_class} signed HKL ({h}, {k}, {l})")
        fig.colorbar(sc, ax=ax, label="target_excitation_Eg")
        path = plot_dir / f"{stem}_Irel_vs_Dexc_norm_by_Eg.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        signs = np.where(group["coupled_excitation_imbalance_norm"] > 0, "positive", "negative")
        signs = np.where(group["coupled_excitation_imbalance_norm"] == 0, "zero", signs)
        for sign in ["negative", "zero", "positive"]:
            sub = group.loc[signs == sign]
            if sub.empty:
                continue
            ax.scatter(
                sub["nonself_local_excitation_raw"],
                sub["I_over_hkl_median"],
                s=14,
                alpha=0.75,
                label=sign,
                color=sign_colors[sign],
                edgecolors="none",
            )
        ax.axhline(1.0, color="0.55", lw=0.8)
        ax.set_xlabel("nonself_local_excitation_raw")
        ax.set_ylabel("I_unmerged / median(I_unmerged for signed HKL)")
        ax.set_title(f"{strength_class} signed HKL ({h}, {k}, {l})")
        ax.legend(title="sign(Dexc norm)", frameon=False)
        path = plot_dir / f"{stem}_Irel_vs_v5_by_Dexc_sign.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    strength_meta: dict[str, Any],
    selected: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    lines = [
        "# V5 Excitation-Imbalance Orientation Diagnostic",
        "",
        "This is an observation-level diagnostic only. It does not filter streams, run Partialator, merge, or refine.",
        "",
        "Dexc is an excitation-imbalance tendency, not a proven gain/loss score.",
        "",
        "## Formulas",
        "",
        f"- `coupling_sum_raw` is read from the existing v5 `{args.coupling_column}` column, which is the v5 neighbour-kernel sum `sum C(g-q)`.",
        f"- `coupled_excitation_imbalance_raw = {args.score_column} - target_excitation_Eg * coupling_sum_raw`.",
        f"- `coupled_excitation_imbalance_norm = {args.score_column} / coupling_sum_raw - target_excitation_Eg`.",
        "- `coupled_excitation_imbalance_norm` is set to 0 when `coupling_sum_raw` is 0.",
        "",
        "## Strength Definition",
        "",
        f"- source: `{strength_meta['strength_source']}`",
        f"- metric: {strength_meta['strength_metric']}",
        f"- minimum accepted observations per signed HKL: {strength_meta['min_accepted_obs']}",
        f"- weak: percentile <= {strength_meta['weak_quantile_max']}",
        f"- medium: percentile in {strength_meta['medium_quantile_range']}",
        f"- strong: percentile >= {strength_meta['strong_quantile_min']}",
        "",
        "## Selected HKLs",
        "",
        "| class | h | k | l | merged I/Fobs | n accepted | percentile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.reflection_strength_class} | {int(row.h)} | {int(row.k)} | {int(row.l)} | "
            f"{float(row.merged_intensity_or_Fobs):.6g} | {int(row.n_accepted)} | {float(row.strength_percentile):.4f} |"
        )
    lines.extend(["", "## Outputs", ""])
    for name in [
        "selected_hkl_strengths.csv",
        "diagnostic_observations.csv",
        "per_hkl_dexc_intensity_stats.csv",
        "orientation_extremes.csv",
        "run_metadata.json",
    ]:
        lines.append(f"- `{out_dir / name}`")
    lines.append(f"- plots: `{out_dir / 'plots'}` ({len(plot_paths)} PNG files)")
    (out_dir / "README_v5_excitation_imbalance_orientation_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {args.out_dir}")

    log("Loading full-data merged intensity table for strength classes")
    strength = read_crystfel_hkl_strength(args.strength_table)
    log(f"Loaded merged strengths: signed_hkls={len(strength):,}")

    log("Counting accepted observations per signed HKL")
    counts = load_accepted_counts(args.accepted, int(args.chunksize))
    selected, strength_meta = choose_selected_hkls(strength, counts, args)
    selected.to_csv(args.out_dir / "selected_hkl_strengths.csv", index=False)
    selected_hkls = [(int(row.h), int(row.k), int(row.l)) for row in selected.itertuples(index=False)]
    log(f"Selected {len(selected_hkls)} signed HKLs: {selected_hkls}")

    log("Loading accepted observation rows for selected HKLs")
    accepted = load_selected_accepted(args.accepted, selected_hkls, int(args.chunksize))
    log(f"Selected accepted rows: {len(accepted):,}")

    log("Loading existing v5 rows for selected accepted HKLs")
    v5 = load_selected_v5(args.v5_scores, selected_hkls, args)
    joined = v5.merge(accepted, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if joined.empty:
        raise SystemExit("No selected accepted observations matched existing v5 rows")
    joined = joined.merge(
        selected.loc[:, [*HKL_COLUMNS, "reflection_strength_class", "merged_intensity_or_Fobs", "strength_percentile"]],
        on=HKL_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    joined = add_dexc_columns(joined, args)
    joined = add_relative_intensity(joined)
    joined = add_orientation_columns(joined, args.stream, int(args.uvw_max))

    output_columns = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "reflection_strength_class",
        "merged_intensity_or_Fobs",
        "I_unmerged",
        "I_over_hkl_median",
        "partiality",
        "target_excitation_Eg",
        "sg_target",
        "nonself_local_excitation_raw",
        "coupling_sum_raw",
        "coupled_excitation_imbalance_raw",
        "coupled_excitation_imbalance_norm",
        "continuous_uvw_x",
        "continuous_uvw_y",
        "continuous_uvw_z",
        "closest_uvw",
        "closest_uvw_angle_deg",
    ]
    observations = joined.loc[:, output_columns].copy()
    observations.to_csv(args.out_dir / "diagnostic_observations.csv", index=False)

    stats = compute_per_hkl_stats(observations, float(args.high_target_quantile), float(args.near_zero_abs_quantile))
    stats.to_csv(args.out_dir / "per_hkl_dexc_intensity_stats.csv", index=False)

    extremes = build_orientation_extremes(observations)
    extreme_columns = [
        "orientation_case",
        *output_columns,
        "continuous_uvw",
    ]
    extremes.loc[:, [column for column in extreme_columns if column in extremes.columns]].to_csv(
        args.out_dir / "orientation_extremes.csv", index=False
    )

    log("Writing per-HKL diagnostic plots")
    plot_paths = plot_per_hkl(observations, args.out_dir / "plots")

    metadata = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stream": str(args.stream),
            "v5_scores": str(args.v5_scores),
            "accepted": str(args.accepted),
            "strength_table": str(args.strength_table),
        },
        "output_dir": str(args.out_dir),
        "strength_definition": strength_meta,
        "dexc_note": "Dexc is only an excitation-imbalance tendency; it is not treated as a proven gain/loss score.",
        "rows": {
            "selected_hkls": int(len(selected)),
            "diagnostic_observations": int(len(observations)),
            "per_hkl_stats": int(len(stats)),
            "orientation_extremes": int(len(extremes)),
            "plots": int(len(plot_paths)),
        },
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(args.out_dir, args, strength_meta, selected, plot_paths)

    print(f"Wrote: {args.out_dir / 'selected_hkl_strengths.csv'}")
    print(f"Wrote: {args.out_dir / 'diagnostic_observations.csv'}")
    print(f"Wrote: {args.out_dir / 'per_hkl_dexc_intensity_stats.csv'}")
    print(f"Wrote: {args.out_dir / 'orientation_extremes.csv'}")
    print(f"Wrote: {args.out_dir / 'README_v5_excitation_imbalance_orientation_diagnostic.md'}")
    print(f"Wrote plots: {args.out_dir / 'plots'}")
    print(stats.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

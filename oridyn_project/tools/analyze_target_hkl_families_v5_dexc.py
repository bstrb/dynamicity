#!/usr/bin/env python3
"""Targeted v5 excitation-deficit diagnostic for selected HKL families.

This script does not run the broad 75-HKL selector. It reads the same upstream
accepted-observation, v5-score, stream-orientation, and merged-intensity inputs
used by the completed broad v2 diagnostic, then restricts analysis to requested
4/mmm reflection families.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_and_crossvalidate_v5_dexc_correction_candidates as cvmod  # noqa: E402
from diagnose_v5_excitation_imbalance_orientation_broad_hkl_v2 import (  # noqa: E402
    DEFAULT_ACCEPTED,
    DEFAULT_CHUNKSIZE,
    DEFAULT_COUPLING_COLUMN,
    DEFAULT_SCORE_COLUMN,
    DEFAULT_SG_COLUMN,
    DEFAULT_STREAM,
    DEFAULT_STRENGTH_TABLE,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_V5_SCORES,
    HKL_COLUMNS,
    KEY_COLUMNS,
    add_dexc_columns,
    add_intensity_responses,
    add_orientation_columns,
    add_reciprocal_descriptors,
    bootstrap_slope_ci,
    choose_intensity_columns,
    four_mmm_orbit_id,
    four_mmm_orbit_variants,
    hkl_geometry_class,
    load_accepted_v5_join,
    normalize_key_columns,
    parse_stream_unit_cell,
    partial_spearman,
    read_crystfel_hkl_strength,
    read_header,
    reciprocal_matrix_from_cell,
    require_columns,
    robust_slope,
    spearman_with_p,
    tertile_labels,
)


BASE = Path("/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524")
DEFAULT_OUT_DIR = BASE / "oridyn_v5_dexc_target_hkl_families_20_0p3_20260712"

ORIGINAL_RESPONSE = "corrected_intensity_residual"
CV_RESPONSE_SOURCE = "observation_intensity"
DEFICIT_COLUMN = "excitation_deficit_norm"
TARGET_COLUMN = "target_excitation_Eg"
PARTIALITY_COLUMN = "partiality"
COUPLING_COLUMN = "coupling_sum_raw"
V5_COLUMN = "nonself_local_excitation_raw"

HH0_SEEDS = [(index, index, 0) for index in range(1, 12)]
SUSPECTED_GAIN_SEEDS = [(2, 2, 4), (6, 0, 1), (5, 0, 3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--v5-scores", type=Path, default=DEFAULT_V5_SCORES)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--strength-table", type=Path, default=DEFAULT_STRENGTH_TABLE)
    parser.add_argument("--model-bias-table", type=Path, default=None, help="Optional existing Fobs/Fcalc or refinement-residual table.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--coupling-column", default=DEFAULT_COUPLING_COLUMN)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--sg-column", default=DEFAULT_SG_COLUMN)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--uvw-max", type=int, default=5)
    parser.add_argument("--min-train-obs", type=int, default=50)
    parser.add_argument("--min-test-obs", type=int, default=30)
    parser.add_argument("--min-transfer-obs", type=int, default=50)
    parser.add_argument("--stable-sign-fraction", type=float, default=0.80)
    parser.add_argument("--plot-dpi", type=int, default=170)
    args = parser.parse_args()

    for label, path in [
        ("--stream", args.stream),
        ("--v5-scores", args.v5_scores),
        ("--accepted", args.accepted),
        ("--strength-table", args.strength_table),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if args.model_bias_table is not None and not args.model_bias_table.is_file():
        raise SystemExit(f"--model-bias-table not found: {args.model_bias_table}")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if int(args.bootstrap_iterations) < 0:
        raise SystemExit("--bootstrap-iterations must be >= 0")
    if int(args.splits) < 1:
        raise SystemExit("--splits must be >= 1")
    if not (0.50 <= float(args.train_fraction) < 1.0):
        raise SystemExit("--train-fraction must satisfy 0.50 <= value < 1.0")
    if int(args.min_train_obs) < 5 or int(args.min_test_obs) < 3 or int(args.min_transfer_obs) < 3:
        raise SystemExit("Minimum observation thresholds are too small")
    if not (0.0 <= float(args.stable_sign_fraction) <= 1.0):
        raise SystemExit("--stable-sign-fraction must be in [0, 1]")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def hkl_label(hkl: tuple[int, int, int]) -> str:
    return f"({hkl[0]},{hkl[1]},{hkl[2]})"


def hkl_slug(hkl: tuple[int, int, int]) -> str:
    def one(value: int) -> str:
        return f"m{abs(int(value))}" if int(value) < 0 else str(int(value))

    return f"{one(hkl[0])}_{one(hkl[1])}_{one(hkl[2])}"


def sign_value(value: float, eps: float = 1e-12) -> int:
    if not np.isfinite(value) or abs(float(value)) <= eps:
        return 0
    return 1 if float(value) > 0.0 else -1


def first_finite(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.iloc[0]) if len(finite) else np.nan


def safe_int_count(value: Any, default: int = 0) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").replace([np.inf, -np.inf], np.nan).iloc[0]
    return int(numeric) if np.isfinite(numeric) else int(default)


def quantile_stats(series: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {f"{prefix}_q10": np.nan, f"{prefix}_median": np.nan, f"{prefix}_q90": np.nan}
    q10_value, median_value, q90_value = np.quantile(values.to_numpy(dtype=float), [0.10, 0.50, 0.90])
    return {f"{prefix}_q10": float(q10_value), f"{prefix}_median": float(median_value), f"{prefix}_q90": float(q90_value)}


def build_target_hkl_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed_hkl in HH0_SEEDS:
        for variant in sorted(four_mmm_orbit_variants(*seed_hkl)):
            rows.append(
                {
                    "h": int(variant[0]),
                    "k": int(variant[1]),
                    "l": int(variant[2]),
                    "target_family": "hh0_depletion_suspected",
                    "target_orbit_id": f"hh0_{abs(seed_hkl[0]):02d}",
                    "target_seed_hkl": hkl_label(seed_hkl),
                    "target_seed_label": f"{seed_hkl[0]}{seed_hkl[1]}0",
                }
            )
    for seed_hkl in SUSPECTED_GAIN_SEEDS:
        seed_label = "".join(str(value) for value in seed_hkl)
        for variant in sorted(four_mmm_orbit_variants(*seed_hkl)):
            rows.append(
                {
                    "h": int(variant[0]),
                    "k": int(variant[1]),
                    "l": int(variant[2]),
                    "target_family": "suspected_gain_reflection",
                    "target_orbit_id": f"gain_{seed_label}",
                    "target_seed_hkl": hkl_label(seed_hkl),
                    "target_seed_label": seed_label,
                }
            )
    table = pd.DataFrame.from_records(rows).drop_duplicates(HKL_COLUMNS + ["target_orbit_id"]).reset_index(drop=True)
    table["hkl"] = [hkl_label((int(row.h), int(row.k), int(row.l))) for row in table.itertuples(index=False)]
    table["four_mmm_orbit_id"] = [four_mmm_orbit_id(int(row.h), int(row.k), int(row.l)) for row in table.itertuples(index=False)]
    table["hkl_geometry_class"] = [hkl_geometry_class(int(row.h), int(row.k), int(row.l)) for row in table.itertuples(index=False)]
    return table.sort_values(["target_family", "target_seed_label", "h", "k", "l"]).reset_index(drop=True)


def hkl_membership_mask(table: pd.DataFrame, hkls: set[tuple[int, int, int]]) -> pd.Series:
    keys = list(zip(table["h"].astype(int), table["k"].astype(int), table["l"].astype(int), strict=False))
    return pd.Series([key in hkls for key in keys], index=table.index)


def load_target_accepted_table(path: Path, target_hkls: set[tuple[int, int, int]], chunksize: int) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    header = read_header(path)
    require_columns(header, KEY_COLUMNS, "accepted survivor table")
    intensity_choice = choose_intensity_columns(header)
    optional_columns = ["partiality", "I_unmerged", intensity_choice.column, *intensity_choice.scale_columns]
    if intensity_choice.sigma_column is not None:
        optional_columns.append(intensity_choice.sigma_column)
    optional_columns = list(dict.fromkeys(column for column in optional_columns if column in header))
    usecols = [*KEY_COLUMNS, *optional_columns]
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_retained = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)), start=1):
        rows_read += int(len(chunk))
        work = normalize_key_columns(chunk)
        work = work.loc[hkl_membership_mask(work, target_hkls)].copy()
        rows_retained += int(len(work))
        if work.empty:
            continue
        work["observation_intensity"] = pd.to_numeric(work[intensity_choice.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work["I_unmerged"] = pd.to_numeric(work["I_unmerged"], errors="coerce").replace([np.inf, -np.inf], np.nan) if "I_unmerged" in work.columns else np.nan
        work["sigma"] = pd.to_numeric(work[intensity_choice.sigma_column], errors="coerce").replace([np.inf, -np.inf], np.nan) if intensity_choice.sigma_column is not None else np.nan
        work["partiality"] = pd.to_numeric(work["partiality"], errors="coerce").replace([np.inf, -np.inf], np.nan) if "partiality" in work.columns else np.nan
        for scale_column in intensity_choice.scale_columns:
            work[scale_column] = pd.to_numeric(work[scale_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        chunks.append(work.loc[:, [*KEY_COLUMNS, "observation_intensity", "I_unmerged", "sigma", "partiality", *intensity_choice.scale_columns]].copy())
        if chunk_index == 1 or chunk_index % 10 == 0:
            log(f"Accepted survivor scan: chunks={chunk_index:,}, rows_read={rows_read:,}, target_rows={rows_retained:,}")
    if chunks:
        table = pd.concat(chunks, ignore_index=True)
    else:
        table = pd.DataFrame(columns=[*KEY_COLUMNS, "observation_intensity", "I_unmerged", "sigma", "partiality", *intensity_choice.scale_columns])
    duplicate_mask = table.duplicated(KEY_COLUMNS, keep=False) if not table.empty else pd.Series(dtype=bool)
    duplicate_rows = int(duplicate_mask.sum()) if len(duplicate_mask) else 0
    duplicate_keys = int(table.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        log("Warning: duplicate exact keys in target accepted survivor table; keeping first")
        table = table.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)
    metadata = {
        "rows_read": int(rows_read),
        "target_rows_retained_before_dedup": int(rows_retained),
        "rows_after_key_cleanup": int(len(table)),
        "duplicate_rows": duplicate_rows,
        "duplicate_keys": duplicate_keys,
        "intensity_column_used": intensity_choice.column,
        "sigma_column_used": intensity_choice.sigma_column,
        "scale_columns_available": list(intensity_choice.scale_columns),
        "intensity_choice_note": intensity_choice.note,
    }
    return table, intensity_choice, metadata


def build_v5_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        v5_scores=args.v5_scores,
        score_column=args.score_column,
        coupling_column=args.coupling_column,
        target_column=args.target_column,
        sg_column=args.sg_column,
        chunksize=args.chunksize,
    )


def prepare_target_observations(args: argparse.Namespace, target_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target_hkls = set(tuple(map(int, row)) for row in target_table.loc[:, HKL_COLUMNS].itertuples(index=False, name=None))
    accepted, intensity_choice, accepted_metadata = load_target_accepted_table(args.accepted, target_hkls, int(args.chunksize))
    if accepted.empty:
        log("No accepted survivor observations were found for target HKLs")
        return pd.DataFrame(), target_table.copy(), {"accepted_table": accepted_metadata}

    v5_args = build_v5_args(args)
    accepted_v5, v5_metadata = load_accepted_v5_join(accepted, v5_args)
    accepted_v5 = add_dexc_columns(accepted_v5, v5_args)

    strength, strength_metadata = read_crystfel_hkl_strength(args.strength_table)
    cell = parse_stream_unit_cell(args.stream)
    reciprocal = reciprocal_matrix_from_cell(cell)
    target_meta = target_table.merge(strength, on=HKL_COLUMNS, how="left", validate="many_to_one")
    target_meta = add_reciprocal_descriptors(target_meta, reciprocal)
    target_meta["resolution_angstrom"] = target_meta["d_spacing_angstrom"]

    metadata_columns = [
        *HKL_COLUMNS,
        "hkl",
        "target_family",
        "target_orbit_id",
        "target_seed_hkl",
        "target_seed_label",
        "four_mmm_orbit_id",
        "merged_intensity_or_Fobs",
        "d_spacing_angstrom",
        "resolution_angstrom",
        "g_norm_invA",
        "reciprocal_direction_class",
        "angle_to_cstar_deg",
        "hkl_geometry_class",
        "symmetry_orbit_id",
    ]
    observations = accepted_v5.merge(target_meta.loc[:, [column for column in metadata_columns if column in target_meta.columns]], on=HKL_COLUMNS, how="left", validate="many_to_one")
    observations = add_intensity_responses(observations, target_meta, tuple(intensity_choice.scale_columns))
    observations = add_orientation_columns(observations, args.stream, int(args.uvw_max))
    metadata = {
        "accepted_table": accepted_metadata,
        "v5_join": v5_metadata,
        "strength": strength_metadata,
        "unit_cell": cell.__dict__,
    }
    return observations, target_meta, metadata


def quintile_medians(group: pd.DataFrame, response_column: str) -> dict[str, Any]:
    nonzero = group.loc[pd.to_numeric(group[COUPLING_COLUMN], errors="coerce") > 0.0].copy()
    if len(nonzero) < 2 or nonzero[DEFICIT_COLUMN].nunique(dropna=True) < 2:
        return {"deficit_quintile_medians": "", **{f"deficit_quintile_{index}_median_{response_column}": np.nan for index in range(1, 6)}}
    labels = pd.qcut(nonzero[DEFICIT_COLUMN].rank(method="first"), q=min(5, len(nonzero)), labels=False, duplicates="drop")
    work = nonzero.assign(_deficit_quintile=np.asarray(labels, dtype=int))
    medians = work.groupby("_deficit_quintile", sort=True)[response_column].median()
    out = {"deficit_quintile_medians": ";".join(f"{float(value):.6g}" for value in medians)}
    for index in range(1, 6):
        out[f"deficit_quintile_{index}_median_{response_column}"] = float(medians.get(index - 1, np.nan))
    return out


def summarize_signed_hkl(group: pd.DataFrame, meta_row: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    hkl = (int(meta_row.h), int(meta_row.k), int(meta_row.l))
    nonzero = group.loc[pd.to_numeric(group.get(COUPLING_COLUMN, pd.Series(dtype=float)), errors="coerce") > 0.0].copy() if not group.empty else pd.DataFrame()
    zero = group.loc[pd.to_numeric(group.get(COUPLING_COLUMN, pd.Series(dtype=float)), errors="coerce") <= 0.0].copy() if not group.empty else pd.DataFrame()
    row: dict[str, Any] = {
        "h": hkl[0],
        "k": hkl[1],
        "l": hkl[2],
        "hkl": hkl_label(hkl),
        "target_family": meta_row.get("target_family", ""),
        "target_orbit_id": meta_row.get("target_orbit_id", ""),
        "target_seed_hkl": meta_row.get("target_seed_hkl", ""),
        "four_mmm_orbit_id": meta_row.get("four_mmm_orbit_id", ""),
        "accepted_observation_count": int(len(group)),
        "nonzero_coupling_count": int(len(nonzero)),
        "zero_coupling_count": int(len(zero)),
        "merged_intensity_or_Fobs": meta_row.get("merged_intensity_or_Fobs", np.nan),
        "resolution_angstrom": meta_row.get("resolution_angstrom", meta_row.get("d_spacing_angstrom", np.nan)),
        "g_norm_invA": meta_row.get("g_norm_invA", np.nan),
        "reciprocal_direction_class": meta_row.get("reciprocal_direction_class", ""),
        "hkl_geometry_class": meta_row.get("hkl_geometry_class", ""),
    }
    if group.empty:
        row["data_status"] = "insufficient data"
        return row
    row.update(quantile_stats(group[TARGET_COLUMN], "target_excitation"))
    row.update(quantile_stats(group[PARTIALITY_COLUMN], "partiality"))
    row.update(quantile_stats(group[V5_COLUMN], "v5"))
    row.update(quantile_stats(nonzero[DEFICIT_COLUMN], "excitation_deficit"))
    row["excitation_deficit_spread_q90_q10"] = row.get("excitation_deficit_q90", np.nan) - row.get("excitation_deficit_q10", np.nan)
    row["median_I_over_hkl_median_in_sample"] = float(pd.to_numeric(group["I_over_hkl_median"], errors="coerce").median())
    row["median_corrected_intensity_residual_in_sample"] = float(pd.to_numeric(group[ORIGINAL_RESPONSE], errors="coerce").median())

    if len(nonzero) >= 3:
        controls = nonzero.loc[:, [TARGET_COLUMN, PARTIALITY_COLUMN]]
        row["original_in_sample_robust_slope"] = robust_slope(nonzero[DEFICIT_COLUMN], nonzero[ORIGINAL_RESPONSE])
        ci_low, ci_high = bootstrap_slope_ci(nonzero[DEFICIT_COLUMN], nonzero[ORIGINAL_RESPONSE], int(args.bootstrap_iterations), int(args.seed) + len(nonzero))
        row["bootstrap_ci95_low"] = ci_low
        row["bootstrap_ci95_high"] = ci_high
        row["spearman_rho"] = spearman_with_p(nonzero[ORIGINAL_RESPONSE], nonzero[DEFICIT_COLUMN])[0]
        row["partial_spearman_rho_ctrl_Eg_partiality"] = partial_spearman(nonzero[ORIGINAL_RESPONSE], nonzero[DEFICIT_COLUMN], controls)
        high_target = nonzero.loc[nonzero[TARGET_COLUMN] >= nonzero[TARGET_COLUMN].quantile(0.75)]
        high_partiality = nonzero.loc[nonzero[PARTIALITY_COLUMN] >= nonzero[PARTIALITY_COLUMN].quantile(0.75)]
        row["high_target_subset_n"] = int(len(high_target))
        row["high_target_subset_slope"] = robust_slope(high_target[DEFICIT_COLUMN], high_target[ORIGINAL_RESPONSE])
        row["high_partiality_subset_n"] = int(len(high_partiality))
        row["high_partiality_subset_slope"] = robust_slope(high_partiality[DEFICIT_COLUMN], high_partiality[ORIGINAL_RESPONSE])
        labels = tertile_labels(nonzero[V5_COLUMN])
        for label in ["low", "medium", "high"]:
            subset = nonzero.loc[labels == label]
            output_label = "mid" if label == "medium" else label
            row[f"v5_{output_label}_tertile_n"] = int(len(subset))
            row[f"v5_{output_label}_tertile_slope"] = robust_slope(subset[DEFICIT_COLUMN], subset[ORIGINAL_RESPONSE])
    else:
        row["original_in_sample_robust_slope"] = np.nan
        row["bootstrap_ci95_low"] = np.nan
        row["bootstrap_ci95_high"] = np.nan
        row["spearman_rho"] = np.nan
        row["partial_spearman_rho_ctrl_Eg_partiality"] = np.nan
    row.update(quintile_medians(group, ORIGINAL_RESPONSE))
    row["data_status"] = "ok" if len(nonzero) >= 3 else "insufficient data"
    return row


def build_signed_hkl_summary(observations: pd.DataFrame, target_meta: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    grouped = {tuple(map(int, key)): group.copy() for key, group in observations.groupby(HKL_COLUMNS, sort=False)} if not observations.empty else {}
    rows = []
    for _, meta_row in target_meta.iterrows():
        hkl = (int(meta_row.h), int(meta_row.k), int(meta_row.l))
        rows.append(summarize_signed_hkl(grouped.get(hkl, pd.DataFrame()), meta_row, args))
    return pd.DataFrame.from_records(rows)


def run_crossvalidation_for_hkls(observations: pd.DataFrame, target_meta: pd.DataFrame, scale_columns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    covariates = [TARGET_COLUMN, PARTIALITY_COLUMN, *scale_columns]
    for _, meta_row in target_meta.iterrows():
        hkl = (int(meta_row.h), int(meta_row.k), int(meta_row.l))
        group = observations.loc[(observations["h"].astype(int) == hkl[0]) & (observations["k"].astype(int) == hkl[1]) & (observations["l"].astype(int) == hkl[2])] if not observations.empty else pd.DataFrame()
        frame = cvmod.clean_model_frame(group, cvmod.CV_RESPONSE_SOURCE_COLUMN, covariates) if not group.empty else pd.DataFrame()
        if len(frame) < int(args.min_train_obs) + int(args.min_test_obs):
            rows.append(
                {
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "hkl": hkl_label(hkl),
                    "target_family": meta_row.get("target_family", ""),
                    "target_orbit_id": meta_row.get("target_orbit_id", ""),
                    "split": -1,
                    "status": "insufficient_data",
                    "n_usable_nonzero_coupling": int(len(frame)),
                }
            )
            continue
        for split_index in range(int(args.splits)):
            train_index, test_index = cvmod.deterministic_stratified_split(frame, split_index, float(args.train_fraction), int(args.seed))
            if len(train_index) < int(args.min_train_obs) or len(test_index) < int(args.min_test_obs):
                rows.append(
                    {
                        "h": hkl[0],
                        "k": hkl[1],
                        "l": hkl[2],
                        "hkl": hkl_label(hkl),
                        "target_family": meta_row.get("target_family", ""),
                        "target_orbit_id": meta_row.get("target_orbit_id", ""),
                        "split": split_index,
                        "status": "split_too_small",
                        "n_usable_nonzero_coupling": int(len(frame)),
                        "n_train": int(len(train_index)),
                        "n_test": int(len(test_index)),
                    }
                )
                continue
            train = frame.iloc[train_index].copy()
            test = frame.iloc[test_index].copy()
            train, test, normalization_denominator, normalization_note = cvmod.add_train_normalized_response(train, test)
            if not np.isfinite(normalization_denominator):
                rows.append({"h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "target_family": meta_row.get("target_family", ""), "target_orbit_id": meta_row.get("target_orbit_id", ""), "split": split_index, "status": "normalization_failed", "n_usable_nonzero_coupling": int(len(frame)), "n_train": int(len(train_index)), "n_test": int(len(test_index))})
                continue
            baseline_fit = cvmod.fit_model(train, cvmod.CV_RESPONSE_COLUMN, covariates)
            train_prediction = cvmod.predict_model(baseline_fit, train)
            test_prediction = cvmod.predict_model(baseline_fit, test)
            response_train = pd.to_numeric(train[cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
            response_test = pd.to_numeric(test[cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
            baseline_residual_train = response_train - train_prediction
            baseline_residual_test = response_test - test_prediction
            beta, reference_deficit = cvmod.fit_deficit_slope_from_training_residuals(train, baseline_residual_train)
            deficit_test = pd.to_numeric(test[DEFICIT_COLUMN], errors="coerce").to_numpy(dtype=float)
            corrected_residual_test = baseline_residual_test - beta * (deficit_test - reference_deficit)
            sham_beta = cvmod.permuted_sham_beta(train, baseline_residual_train, beta, int(args.seed) + 1009 * (split_index + 1) + 31 * sum(abs(value) for value in hkl))
            sham_residual_test = baseline_residual_test - sham_beta * (deficit_test - reference_deficit)
            before = cvmod.residual_metrics(response_test, baseline_residual_test, deficit_test)
            after = cvmod.residual_metrics(response_test, corrected_residual_test, deficit_test)
            sham = cvmod.residual_metrics(response_test, sham_residual_test, deficit_test)
            rows.append(
                {
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "hkl": hkl_label(hkl),
                    "target_family": meta_row.get("target_family", ""),
                    "target_orbit_id": meta_row.get("target_orbit_id", ""),
                    "split": split_index,
                    "status": "ok",
                    "n_usable_nonzero_coupling": int(len(frame)),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "cv_response_column": cvmod.CV_RESPONSE_COLUMN,
                    "cv_response_source_column": cvmod.CV_RESPONSE_SOURCE_COLUMN,
                    "response_normalization_source": normalization_note,
                    "response_normalization_denominator_train": normalization_denominator,
                    "baseline_predictors": ";".join(covariates),
                    "test_data_used_for_normalization_or_fitting": False,
                    "beta_excitation_deficit_train": beta,
                    "reference_deficit_train": reference_deficit,
                    "heldout_baseline_mae": before["mae"],
                    "heldout_corrected_mae": after["mae"],
                    "heldout_delta_mae_positive_improves": before["mae"] - after["mae"],
                    "heldout_baseline_median_abs_residual": before["median_abs_residual"],
                    "heldout_corrected_median_abs_residual": after["median_abs_residual"],
                    "heldout_delta_median_abs_residual_positive_improves": before["median_abs_residual"] - after["median_abs_residual"],
                    "heldout_residual_deficit_rho_before": before["residual_spearman_vs_deficit"],
                    "heldout_residual_deficit_rho_after": after["residual_spearman_vs_deficit"],
                    "heldout_abs_rho_reduction_positive_improves": abs(before["residual_spearman_vs_deficit"]) - abs(after["residual_spearman_vs_deficit"]),
                    "sham_beta_same_magnitude_permuted_train_deficit": sham_beta,
                    "sham_heldout_delta_mae_positive_improves": before["mae"] - sham["mae"],
                    "correction_beats_sham_mae": bool((before["mae"] - after["mae"]) > (before["mae"] - sham["mae"])),
                }
            )
    out = pd.DataFrame.from_records(rows)
    return add_crossvalidation_aggregates(out) if not out.empty else out


def add_crossvalidation_aggregates(crossvalidation: pd.DataFrame) -> pd.DataFrame:
    out = crossvalidation.copy()
    ok = out.get("status", pd.Series(dtype=object)) == "ok"
    for hkl, group in out.loc[ok].groupby(HKL_COLUMNS, sort=False):
        slopes = pd.to_numeric(group["beta_excitation_deficit_train"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(slopes):
            median_slope = float(slopes.median())
            median_sign = sign_value(median_slope)
            sign_stability = float((slopes.apply(sign_value) == median_sign).mean()) if median_sign else np.nan
        else:
            median_slope = np.nan
            median_sign = 0
            sign_stability = np.nan
        idx = group.index
        out.loc[idx, "median_heldout_deficit_slope"] = median_slope
        out.loc[idx, "heldout_deficit_slope_sign"] = "positive" if median_sign > 0 else "negative" if median_sign < 0 else "zero_or_nan"
        out.loc[idx, "slope_sign_stability"] = sign_stability
        out.loc[idx, "median_heldout_delta_mae"] = float(pd.to_numeric(group["heldout_delta_mae_positive_improves"], errors="coerce").median())
        out.loc[idx, "median_heldout_rho_before"] = float(pd.to_numeric(group["heldout_residual_deficit_rho_before"], errors="coerce").median())
        out.loc[idx, "median_heldout_rho_after"] = float(pd.to_numeric(group["heldout_residual_deficit_rho_after"], errors="coerce").median())
        out.loc[idx, "fraction_splits_beating_sham_mae"] = float(group["correction_beats_sham_mae"].astype(bool).mean())
        out.loc[idx, "n_ok_splits"] = int(len(group))
    return out


def aggregate_crossvalidation(crossvalidation: pd.DataFrame) -> pd.DataFrame:
    if crossvalidation.empty or "status" not in crossvalidation.columns:
        return pd.DataFrame()
    ok = crossvalidation.loc[crossvalidation["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for hkl, group in ok.groupby(HKL_COLUMNS, sort=False):
        hkl_tuple = tuple(map(int, hkl))
        rows.append(
            {
                "h": hkl_tuple[0],
                "k": hkl_tuple[1],
                "l": hkl_tuple[2],
                "hkl": hkl_label(hkl_tuple),
                "n_ok_splits": int(len(group)),
                "median_heldout_deficit_slope": first_finite(group["median_heldout_deficit_slope"]),
                "heldout_deficit_slope_sign": str(group["heldout_deficit_slope_sign"].dropna().iloc[0]) if group["heldout_deficit_slope_sign"].dropna().any() else "zero_or_nan",
                "slope_sign_stability": first_finite(group["slope_sign_stability"]),
                "median_heldout_delta_mae": first_finite(group["median_heldout_delta_mae"]),
                "median_heldout_rho_before": first_finite(group["median_heldout_rho_before"]),
                "median_heldout_rho_after": first_finite(group["median_heldout_rho_after"]),
                "fraction_splits_beating_sham_mae": first_finite(group["fraction_splits_beating_sham_mae"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def classify_orientation_trends(summary: pd.DataFrame, cv_summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    cv_columns = [
        *HKL_COLUMNS,
        "hkl",
        "n_ok_splits",
        "median_heldout_deficit_slope",
        "heldout_deficit_slope_sign",
        "slope_sign_stability",
        "median_heldout_delta_mae",
        "median_heldout_rho_before",
        "median_heldout_rho_after",
        "fraction_splits_beating_sham_mae",
    ]
    if cv_summary.empty:
        cv_summary = pd.DataFrame(columns=cv_columns)
    out = summary.merge(cv_summary.loc[:, [column for column in cv_columns if column in cv_summary.columns]], on=HKL_COLUMNS + ["hkl"], how="left", validate="one_to_one")
    classifications: list[str] = []
    for _, row in out.iterrows():
        n_ok_splits = safe_int_count(row.get("n_ok_splits", 0), default=0)
        if str(row.get("data_status", "")) != "ok" or n_ok_splits < max(3, int(args.splits) // 2):
            classifications.append("insufficient data")
            continue
        stability = float(row.get("slope_sign_stability", np.nan))
        slope = float(row.get("median_heldout_deficit_slope", np.nan))
        if not np.isfinite(stability) or not np.isfinite(slope) or stability < float(args.stable_sign_fraction):
            classifications.append("no stable trend")
        elif slope < 0.0:
            classifications.append("depletion-like")
        elif slope > 0.0:
            classifications.append("enhancement-like")
        else:
            classifications.append("no stable trend")
    out["orientation_trend_classification"] = classifications
    out["classification_scope_note"] = "within-HKL orientation dependence; not automatically the same as merged Fobs/Fcalc bias"
    return out


def run_symmetry_transfer(observations: pd.DataFrame, target_meta: pd.DataFrame, scale_columns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    covariates = [TARGET_COLUMN, PARTIALITY_COLUMN, *scale_columns]
    for orbit_id, orbit_rows in target_meta.groupby("target_orbit_id", sort=False):
        frames: dict[tuple[int, int, int], pd.DataFrame] = {}
        for _, meta_row in orbit_rows.iterrows():
            hkl = (int(meta_row.h), int(meta_row.k), int(meta_row.l))
            group = observations.loc[(observations["h"].astype(int) == hkl[0]) & (observations["k"].astype(int) == hkl[1]) & (observations["l"].astype(int) == hkl[2])] if not observations.empty else pd.DataFrame()
            frames[hkl] = cvmod.clean_model_frame(group, cvmod.CV_RESPONSE_SOURCE_COLUMN, covariates) if not group.empty else pd.DataFrame()
        hkls = sorted(frames)
        for train_hkl in hkls:
            for test_hkl in hkls:
                if train_hkl == test_hkl:
                    continue
                train_frame = frames[train_hkl]
                test_frame = frames[test_hkl]
                if len(train_frame) < int(args.min_transfer_obs) or len(test_frame) < int(args.min_transfer_obs):
                    rows.append(
                        {
                            "target_orbit_id": orbit_id,
                            "train_hkl": hkl_label(train_hkl),
                            "test_hkl": hkl_label(test_hkl),
                            "status": "insufficient_data",
                            "n_train": int(len(train_frame)),
                            "n_test": int(len(test_frame)),
                        }
                    )
                    continue
                result = cvmod.transfer_one_direction(str(orbit_id), train_hkl, test_hkl, train_frame, test_frame, covariates)
                result["target_orbit_id"] = orbit_id
                rows.append(result)
    return pd.DataFrame.from_records(rows)


def candidate_model_bias_paths(root: Path) -> list[Path]:
    patterns = ["*fcalc*.csv", "*Fcalc*.csv", "*fobs*.csv", "*Fobs*.csv", "*refinement*residual*.csv", "*fofc*.csv", "*FOFC*.csv"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return sorted(set(path for path in paths if path.is_file()))


def choose_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {column.lower().replace("_", ""): column for column in columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "")
        if key in lookup:
            return lookup[key]
    return None


def load_model_bias_table(path: Path | None, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    chosen_path = path
    auto_candidates: list[str] = []
    if chosen_path is None:
        candidates = candidate_model_bias_paths(root)
        auto_candidates = [str(candidate) for candidate in candidates[:20]]
        chosen_path = candidates[0] if candidates else None
    if chosen_path is None:
        return pd.DataFrame(), {"model_bias_table": None, "auto_candidates_checked": auto_candidates, "note": "No existing Fobs/Fcalc or refinement-residual table found automatically."}
    try:
        if chosen_path.suffix.lower() == ".csv":
            table = pd.read_csv(chosen_path)
        else:
            table = pd.read_csv(chosen_path, sep=r"\s+|,", engine="python", comment="#")
    except Exception as exc:
        return pd.DataFrame(), {"model_bias_table": str(chosen_path), "error": str(exc)}
    h_col = choose_column(table.columns, ["h", "H"])
    k_col = choose_column(table.columns, ["k", "K"])
    l_col = choose_column(table.columns, ["l", "L"])
    if h_col is None or k_col is None or l_col is None:
        return pd.DataFrame(), {"model_bias_table": str(chosen_path), "note": "Table has no recognizable h,k,l columns."}
    out = pd.DataFrame(
        {
            "h": pd.to_numeric(table[h_col], errors="coerce"),
            "k": pd.to_numeric(table[k_col], errors="coerce"),
            "l": pd.to_numeric(table[l_col], errors="coerce"),
        }
    ).dropna()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype(int)
    fobs_col = choose_column(table.columns, ["Fobs", "F_obs", "FOBS", "Fo", "F_o"])
    fcalc_col = choose_column(table.columns, ["Fcalc", "F_calc", "FCALC", "Fc", "F_c"])
    residual_col = choose_column(table.columns, ["Fobs_minus_Fcalc", "FoFc", "FOFC", "refinement_residual", "residual", "deltaF"])
    if fobs_col is not None:
        out["Fobs"] = pd.to_numeric(table.loc[out.index, fobs_col], errors="coerce").to_numpy(dtype=float)
    if fcalc_col is not None:
        out["Fcalc"] = pd.to_numeric(table.loc[out.index, fcalc_col], errors="coerce").to_numpy(dtype=float)
    if residual_col is not None:
        out["model_residual"] = pd.to_numeric(table.loc[out.index, residual_col], errors="coerce").to_numpy(dtype=float)
    if "Fobs" in out.columns and "Fcalc" in out.columns:
        fcalc = out["Fcalc"].to_numpy(dtype=float)
        fobs = out["Fobs"].to_numpy(dtype=float)
        out["Fobs_over_Fcalc"] = np.divide(fobs, fcalc, out=np.full_like(fobs, np.nan), where=np.isfinite(fcalc) & (np.abs(fcalc) > 1e-12))
        out["Fobs_minus_Fcalc"] = fobs - fcalc
        out["merged_model_bias_metric"] = out["Fobs_over_Fcalc"]
        out["merged_model_bias_kind"] = "Fobs_over_Fcalc"
        out["global_merged_bias_class"] = np.select([out["Fobs_over_Fcalc"] < 1.0, out["Fobs_over_Fcalc"] > 1.0], ["globally_depleted", "globally_enhanced"], default="neutral_or_unknown")
    elif "model_residual" in out.columns:
        out["merged_model_bias_metric"] = out["model_residual"]
        out["merged_model_bias_kind"] = "model_residual"
        out["global_merged_bias_class"] = np.select([out["model_residual"] < 0.0, out["model_residual"] > 0.0], ["globally_depleted", "globally_enhanced"], default="neutral_or_unknown")
    else:
        out["merged_model_bias_metric"] = np.nan
        out["merged_model_bias_kind"] = "none"
        out["global_merged_bias_class"] = "neutral_or_unknown"
    out = out.drop_duplicates(HKL_COLUMNS, keep="first").reset_index(drop=True)
    return out, {"model_bias_table": str(chosen_path), "auto_candidates_checked": auto_candidates, "columns": list(table.columns)}


def add_bias_comparison(summary: pd.DataFrame, bias_table: pd.DataFrame) -> pd.DataFrame:
    if bias_table.empty:
        out = summary.copy()
        out["merged_model_bias_metric"] = np.nan
        out["merged_model_bias_kind"] = "not_available"
        out["global_merged_bias_class"] = "not_available"
    else:
        payload = [column for column in [*HKL_COLUMNS, "Fobs", "Fcalc", "Fobs_over_Fcalc", "Fobs_minus_Fcalc", "model_residual", "merged_model_bias_metric", "merged_model_bias_kind", "global_merged_bias_class"] if column in bias_table.columns]
        out = summary.merge(bias_table.loc[:, payload], on=HKL_COLUMNS, how="left", validate="many_to_one")
        out["global_merged_bias_class"] = out["global_merged_bias_class"].fillna("not_available")
        out["merged_model_bias_kind"] = out["merged_model_bias_kind"].fillna("not_available")
    agreements = []
    for _, row in out.iterrows():
        global_class = str(row.get("global_merged_bias_class", "not_available"))
        trend_class = str(row.get("orientation_trend_classification", ""))
        if global_class == "not_available" or trend_class in {"insufficient data", "no stable trend"}:
            agreements.append("not_assessable")
        elif (global_class == "globally_depleted" and trend_class == "depletion-like") or (global_class == "globally_enhanced" and trend_class == "enhancement-like"):
            agreements.append("same_direction")
        else:
            agreements.append("opposite_or_mixed")
    out["global_bias_vs_heldout_dexc_trend"] = agreements
    out["comparison_note"] = "global merged/model bias and within-HKL excitation-deficit orientation trend are distinct diagnostics"
    return out


def observation_output_columns(observations: pd.DataFrame) -> list[str]:
    preferred = [
        *KEY_COLUMNS,
        "target_family",
        "target_orbit_id",
        "target_seed_hkl",
        "hkl",
        "merged_intensity_or_Fobs",
        "d_spacing_angstrom",
        "resolution_angstrom",
        "g_norm_invA",
        "reciprocal_direction_class",
        "hkl_geometry_class",
        "continuous_uvw_x",
        "continuous_uvw_y",
        "continuous_uvw_z",
        "closest_uvw",
        "closest_zone_axis_angle_deg",
        "observation_intensity",
        "I_unmerged",
        "sigma",
        "partiality",
        TARGET_COLUMN,
        "target_excitation_error",
        V5_COLUMN,
        COUPLING_COLUMN,
        "neighbor_excitation_weighted_mean",
        "coupled_excitation_imbalance_norm",
        DEFICIT_COLUMN,
        "I_over_merged",
        "I_over_hkl_median",
        "I_robust_z_within_hkl",
        ORIGINAL_RESPONSE,
    ]
    return [column for column in preferred if column in observations.columns]


def make_plots(observations: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, dpi: int) -> list[Path]:
    plot_dir = out_dir / "plots" / "per_reflection"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if observations.empty:
        return paths
    for hkl, group in observations.groupby(HKL_COLUMNS, sort=False):
        hkl_tuple = tuple(map(int, hkl))
        nonzero = group.loc[pd.to_numeric(group[COUPLING_COLUMN], errors="coerce") > 0.0].copy()
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        axes[0].scatter(nonzero[DEFICIT_COLUMN], nonzero[ORIGINAL_RESPONSE], s=12, alpha=0.65)
        axes[0].axhline(0.0, color="0.5", lw=0.8)
        axes[0].set_xlabel("excitation_deficit_norm")
        axes[0].set_ylabel("in-sample corrected residual")
        if len(nonzero) >= 5:
            labels = pd.qcut(nonzero[DEFICIT_COLUMN].rank(method="first"), q=min(5, len(nonzero)), labels=False, duplicates="drop")
            med = nonzero.assign(_q=np.asarray(labels, dtype=int)).groupby("_q", sort=True)[ORIGINAL_RESPONSE].median()
            axes[1].plot(np.arange(1, len(med) + 1), med.to_numpy(dtype=float), marker="o")
        axes[1].axhline(0.0, color="0.5", lw=0.8)
        axes[1].set_xlabel("deficit quintile")
        axes[1].set_ylabel("median residual")
        if {"continuous_uvw_x", "continuous_uvw_y", "continuous_uvw_z"}.issubset(group.columns):
            scatter = axes[2].scatter(group["continuous_uvw_x"], group["continuous_uvw_y"], c=group[DEFICIT_COLUMN], s=12, alpha=0.75, cmap="coolwarm")
            fig.colorbar(scatter, ax=axes[2], label="deficit")
        axes[2].set_xlabel("continuous UVW x")
        axes[2].set_ylabel("continuous UVW y")
        row = summary.loc[(summary["h"] == hkl_tuple[0]) & (summary["k"] == hkl_tuple[1]) & (summary["l"] == hkl_tuple[2])]
        classification = str(row["orientation_trend_classification"].iloc[0]) if not row.empty and "orientation_trend_classification" in row else ""
        fig.suptitle(f"HKL {hkl_label(hkl_tuple)} | {classification}")
        path = plot_dir / f"target_hkl_{hkl_slug(hkl_tuple)}.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths.append(path)
    return paths


def write_readme(out_dir: Path, args: argparse.Namespace, metadata: dict[str, Any], plot_count: int) -> None:
    lines = [
        "# Target HKL Family V5 Excitation-Deficit Diagnostic",
        "",
        "Targeted observation-level diagnostic for hh0 reflections and suspected enhanced 224, 601, and 503 orbits.",
        "",
        "## Scope",
        "",
        "- The broad 75-HKL selector is not run.",
        "- Existing Partialator-accepted survivor observations and existing v5 scores are joined by exact source/event/signed-HKL key.",
        "- v5 scores are not recomputed or changed.",
        "- No stream rewrite, merge, Partialator run, or refinement is performed.",
        "",
        "## Target Families",
        "",
        "- hh0 seeds: 110 through 11 11 0, with all signed observed 4/mmm mates preserved.",
        "- suspected enhanced seeds: 224, 601, 503, with all signed observed 4/mmm mates preserved.",
        "",
        "## Dexc Definitions",
        "",
        "- `coupling_sum_raw = nonself_neighbor_count_effective`",
        "- `neighbor_excitation_weighted_mean = nonself_local_excitation_raw / coupling_sum_raw`",
        "- `excitation_deficit_norm = target_excitation_Eg - neighbor_excitation_weighted_mean`",
        "- `coupled_excitation_imbalance_norm = -excitation_deficit_norm`",
        "- derived normalized values are set to 0 when coupling sum is 0.",
        "",
        "## Held-Out Validation",
        "",
        "Held-out validation uses the leakage-free procedure from `analyze_and_crossvalidate_v5_dexc_correction_candidates.py`: train-only intensity normalization, train-only baseline fit, train-only Dexc fit, and train-only quintile/correction parameters.",
        "",
        "The orientation-trend classification describes within-HKL orientation dependence. It is not automatically the same as merged Fobs/Fcalc bias.",
        "",
        "## Outputs",
        "",
        "- `hh0_family_summary.csv`",
        "- `suspected_gain_reflections_summary.csv`",
        "- `target_hkl_observation_diagnostics.csv`",
        "- `target_hkl_crossvalidation.csv`",
        "- `target_hkl_symmetry_transfer.csv`",
        "- `merged_bias_vs_dexc_trend.csv`",
        "- `run_metadata.json`",
        f"- compact plots: {plot_count} PNG files under `plots/per_reflection/`",
        "",
        "## Model Bias Table",
        "",
        f"- source: `{metadata.get('model_bias', {}).get('model_bias_table')}`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(out_dir: Path, args: argparse.Namespace, metadata: dict[str, Any], outputs: dict[str, Path]) -> None:
    run_metadata = {
        "command": "tools/analyze_target_hkl_families_v5_dexc.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": {
            "stream": str(args.stream),
            "v5_scores": str(args.v5_scores),
            "accepted": str(args.accepted),
            "strength_table": str(args.strength_table),
            "model_bias_table": str(args.model_bias_table) if args.model_bias_table else None,
        },
        "target_seeds": {"hh0": [hkl_label(hkl) for hkl in HH0_SEEDS], "suspected_gain": [hkl_label(hkl) for hkl in SUSPECTED_GAIN_SEEDS]},
        "metadata": metadata,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "leakage_controls": {
            "cv_response": cvmod.CV_RESPONSE_COLUMN,
            "cv_response_source": cvmod.CV_RESPONSE_SOURCE_COLUMN,
            "normalization": "training median observation intensity only",
            "baseline_fit": "training observations only",
            "dexc_fit": "training baseline residuals only",
            "test_observations_used_for_fitting_or_centering": False,
        },
        "did_not_run": ["broad 75-HKL selection", "stream rewrite", "Partialator", "merge", "structure refinement", "v5 recomputation"],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_terminal_summary(summary: pd.DataFrame) -> None:
    columns = [
        "hkl",
        "global_merged_bias_class",
        "merged_model_bias_metric",
        "heldout_deficit_slope_sign",
        "median_heldout_delta_mae",
        "slope_sign_stability",
        "orientation_trend_classification",
    ]
    available = [column for column in columns if column in summary.columns]
    terminal = summary.loc[:, available].copy()
    if "median_heldout_delta_mae" in terminal.columns:
        terminal["median_heldout_delta_mae"] = pd.to_numeric(terminal["median_heldout_delta_mae"], errors="coerce").map(lambda value: f"{value:.6g}" if np.isfinite(value) else "")
    if "slope_sign_stability" in terminal.columns:
        terminal["slope_sign_stability"] = pd.to_numeric(terminal["slope_sign_stability"], errors="coerce").map(lambda value: f"{value:.3g}" if np.isfinite(value) else "")
    if "merged_model_bias_metric" in terminal.columns:
        terminal["merged_model_bias_metric"] = pd.to_numeric(terminal["merged_model_bias_metric"], errors="coerce").map(lambda value: f"{value:.6g}" if np.isfinite(value) else "")
    print("\nTarget HKL family diagnostic summary:")
    print(terminal.to_string(index=False))


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_dir.resolve() in {args.accepted.resolve(), args.v5_scores.resolve(), args.stream.resolve(), args.strength_table.resolve()}:
        raise SystemExit("--out-dir must be a directory distinct from input files")

    log("Building requested target 4/mmm HKL families")
    target_table = build_target_hkl_table()
    log(f"Target signed HKLs requested: {len(target_table):,}")

    log("Loading exact-key accepted/v5 target observations")
    observations, target_meta, metadata = prepare_target_observations(args, target_table)

    log("Writing observation-level diagnostics and in-sample target summaries")
    signed_summary = build_signed_hkl_summary(observations, target_meta, args)
    scale_columns = list(metadata.get("accepted_table", {}).get("scale_columns_available", []) or [])
    scale_columns = [column for column in scale_columns if column in observations.columns]

    log("Running leakage-free held-out validation for target signed HKLs")
    crossvalidation = run_crossvalidation_for_hkls(observations, target_meta, scale_columns, args)
    cv_summary = aggregate_crossvalidation(crossvalidation)
    classified = classify_orientation_trends(signed_summary, cv_summary, args)

    log("Running symmetry-mate transfer validation where enough observations exist")
    transfer = run_symmetry_transfer(observations, target_meta, scale_columns, args)

    log("Checking for optional merged/model bias table")
    model_bias, bias_metadata = load_model_bias_table(args.model_bias_table, BASE)
    metadata["model_bias"] = bias_metadata
    bias_comparison = add_bias_comparison(classified, model_bias)

    outputs = {
        "hh0_family_summary": args.out_dir / "hh0_family_summary.csv",
        "suspected_gain_reflections_summary": args.out_dir / "suspected_gain_reflections_summary.csv",
        "target_hkl_observation_diagnostics": args.out_dir / "target_hkl_observation_diagnostics.csv",
        "target_hkl_crossvalidation": args.out_dir / "target_hkl_crossvalidation.csv",
        "target_hkl_symmetry_transfer": args.out_dir / "target_hkl_symmetry_transfer.csv",
        "merged_bias_vs_dexc_trend": args.out_dir / "merged_bias_vs_dexc_trend.csv",
        "README": args.out_dir / "README.md",
        "run_metadata": args.out_dir / "run_metadata.json",
    }
    bias_comparison.loc[bias_comparison["target_family"] == "hh0_depletion_suspected"].to_csv(outputs["hh0_family_summary"], index=False)
    bias_comparison.loc[bias_comparison["target_family"] == "suspected_gain_reflection"].to_csv(outputs["suspected_gain_reflections_summary"], index=False)
    if observations.empty:
        pd.DataFrame().to_csv(outputs["target_hkl_observation_diagnostics"], index=False)
    else:
        observations.loc[:, observation_output_columns(observations)].to_csv(outputs["target_hkl_observation_diagnostics"], index=False)
    crossvalidation.to_csv(outputs["target_hkl_crossvalidation"], index=False)
    transfer.to_csv(outputs["target_hkl_symmetry_transfer"], index=False)
    bias_comparison.to_csv(outputs["merged_bias_vs_dexc_trend"], index=False)

    log("Writing compact per-reflection plots and documentation")
    plot_paths = make_plots(observations, bias_comparison, args.out_dir, int(args.plot_dpi))
    write_readme(args.out_dir, args, metadata, len(plot_paths))
    write_metadata(args.out_dir, args, metadata, outputs)
    print_terminal_summary(bias_comparison)
    print(f"\nOutputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
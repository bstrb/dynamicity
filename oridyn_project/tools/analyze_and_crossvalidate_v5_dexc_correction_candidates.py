#!/usr/bin/env python3
"""Cross-validate v5 excitation-deficit correction candidates.

This script is intentionally read-only with respect to the completed broad
diagnostic. It consumes the diagnostic CSVs, runs held-out candidate checks, and
writes a separate validation report directory. It does not rewrite streams, run
Partialator, merge data, or refine structures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


BASE = Path("/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524")
DEFAULT_DIAGNOSTIC_DIR = BASE / "oridyn_v5_excitation_imbalance_orientation_broad_hkl_v2_20_0p3_20260712"
DEFAULT_OUT_DIR = BASE / "oridyn_v5_dexc_correction_candidate_validation_20_0p3_20260712"

HKL_COLUMNS = ["h", "k", "l"]
ORIGINAL_DIAGNOSTIC_RESPONSE_COLUMN = "corrected_intensity_residual"
UPSTREAM_NORMALIZED_RESPONSE_COLUMN = "I_over_hkl_median"
CV_RESPONSE_SOURCE_COLUMN = "observation_intensity"
CV_RESPONSE_COLUMN = "I_over_train_hkl_median"
DEFICIT_COLUMN = "excitation_deficit_norm"
TARGET_COLUMN = "target_excitation_Eg"
PARTIALITY_COLUMN = "partiality"
COUPLING_COLUMN = "coupling_sum_raw"
PRIMARY_SLOPE_COLUMN = "slope_corrected_residual_vs_excitation_deficit_norm"
PRIMARY_RHO_COLUMN = "spearman_corrected_intensity_residual_vs_excitation_deficit_norm"
PRIMARY_PARTIAL_RHO_COLUMN = "partial_spearman_corrected_intensity_residual_vs_excitation_deficit_norm_ctrl_Eg_partiality"

FOCUS_PAIRS = [
    ((0, 0, -4), (0, 0, 4)),
    ((-4, 0, 2), (4, 0, -2)),
    ((3, 3, 0), (-3, -3, 0)),
    ((0, 1, -7), (1, 0, -7)),
]

SCIENTIFIC_WARNING = (
    "Thick strongly dynamical crystals, visible Kikuchi lines, imperfect indexing/orientation refinement, "
    "fixed-radius CrystFEL ring integration, imperfect partiality/scaling, and high refinement R values mean "
    "these diagnostics are correction-screening evidence only."
)


@dataclass(frozen=True)
class InputPaths:
    diagnostic_dir: Path
    observations: Path
    per_hkl: Path
    quintiles: Path
    symmetry: Path
    aggregate: Path
    readme: Path
    metadata: Path


@dataclass(frozen=True)
class ModelFit:
    beta: np.ndarray
    columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR, help="Completed broad v5 diagnostic directory.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="New output directory for validation products.")
    parser.add_argument("--splits", type=int, default=20, help="Deterministic repeated held-out splits per signed HKL.")
    parser.add_argument("--train-fraction", type=float, default=0.70, help="Approximate within-stratum training fraction.")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--min-nonzero-coupling-obs", type=int, default=100)
    parser.add_argument("--min-train-obs", type=int, default=50)
    parser.add_argument("--min-test-obs", type=int, default=30)
    parser.add_argument("--candidate-only-focus-pairs", action="store_true", help="Run held-out stages only for the four named focus pairs.")
    parser.add_argument("--cv-all-symmetry-pairs", action="store_true", help="Run held-out stages for every symmetry pair, not only promising/focus pairs.")
    parser.add_argument("--plot-dpi", type=int, default=170)
    args = parser.parse_args()

    if int(args.splits) < 1:
        raise SystemExit("--splits must be >= 1")
    if not (0.50 <= float(args.train_fraction) < 1.0):
        raise SystemExit("--train-fraction must satisfy 0.50 <= value < 1.0")
    if int(args.min_nonzero_coupling_obs) < 1:
        raise SystemExit("--min-nonzero-coupling-obs must be >= 1")
    if int(args.min_train_obs) < 5 or int(args.min_test_obs) < 3:
        raise SystemExit("--min-train-obs must be >= 5 and --min-test-obs must be >= 3")
    if int(args.plot_dpi) < 60:
        raise SystemExit("--plot-dpi must be >= 60")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def hkl_tuple(row: pd.Series | Iterable[Any]) -> tuple[int, int, int]:
    if isinstance(row, pd.Series):
        return int(row["h"]), int(row["k"]), int(row["l"])
    values = list(row)
    return int(values[0]), int(values[1]), int(values[2])


def hkl_label(hkl: tuple[int, int, int]) -> str:
    return f"({hkl[0]},{hkl[1]},{hkl[2]})"


def hkl_slug(hkl: tuple[int, int, int]) -> str:
    def one(value: int) -> str:
        return f"m{abs(int(value))}" if int(value) < 0 else str(int(value))

    return f"{one(hkl[0])}_{one(hkl[1])}_{one(hkl[2])}"


def pair_key(a: tuple[int, int, int], b: tuple[int, int, int]) -> frozenset[tuple[int, int, int]]:
    return frozenset((tuple(map(int, a)), tuple(map(int, b))))


def focus_pair_id(a: tuple[int, int, int], b: tuple[int, int, int]) -> str:
    key = pair_key(a, b)
    for idx, (left, right) in enumerate(FOCUS_PAIRS, start=1):
        if key == pair_key(left, right):
            return f"focus_{idx}"
    return ""


def sign_value(value: float, eps: float = 1e-12) -> int:
    if not np.isfinite(value) or abs(float(value)) <= eps:
        return 0
    return 1 if float(value) > 0.0 else -1


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def qspread(values: pd.Series, low: float = 0.10, high: float = 0.90) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 2:
        return np.nan
    q_low, q_high = np.quantile(finite.to_numpy(dtype=float), [low, high])
    return float(q_high - q_low)


def mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan
    center = np.median(finite)
    return float(np.median(np.abs(finite - center)))


def spearman_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"), "y": pd.to_numeric(pd.Series(y), errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame["x"].rank(method="average").corr(frame["y"].rank(method="average")))


def robust_slope_simple(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"), "y": pd.to_numeric(pd.Series(y), errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2:
        return np.nan
    design = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy(dtype=float)])
    beta = huber_irls(design, frame["y"].to_numpy(dtype=float))
    return float(beta[1]) if len(beta) > 1 else np.nan


def huber_irls(x: np.ndarray, y: np.ndarray, c: float = 1.345, max_iter: int = 35) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    if len(y) == 0:
        return np.full(x.shape[1] if x.ndim == 2 else 0, np.nan)
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(x.shape[1], np.nan)
    for _ in range(int(max_iter)):
        resid = y - x @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if not np.isfinite(scale) or scale <= 1e-12:
            break
        weights = np.ones_like(resid)
        large = np.abs(resid) > c * scale
        weights[large] = (c * scale) / np.abs(resid[large])
        try:
            new_beta, *_ = np.linalg.lstsq(x * weights[:, None], y * weights, rcond=None)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(new_beta - beta) < 1e-8:
            beta = new_beta
            break
        beta = new_beta
    return beta


def fit_model(frame: pd.DataFrame, response: str, covariates: list[str]) -> ModelFit:
    columns = ["intercept", *covariates]
    y = pd.to_numeric(frame[response], errors="coerce").to_numpy(dtype=float)
    parts = [np.ones(len(frame), dtype=float)]
    for column in covariates:
        parts.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
    x = np.column_stack(parts)
    beta = huber_irls(x, y)
    return ModelFit(beta=beta, columns=columns)


def predict_model(fit: ModelFit, frame: pd.DataFrame) -> np.ndarray:
    parts = []
    for column in fit.columns:
        if column == "intercept":
            parts.append(np.ones(len(frame), dtype=float))
        else:
            parts.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
    x = np.column_stack(parts)
    return x @ fit.beta


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 3:
        return np.nan
    denom = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - np.sum((y[mask] - pred[mask]) ** 2) / denom)


def explained_variance(y: np.ndarray, residual: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    residual = np.asarray(residual, dtype=float)
    mask = np.isfinite(y) & np.isfinite(residual)
    if int(mask.sum()) < 3:
        return np.nan
    var_y = float(np.var(y[mask]))
    if var_y <= 1e-12:
        return np.nan
    return float(1.0 - np.var(residual[mask]) / var_y)


def require_columns(table: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def input_paths(diagnostic_dir: Path) -> InputPaths:
    paths = InputPaths(
        diagnostic_dir=diagnostic_dir,
        observations=diagnostic_dir / "diagnostic_observations.csv",
        per_hkl=diagnostic_dir / "per_hkl_dexc_intensity_stats.csv",
        quintiles=diagnostic_dir / "per_hkl_quintile_trends.csv",
        symmetry=diagnostic_dir / "symmetry_mate_comparison.csv",
        aggregate=diagnostic_dir / "aggregate_trend_summary.csv",
        readme=diagnostic_dir / "README_v5_excitation_imbalance_orientation_broad_hkl.md",
        metadata=diagnostic_dir / "run_metadata.json",
    )
    for label, path in paths.__dict__.items():
        if label == "diagnostic_dir":
            if not Path(path).is_dir():
                raise SystemExit(f"--diagnostic-dir not found: {path}")
            continue
        if not Path(path).is_file():
            raise SystemExit(f"Required diagnostic file not found: {path}")
    return paths


def load_inputs(paths: InputPaths) -> dict[str, Any]:
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    readme_text = paths.readme.read_text(encoding="utf-8", errors="replace")
    observations = pd.read_csv(paths.observations)
    per_hkl = pd.read_csv(paths.per_hkl)
    quintiles = pd.read_csv(paths.quintiles)
    symmetry = pd.read_csv(paths.symmetry)
    aggregate = pd.read_csv(paths.aggregate)

    require_columns(observations, [*HKL_COLUMNS, ORIGINAL_DIAGNOSTIC_RESPONSE_COLUMN, UPSTREAM_NORMALIZED_RESPONSE_COLUMN, CV_RESPONSE_SOURCE_COLUMN, DEFICIT_COLUMN, TARGET_COLUMN, PARTIALITY_COLUMN, COUPLING_COLUMN], "diagnostic_observations.csv")
    require_columns(per_hkl, [*HKL_COLUMNS, PRIMARY_SLOPE_COLUMN, "slope_ci95_low", "slope_ci95_high", PRIMARY_RHO_COLUMN, PRIMARY_PARTIAL_RHO_COLUMN], "per_hkl_dexc_intensity_stats.csv")
    require_columns(quintiles, [*HKL_COLUMNS, "quintile_variable", "quintile_index", "median_corrected_intensity_residual"], "per_hkl_quintile_trends.csv")
    require_columns(symmetry, ["symmetry_pair_id", "h1", "k1", "l1", "h2", "k2", "l2"], "symmetry_mate_comparison.csv")

    for table in [observations, per_hkl, quintiles, symmetry]:
        for column in [c for c in HKL_COLUMNS if c in table.columns]:
            table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    return {
        "metadata": metadata,
        "readme_text": readme_text,
        "observations": observations,
        "per_hkl": per_hkl,
        "quintiles": quintiles,
        "symmetry": symmetry,
        "aggregate": aggregate,
    }


def response_control_summary(metadata: dict[str, Any], observations: pd.DataFrame) -> dict[str, Any]:
    accepted_meta = metadata.get("accepted_table", {}) if isinstance(metadata, dict) else {}
    scale_columns = list(accepted_meta.get("scale_columns_available", []) or [])
    scale_like_obs = [column for column in observations.columns if "scale" in column.lower()]
    return {
        "reported_primary_slope_column": PRIMARY_SLOPE_COLUMN,
        "reported_primary_response": ORIGINAL_DIAGNOSTIC_RESPONSE_COLUMN,
        "reported_primary_x": DEFICIT_COLUMN,
        "cv_response_column": CV_RESPONSE_COLUMN,
        "cv_response_source_column": CV_RESPONSE_SOURCE_COLUMN,
        "upstream_normalized_response_column_not_used_directly_for_cv": UPSTREAM_NORMALIZED_RESPONSE_COLUMN,
        "cv_response_reason": "A train-only equivalent of I_over_hkl_median is recomputed as observation_intensity / median_train_observation_intensity so test observations do not enter response normalization.",
        "observation_intensity_used": accepted_meta.get("intensity_column_used", "unknown"),
        "response_precontrols_target_excitation": True,
        "response_precontrols_partiality": True,
        "response_precontrols_scale_columns": bool(scale_columns),
        "heldout_baseline_predictors": [TARGET_COLUMN, PARTIALITY_COLUMN, *scale_columns],
        "heldout_scale_term_found": bool(scale_columns),
        "scale_columns_from_metadata": scale_columns,
        "scale_like_columns_in_observations": scale_like_obs,
        "heldout_leakage_controls": {
            "uses_precomputed_corrected_residual_as_cv_response": False,
            "uses_precomputed_I_over_hkl_median_directly_as_cv_response": False,
            "stage1_only_uses_precomputed_corrected_residual": True,
            "baseline_fit_uses_training_observations_only": True,
            "normalization_and_centering_estimated_from_training_only": True,
            "quintile_boundaries_estimated_from_training_only": True,
            "correction_parameters_estimated_from_training_only": True,
            "test_observations_enter_fit_or_centering": False,
        },
        "note": (
            "The broad diagnostic defines corrected_intensity_residual as a robust residual of "
            "I_over_hkl_median against target_excitation_Eg, partiality, and any scale covariates. "
            "This script keeps corrected_intensity_residual only for Stage 1 in-sample descriptive reproduction; "
            "all held-out validation uses train-median-normalized observation_intensity and train-fitted baseline residuals."
        ),
    }


def available_stat_columns(per_hkl: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(per_hkl.columns)
    return {
        "slope_columns": [c for c in columns if c.startswith("slope_")],
        "rho_columns": [c for c in columns if c.startswith("spearman_") and not c.startswith("spearman_p_")],
        "partial_rho_columns": [c for c in columns if c.startswith("partial_spearman_")],
        "confidence_interval_columns": [c for c in columns if "ci95" in c.lower()],
        "subset_columns": [c for c in columns if "quartile" in c or "tertile" in c],
    }


def hkl_index(table: pd.DataFrame) -> dict[tuple[int, int, int], pd.Series]:
    out: dict[tuple[int, int, int], pd.Series] = {}
    for _, row in table.iterrows():
        if pd.isna(row["h"]) or pd.isna(row["k"]) or pd.isna(row["l"]):
            continue
        out[hkl_tuple(row)] = row
    return out


def observation_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hkl, group in observations.groupby(HKL_COLUMNS, sort=False, dropna=True):
        hkl_int = tuple(map(int, hkl))
        nonzero = group.loc[pd.to_numeric(group[COUPLING_COLUMN], errors="coerce") > 0.0].copy()
        robust_z = pd.to_numeric(group.get("I_robust_z_within_hkl", pd.Series(index=group.index, dtype=float)), errors="coerce")
        outlier_fraction = float((robust_z.abs() > 3.0).mean()) if len(robust_z.dropna()) else np.nan
        if {"continuous_uvw_x", "continuous_uvw_y", "continuous_uvw_z"}.issubset(group.columns):
            uvw = group.loc[:, ["continuous_uvw_x", "continuous_uvw_y", "continuous_uvw_z"]].apply(pd.to_numeric, errors="coerce")
            uvw_spread = float(np.sqrt(np.nansum(np.nanvar(uvw.to_numpy(dtype=float), axis=0)))) if len(uvw.dropna()) else np.nan
        else:
            uvw_spread = np.nan
        if "closest_uvw" in group.columns:
            counts = group["closest_uvw"].astype(str).value_counts(dropna=True)
            orientation_regions = int(len(counts))
            dominant_fraction = float(counts.iloc[0] / len(group)) if len(counts) and len(group) else np.nan
        else:
            orientation_regions = 0
            dominant_fraction = np.nan
        rows.append(
            {
                "h": hkl_int[0],
                "k": hkl_int[1],
                "l": hkl_int[2],
                "accepted_observation_count_from_observations": int(len(group)),
                "nonzero_coupling_count_from_observations": int(len(nonzero)),
                "excitation_deficit_spread_q90_q10": qspread(nonzero[DEFICIT_COLUMN]) if len(nonzero) else np.nan,
                "excitation_deficit_min": finite_float(nonzero[DEFICIT_COLUMN].min()) if len(nonzero) else np.nan,
                "excitation_deficit_max": finite_float(nonzero[DEFICIT_COLUMN].max()) if len(nonzero) else np.nan,
                "fraction_intensity_outliers_abs_robust_z_gt_3": outlier_fraction,
                "orientation_region_count": orientation_regions,
                "orientation_dominant_region_fraction": dominant_fraction,
                "orientation_vector_rms_spread": uvw_spread,
            }
        )
    return pd.DataFrame.from_records(rows)


def quintile_shape_for_hkl(quintiles: pd.DataFrame, hkl: tuple[int, int, int]) -> dict[str, Any]:
    subset = quintiles.loc[
        (quintiles["h"].astype(int) == hkl[0])
        & (quintiles["k"].astype(int) == hkl[1])
        & (quintiles["l"].astype(int) == hkl[2])
        & (quintiles["quintile_variable"].astype(str) == DEFICIT_COLUMN)
    ].sort_values("quintile_index")
    medians = pd.to_numeric(subset["median_corrected_intensity_residual"], errors="coerce").to_numpy(dtype=float)
    medians = medians[np.isfinite(medians)]
    if len(medians) < 2:
        return {
            "quintile_medians": "",
            "quintile_monotonic": False,
            "quintile_direction": "insufficient",
            "extreme_quintile_drives_slope": False,
            "extreme_step_fraction": np.nan,
            "quintile_shape_code": "",
            "quintile_total_change": np.nan,
        }
    diffs = np.diff(medians)
    scale = max(float(np.nanmedian(np.abs(medians))), 1.0)
    tol = 0.02 * scale
    nondecreasing = bool(np.all(diffs >= -tol))
    nonincreasing = bool(np.all(diffs <= tol))
    total_change = float(medians[-1] - medians[0])
    if nondecreasing and total_change > tol:
        direction = "increasing"
    elif nonincreasing and total_change < -tol:
        direction = "decreasing"
    elif nondecreasing or nonincreasing:
        direction = "flat"
    else:
        direction = "nonmonotonic"
    abs_steps = np.abs(diffs)
    step_total = float(np.sum(abs_steps))
    extreme_step = float(max(abs_steps[0], abs_steps[-1])) if len(abs_steps) else np.nan
    extreme_fraction = float(extreme_step / step_total) if step_total > 1e-12 else np.nan
    inner_driven_flag = False
    if len(medians) >= 5:
        inner_slope = robust_slope_simple(np.arange(2, len(medians)), medians[1:-1])
        inner_driven_flag = sign_value(inner_slope) not in (0, sign_value(total_change)) and abs(total_change) > tol
    extreme_driven = bool(np.isfinite(extreme_fraction) and extreme_fraction >= 0.60 and abs(total_change) > tol) or inner_driven_flag
    shape_code = ",".join("+" if d > tol else "-" if d < -tol else "0" for d in diffs)
    return {
        "quintile_medians": ";".join(f"{value:.6g}" for value in medians),
        "quintile_monotonic": bool(direction in {"increasing", "decreasing", "flat"}),
        "quintile_direction": direction,
        "extreme_quintile_drives_slope": extreme_driven,
        "extreme_step_fraction": extreme_fraction,
        "quintile_shape_code": shape_code,
        "quintile_total_change": total_change,
    }


def build_candidate_quintile_diagnostics(quintiles: pd.DataFrame, symmetry: pd.DataFrame) -> pd.DataFrame:
    candidate_hkls = set()
    for _, row in symmetry.iterrows():
        candidate_hkls.add((int(row.h1), int(row.k1), int(row.l1)))
        candidate_hkls.add((int(row.h2), int(row.k2), int(row.l2)))
    rows: list[dict[str, Any]] = []
    for hkl, group in quintiles.groupby(HKL_COLUMNS, sort=False, dropna=True):
        hkl_int = tuple(map(int, hkl))
        if hkl_int not in candidate_hkls:
            continue
        for variable, var_group in group.groupby("quintile_variable", sort=False):
            ordered = var_group.sort_values("quintile_index")
            med = pd.to_numeric(ordered["median_corrected_intensity_residual"], errors="coerce").to_numpy(dtype=float)
            med = med[np.isfinite(med)]
            if len(med) >= 2:
                diffs = np.diff(med)
                scale = max(float(np.nanmedian(np.abs(med))), 1.0)
                tol = 0.02 * scale
                monotonic = bool(np.all(diffs >= -tol) or np.all(diffs <= tol))
                total_change = float(med[-1] - med[0])
                abs_steps = np.abs(diffs)
                step_total = float(np.sum(abs_steps))
                extreme_fraction = float(max(abs_steps[0], abs_steps[-1]) / step_total) if step_total > 1e-12 else np.nan
                extreme_driven = bool(np.isfinite(extreme_fraction) and extreme_fraction >= 0.60 and abs(total_change) > tol)
                shape_code = ",".join("+" if d > tol else "-" if d < -tol else "0" for d in diffs)
            else:
                monotonic = False
                total_change = np.nan
                extreme_fraction = np.nan
                extreme_driven = False
                shape_code = ""
            rows.append(
                {
                    "h": hkl_int[0],
                    "k": hkl_int[1],
                    "l": hkl_int[2],
                    "hkl": hkl_label(hkl_int),
                    "quintile_variable": variable,
                    "n_quintiles": int(len(med)),
                    "quintile_medians": ";".join(f"{value:.6g}" for value in med),
                    "monotonic": monotonic,
                    "shape_code": shape_code,
                    "total_change": total_change,
                    "extreme_step_fraction": extreme_fraction,
                    "extreme_quintile_drives_slope": extreme_driven,
                }
            )
    return pd.DataFrame.from_records(rows)


def pair_metrics(first: dict[str, Any], second: dict[str, Any], min_nonzero: int) -> dict[str, Any]:
    slope1 = finite_float(first.get(PRIMARY_SLOPE_COLUMN))
    slope2 = finite_float(second.get(PRIMARY_SLOPE_COLUMN))
    rho1 = finite_float(first.get(PRIMARY_RHO_COLUMN))
    rho2 = finite_float(second.get(PRIMARY_RHO_COLUMN))
    partial1 = finite_float(first.get(PRIMARY_PARTIAL_RHO_COLUMN))
    partial2 = finite_float(second.get(PRIMARY_PARTIAL_RHO_COLUMN))
    ci1 = (finite_float(first.get("slope_ci95_low")), finite_float(first.get("slope_ci95_high")))
    ci2 = (finite_float(second.get("slope_ci95_low")), finite_float(second.get("slope_ci95_high")))
    slope_sign_agreement = sign_value(slope1) != 0 and sign_value(slope1) == sign_value(slope2)
    rho_agreement = sign_value(rho1) != 0 and sign_value(rho1) == sign_value(rho2) and abs(rho1 - rho2) <= 0.25
    partial_agreement = (sign_value(partial1) == sign_value(partial2) and sign_value(partial1) != 0) or (abs(partial1) < 0.05 and abs(partial2) < 0.05)
    mag_low = min(abs(slope1), abs(slope2))
    mag_high = max(abs(slope1), abs(slope2))
    mag_ratio = float(mag_high / mag_low) if mag_low > 1e-12 and np.isfinite(mag_high) else np.inf
    ci_overlap = bool(np.isfinite(ci1[0]) and np.isfinite(ci1[1]) and np.isfinite(ci2[0]) and np.isfinite(ci2[1]) and max(ci1[0], ci2[0]) <= min(ci1[1], ci2[1]))
    ci_both_exclude_zero = bool(
        np.isfinite(ci1[0])
        and np.isfinite(ci1[1])
        and np.isfinite(ci2[0])
        and np.isfinite(ci2[1])
        and not (ci1[0] <= 0.0 <= ci1[1])
        and not (ci2[0] <= 0.0 <= ci2[1])
    )

    high_target_agreement = signs_agree_for_columns(first, second, "slope_corrected_residual_vs_deficit_high_target_quartile")
    high_partial_agreement = signs_agree_for_columns(first, second, "slope_corrected_residual_vs_deficit_high_partiality_quartile")
    shape_corr = quintile_shape_correlation(first.get("quintile_medians", ""), second.get("quintile_medians", ""))
    shape_agreement = bool(
        first.get("quintile_direction") == second.get("quintile_direction")
        and first.get("quintile_direction") not in {"", "insufficient", "nonmonotonic"}
        and np.isfinite(shape_corr)
        and shape_corr >= 0.50
    )
    coverage_similarity = orientation_coverage_similarity(first, second)
    enough_nonzero = int(first.get("nonzero_coupling_observation_count", 0)) >= min_nonzero and int(second.get("nonzero_coupling_observation_count", 0)) >= min_nonzero
    extreme_free = not bool(first.get("extreme_quintile_drives_slope", False)) and not bool(second.get("extreme_quintile_drives_slope", False))

    classification = classify_stage1(
        enough_nonzero=enough_nonzero,
        slope_sign_agreement=slope_sign_agreement,
        rho_agreement=rho_agreement,
        partial_agreement=partial_agreement,
        ci_both_exclude_zero=ci_both_exclude_zero,
        ci_overlap=ci_overlap,
        high_target_agreement=high_target_agreement,
        high_partial_agreement=high_partial_agreement,
        shape_agreement=shape_agreement,
        coverage_similarity=coverage_similarity,
        extreme_free=extreme_free,
        min_abs_rho=min(abs(rho1), abs(rho2)) if np.isfinite(rho1) and np.isfinite(rho2) else np.nan,
        mag_ratio=mag_ratio,
    )
    return {
        "slope_sign_agreement": slope_sign_agreement,
        "slope_magnitude_ratio": mag_ratio,
        "confidence_interval_overlap": ci_overlap,
        "confidence_intervals_both_exclude_zero": ci_both_exclude_zero,
        "rho_agreement": rho_agreement,
        "partial_rho_agreement": partial_agreement,
        "high_target_agreement": high_target_agreement,
        "high_partiality_agreement": high_partial_agreement,
        "quintile_shape_agreement": shape_agreement,
        "quintile_shape_correlation": shape_corr,
        "orientation_coverage_similarity": coverage_similarity,
        "enough_nonzero_coupling_observations": enough_nonzero,
        "trend_not_extreme_quintile_driven": extreme_free,
        "stage1_pair_classification": classification,
    }


def signs_agree_for_columns(first: dict[str, Any], second: dict[str, Any], column: str) -> bool:
    v1 = finite_float(first.get(column))
    v2 = finite_float(second.get(column))
    return sign_value(v1) != 0 and sign_value(v1) == sign_value(v2)


def quintile_shape_correlation(first_text: Any, second_text: Any) -> float:
    first = parse_float_list(first_text)
    second = parse_float_list(second_text)
    n = min(len(first), len(second))
    if n < 3:
        return np.nan
    return spearman_corr(np.asarray(first[:n]), np.asarray(second[:n]))


def parse_float_list(text: Any) -> list[float]:
    if not isinstance(text, str) or not text:
        return []
    out: list[float] = []
    for part in text.split(";"):
        value = finite_float(part)
        if np.isfinite(value):
            out.append(value)
    return out


def orientation_coverage_similarity(first: dict[str, Any], second: dict[str, Any]) -> bool:
    n1 = finite_float(first.get("orientation_region_count"), 0.0)
    n2 = finite_float(second.get("orientation_region_count"), 0.0)
    dom1 = finite_float(first.get("orientation_dominant_region_fraction"))
    dom2 = finite_float(second.get("orientation_dominant_region_fraction"))
    spread1 = finite_float(first.get("orientation_vector_rms_spread"))
    spread2 = finite_float(second.get("orientation_vector_rms_spread"))
    n_ratio = min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 0.0
    spread_ratio = min(spread1, spread2) / max(spread1, spread2) if np.isfinite(spread1) and np.isfinite(spread2) and max(spread1, spread2) > 0 else 0.0
    dom_close = np.isfinite(dom1) and np.isfinite(dom2) and abs(dom1 - dom2) <= 0.25
    return bool(n_ratio >= 0.50 and spread_ratio >= 0.40 and dom_close)


def classify_stage1(
    *,
    enough_nonzero: bool,
    slope_sign_agreement: bool,
    rho_agreement: bool,
    partial_agreement: bool,
    ci_both_exclude_zero: bool,
    ci_overlap: bool,
    high_target_agreement: bool,
    high_partial_agreement: bool,
    shape_agreement: bool,
    coverage_similarity: bool,
    extreme_free: bool,
    min_abs_rho: float,
    mag_ratio: float,
) -> str:
    if not enough_nonzero:
        return "insufficient data"
    if not slope_sign_agreement:
        return "inconsistent"
    if not rho_agreement and not partial_agreement:
        return "inconsistent"
    strong = (
        ci_both_exclude_zero
        and ci_overlap
        and rho_agreement
        and partial_agreement
        and high_target_agreement
        and high_partial_agreement
        and shape_agreement
        and coverage_similarity
        and extreme_free
        and np.isfinite(min_abs_rho)
        and min_abs_rho >= 0.25
        and np.isfinite(mag_ratio)
        and mag_ratio <= 2.5
    )
    if strong:
        return "robust correction candidate"
    uncertain = (
        slope_sign_agreement
        and np.isfinite(mag_ratio)
        and mag_ratio <= 5.0
        and (rho_agreement or partial_agreement)
        and (high_target_agreement or high_partial_agreement or shape_agreement)
    )
    if uncertain:
        return "promising but uncertain"
    return "inconsistent"


def build_candidate_pair_validation_summary(
    per_hkl: pd.DataFrame,
    quintiles: pd.DataFrame,
    observations: pd.DataFrame,
    symmetry: pd.DataFrame,
    min_nonzero: int,
) -> pd.DataFrame:
    obs_metrics = observation_metrics(observations)
    enriched = per_hkl.merge(obs_metrics, on=HKL_COLUMNS, how="left", validate="one_to_one")
    rows_by_hkl = hkl_index(enriched)
    signed_rows: list[dict[str, Any]] = []

    for _, pair in symmetry.iterrows():
        hkl1 = (int(pair.h1), int(pair.k1), int(pair.l1))
        hkl2 = (int(pair.h2), int(pair.k2), int(pair.l2))
        if hkl1 not in rows_by_hkl or hkl2 not in rows_by_hkl:
            continue
        base1 = rows_by_hkl[hkl1].to_dict()
        base2 = rows_by_hkl[hkl2].to_dict()
        base1.update(quintile_shape_for_hkl(quintiles, hkl1))
        base2.update(quintile_shape_for_hkl(quintiles, hkl2))
        metrics = pair_metrics(base1, base2, int(min_nonzero))
        focus_id = focus_pair_id(hkl1, hkl2)
        for role, hkl, mate, base in [("A", hkl1, hkl2, base1), ("B", hkl2, hkl1, base2)]:
            row = {
                "symmetry_pair_id": pair.symmetry_pair_id,
                "focus_pair_id": focus_id,
                "is_four_main_candidate_pair": bool(focus_id),
                "pair_role": role,
                "h": hkl[0],
                "k": hkl[1],
                "l": hkl[2],
                "hkl": hkl_label(hkl),
                "mate_hkl": hkl_label(mate),
                "accepted_observation_count": int(base.get("n_observations", base.get("accepted_observation_count_from_observations", 0))),
                "nonzero_coupling_count": int(base.get("nonzero_coupling_observation_count", base.get("nonzero_coupling_count_from_observations", 0))),
                "zero_coupling_observation_count": int(base.get("zero_coupling_observation_count", 0)),
                "excitation_deficit_spread_q90_q10": base.get("excitation_deficit_spread_q90_q10", np.nan),
                "robust_slope": base.get(PRIMARY_SLOPE_COLUMN, np.nan),
                "bootstrap_ci95_low": base.get("slope_ci95_low", np.nan),
                "bootstrap_ci95_high": base.get("slope_ci95_high", np.nan),
                "spearman_rho": base.get(PRIMARY_RHO_COLUMN, np.nan),
                "partial_spearman_rho_ctrl_Eg_partiality": base.get(PRIMARY_PARTIAL_RHO_COLUMN, np.nan),
                "high_target_n": base.get("n_high_target_quartile", np.nan),
                "high_target_slope": base.get("slope_corrected_residual_vs_deficit_high_target_quartile", np.nan),
                "high_target_rho": base.get("spearman_corrected_residual_vs_deficit_high_target_quartile", np.nan),
                "high_partiality_n": base.get("n_high_partiality_quartile", np.nan),
                "high_partiality_slope": base.get("slope_corrected_residual_vs_deficit_high_partiality_quartile", np.nan),
                "high_partiality_rho": base.get("spearman_corrected_residual_vs_deficit_high_partiality_quartile", np.nan),
                "slope_v5_low_tertile": base.get("slope_corrected_residual_vs_deficit_v5_low_tertile", np.nan),
                "slope_v5_medium_tertile": base.get("slope_corrected_residual_vs_deficit_v5_medium_tertile", np.nan),
                "slope_v5_high_tertile": base.get("slope_corrected_residual_vs_deficit_v5_high_tertile", np.nan),
                "rho_v5_low_tertile": base.get("spearman_corrected_residual_vs_deficit_v5_low_tertile", np.nan),
                "rho_v5_medium_tertile": base.get("spearman_corrected_residual_vs_deficit_v5_medium_tertile", np.nan),
                "rho_v5_high_tertile": base.get("spearman_corrected_residual_vs_deficit_v5_high_tertile", np.nan),
                "quintile_medians": base.get("quintile_medians", ""),
                "quintile_monotonic": base.get("quintile_monotonic", False),
                "quintile_direction": base.get("quintile_direction", ""),
                "extreme_quintile_drives_slope": base.get("extreme_quintile_drives_slope", False),
                "extreme_step_fraction": base.get("extreme_step_fraction", np.nan),
                "fraction_intensity_outliers": base.get("fraction_intensity_outliers_abs_robust_z_gt_3", np.nan),
                "orientation_region_count": base.get("orientation_region_count", np.nan),
                "orientation_dominant_region_fraction": base.get("orientation_dominant_region_fraction", np.nan),
                "orientation_vector_rms_spread": base.get("orientation_vector_rms_spread", np.nan),
            }
            row.update(metrics)
            signed_rows.append(row)
    return pd.DataFrame.from_records(signed_rows)


def candidate_pairs_for_validation(summary: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, tuple[int, int, int], tuple[int, int, int]]]:
    pairs: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]] = []
    for pair_id, group in summary.groupby("symmetry_pair_id", sort=False):
        if len(group) < 2:
            continue
        rows = group.sort_values("pair_role")
        first = rows.iloc[0]
        second = rows.iloc[1]
        hkl1 = (int(first.h), int(first.k), int(first.l))
        hkl2 = (int(second.h), int(second.k), int(second.l))
        is_focus = bool(first.get("is_four_main_candidate_pair", False))
        classification = str(first.get("stage1_pair_classification", ""))
        include = False
        if args.cv_all_symmetry_pairs:
            include = True
        elif args.candidate_only_focus_pairs:
            include = is_focus
        else:
            include = is_focus or classification in {"robust correction candidate", "promising but uncertain"}
        if include:
            pairs.append((str(pair_id), hkl1, hkl2))
    return pairs


def clean_model_frame(group: pd.DataFrame, response: str, covariates: list[str], include_deficit: bool = True) -> pd.DataFrame:
    required = [response, *covariates]
    if include_deficit and DEFICIT_COLUMN not in required:
        required.append(DEFICIT_COLUMN)
    required.append(COUPLING_COLUMN)
    if "I_robust_z_within_hkl" in group.columns:
        required.append("I_robust_z_within_hkl")
    if TARGET_COLUMN not in required:
        required.append(TARGET_COLUMN)
    if PARTIALITY_COLUMN not in required:
        required.append(PARTIALITY_COLUMN)
    keep = [column for column in dict.fromkeys([*HKL_COLUMNS, "source_filename", "event", "closest_uvw", "continuous_uvw_x", "continuous_uvw_y", "continuous_uvw_z", *required]) if column in group.columns]
    out = group.loc[:, keep].copy()
    for column in [response, *covariates, DEFICIT_COLUMN, COUPLING_COLUMN, TARGET_COLUMN, PARTIALITY_COLUMN, "I_robust_z_within_hkl"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.loc[out[COUPLING_COLUMN] > 0.0].dropna(subset=[response, *covariates, DEFICIT_COLUMN]).reset_index(drop=True)
    return out


def add_train_normalized_response(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, str]:
    train_out = train.copy()
    test_out = test.copy()
    train_intensity = pd.to_numeric(train_out[CV_RESPONSE_SOURCE_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
    denom = float(train_intensity.median()) if len(train_intensity.dropna()) else np.nan
    note = "median_train_observation_intensity"
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        denom = float(train_intensity.abs().median()) if len(train_intensity.dropna()) else np.nan
        note = "fallback_median_abs_train_observation_intensity"
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        train_out[CV_RESPONSE_COLUMN] = np.nan
        test_out[CV_RESPONSE_COLUMN] = np.nan
        return train_out, test_out, np.nan, "normalization_failed"
    train_out[CV_RESPONSE_COLUMN] = pd.to_numeric(train_out[CV_RESPONSE_SOURCE_COLUMN], errors="coerce").to_numpy(dtype=float) / denom
    test_out[CV_RESPONSE_COLUMN] = pd.to_numeric(test_out[CV_RESPONSE_SOURCE_COLUMN], errors="coerce").to_numpy(dtype=float) / denom
    return train_out, test_out, denom, note


def orientation_region(frame: pd.DataFrame) -> pd.Series:
    if "closest_uvw" in frame.columns:
        values = frame["closest_uvw"].astype(str).fillna("unknown")
        counts = values.value_counts()
        rare = set(counts[counts < 8].index)
        return values.where(~values.isin(rare), other="other_orientation")
    if {"continuous_uvw_x", "continuous_uvw_y", "continuous_uvw_z"}.issubset(frame.columns):
        x = pd.to_numeric(frame["continuous_uvw_x"], errors="coerce").fillna(0.0)
        y = pd.to_numeric(frame["continuous_uvw_y"], errors="coerce").fillna(0.0)
        z = pd.to_numeric(frame["continuous_uvw_z"], errors="coerce").abs().fillna(0.0)
        z_bin = qbin_labels(z, 2, "z")
        labels = [
            f"{'xp' if xv >= 0 else 'xn'}_{'yp' if yv >= 0 else 'yn'}_{zv}"
            for xv, yv, zv in zip(x.to_numpy(dtype=float), y.to_numpy(dtype=float), z_bin.astype(str), strict=False)
        ]
        return pd.Series(labels, index=frame.index)
    return pd.Series("all_orientation", index=frame.index)


def qbin_labels(values: pd.Series, bins: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.dropna().nunique() < 2:
        return pd.Series(f"{prefix}0", index=values.index)
    ranks = numeric.rank(method="first", pct=True)
    idx = np.minimum(np.floor(ranks.fillna(0.0).to_numpy(dtype=float) * bins).astype(int), bins - 1)
    return pd.Series([f"{prefix}{value}" for value in idx], index=values.index)


def split_strata(frame: pd.DataFrame) -> pd.Series:
    eg = qbin_labels(frame[TARGET_COLUMN], 3, "eg")
    part = qbin_labels(frame[PARTIALITY_COLUMN], 3, "p")
    deficit = qbin_labels(frame[DEFICIT_COLUMN], 3, "d")
    orient = orientation_region(frame)
    strata = eg.astype(str) + "|" + part.astype(str) + "|" + deficit.astype(str) + "|" + orient.astype(str)
    counts = strata.value_counts()
    small = set(counts[counts < 4].index)
    coarse = eg.astype(str) + "|" + deficit.astype(str) + "|" + orient.astype(str)
    return strata.where(~strata.isin(small), other=coarse)


def deterministic_stratified_split(frame: pd.DataFrame, split_index: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 1009 * int(split_index) + 17 * len(frame))
    strata = split_strata(frame)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for _, idx_values in strata.groupby(strata, sort=False).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        rng.shuffle(idx)
        if len(idx) < 4:
            cutoff = int(round(len(idx) * train_fraction))
            cutoff = min(max(cutoff, 1), len(idx))
        else:
            cutoff = int(round(len(idx) * train_fraction))
            cutoff = min(max(cutoff, 1), len(idx) - 1)
        train_indices.extend(idx[:cutoff].tolist())
        test_indices.extend(idx[cutoff:].tolist())
    train = np.asarray(sorted(set(train_indices)), dtype=int)
    test = np.asarray(sorted(set(test_indices)), dtype=int)
    if len(test) == 0 or len(train) == 0:
        all_idx = np.arange(len(frame), dtype=int)
        rng.shuffle(all_idx)
        cutoff = min(max(int(round(len(frame) * train_fraction)), 1), len(frame) - 1)
        train = np.sort(all_idx[:cutoff])
        test = np.sort(all_idx[cutoff:])
    return train, test


def fit_deficit_slope_from_training_residuals(train: pd.DataFrame, baseline_residual_train: np.ndarray) -> tuple[float, float]:
    deficit = pd.to_numeric(train[DEFICIT_COLUMN], errors="coerce").to_numpy(dtype=float)
    residual = np.asarray(baseline_residual_train, dtype=float)
    mask = np.isfinite(deficit) & np.isfinite(residual)
    if int(mask.sum()) < 3 or len(np.unique(deficit[mask])) < 2:
        return np.nan, np.nan
    reference_deficit = float(np.nanmedian(deficit[mask]))
    beta = robust_slope_simple(deficit[mask] - reference_deficit, residual[mask])
    return beta, reference_deficit


def train_quintile_offsets_from_residuals(train: pd.DataFrame, baseline_residual_train: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(baseline_residual_train, dtype=float)
    deficit = pd.to_numeric(train[DEFICIT_COLUMN], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(residual) & np.isfinite(deficit)
    if int(mask.sum()) < 10 or len(np.unique(deficit[mask])) < 2:
        return {"edges": np.asarray([], dtype=float), "offsets": {}, "default_offset": 0.0, "n_bins": 0}
    ranks = pd.Series(deficit[mask]).rank(method="first")
    qlabels = pd.qcut(ranks, q=min(5, int(mask.sum())), labels=False, duplicates="drop")
    train_valid = pd.DataFrame({"deficit": deficit[mask], "residual": residual[mask], "q": np.asarray(qlabels, dtype=int)})
    edges = np.quantile(train_valid["deficit"].to_numpy(dtype=float), np.linspace(0.0, 1.0, int(train_valid["q"].nunique()) + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    offsets = train_valid.groupby("q", sort=True)["residual"].median().to_dict()
    default_offset = float(np.median(residual[mask]))
    centered = {int(key): float(value - default_offset) for key, value in offsets.items()}
    return {"edges": edges, "offsets": centered, "default_offset": 0.0, "n_bins": int(len(centered))}


def apply_quintile_offsets_to_residuals(frame: pd.DataFrame, baseline_residual: np.ndarray, offsets: dict[str, Any]) -> np.ndarray:
    if int(offsets.get("n_bins", 0)) <= 0:
        return np.asarray(baseline_residual, dtype=float).copy()
    deficit = pd.to_numeric(frame[DEFICIT_COLUMN], errors="coerce").to_numpy(dtype=float)
    edges = np.asarray(offsets["edges"], dtype=float)
    bins = np.searchsorted(edges, deficit, side="right") - 1
    max_bin = max(offsets["offsets"].keys()) if offsets["offsets"] else 0
    bins = np.clip(bins, 0, max_bin)
    adjustment = np.asarray([offsets["offsets"].get(int(bin_idx), 0.0) for bin_idx in bins], dtype=float)
    return np.asarray(baseline_residual, dtype=float) - adjustment


def residual_metrics(y: np.ndarray, residual: np.ndarray, deficit: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    y = np.asarray(y, dtype=float)
    return {
        "mae": float(np.nanmean(np.abs(residual))) if len(residual) else np.nan,
        "median_abs_residual": float(np.nanmedian(np.abs(residual))) if len(residual) else np.nan,
        "residual_spearman_vs_deficit": spearman_corr(residual, deficit),
        "r2": r2_score(y, y - residual),
        "explained_variance": explained_variance(y, residual),
    }


def run_within_hkl_crossvalidation(
    observations: pd.DataFrame,
    candidate_pairs: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]],
    scale_columns: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    covariates = [TARGET_COLUMN, PARTIALITY_COLUMN, *scale_columns]
    hkls: dict[tuple[int, int, int], str] = {}
    for pair_id, hkl1, hkl2 in candidate_pairs:
        hkls[hkl1] = pair_id
        hkls[hkl2] = pair_id
    for hkl, pair_id in hkls.items():
        group = observations.loc[(observations["h"].astype(int) == hkl[0]) & (observations["k"].astype(int) == hkl[1]) & (observations["l"].astype(int) == hkl[2])]
        frame = clean_model_frame(group, CV_RESPONSE_SOURCE_COLUMN, covariates)
        if len(frame) < int(args.min_train_obs) + int(args.min_test_obs):
            rows.append({"symmetry_pair_id": pair_id, "h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "split": -1, "status": "insufficient_data", "n_usable": int(len(frame))})
            continue
        for split in range(int(args.splits)):
            train_idx, test_idx = deterministic_stratified_split(frame, split, float(args.train_fraction), int(args.seed))
            if len(train_idx) < int(args.min_train_obs) or len(test_idx) < int(args.min_test_obs):
                rows.append({"symmetry_pair_id": pair_id, "h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "split": split, "status": "split_too_small", "n_usable": int(len(frame)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
                continue
            train = frame.iloc[train_idx].copy()
            test = frame.iloc[test_idx].copy()
            train, test, response_norm_denom, response_norm_note = add_train_normalized_response(train, test)
            if not np.isfinite(response_norm_denom):
                rows.append({"symmetry_pair_id": pair_id, "h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "split": split, "status": "normalization_failed", "n_usable": int(len(frame)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
                continue
            base_fit = fit_model(train, CV_RESPONSE_COLUMN, covariates)
            base_pred_train = predict_model(base_fit, train)
            base_pred_test = predict_model(base_fit, test)
            y_train = pd.to_numeric(train[CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(test[CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
            baseline_residual_train = y_train - base_pred_train
            baseline_residual_test = y - base_pred_test
            beta_deficit, reference_deficit = fit_deficit_slope_from_training_residuals(train, baseline_residual_train)
            deficit = pd.to_numeric(test[DEFICIT_COLUMN], errors="coerce").to_numpy(dtype=float)
            extended_residual_test = baseline_residual_test - beta_deficit * (deficit - reference_deficit)
            offsets = train_quintile_offsets_from_residuals(train, baseline_residual_train)
            quintile_residual_test = apply_quintile_offsets_to_residuals(test, baseline_residual_test, offsets)
            base_metrics = residual_metrics(y, baseline_residual_test, deficit)
            ext_metrics = residual_metrics(y, extended_residual_test, deficit)
            quint_metrics = residual_metrics(y, quintile_residual_test, deficit)
            rows.append(
                {
                    "symmetry_pair_id": pair_id,
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "hkl": hkl_label(hkl),
                    "split": split,
                    "status": "ok",
                    "n_usable": int(len(frame)),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "cv_response_column": CV_RESPONSE_COLUMN,
                    "cv_response_source_column": CV_RESPONSE_SOURCE_COLUMN,
                    "response_normalization_source": response_norm_note,
                    "response_normalization_denominator_train": response_norm_denom,
                    "stratification": "target_excitation_x_partiality_x_excitation_deficit_x_orientation_region",
                    "baseline_covariates": ";".join(covariates),
                    "extended_model": "training_baseline_residual ~ excitation_deficit_norm; centered correction applied to held-out baseline residuals",
                    "beta_excitation_deficit_train": beta_deficit,
                    "reference_deficit_train": reference_deficit,
                    "test_data_used_for_baseline_fit_or_correction_fit": False,
                    "baseline_heldout_mae": base_metrics["mae"],
                    "extended_heldout_mae": ext_metrics["mae"],
                    "linear_delta_mae_positive_improves": base_metrics["mae"] - ext_metrics["mae"],
                    "baseline_heldout_median_abs_residual": base_metrics["median_abs_residual"],
                    "extended_heldout_median_abs_residual": ext_metrics["median_abs_residual"],
                    "linear_delta_median_abs_residual_positive_improves": base_metrics["median_abs_residual"] - ext_metrics["median_abs_residual"],
                    "baseline_residual_spearman_vs_deficit": base_metrics["residual_spearman_vs_deficit"],
                    "extended_residual_spearman_vs_deficit": ext_metrics["residual_spearman_vs_deficit"],
                    "abs_residual_rho_reduction_positive_improves": abs(base_metrics["residual_spearman_vs_deficit"]) - abs(ext_metrics["residual_spearman_vs_deficit"]),
                    "baseline_heldout_r2": base_metrics["r2"],
                    "extended_heldout_r2": ext_metrics["r2"],
                    "linear_delta_r2_positive_improves": ext_metrics["r2"] - base_metrics["r2"],
                    "baseline_explained_variance": base_metrics["explained_variance"],
                    "extended_explained_variance": ext_metrics["explained_variance"],
                    "linear_improves_mae": bool(ext_metrics["mae"] < base_metrics["mae"]),
                    "linear_improves_median_abs_residual": bool(ext_metrics["median_abs_residual"] < base_metrics["median_abs_residual"]),
                    "quintile_n_bins": int(offsets.get("n_bins", 0)),
                    "quintile_heldout_mae": quint_metrics["mae"],
                    "quintile_delta_mae_positive_improves": base_metrics["mae"] - quint_metrics["mae"],
                    "quintile_heldout_median_abs_residual": quint_metrics["median_abs_residual"],
                    "quintile_delta_median_abs_residual_positive_improves": base_metrics["median_abs_residual"] - quint_metrics["median_abs_residual"],
                    "quintile_residual_spearman_vs_deficit": quint_metrics["residual_spearman_vs_deficit"],
                    "quintile_heldout_r2": quint_metrics["r2"],
                    "quintile_beats_linear_mae": bool(quint_metrics["mae"] < ext_metrics["mae"]),
                }
            )
    out = pd.DataFrame.from_records(rows)
    if not out.empty and "status" in out.columns:
        out = add_cv_group_summaries(out)
    return out


def add_cv_group_summaries(cv: pd.DataFrame) -> pd.DataFrame:
    out = cv.copy()
    ok = out["status"] == "ok"
    for hkl, group in out.loc[ok].groupby(HKL_COLUMNS, sort=False):
        idx = group.index
        slopes = pd.to_numeric(group["beta_excitation_deficit_train"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(slopes):
            median_slope = float(slopes.median())
            sign = sign_value(median_slope)
            fraction_same = float((slopes.apply(sign_value) == sign).mean()) if sign else np.nan
            ci_low, ci_high = np.quantile(slopes.to_numpy(dtype=float), [0.025, 0.975]) if len(slopes) >= 2 else (np.nan, np.nan)
        else:
            median_slope = np.nan
            fraction_same = np.nan
            ci_low = np.nan
            ci_high = np.nan
        out.loc[idx, "split_slope_median"] = median_slope
        out.loc[idx, "split_slope_ci95_low"] = ci_low
        out.loc[idx, "split_slope_ci95_high"] = ci_high
        out.loc[idx, "fraction_splits_same_slope_sign_as_median"] = fraction_same
        out.loc[idx, "fraction_splits_linear_mae_improves"] = float(group["linear_improves_mae"].astype(bool).mean())
        out.loc[idx, "median_linear_delta_mae_positive_improves"] = float(pd.to_numeric(group["linear_delta_mae_positive_improves"], errors="coerce").median())
        out.loc[idx, "median_quintile_delta_mae_positive_improves"] = float(pd.to_numeric(group["quintile_delta_mae_positive_improves"], errors="coerce").median())
        out.loc[idx, "fraction_splits_quintile_beats_linear_mae"] = float(group["quintile_beats_linear_mae"].astype(bool).mean())
    return out


def pair_frame(observations: pd.DataFrame, hkl: tuple[int, int, int], covariates: list[str]) -> pd.DataFrame:
    group = observations.loc[(observations["h"].astype(int) == hkl[0]) & (observations["k"].astype(int) == hkl[1]) & (observations["l"].astype(int) == hkl[2])]
    return clean_model_frame(group, CV_RESPONSE_SOURCE_COLUMN, covariates)


def transfer_one_direction(
    pair_id: str,
    train_hkl: tuple[int, int, int],
    test_hkl: tuple[int, int, int],
    train: pd.DataFrame,
    test: pd.DataFrame,
    covariates: list[str],
) -> dict[str, Any]:
    if len(train) < 20 or len(test) < 20:
        return {"symmetry_pair_id": pair_id, "train_hkl": hkl_label(train_hkl), "test_hkl": hkl_label(test_hkl), "status": "insufficient_data", "n_train": int(len(train)), "n_test": int(len(test))}
    train, test, response_norm_denom, response_norm_note = add_train_normalized_response(train, test)
    if not np.isfinite(response_norm_denom):
        return {"symmetry_pair_id": pair_id, "train_hkl": hkl_label(train_hkl), "test_hkl": hkl_label(test_hkl), "status": "normalization_failed", "n_train": int(len(train)), "n_test": int(len(test))}
    baseline_fit = fit_model(train, CV_RESPONSE_COLUMN, covariates)
    train_baseline_pred = predict_model(baseline_fit, train)
    test_baseline_pred = predict_model(baseline_fit, test)
    y_train = pd.to_numeric(train[CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test[CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
    train_baseline_residual = y_train - train_baseline_pred
    test_baseline_residual = y_test - test_baseline_pred
    beta, reference_deficit = fit_deficit_slope_from_training_residuals(train, train_baseline_residual)
    own_beta_test = robust_slope_simple(test[DEFICIT_COLUMN], test_baseline_residual)
    deficit_test = pd.to_numeric(test[DEFICIT_COLUMN], errors="coerce").to_numpy(dtype=float)
    after_residual = test_baseline_residual - beta * (deficit_test - reference_deficit)
    before_corr = spearman_corr(test_baseline_residual, deficit_test)
    after_corr = spearman_corr(after_residual, deficit_test)
    before_mad = mad(test_baseline_residual)
    after_mad = mad(after_residual)
    slope_ratio = abs(beta) / abs(own_beta_test) if np.isfinite(own_beta_test) and abs(own_beta_test) > 1e-12 else np.nan
    return {
        "symmetry_pair_id": pair_id,
        "train_hkl": hkl_label(train_hkl),
        "test_hkl": hkl_label(test_hkl),
        "status": "ok",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "cv_response_column": CV_RESPONSE_COLUMN,
        "cv_response_source_column": CV_RESPONSE_SOURCE_COLUMN,
        "response_normalization_source": response_norm_note,
        "response_normalization_denominator_train_mate": response_norm_denom,
        "baseline_fit_source": "training_mate_only",
        "dexc_fit_source": "training_mate_baseline_residuals_only",
        "test_mate_intercept_or_normalization_used": "none",
        "fully_transferred_prediction_without_test_mate_refit": True,
        "reference_deficit_train": reference_deficit,
        "transferred_beta_excitation_deficit": beta,
        "test_apparent_beta_after_train_baseline_not_used_for_correction": own_beta_test,
        "transferred_slope_sign_agrees_with_test_own_slope": bool(sign_value(beta) != 0 and sign_value(beta) == sign_value(own_beta_test)),
        "transferred_to_test_abs_slope_ratio": slope_ratio,
        "before_spearman_baseline_residual_vs_deficit": before_corr,
        "after_spearman_corrected_residual_vs_deficit": after_corr,
        "abs_correlation_reduction_positive_improves": abs(before_corr) - abs(after_corr),
        "before_median_abs_normalized_residual": before_mad,
        "after_median_abs_normalized_residual": after_mad,
        "median_abs_normalized_residual_improvement": before_mad - after_mad,
        "transfer_improves_residual_and_correlation": bool(after_mad < before_mad and abs(after_corr) < abs(before_corr)),
        "common_slope_plausible_this_direction": bool(np.isfinite(slope_ratio) and 0.5 <= slope_ratio <= 2.0 and sign_value(beta) == sign_value(own_beta_test)),
    }


def run_symmetry_transfer_validation(
    observations: pd.DataFrame,
    candidate_pairs: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]],
    scale_columns: list[str],
) -> pd.DataFrame:
    covariates = [TARGET_COLUMN, PARTIALITY_COLUMN, *scale_columns]
    rows: list[dict[str, Any]] = []
    for pair_id, hkl1, hkl2 in candidate_pairs:
        frame1 = pair_frame(observations, hkl1, covariates)
        frame2 = pair_frame(observations, hkl2, covariates)
        rows.append(transfer_one_direction(pair_id, hkl1, hkl2, frame1, frame2, covariates))
        rows.append(transfer_one_direction(pair_id, hkl2, hkl1, frame2, frame1, covariates))
    out = pd.DataFrame.from_records(rows)
    if not out.empty and "status" in out.columns:
        ok = out["status"] == "ok"
        for pair_id, group in out.loc[ok].groupby("symmetry_pair_id", sort=False):
            idx = group.index
            both_improve = bool(group["transfer_improves_residual_and_correlation"].astype(bool).all()) if len(group) == 2 else False
            common = bool(group["common_slope_plausible_this_direction"].astype(bool).all()) if len(group) == 2 else False
            out.loc[idx, "transfer_improves_both_directions"] = both_improve
            out.loc[idx, "correction_strength_assessment"] = "common_slope_plausible" if common else "reflection_specific_slope_likely_needed"
    return out


def permuted_sham_beta(train: pd.DataFrame, baseline_residual_train: np.ndarray, beta_magnitude: float, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    values = pd.to_numeric(train[DEFICIT_COLUMN], errors="coerce").to_numpy(dtype=float).copy()
    rng.shuffle(values)
    residual = np.asarray(baseline_residual_train, dtype=float)
    mask = np.isfinite(values) & np.isfinite(residual)
    beta_perm = robust_slope_simple(values[mask] - np.nanmedian(values[mask]), residual[mask]) if int(mask.sum()) >= 3 else np.nan
    sign = sign_value(beta_perm)
    if sign == 0:
        sign = -sign_value(beta_magnitude) or 1
    return float(sign * abs(beta_magnitude))


def correction_sim_metrics(
    response_before: np.ndarray,
    response_after: np.ndarray,
    residual_before: np.ndarray,
    residual_after: np.ndarray,
    deficit: np.ndarray,
    robust_z: np.ndarray | None = None,
) -> dict[str, float]:
    high_target_placeholder = np.nan
    out = {
        "scatter_std_before": float(np.nanstd(residual_before)),
        "scatter_std_after": float(np.nanstd(residual_after)),
        "mad_before": mad(residual_before),
        "mad_after": mad(residual_after),
        "mad_improvement_positive_improves": mad(residual_before) - mad(residual_after),
        "spearman_before_vs_deficit": spearman_corr(residual_before, deficit),
        "spearman_after_vs_deficit": spearman_corr(residual_after, deficit),
        "abs_spearman_reduction_positive_improves": abs(spearman_corr(residual_before, deficit)) - abs(spearman_corr(residual_after, deficit)),
        "mean_before": float(np.nanmean(response_before)),
        "mean_after": float(np.nanmean(response_after)),
        "mean_shift_after_minus_before": float(np.nanmean(response_after) - np.nanmean(response_before)),
        "median_before": float(np.nanmedian(response_before)),
        "median_after": float(np.nanmedian(response_after)),
        "median_shift_after_minus_before": float(np.nanmedian(response_after) - np.nanmedian(response_before)),
        "high_target_placeholder": high_target_placeholder,
    }
    if robust_z is not None:
        mask = np.isfinite(robust_z) & (np.abs(robust_z) <= 3.0)
        out["nonoutlier_fraction_test"] = float(mask.mean()) if len(mask) else np.nan
        out["nonoutlier_mad_before"] = mad(residual_before[mask]) if np.any(mask) else np.nan
        out["nonoutlier_mad_after"] = mad(residual_after[mask]) if np.any(mask) else np.nan
        out["nonoutlier_mad_improvement_positive_improves"] = out["nonoutlier_mad_before"] - out["nonoutlier_mad_after"]
    return out


def subset_mad_improvement(test: pd.DataFrame, residual_before: np.ndarray, residual_after: np.ndarray, subset_column: str) -> tuple[int, float, float, float]:
    values = pd.to_numeric(test[subset_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(values.dropna()) < 4:
        return 0, np.nan, np.nan, np.nan
    threshold = float(values.quantile(0.75))
    mask = values.to_numpy(dtype=float) >= threshold
    if int(mask.sum()) < 3:
        return int(mask.sum()), np.nan, np.nan, np.nan
    before = mad(residual_before[mask])
    after = mad(residual_after[mask])
    return int(mask.sum()), before, after, before - after


def run_heldout_correction_simulation(
    observations: pd.DataFrame,
    candidate_pairs: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]],
    scale_columns: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    covariates = [TARGET_COLUMN, PARTIALITY_COLUMN, *scale_columns]
    for pair_id, hkl1, hkl2 in candidate_pairs:
        split_rows_for_pair: list[dict[str, Any]] = []
        for hkl in [hkl1, hkl2]:
            frame = pair_frame(observations, hkl, covariates)
            if len(frame) < int(args.min_train_obs) + int(args.min_test_obs):
                split_rows_for_pair.append({"symmetry_pair_id": pair_id, "h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "split": -1, "status": "insufficient_data", "n_usable": int(len(frame))})
                continue
            for split in range(int(args.splits)):
                train_idx, test_idx = deterministic_stratified_split(frame, split, float(args.train_fraction), int(args.seed) + 53)
                if len(train_idx) < int(args.min_train_obs) or len(test_idx) < int(args.min_test_obs):
                    split_rows_for_pair.append({"symmetry_pair_id": pair_id, "h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "split": split, "status": "split_too_small", "n_usable": int(len(frame)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
                    continue
                train = frame.iloc[train_idx].copy()
                test = frame.iloc[test_idx].copy()
                train, test, response_norm_denom, response_norm_note = add_train_normalized_response(train, test)
                if not np.isfinite(response_norm_denom):
                    split_rows_for_pair.append({"symmetry_pair_id": pair_id, "h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "split": split, "status": "normalization_failed", "n_usable": int(len(frame)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
                    continue
                baseline_fit = fit_model(train, CV_RESPONSE_COLUMN, covariates)
                train_baseline_pred = predict_model(baseline_fit, train)
                test_baseline_pred = predict_model(baseline_fit, test)
                y_train = pd.to_numeric(train[CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
                baseline_residual_train = y_train - train_baseline_pred
                beta, reference_deficit = fit_deficit_slope_from_training_residuals(train, baseline_residual_train)
                deficit_test = pd.to_numeric(test[DEFICIT_COLUMN], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(test[CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
                baseline_residual_test = y - test_baseline_pred
                correction = beta * (deficit_test - reference_deficit)
                y_corr = y - correction
                corrected_residual_test = baseline_residual_test - correction
                beta_sham = permuted_sham_beta(train, baseline_residual_train, beta, int(args.seed) + 7919 * (split + 1) + 97 * (abs(hkl[0]) + abs(hkl[1]) + abs(hkl[2])))
                sham_correction = beta_sham * (deficit_test - reference_deficit)
                y_sham = y - sham_correction
                sham_residual_test = baseline_residual_test - sham_correction
                robust_z = pd.to_numeric(test.get("I_robust_z_within_hkl", pd.Series(index=test.index, dtype=float)), errors="coerce").to_numpy(dtype=float)
                metrics = correction_sim_metrics(y, y_corr, baseline_residual_test, corrected_residual_test, deficit_test, robust_z)
                sham_metrics = correction_sim_metrics(y, y_sham, baseline_residual_test, sham_residual_test, deficit_test, robust_z)
                high_target_n, high_target_before, high_target_after, high_target_improve = subset_mad_improvement(test, baseline_residual_test, corrected_residual_test, TARGET_COLUMN)
                high_part_n, high_part_before, high_part_after, high_part_improve = subset_mad_improvement(test, baseline_residual_test, corrected_residual_test, PARTIALITY_COLUMN)
                row = {
                    "symmetry_pair_id": pair_id,
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "hkl": hkl_label(hkl),
                    "split": split,
                    "status": "ok",
                    "n_usable": int(len(frame)),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "response_column": CV_RESPONSE_COLUMN,
                    "response_source_column": CV_RESPONSE_SOURCE_COLUMN,
                    "response_normalization_source": response_norm_note,
                    "response_normalization_denominator_train": response_norm_denom,
                    "baseline_covariates": ";".join(covariates),
                    "correction_formula": "response_corrected=response_observed-beta_train*(excitation_deficit_norm-median_train_deficit); residual_after=baseline_residual_test-beta_train*(excitation_deficit_norm-median_train_deficit)",
                    "baseline_fit_source": "training_observations_only",
                    "dexc_fit_source": "training_baseline_residuals_only",
                    "test_data_used_for_baseline_fit_or_correction_fit": False,
                    "beta_train": beta,
                    "median_train_deficit": reference_deficit,
                    "sham_beta_same_magnitude_permuted_train_deficit": beta_sham,
                    "high_target_n_test": high_target_n,
                    "high_target_mad_before": high_target_before,
                    "high_target_mad_after": high_target_after,
                    "high_target_mad_improvement_positive_improves": high_target_improve,
                    "high_partiality_n_test": high_part_n,
                    "high_partiality_mad_before": high_part_before,
                    "high_partiality_mad_after": high_part_after,
                    "high_partiality_mad_improvement_positive_improves": high_part_improve,
                    "sham_mad_after": sham_metrics["mad_after"],
                    "sham_mad_improvement_positive_improves": sham_metrics["mad_improvement_positive_improves"],
                    "correction_beats_sham_mad": bool(metrics["mad_improvement_positive_improves"] > sham_metrics["mad_improvement_positive_improves"]),
                }
                row.update({key: value for key, value in metrics.items() if key != "high_target_placeholder"})
                split_rows_for_pair.append(row)
        rows.extend(add_pair_agreement_to_sim_rows(split_rows_for_pair))
    return pd.DataFrame.from_records(rows)


def add_pair_agreement_to_sim_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame.from_records(rows)
    if frame.empty or "status" not in frame.columns:
        return rows
    ok = frame["status"] == "ok"
    for split, group in frame.loc[ok].groupby("split", sort=False):
        if len(group) < 2:
            continue
        med_before = pd.to_numeric(group["median_before"], errors="coerce").to_numpy(dtype=float)
        med_after = pd.to_numeric(group["median_after"], errors="coerce").to_numpy(dtype=float)
        if len(med_before) != 2 or len(med_after) != 2:
            continue
        gap_before = float(abs(med_before[0] - med_before[1]))
        gap_after = float(abs(med_after[0] - med_after[1]))
        idx = group.index
        frame.loc[idx, "symmetry_mate_median_gap_before"] = gap_before
        frame.loc[idx, "symmetry_mate_median_gap_after"] = gap_after
        frame.loc[idx, "symmetry_mate_median_gap_improvement_positive_improves"] = gap_before - gap_after
        frame.loc[idx, "symmetry_mate_agreement_improves"] = bool(gap_after < gap_before)
    return frame.to_dict("records")


def aggregate_hkl_cv(cv: pd.DataFrame) -> pd.DataFrame:
    if cv.empty or "status" not in cv.columns:
        return pd.DataFrame()
    ok = cv.loc[cv["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for hkl, group in ok.groupby(HKL_COLUMNS, sort=False):
        hkl_int = tuple(map(int, hkl))
        slopes = pd.to_numeric(group["beta_excitation_deficit_train"], errors="coerce").dropna()
        if len(slopes) >= 2:
            slope_ci_low, slope_ci_high = np.quantile(slopes.to_numpy(dtype=float), [0.025, 0.975])
        else:
            slope_ci_low, slope_ci_high = np.nan, np.nan
        slope_sign_fraction = group["fraction_splits_same_slope_sign_as_median"].dropna()
        rows.append(
            {
                "h": hkl_int[0],
                "k": hkl_int[1],
                "l": hkl_int[2],
                "median_cv_delta_mae": float(pd.to_numeric(group["linear_delta_mae_positive_improves"], errors="coerce").median()),
                "fraction_cv_splits_improved_mae": float(group["linear_improves_mae"].astype(bool).mean()),
                "median_cv_delta_abs_rho": float(pd.to_numeric(group["abs_residual_rho_reduction_positive_improves"], errors="coerce").median()),
                "median_quintile_delta_mae": float(pd.to_numeric(group["quintile_delta_mae_positive_improves"], errors="coerce").median()),
                "fraction_quintile_beats_linear": float(group["quintile_beats_linear_mae"].astype(bool).mean()),
                "slope_median": float(slopes.median()) if len(slopes) else np.nan,
                "slope_ci95_low": float(slope_ci_low),
                "slope_ci95_high": float(slope_ci_high),
                "fraction_splits_same_slope_sign_as_median": float(slope_sign_fraction.iloc[0]) if not slope_sign_fraction.empty else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def aggregate_transfer(transfer: pd.DataFrame) -> pd.DataFrame:
    if transfer.empty or "status" not in transfer.columns:
        return pd.DataFrame()
    ok = transfer.loc[transfer["status"] == "ok"].copy()
    rows: list[dict[str, Any]] = []
    for pair_id, group in ok.groupby("symmetry_pair_id", sort=False):
        rows.append(
            {
                "symmetry_pair_id": pair_id,
                "transfer_improves_both_directions": bool(group["transfer_improves_residual_and_correlation"].astype(bool).all()) if len(group) == 2 else False,
                "median_transfer_mad_improvement": float(pd.to_numeric(group["median_abs_normalized_residual_improvement"], errors="coerce").median()),
                "median_transfer_abs_rho_reduction": float(pd.to_numeric(group["abs_correlation_reduction_positive_improves"], errors="coerce").median()),
                "all_transfer_slope_signs_agree": bool(group["transferred_slope_sign_agrees_with_test_own_slope"].astype(bool).all()) if len(group) == 2 else False,
                "correction_strength_assessment": str(group["correction_strength_assessment"].dropna().iloc[0]) if "correction_strength_assessment" in group and group["correction_strength_assessment"].dropna().any() else "unknown",
            }
        )
    return pd.DataFrame.from_records(rows)


def aggregate_simulation(sim: pd.DataFrame) -> pd.DataFrame:
    if sim.empty or "status" not in sim.columns:
        return pd.DataFrame()
    ok = sim.loc[sim["status"] == "ok"].copy()
    rows: list[dict[str, Any]] = []
    for hkl, group in ok.groupby(HKL_COLUMNS, sort=False):
        hkl_int = tuple(map(int, hkl))
        rows.append(
            {
                "h": hkl_int[0],
                "k": hkl_int[1],
                "l": hkl_int[2],
                "median_sim_mad_improvement": float(pd.to_numeric(group["mad_improvement_positive_improves"], errors="coerce").median()),
                "median_sim_mean_shift_abs": float(pd.to_numeric(group["mean_shift_after_minus_before"], errors="coerce").abs().median()),
                "median_sim_median_shift_abs": float(pd.to_numeric(group["median_shift_after_minus_before"], errors="coerce").abs().median()),
                "fraction_sim_beats_sham_mad": float(group["correction_beats_sham_mad"].astype(bool).mean()),
                "median_high_target_mad_improvement": float(pd.to_numeric(group["high_target_mad_improvement_positive_improves"], errors="coerce").median()),
                "median_high_partiality_mad_improvement": float(pd.to_numeric(group["high_partiality_mad_improvement_positive_improves"], errors="coerce").median()),
                "fraction_symmetry_mate_agreement_improves": float(group.get("symmetry_mate_agreement_improves", pd.Series(False, index=group.index)).astype(bool).mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_decision_table(summary: pd.DataFrame, cv: pd.DataFrame, transfer: pd.DataFrame, sim: pd.DataFrame, min_nonzero: int) -> pd.DataFrame:
    cv_hkl = hkl_index(aggregate_hkl_cv(cv)) if not cv.empty else {}
    sim_hkl = hkl_index(aggregate_simulation(sim)) if not sim.empty else {}
    transfer_pair = {str(row.symmetry_pair_id): row for _, row in aggregate_transfer(transfer).iterrows()} if not transfer.empty else {}
    rows: list[dict[str, Any]] = []
    for pair_id, group in summary.groupby("symmetry_pair_id", sort=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values("pair_role")
        first = ordered.iloc[0]
        second = ordered.iloc[1]
        hkl1 = (int(first.h), int(first.k), int(first.l))
        hkl2 = (int(second.h), int(second.k), int(second.l))
        cv1 = cv_hkl.get(hkl1, pd.Series(dtype=object))
        cv2 = cv_hkl.get(hkl2, pd.Series(dtype=object))
        sim1 = sim_hkl.get(hkl1, pd.Series(dtype=object))
        sim2 = sim_hkl.get(hkl2, pd.Series(dtype=object))
        tr = transfer_pair.get(str(pair_id), pd.Series(dtype=object))
        same_slope_sign = bool(first.get("slope_sign_agreement", False))
        stable_sign = bool(finite_float(cv1.get("fraction_splits_same_slope_sign_as_median")) >= 0.80 and finite_float(cv2.get("fraction_splits_same_slope_sign_as_median")) >= 0.80)
        heldout_improves = bool(finite_float(cv1.get("median_cv_delta_mae")) > 0.0 and finite_float(cv2.get("median_cv_delta_mae")) > 0.0 and finite_float(cv1.get("fraction_cv_splits_improved_mae")) >= 0.55 and finite_float(cv2.get("fraction_cv_splits_improved_mae")) >= 0.55)
        transfer_improves = bool(tr.get("transfer_improves_both_directions", False)) if len(tr) else False
        not_extreme = bool(first.get("trend_not_extreme_quintile_driven", False))
        enough_nonzero = bool(first.get("enough_nonzero_coupling_observations", False))
        mean_ok = bool(
            finite_float(sim1.get("median_sim_mean_shift_abs"), np.inf) <= max(0.10, 0.25 * abs(finite_float(sim1.get("median_sim_mad_improvement"), 0.0)) + 0.10)
            and finite_float(sim2.get("median_sim_mean_shift_abs"), np.inf) <= max(0.10, 0.25 * abs(finite_float(sim2.get("median_sim_mad_improvement"), 0.0)) + 0.10)
        )
        high_quality_ok = bool(
            finite_float(sim1.get("median_high_target_mad_improvement"), -np.inf) >= -0.02
            and finite_float(sim2.get("median_high_target_mad_improvement"), -np.inf) >= -0.02
            and finite_float(sim1.get("median_high_partiality_mad_improvement"), -np.inf) >= -0.02
            and finite_float(sim2.get("median_high_partiality_mad_improvement"), -np.inf) >= -0.02
        )
        no_major_degradation = mean_ok and high_quality_ok
        ready = same_slope_sign and stable_sign and heldout_improves and transfer_improves and not_extreme and enough_nonzero and no_major_degradation
        nonlinear_preferred = bool(finite_float(cv1.get("median_quintile_delta_mae")) > finite_float(cv1.get("median_cv_delta_mae")) and finite_float(cv2.get("median_quintile_delta_mae")) > finite_float(cv2.get("median_cv_delta_mae")))
        common_slope = str(tr.get("correction_strength_assessment", "")) == "common_slope_plausible"
        if not ready:
            recommendation = "none"
        elif nonlinear_preferred:
            recommendation = "nonlinear quintile correction"
        elif common_slope:
            recommendation = "common linear slope"
        else:
            recommendation = "reflection-specific linear slope"
        concerns = decision_concerns(
            same_slope_sign=same_slope_sign,
            stable_sign=stable_sign,
            heldout_improves=heldout_improves,
            transfer_improves=transfer_improves,
            not_extreme=not_extreme,
            enough_nonzero=enough_nonzero,
            no_major_degradation=no_major_degradation,
            first=first,
            second=second,
        )
        rows.append(
            {
                "symmetry_pair_id": pair_id,
                "orbit_or_pair": f"{hkl_label(hkl1)} <-> {hkl_label(hkl2)}",
                "focus_pair_id": first.get("focus_pair_id", ""),
                "stage1_pair_classification": first.get("stage1_pair_classification", ""),
                "correction_ready": "yes" if ready else "no",
                "recommended_correction_form": recommendation,
                "same_slope_sign_for_both_mates": same_slope_sign,
                "stable_heldout_slope_sign": stable_sign,
                "heldout_improvement_over_baseline": heldout_improves,
                "transferred_correction_improves_opposite_mate": transfer_improves,
                "trend_not_driven_only_by_one_extreme_quintile": not_extreme,
                "enough_nonzero_coupling_observations": enough_nonzero,
                "no_major_degradation_mean_or_high_quality_subsets": no_major_degradation,
                "cross_validated_improvement": f"median delta MAE {finite_float(cv1.get('median_cv_delta_mae')):.6g}, {finite_float(cv2.get('median_cv_delta_mae')):.6g}",
                "symmetry_transfer_improvement": f"median transfer MAD delta {finite_float(tr.get('median_transfer_mad_improvement')):.6g}" if len(tr) else "not run",
                "slope_stability": f"same-sign split fractions {finite_float(cv1.get('fraction_splits_same_slope_sign_as_median')):.3g}, {finite_float(cv2.get('fraction_splits_same_slope_sign_as_median')):.3g}",
                "residual_concerns": concerns,
                "sample_size_limitations": sample_size_note(first, second, min_nonzero),
                "preprocessing_confounding_warning": SCIENTIFIC_WARNING,
            }
        )
    return pd.DataFrame.from_records(rows)


def decision_concerns(**kwargs: Any) -> str:
    labels = []
    for key, label in [
        ("same_slope_sign", "mate slope signs disagree"),
        ("stable_sign", "held-out slope sign unstable"),
        ("heldout_improves", "no consistent held-out improvement"),
        ("transfer_improves", "symmetry transfer does not improve both directions"),
        ("not_extreme", "quintile trend may be extreme-bin driven"),
        ("enough_nonzero", "limited nonzero-coupling observations"),
        ("no_major_degradation", "mean or high-quality subset degradation"),
    ]:
        if not bool(kwargs[key]):
            labels.append(label)
    first = kwargs["first"]
    second = kwargs["second"]
    if not bool(first.get("partial_rho_agreement", False)):
        labels.append("partial-rho agreement weak")
    if not bool(first.get("quintile_shape_agreement", False)):
        labels.append("quintile shape agreement weak")
    if not labels:
        return "none flagged by scripted gates"
    # Keep the CSV readable.
    return "; ".join(dict.fromkeys(labels))


def sample_size_note(first: pd.Series, second: pd.Series, min_nonzero: int) -> str:
    n1 = int(first.get("nonzero_coupling_count", 0))
    n2 = int(second.get("nonzero_coupling_count", 0))
    if n1 < min_nonzero or n2 < min_nonzero:
        return f"nonzero counts below threshold {min_nonzero}: {n1}, {n2}"
    if min(n1, n2) < 200:
        return f"modest nonzero counts: {n1}, {n2}"
    return f"nonzero counts: {n1}, {n2}"


def make_validation_plots(
    out_dir: Path,
    summary: pd.DataFrame,
    cv: pd.DataFrame,
    transfer: pd.DataFrame,
    sim: pd.DataFrame,
    dpi: int,
) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pair_id, group in summary.groupby("symmetry_pair_id", sort=False):
        if not bool(group["is_four_main_candidate_pair"].any()) and str(group["stage1_pair_classification"].iloc[0]) not in {"robust correction candidate", "promising but uncertain"}:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        ax = axes[0, 0]
        for _, row in group.iterrows():
            medians = parse_float_list(row.get("quintile_medians", ""))
            if medians:
                ax.plot(np.arange(1, len(medians) + 1), medians, marker="o", label=str(row["hkl"]))
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_xlabel("excitation-deficit quintile")
        ax.set_ylabel("median corrected residual")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[0, 1]
        cv_pair = cv.loc[(cv.get("symmetry_pair_id", pd.Series(dtype=object)) == pair_id) & (cv.get("status", pd.Series(dtype=object)) == "ok")].copy() if not cv.empty else pd.DataFrame()
        if not cv_pair.empty:
            data = [pd.to_numeric(sub["linear_delta_mae_positive_improves"], errors="coerce").dropna().to_numpy(dtype=float) for _, sub in cv_pair.groupby("hkl", sort=False)]
            labels = [str(label) for label, _ in cv_pair.groupby("hkl", sort=False)]
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_ylabel("held-out delta MAE")
        ax.tick_params(axis="x", rotation=25)

        ax = axes[1, 0]
        transfer_pair = transfer.loc[(transfer.get("symmetry_pair_id", pd.Series(dtype=object)) == pair_id) & (transfer.get("status", pd.Series(dtype=object)) == "ok")].copy() if not transfer.empty else pd.DataFrame()
        if not transfer_pair.empty:
            x = np.arange(len(transfer_pair))
            before = pd.to_numeric(transfer_pair["before_median_abs_normalized_residual"], errors="coerce").to_numpy(dtype=float)
            after = pd.to_numeric(transfer_pair["after_median_abs_normalized_residual"], errors="coerce").to_numpy(dtype=float)
            ax.bar(x - 0.18, before, width=0.36, label="before")
            ax.bar(x + 0.18, after, width=0.36, label="after")
            ax.set_xticks(x, [f"{r.train_hkl}->\n{r.test_hkl}" for _, r in transfer_pair.iterrows()], fontsize=7)
            ax.legend(frameon=False, fontsize=8)
        ax.set_ylabel("transfer median abs residual")

        ax = axes[1, 1]
        sim_pair = sim.loc[(sim.get("symmetry_pair_id", pd.Series(dtype=object)) == pair_id) & (sim.get("status", pd.Series(dtype=object)) == "ok")].copy() if not sim.empty else pd.DataFrame()
        if not sim_pair.empty:
            data = [pd.to_numeric(sub["mad_improvement_positive_improves"], errors="coerce").dropna().to_numpy(dtype=float) for _, sub in sim_pair.groupby("hkl", sort=False)]
            labels = [str(label) for label, _ in sim_pair.groupby("hkl", sort=False)]
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_ylabel("simulation delta MAD")
        ax.tick_params(axis="x", rotation=25)
        fig.suptitle(f"{pair_id}: {', '.join(group['hkl'].astype(str).tolist())}")
        path = plot_dir / f"candidate_validation_{pair_id}.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths.append(path)
    return paths


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    paths: InputPaths,
    control_summary: dict[str, Any],
    stat_columns: dict[str, list[str]],
    candidate_pairs: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]],
    plot_paths: list[Path],
) -> None:
    lines = [
        "# V5 Excitation-Deficit Correction Candidate Validation",
        "",
        "Read-only validation of candidate v5 excitation-deficit correction relationships. The completed broad diagnostic was not modified.",
        "",
        "## Inputs",
        "",
        f"- completed diagnostic directory: `{paths.diagnostic_dir}`",
        f"- diagnostic observations: `{paths.observations}`",
        f"- per-HKL stats: `{paths.per_hkl}`",
        f"- symmetry mate comparison: `{paths.symmetry}`",
        "",
        "## Response and Controls",
        "",
        f"- reported primary slope column: `{control_summary['reported_primary_slope_column']}`",
        f"- original in-sample diagnostic response: `{control_summary['reported_primary_response']}`",
        f"- leakage-free held-out CV response: `{control_summary['cv_response_column']}`",
        f"- x variable: `{control_summary['reported_primary_x']}`",
        f"- observation intensity used upstream: `{control_summary['observation_intensity_used']}`",
        f"- upstream response already controls target excitation: `{control_summary['response_precontrols_target_excitation']}`",
        f"- upstream response already controls partiality: `{control_summary['response_precontrols_partiality']}`",
        f"- held-out baseline predictors: `{control_summary['heldout_baseline_predictors']}`",
        f"- held-out scale term found: `{control_summary['heldout_scale_term_found']}`",
        f"- upstream scale columns from metadata: `{control_summary['scale_columns_from_metadata']}`",
        f"- scale-like columns in diagnostic observations: `{control_summary['scale_like_columns_in_observations']}`",
        "",
        "Stage 1 reproduces the original in-sample descriptive statistics from `corrected_intensity_residual`. All held-out stages use a train-only equivalent of `I_over_hkl_median`, computed as `observation_intensity / median_train_observation_intensity`, fit baseline coefficients on training observations only, compute training/test baseline residuals from that training-fitted baseline, and fit/apply Dexc corrections from training residuals only.",
        "",
        "Leakage-control confirmation: no test observations enter baseline fitting, normalization, centering, quintile boundaries, or correction fitting.",
        "",
        "## Definitions Read from the Diagnostic",
        "",
        "- zero-coupling observations are those with `coupling_sum_raw <= 0`; the completed per-HKL slopes and quintiles use nonzero-coupling observations.",
        "- quintiles are rank-based within signed HKL for `excitation_deficit_norm`, `nonself_local_excitation_raw`, or `coupling_sum_raw`, with median normalized responses reported per quintile.",
        "- no raw-intensity correction is applied here; correction simulation stays in train-median-normalized response units and reports held-out baseline-residual changes.",
        "",
        "## Available Statistic Columns",
        "",
        f"- slope columns: `{stat_columns['slope_columns']}`",
        f"- rho columns: `{stat_columns['rho_columns']}`",
        f"- partial-rho columns: `{stat_columns['partial_rho_columns']}`",
        f"- confidence interval columns: `{stat_columns['confidence_interval_columns']}`",
        f"- subset columns: `{stat_columns['subset_columns']}`",
        "",
        "## Held-Out Scope",
        "",
        f"- repeated splits per signed HKL: `{args.splits}`",
        f"- approximate train/test fraction: `{args.train_fraction:.2f}/{1.0 - float(args.train_fraction):.2f}`",
        "- split stratification: target excitation, partiality, excitation deficit, and orientation region.",
        f"- candidate pairs entering held-out stages: `{[(pair_id, hkl_label(a), hkl_label(b)) for pair_id, a, b in candidate_pairs]}`",
        "",
        "## Outputs",
        "",
        "- `candidate_pair_validation_summary.csv`",
        "- `within_hkl_crossvalidation_results.csv`",
        "- `symmetry_transfer_validation.csv`",
        "- `heldout_correction_simulation.csv`",
        "- `correction_candidate_decision_table.csv`",
        "- `candidate_quintile_diagnostics.csv`",
        "- `run_metadata.json`",
        f"- compact plots: `{len(plot_paths)}` PNG files under `plots/`",
        "",
        "## Scientific Warning",
        "",
        SCIENTIFIC_WARNING,
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    out_dir: Path,
    args: argparse.Namespace,
    paths: InputPaths,
    control_summary: dict[str, Any],
    stat_columns: dict[str, list[str]],
    candidate_pairs: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]],
    outputs: dict[str, Path],
) -> None:
    metadata = {
        "command": "tools/analyze_and_crossvalidate_v5_dexc_correction_candidates.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_input_dir": str(paths.diagnostic_dir),
        "output_dir": str(out_dir),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "response_and_controls": control_summary,
        "heldout_validation_leakage_controls": control_summary["heldout_leakage_controls"],
        "cv_response_column": control_summary["cv_response_column"],
        "baseline_predictors": control_summary["heldout_baseline_predictors"],
        "scale_term_found": control_summary["heldout_scale_term_found"],
        "available_stat_columns": stat_columns,
        "candidate_pairs_validated": [
            {"symmetry_pair_id": pair_id, "hkl_a": hkl_label(a), "hkl_b": hkl_label(b)} for pair_id, a, b in candidate_pairs
        ],
        "outputs": {key: str(path) for key, path in outputs.items()},
        "did_not_run": ["stream rewrite", "Partialator", "merge", "structure refinement", "v5 recomputation"],
        "warning": SCIENTIFIC_WARNING,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_focus_summary(decision: pd.DataFrame, summary: pd.DataFrame, cv: pd.DataFrame, transfer: pd.DataFrame) -> None:
    print("\nFour main candidate-pair summary:")
    if decision.empty:
        print("  No decision rows were produced.")
        return
    for idx, (left, right) in enumerate(FOCUS_PAIRS, start=1):
        key = pair_key(left, right)
        match = None
        for _, row in decision.iterrows():
            text = str(row.get("orbit_or_pair", ""))
            if hkl_label(left) in text and hkl_label(right) in text:
                match = row
                break
        if match is None:
            print(f"  {idx}. {hkl_label(left)} <-> {hkl_label(right)}: not present in symmetry comparison")
            continue
        pair_id = str(match["symmetry_pair_id"])
        cv_pair = cv.loc[(cv.get("symmetry_pair_id", pd.Series(dtype=object)) == pair_id) & (cv.get("status", pd.Series(dtype=object)) == "ok")] if not cv.empty else pd.DataFrame()
        transfer_pair = transfer.loc[(transfer.get("symmetry_pair_id", pd.Series(dtype=object)) == pair_id) & (transfer.get("status", pd.Series(dtype=object)) == "ok")] if not transfer.empty else pd.DataFrame()
        cv_delta = finite_float(pd.to_numeric(cv_pair.get("linear_delta_mae_positive_improves", pd.Series(dtype=float)), errors="coerce").median()) if not cv_pair.empty else np.nan
        transfer_delta = finite_float(pd.to_numeric(transfer_pair.get("median_abs_normalized_residual_improvement", pd.Series(dtype=float)), errors="coerce").median()) if not transfer_pair.empty else np.nan
        print(
            f"  {idx}. {match['orbit_or_pair']}: ready={match['correction_ready']}, "
            f"form={match['recommended_correction_form']}, stage1={match['stage1_pair_classification']}, "
            f"median CV delta MAE={cv_delta:.6g}, median transfer delta MAD={transfer_delta:.6g}"
        )


def main() -> int:
    args = parse_args()
    paths = input_paths(args.diagnostic_dir)
    if paths.diagnostic_dir.resolve() == args.out_dir.resolve():
        raise SystemExit("--out-dir must be different from --diagnostic-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading completed broad diagnostic CSVs")
    data = load_inputs(paths)
    observations: pd.DataFrame = data["observations"]
    per_hkl: pd.DataFrame = data["per_hkl"]
    quintiles: pd.DataFrame = data["quintiles"]
    symmetry: pd.DataFrame = data["symmetry"]

    control_summary = response_control_summary(data["metadata"], observations)
    stat_columns = available_stat_columns(per_hkl)
    scale_columns = [column for column in control_summary["scale_columns_from_metadata"] if column in observations.columns]

    log("Building Stage 1 candidate-pair validation summary")
    candidate_summary = build_candidate_pair_validation_summary(per_hkl, quintiles, observations, symmetry, int(args.min_nonzero_coupling_obs))
    quintile_diagnostics = build_candidate_quintile_diagnostics(quintiles, symmetry)
    candidate_pairs = candidate_pairs_for_validation(candidate_summary, args)
    if not candidate_pairs:
        log("No candidate pairs passed the held-out-stage selection gates; Stage 2-4 outputs will be empty.")

    log(f"Running held-out within-HKL validation for {len(candidate_pairs)} symmetry pair(s)")
    cv = run_within_hkl_crossvalidation(observations, candidate_pairs, scale_columns, args)
    log("Running leave-one-symmetry-mate-out transfer validation")
    transfer = run_symmetry_transfer_validation(observations, candidate_pairs, scale_columns)
    log("Running held-out correction simulation in normalized response units")
    simulation = run_heldout_correction_simulation(observations, candidate_pairs, scale_columns, args)
    log("Building correction decision table")
    decision = build_decision_table(candidate_summary, cv, transfer, simulation, int(args.min_nonzero_coupling_obs))

    outputs = {
        "candidate_pair_validation_summary": args.out_dir / "candidate_pair_validation_summary.csv",
        "within_hkl_crossvalidation_results": args.out_dir / "within_hkl_crossvalidation_results.csv",
        "symmetry_transfer_validation": args.out_dir / "symmetry_transfer_validation.csv",
        "heldout_correction_simulation": args.out_dir / "heldout_correction_simulation.csv",
        "correction_candidate_decision_table": args.out_dir / "correction_candidate_decision_table.csv",
        "candidate_quintile_diagnostics": args.out_dir / "candidate_quintile_diagnostics.csv",
    }
    candidate_summary.to_csv(outputs["candidate_pair_validation_summary"], index=False)
    cv.to_csv(outputs["within_hkl_crossvalidation_results"], index=False)
    transfer.to_csv(outputs["symmetry_transfer_validation"], index=False)
    simulation.to_csv(outputs["heldout_correction_simulation"], index=False)
    decision.to_csv(outputs["correction_candidate_decision_table"], index=False)
    quintile_diagnostics.to_csv(outputs["candidate_quintile_diagnostics"], index=False)

    log("Writing compact validation plots and documentation")
    plot_paths = make_validation_plots(args.out_dir, candidate_summary, cv, transfer, simulation, int(args.plot_dpi))
    write_readme(args.out_dir, args, paths, control_summary, stat_columns, candidate_pairs, plot_paths)
    outputs["README"] = args.out_dir / "README.md"
    outputs["run_metadata"] = args.out_dir / "run_metadata.json"
    write_metadata(args.out_dir, args, paths, control_summary, stat_columns, candidate_pairs, outputs)

    print_focus_summary(decision, candidate_summary, cv, transfer)
    print(f"\nValidation outputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
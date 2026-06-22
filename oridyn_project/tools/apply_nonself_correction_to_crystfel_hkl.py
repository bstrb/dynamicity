#!/usr/bin/env python3
"""Apply a lambda=0.5 nonself correction to an existing CrystFEL merged HKL file.

This is a proof-of-concept diagnostic tool. It does not rerun partialator,
OriDyn, or any merging. It derives observation-level nonself trends from an
existing joined component observations CSV, converts those trends to HKL-level
intensity deltas, then applies deltas onto an existing merged CrystFEL HKL file.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
from typing import NamedTuple

import numpy as np
import pandas as pd


NONSELF_COMPONENT_COLUMNS = [
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

REQUIRED_JOINED_COLUMNS = [
    "h",
    "k",
    "l",
    "I_pr",
    "residual",
    "d_spacing",
    "shell_label",
    *NONSELF_COMPONENT_COLUMNS,
]


class ReflectionRow(NamedTuple):
    line_idx: int
    line_no: int
    h: int
    k: int
    l: int
    intensity: float
    phase: str
    sigma_i: float
    nmeas: int
    raw_line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply proof-of-concept nonself correction deltas (lambda=0.5) to an "
            "existing merged CrystFEL HKL file."
        )
    )
    parser.add_argument(
        "--joined-component-observations",
        required=True,
        type=Path,
        help="Path to sdyn_component_diagnostics/joined_component_observations.csv",
    )
    parser.add_argument("--input-hkl", required=True, type=Path, help="Input merged CrystFEL HKL file")
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder for corrected HKL and diagnostics")

    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--min-obs", type=int, default=100)
    parser.add_argument("--min-tail-obs", type=int, default=8)
    parser.add_argument("--min-abs-relative-shift", type=float, default=0.5)
    parser.add_argument("--min-abs-rho", type=float, default=0.15)
    parser.add_argument("--min-nonself-spread", type=float, default=0.05)
    parser.add_argument("--intensity-floor", type=float, default=1e-9)
    parser.add_argument("--delta-method", choices=["mean", "median"], default="median")
    parser.add_argument(
        "--match-abs-hkl",
        action="store_true",
        help="Diagnostic mode: match by abs(h),abs(k),abs(l) instead of exact signed HKL",
    )

    args = parser.parse_args()

    if not args.joined_component_observations.exists():
        raise SystemExit(f"--joined-component-observations not found: {args.joined_component_observations}")
    if not args.input_hkl.exists():
        raise SystemExit(f"--input-hkl not found: {args.input_hkl}")
    if args.min_obs <= 0:
        raise SystemExit("--min-obs must be > 0")
    if args.min_tail_obs <= 0:
        raise SystemExit("--min-tail-obs must be > 0")
    if args.min_abs_relative_shift < 0.0:
        raise SystemExit("--min-abs-relative-shift must be >= 0")
    if args.min_abs_rho < 0.0:
        raise SystemExit("--min-abs-rho must be >= 0")
    if args.min_nonself_spread < 0.0:
        raise SystemExit("--min-nonself-spread must be >= 0")
    if args.intensity_floor <= 0.0:
        raise SystemExit("--intensity-floor must be > 0")
    if not (0.0 <= args.low_quantile < args.high_quantile <= 1.0):
        raise SystemExit("Expected 0 <= --low-quantile < --high-quantile <= 1")

    # Intentionally fixed for this proof-of-concept mode.
    if not np.isclose(float(args.lambda_value), 0.5, atol=1e-12):
        raise SystemExit("This proof-of-concept currently supports only --lambda-value 0.5")

    return args


def hkl_text(h: int, k: int, l: int) -> str:
    return f"({h},{k},{l})"


def ensure_output_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "corrected_hkl": root / "crystfel_nonself_lambda05.hkl",
        "original_copy_hkl": root / "crystfel_original_copy.hkl",
        "correction_table_csv": root / "hkl_correction_table.csv",
        "applied_rows_csv": root / "applied_hkl_corrections.csv",
        "unmatched_csv": root / "unmatched_correction_hkls.csv",
        "readme": root / "README_nonself_lambda05_hkl_correction.txt",
    }
    root.mkdir(parents=True, exist_ok=True)
    return paths


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].corr(frame["y"], method="spearman"))


def sign_agrees(a: float, b: float) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    if a == 0.0 or b == 0.0:
        return False
    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


def load_joined_observations(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [c for c in REQUIRED_JOINED_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"joined_component_observations.csv missing required column(s): {missing}")

    out = table.copy()
    out["h"] = pd.to_numeric(out["h"], errors="coerce")
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["l"] = pd.to_numeric(out["l"], errors="coerce")
    out["I_pr"] = pd.to_numeric(out["I_pr"], errors="coerce")
    out["residual"] = pd.to_numeric(out["residual"], errors="coerce")
    out["d_spacing"] = pd.to_numeric(out["d_spacing"], errors="coerce")
    for col in NONSELF_COMPONENT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["h", "k", "l", "I_pr", "residual"]).copy()
    out[["h", "k", "l"]] = out[["h", "k", "l"]].astype("int64")

    out["nonself_mean"] = out[NONSELF_COMPONENT_COLUMNS].mean(axis=1, skipna=True)
    out = out.dropna(subset=["nonself_mean"]).copy()
    out["hkl"] = "(" + out["h"].astype(str) + "," + out["k"].astype(str) + "," + out["l"].astype(str) + ")"
    return out


def compute_hkl_correction_table(
    joined: pd.DataFrame,
    min_obs: int,
    low_quantile: float,
    high_quantile: float,
    min_tail_obs: int,
    min_abs_relative_shift: float,
    min_abs_rho: float,
    min_nonself_spread: float,
    lambda_value: float,
    delta_method: str,
    intensity_floor: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = joined.groupby(["h", "k", "l"], sort=True)
    for (h, k, l), g in grouped:
        n_obs = int(len(g))
        nonself = pd.to_numeric(g["nonself_mean"], errors="coerce")

        q_low = float(nonself.quantile(low_quantile))
        q_high = float(nonself.quantile(high_quantile))
        low_mask = nonself <= q_low
        high_mask = nonself >= q_high
        g_low = g.loc[low_mask]
        g_high = g.loc[high_mask]
        n_low = int(len(g_low))
        n_high = int(len(g_high))

        can_compute = n_obs >= int(min_obs) and n_low >= int(min_tail_obs) and n_high >= int(min_tail_obs)

        median_low_i_pr = np.nan
        median_high_i_pr = np.nan
        shift = np.nan
        relative_shift_low_baseline = np.nan
        rho = np.nan
        nonself_p05 = float(nonself.quantile(0.05)) if len(nonself) > 0 else np.nan
        nonself_p95 = float(nonself.quantile(0.95)) if len(nonself) > 0 else np.nan
        nonself_spread = float(nonself_p95 - nonself_p05) if np.isfinite(nonself_p05) and np.isfinite(nonself_p95) else np.nan
        x_low = np.nan
        x_high = np.nan
        slope = np.nan

        if can_compute:
            median_low_i_pr = float(pd.to_numeric(g_low["I_pr"], errors="coerce").median())
            median_high_i_pr = float(pd.to_numeric(g_high["I_pr"], errors="coerce").median())
            shift = float(median_high_i_pr - median_low_i_pr)

            baseline = max(abs(median_low_i_pr), float(intensity_floor))
            relative_shift_low_baseline = float(shift / baseline)
            rho = spearman_corr(g["nonself_mean"], g["residual"])

            x_low = float(pd.to_numeric(g_low["nonself_mean"], errors="coerce").median())
            x_high = float(pd.to_numeric(g_high["nonself_mean"], errors="coerce").median())
            x_span = x_high - x_low
            if np.isfinite(x_span) and abs(x_span) > 0.0:
                slope = float(shift / x_span)

        sign_match = sign_agrees(float(shift), float(rho)) if np.isfinite(shift) and np.isfinite(rho) else False
        correction_eligible = bool(
            can_compute
            and np.isfinite(relative_shift_low_baseline)
            and np.isfinite(rho)
            and np.isfinite(nonself_spread)
            and np.isfinite(slope)
            and abs(float(relative_shift_low_baseline)) >= float(min_abs_relative_shift)
            and abs(float(rho)) >= float(min_abs_rho)
            and sign_match
            and float(nonself_spread) >= float(min_nonself_spread)
        )

        i_pr = pd.to_numeric(g["I_pr"], errors="coerce")
        if correction_eligible:
            nonself_centered = pd.to_numeric(g["nonself_mean"], errors="coerce") - float(x_low)
            i_pr_corr = i_pr - float(lambda_value) * float(slope) * nonself_centered
        else:
            i_pr_corr = i_pr.copy()

        delta_values = i_pr_corr - i_pr
        delta_mean = float(pd.to_numeric(delta_values, errors="coerce").mean())
        delta_median = float(pd.to_numeric(delta_values, errors="coerce").median())
        if not correction_eligible:
            delta_mean = 0.0
            delta_median = 0.0

        apply_delta = float(delta_median if delta_method == "median" else delta_mean)

        rows.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "hkl": hkl_text(int(h), int(k), int(l)),
                "d_spacing": first_non_null(pd.to_numeric(g["d_spacing"], errors="coerce")),
                "shell_label": str(first_non_null(g["shell_label"])),
                "n_obs": n_obs,
                "n_low": n_low,
                "n_high": n_high,
                "median_low_I_pr": median_low_i_pr,
                "median_high_I_pr": median_high_i_pr,
                "shift": shift,
                "relative_shift_low_baseline": relative_shift_low_baseline,
                "rho": rho,
                "nonself_p05": nonself_p05,
                "nonself_p95": nonself_p95,
                "nonself_spread": nonself_spread,
                "x_low": x_low,
                "x_high": x_high,
                "slope": slope,
                "sign_agree": bool(sign_match),
                "correction_eligible": bool(correction_eligible),
                "delta_mean": float(delta_mean),
                "delta_median": float(delta_median),
                "apply_delta": float(apply_delta),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["lambda_value"] = float(lambda_value)
    table["delta_method"] = str(delta_method)
    table["min_obs"] = int(min_obs)
    table["min_tail_obs"] = int(min_tail_obs)
    table["low_quantile"] = float(low_quantile)
    table["high_quantile"] = float(high_quantile)
    table["min_abs_relative_shift"] = float(min_abs_relative_shift)
    table["min_abs_rho"] = float(min_abs_rho)
    table["min_nonself_spread"] = float(min_nonself_spread)

    table = table.sort_values(["d_spacing", "h", "k", "l"], ascending=[False, True, True, True]).reset_index(drop=True)
    return table


def parse_reflection_row(line: str, line_idx: int, line_no: int) -> ReflectionRow | None:
    parts = line.split()
    if len(parts) < 7:
        return None

    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        intensity = float(parts[3])
        phase = parts[4]
        sigma_i = float(parts[5])
        nmeas = int(parts[6])
    except ValueError:
        return None

    return ReflectionRow(
        line_idx=line_idx,
        line_no=line_no,
        h=h,
        k=k,
        l=l,
        intensity=intensity,
        phase=phase,
        sigma_i=sigma_i,
        nmeas=nmeas,
        raw_line=line,
    )


def format_reflection_line(h: int, k: int, l: int, intensity: float, phase: str, sigma_i: float, nmeas: int) -> str:
    # Keep phase right-aligned so intensity and phase remain split-token compatible.
    return f"{h:4d}{k:5d}{l:5d}{intensity:12.2f}{phase:>9s}{sigma_i:12.2f}{nmeas:8d}\n"


def read_crystfel_hkl(path: Path) -> tuple[list[str], int, int, list[ReflectionRow]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    header_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if line.strip().startswith("h") and "sigma(I)" in line and "nmeas" in line:
            header_idx = i
            break

    if header_idx is None:
        raise SystemExit(f"Could not find CrystFEL HKL table header in input file: {path}")

    for i in range(header_idx + 1, len(lines)):
        if lines[i].strip() == "End of reflections":
            end_idx = i
            break

    if end_idx is None:
        raise SystemExit(f"Could not find 'End of reflections' marker in input file: {path}")

    reflections: list[ReflectionRow] = []
    for i in range(header_idx + 1, end_idx):
        parsed = parse_reflection_row(lines[i], line_idx=i, line_no=i + 1)
        if parsed is not None:
            reflections.append(parsed)

    return lines, header_idx, end_idx, reflections


def build_lookup(table: pd.DataFrame, match_abs_hkl: bool) -> tuple[dict[tuple[int, int, int], dict[str, object]], int]:
    if table.empty:
        return {}, 0

    payload_cols = [
        "h",
        "k",
        "l",
        "hkl",
        "d_spacing",
        "shell_label",
        "n_obs",
        "n_low",
        "n_high",
        "median_low_I_pr",
        "median_high_I_pr",
        "shift",
        "relative_shift_low_baseline",
        "rho",
        "nonself_spread",
        "x_low",
        "x_high",
        "slope",
        "sign_agree",
        "correction_eligible",
        "delta_mean",
        "delta_median",
        "apply_delta",
        "delta_method",
    ]

    work = table[payload_cols].copy()
    if match_abs_hkl:
        work["key_h"] = work["h"].abs().astype(int)
        work["key_k"] = work["k"].abs().astype(int)
        work["key_l"] = work["l"].abs().astype(int)
    else:
        work["key_h"] = work["h"].astype(int)
        work["key_k"] = work["k"].astype(int)
        work["key_l"] = work["l"].astype(int)

    work["abs_apply_delta"] = pd.to_numeric(work["apply_delta"], errors="coerce").abs()
    work = work.sort_values(["abs_apply_delta", "n_obs"], ascending=[False, False])

    duplicate_count = int(work.duplicated(["key_h", "key_k", "key_l"], keep="first").sum())
    dedup = work.drop_duplicates(["key_h", "key_k", "key_l"], keep="first").copy()

    lookup: dict[tuple[int, int, int], dict[str, object]] = {}
    for row in dedup.itertuples(index=False):
        key = (int(row.key_h), int(row.key_k), int(row.key_l))
        lookup[key] = {
            "h": int(row.h),
            "k": int(row.k),
            "l": int(row.l),
            "hkl": str(row.hkl),
            "d_spacing": row.d_spacing,
            "shell_label": row.shell_label,
            "n_obs": int(row.n_obs),
            "shift": float(row.shift) if pd.notna(row.shift) else np.nan,
            "relative_shift_low_baseline": float(row.relative_shift_low_baseline)
            if pd.notna(row.relative_shift_low_baseline)
            else np.nan,
            "rho": float(row.rho) if pd.notna(row.rho) else np.nan,
            "nonself_spread": float(row.nonself_spread) if pd.notna(row.nonself_spread) else np.nan,
            "sign_agree": bool(row.sign_agree),
            "correction_eligible": bool(row.correction_eligible),
            "delta_mean": float(row.delta_mean) if pd.notna(row.delta_mean) else 0.0,
            "delta_median": float(row.delta_median) if pd.notna(row.delta_median) else 0.0,
            "apply_delta": float(row.apply_delta) if pd.notna(row.apply_delta) else 0.0,
            "delta_method": str(row.delta_method),
        }

    return lookup, duplicate_count


def apply_corrections_to_hkl(
    lines: list[str],
    reflections: list[ReflectionRow],
    lookup: dict[tuple[int, int, int], dict[str, object]],
    match_abs_hkl: bool,
) -> tuple[list[str], pd.DataFrame, set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    out_lines = list(lines)
    diagnostics: list[dict[str, object]] = []

    matched_keys: set[tuple[int, int, int]] = set()
    matched_signed_keys: set[tuple[int, int, int]] = set()

    for row in reflections:
        exact_key = (int(row.h), int(row.k), int(row.l))
        key = (abs(row.h), abs(row.k), abs(row.l)) if match_abs_hkl else exact_key

        info = lookup.get(key)
        matched = info is not None
        if matched:
            matched_keys.add(key)
            matched_signed_keys.add(exact_key)

        if not matched:
            out_line = row.raw_line if row.raw_line.endswith("\n") else row.raw_line + "\n"
            out_lines[row.line_idx] = out_line
            diagnostics.append(
                {
                    "line_no": int(row.line_no),
                    "h": int(row.h),
                    "k": int(row.k),
                    "l": int(row.l),
                    "hkl": hkl_text(int(row.h), int(row.k), int(row.l)),
                    "matched": False,
                    "match_mode": "abs" if match_abs_hkl else "exact",
                    "correction_eligible": False,
                    "delta_method": np.nan,
                    "delta_mean": np.nan,
                    "delta_median": np.nan,
                    "delta_applied": 0.0,
                    "intensity_old": float(row.intensity),
                    "intensity_new": float(row.intensity),
                    "sigma_i": float(row.sigma_i),
                    "nmeas": int(row.nmeas),
                    "source_h": np.nan,
                    "source_k": np.nan,
                    "source_l": np.nan,
                    "source_hkl": np.nan,
                    "source_n_obs": np.nan,
                    "source_shell_label": np.nan,
                    "source_d_spacing": np.nan,
                    "source_shift": np.nan,
                    "source_relative_shift_low_baseline": np.nan,
                    "source_rho": np.nan,
                    "source_nonself_spread": np.nan,
                    "source_sign_agree": np.nan,
                }
            )
            continue

        delta_applied = float(info["apply_delta"])
        intensity_new = float(row.intensity + delta_applied)

        out_lines[row.line_idx] = format_reflection_line(
            int(row.h),
            int(row.k),
            int(row.l),
            intensity_new,
            str(row.phase),
            float(row.sigma_i),
            int(row.nmeas),
        )

        diagnostics.append(
            {
                "line_no": int(row.line_no),
                "h": int(row.h),
                "k": int(row.k),
                "l": int(row.l),
                "hkl": hkl_text(int(row.h), int(row.k), int(row.l)),
                "matched": True,
                "match_mode": "abs" if match_abs_hkl else "exact",
                "correction_eligible": bool(info["correction_eligible"]),
                "delta_method": str(info["delta_method"]),
                "delta_mean": float(info["delta_mean"]),
                "delta_median": float(info["delta_median"]),
                "delta_applied": float(delta_applied),
                "intensity_old": float(row.intensity),
                "intensity_new": float(intensity_new),
                "sigma_i": float(row.sigma_i),
                "nmeas": int(row.nmeas),
                "source_h": int(info["h"]),
                "source_k": int(info["k"]),
                "source_l": int(info["l"]),
                "source_hkl": str(info["hkl"]),
                "source_n_obs": int(info["n_obs"]),
                "source_shell_label": str(info["shell_label"]),
                "source_d_spacing": float(info["d_spacing"]) if pd.notna(info["d_spacing"]) else np.nan,
                "source_shift": float(info["shift"]) if pd.notna(info["shift"]) else np.nan,
                "source_relative_shift_low_baseline": float(info["relative_shift_low_baseline"])
                if pd.notna(info["relative_shift_low_baseline"])
                else np.nan,
                "source_rho": float(info["rho"]) if pd.notna(info["rho"]) else np.nan,
                "source_nonself_spread": float(info["nonself_spread"]) if pd.notna(info["nonself_spread"]) else np.nan,
                "source_sign_agree": bool(info["sign_agree"]),
            }
        )

    diag_df = pd.DataFrame(diagnostics)
    return out_lines, diag_df, matched_keys, matched_signed_keys


def write_summary(
    path: Path,
    input_hkl: Path,
    output_hkl: Path,
    original_copy_hkl: Path,
    correction_table: pd.DataFrame,
    applied_diag: pd.DataFrame,
    unmatched_table: pd.DataFrame,
    n_reflection_rows: int,
    n_unique_reflections: int,
    exact_match_count: int,
    mode_match_count: int,
    eligible_exact_match_count: int,
    eligible_mode_match_count: int,
    modified_count: int,
    match_abs_hkl: bool,
    duplicate_key_choices: int,
) -> None:
    eligible = correction_table.loc[correction_table["correction_eligible"]].copy() if not correction_table.empty else correction_table

    matched_applied = applied_diag.loc[applied_diag["matched"]].copy() if not applied_diag.empty else applied_diag
    changed = applied_diag.loc[(applied_diag["matched"]) & (applied_diag["delta_applied"].abs() > 0.0)].copy() if not applied_diag.empty else applied_diag

    if changed.empty:
        delta_min = 0.0
        delta_med = 0.0
        delta_max = 0.0
        top20 = pd.DataFrame(columns=["hkl", "delta_applied", "intensity_old", "intensity_new", "source_hkl", "source_n_obs"])
    else:
        delta_min = float(changed["delta_applied"].min())
        delta_med = float(changed["delta_applied"].median())
        delta_max = float(changed["delta_applied"].max())
        top20 = changed.copy()
        top20["abs_delta"] = top20["delta_applied"].abs()
        top20 = top20.sort_values(["abs_delta", "source_n_obs"], ascending=[False, False]).head(20)
        top20 = top20[["hkl", "delta_applied", "intensity_old", "intensity_new", "source_hkl", "source_n_obs"]]

    lines: list[str] = []
    lines.append("Proof-of-concept nonself lambda=0.5 correction applied to merged CrystFEL HKL")
    lines.append("")
    lines.append(f"Input merged HKL: {input_hkl}")
    lines.append(f"Output corrected HKL: {output_hkl}")
    lines.append(f"Original copy: {original_copy_hkl}")
    lines.append("")
    lines.append("Summary:")
    lines.append(f"- Number of reflection rows in original merged file: {n_reflection_rows:,}")
    lines.append(f"- Number of unique HKLs in original merged file: {n_unique_reflections:,}")
    lines.append(f"- Number of correction-eligible HKLs: {len(eligible):,}")
    lines.append(f"- Number matched exactly: {exact_match_count:,}")
    lines.append(f"- Number matched in selected mode ({'abs' if match_abs_hkl else 'exact'}): {mode_match_count:,}")
    lines.append(f"- Number of correction-eligible HKLs matched exactly: {eligible_exact_match_count:,}")
    lines.append(
        f"- Number of correction-eligible HKLs matched in selected mode ({'abs' if match_abs_hkl else 'exact'}): "
        f"{eligible_mode_match_count:,}"
    )
    lines.append(f"- Number actually modified (delta != 0): {modified_count:,}")
    lines.append(f"- Number of eligible HKLs unmatched in selected mode: {len(unmatched_table):,}")
    if duplicate_key_choices > 0:
        lines.append(
            "- Diagnostic note: abs-HKL matching collapsed duplicate correction keys and kept the "
            f"largest |delta| for {duplicate_key_choices:,} duplicate key entries."
        )
    lines.append("")
    lines.append("Applied intensity delta statistics (delta_applied):")
    lines.append(f"- min: {delta_min:.6g}")
    lines.append(f"- median: {delta_med:.6g}")
    lines.append(f"- max: {delta_max:.6g}")
    lines.append("")
    lines.append("Top 20 absolute applied corrections:")
    if top20.empty:
        lines.append("(none)")
    else:
        lines.extend(top20.to_string(index=False).splitlines())
    lines.append("")
    lines.append("Warnings:")
    lines.append("- Sigma/ESD values were left unchanged in this proof-of-concept output.")
    lines.append("- This output is diagnostic only; not intended as a final physically validated merge.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure_output_layout(args.output_root)

    joined = load_joined_observations(args.joined_component_observations)

    correction_table = compute_hkl_correction_table(
        joined=joined,
        min_obs=int(args.min_obs),
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
        min_tail_obs=int(args.min_tail_obs),
        min_abs_relative_shift=float(args.min_abs_relative_shift),
        min_abs_rho=float(args.min_abs_rho),
        min_nonself_spread=float(args.min_nonself_spread),
        lambda_value=float(args.lambda_value),
        delta_method=str(args.delta_method),
        intensity_floor=float(args.intensity_floor),
    )

    correction_table.to_csv(out["correction_table_csv"], index=False)

    lines, _header_idx, _end_idx, reflections = read_crystfel_hkl(args.input_hkl)
    n_reflection_rows = int(len(reflections))
    unique_reflection_hkls = {(r.h, r.k, r.l) for r in reflections}
    n_unique_reflections = int(len(unique_reflection_hkls))

    exact_lookup, _exact_dup = build_lookup(correction_table, match_abs_hkl=False)
    mode_lookup, duplicate_key_choices = build_lookup(correction_table, match_abs_hkl=bool(args.match_abs_hkl))

    out_lines, applied_diag, matched_mode_keys, _matched_exact_signed_keys = apply_corrections_to_hkl(
        lines=lines,
        reflections=reflections,
        lookup=mode_lookup,
        match_abs_hkl=bool(args.match_abs_hkl),
    )

    if not correction_table.empty:
        correction_table["exact_key"] = list(zip(correction_table["h"].astype(int), correction_table["k"].astype(int), correction_table["l"].astype(int)))
        correction_table["abs_key"] = list(
            zip(correction_table["h"].abs().astype(int), correction_table["k"].abs().astype(int), correction_table["l"].abs().astype(int))
        )

    eligible = correction_table.loc[correction_table["correction_eligible"]].copy() if not correction_table.empty else correction_table

    if args.match_abs_hkl:
        matched_key_set = set(matched_mode_keys)
        eligible["matched_in_selected_mode"] = [tuple(k) in matched_key_set for k in eligible["abs_key"]]
    else:
        matched_key_set = set(matched_mode_keys)
        eligible["matched_in_selected_mode"] = [tuple(k) in matched_key_set for k in eligible["exact_key"]]

    exact_key_set = set(unique_reflection_hkls)
    if eligible.empty:
        eligible_exact_match_count = 0
        eligible_mode_match_count = 0
    else:
        eligible_exact_match_count = int(sum(tuple(k) in exact_key_set for k in eligible["exact_key"]))
        eligible_mode_match_count = int(eligible["matched_in_selected_mode"].sum())

    unmatched = eligible.loc[~eligible["matched_in_selected_mode"]].copy() if not eligible.empty else eligible
    unmatched = unmatched.drop(columns=[c for c in ["exact_key", "abs_key", "matched_in_selected_mode"] if c in unmatched.columns])
    unmatched.to_csv(out["unmatched_csv"], index=False)

    if "exact_key" in correction_table.columns:
        correction_table = correction_table.drop(columns=["exact_key", "abs_key"])

    # Save original copy and corrected HKL output.
    shutil.copy2(args.input_hkl, out["original_copy_hkl"])
    out["corrected_hkl"].write_text("".join(out_lines), encoding="utf-8")

    applied_diag.to_csv(out["applied_rows_csv"], index=False)

    exact_match_count = int(sum(1 for key in unique_reflection_hkls if key in exact_lookup))
    mode_match_count = int(sum(1 for key in unique_reflection_hkls if key in mode_lookup)) if not args.match_abs_hkl else int(len(matched_mode_keys))
    modified_count = int(((applied_diag["matched"] == True) & (applied_diag["delta_applied"].abs() > 0.0)).sum()) if not applied_diag.empty else 0

    write_summary(
        path=out["readme"],
        input_hkl=args.input_hkl,
        output_hkl=out["corrected_hkl"],
        original_copy_hkl=out["original_copy_hkl"],
        correction_table=correction_table,
        applied_diag=applied_diag,
        unmatched_table=unmatched,
        n_reflection_rows=n_reflection_rows,
        n_unique_reflections=n_unique_reflections,
        exact_match_count=exact_match_count,
        mode_match_count=mode_match_count,
        eligible_exact_match_count=eligible_exact_match_count,
        eligible_mode_match_count=eligible_mode_match_count,
        modified_count=modified_count,
        match_abs_hkl=bool(args.match_abs_hkl),
        duplicate_key_choices=duplicate_key_choices,
    )

    print("NONSELF_APPLY_TO_MERGED_HKL_OK")
    print(f"joined_rows={len(joined):,}")
    print(f"correction_table_rows={len(correction_table):,}")
    print(f"eligible_hkls={int((correction_table['correction_eligible'] == True).sum()) if not correction_table.empty else 0:,}")
    print(f"reflection_rows_in_input_hkl={n_reflection_rows:,}")
    print(f"unique_hkls_in_input_hkl={n_unique_reflections:,}")
    print(f"exact_match_count={exact_match_count:,}")
    print(f"mode_match_count={mode_match_count:,}")
    print(f"eligible_exact_match_count={eligible_exact_match_count:,}")
    print(f"eligible_mode_match_count={eligible_mode_match_count:,}")
    print(f"modified_rows={modified_count:,}")
    print(f"corrected_hkl={out['corrected_hkl']}")
    print(f"original_copy_hkl={out['original_copy_hkl']}")
    print(f"correction_table_csv={out['correction_table_csv']}")
    print(f"applied_rows_csv={out['applied_rows_csv']}")
    print(f"unmatched_corrections_csv={out['unmatched_csv']}")
    print(f"summary_readme={out['readme']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply lambda=0.5 nonself correction to merged CrystFEL HKL using full observation join.

This proof-of-concept script rebuilds observation-level rows from:
- partialator --unmerged-output file
- OriDyn reflection_scores.csv

It does not rerun partialator/OriDyn and does not modify the original HKL in-place.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
from typing import NamedTuple

import numpy as np
import pandas as pd


SOURCE_COLUMNS = (
    "source_filename",
    "target_source",
    "source",
    "filename",
    "image_filename",
    "image",
    "file",
    "Image filename",
)

UNMERGED_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+(\S+))?\s*$")
UNMERGED_FLAGGED_RE = re.compile(r"^\s*Flagged:\s*(\S+)\s*$", re.IGNORECASE)

NONSELF_COMPONENT_COLUMNS = [
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

TARGET_00L = [
    (0, 0, 4),
    (0, 0, 6),
    (0, 0, 8),
    (0, 0, 10),
    (0, 0, 12),
    (0, 0, 14),
    (0, 0, 16),
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
            "Apply proof-of-concept lambda=0.5 nonself correction to merged CrystFEL HKL "
            "using full unmerged+scores observation join."
        )
    )
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator --unmerged-output file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument("--input-hkl", required=True, type=Path, help="Input merged CrystFEL HKL")
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder")

    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--min-obs", type=int, default=100)
    parser.add_argument("--min-tail-obs", type=int, default=8)
    parser.add_argument("--min-abs-relative-shift", type=float, default=0.5)
    parser.add_argument("--min-abs-rho", type=float, default=0.15)
    parser.add_argument("--min-nonself-spread", type=float, default=0.05)
    parser.add_argument("--delta-method", choices=["mean", "median"], default="median")
    parser.add_argument("--intensity-floor", type=float, default=1e-9)
    parser.add_argument(
        "--match-abs-hkl",
        action="store_true",
        help="Diagnostic mode: match by abs(h),abs(k),abs(l) when applying to merged HKL",
    )
    parser.add_argument(
        "--stream",
        type=Path,
        default=None,
        help="Optional stream path (not required by default parser path).",
    )
    parser.add_argument("--score-chunksize", type=int, default=1_000_000)

    args = parser.parse_args()

    if not args.unmerged.exists():
        raise SystemExit(f"--unmerged not found: {args.unmerged}")
    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
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
    if args.score_chunksize <= 0:
        raise SystemExit("--score-chunksize must be > 0")
    if not (0.0 <= args.low_quantile < args.high_quantile <= 1.0):
        raise SystemExit("Expected 0 <= --low-quantile < --high-quantile <= 1")

    # Keep this POC constrained to the requested mode.
    if not np.isclose(float(args.lambda_value), 0.5, atol=1e-12):
        raise SystemExit("This proof-of-concept currently supports only --lambda-value 0.5")

    if args.stream is not None and not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")

    return args


def normalize_source(value: object) -> str:
    return str(value).strip()


def normalize_event(value: object) -> str:
    return str(value).strip()


def source_basename(value: object) -> str:
    return Path(str(value).strip()).name


def hkl_text(h: int, k: int, l: int) -> str:
    return f"({h},{k},{l})"


def looks_like_source_filename(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(1000)
    if values.empty:
        return False
    return bool(values.str.contains(r"\.h5\b|/|\\", regex=True).any())


def choose_score_source_column(scores_path: Path, header: list[str]) -> str:
    candidates = [column for column in SOURCE_COLUMNS if column in header]
    if not candidates:
        raise SystemExit(
            "Scores file does not contain a recognized source column. "
            f"Checked {list(SOURCE_COLUMNS)}. Available columns: {header}"
        )

    sample = pd.read_csv(scores_path, usecols=candidates, nrows=1000)
    for column in candidates:
        if looks_like_source_filename(sample[column]):
            return column
    return candidates[0]


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
    return f"{h:4d}{k:5d}{l:5d}{intensity:12.2f}{phase:>9s}{sigma_i:12.2f}{nmeas:8d}\n"


def read_crystfel_hkl(path: Path) -> tuple[list[str], int, int, list[ReflectionRow], set[tuple[int, int, int]]]:
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
    target_hkls: set[tuple[int, int, int]] = set()
    for i in range(header_idx + 1, end_idx):
        parsed = parse_reflection_row(lines[i], line_idx=i, line_no=i + 1)
        if parsed is None:
            continue
        reflections.append(parsed)
        target_hkls.add((parsed.h, parsed.k, parsed.l))

    return lines, header_idx, end_idx, reflections, target_hkls


def parse_unmerged_observations(unmerged_path: Path, target_hkls: set[tuple[int, int, int]]) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []

    stats = {
        "rows_parsed": 0,
        "excluded_flagged_crystal": 0,
        "excluded_partiality_too_small": 0,
        "excluded_nan_esd": 0,
        "excluded_nonpositive_partiality": 0,
        "excluded_not_target_hkl": 0,
        "eligible_rows": 0,
    }

    current_source = ""
    current_event = ""
    current_crystal_flagged = False

    with unmerged_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Crystal "):
                current_crystal_flagged = False
                continue

            filename_match = UNMERGED_FILENAME_RE.match(line)
            if filename_match:
                current_source = normalize_source(filename_match.group(1))
                current_event = normalize_event(filename_match.group(2) or "")
                continue

            flagged_match = UNMERGED_FLAGGED_RE.match(line)
            if flagged_match:
                current_crystal_flagged = flagged_match.group(1).strip().lower() in {"yes", "y", "true", "1"}
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                i_unmerged = float(parts[3])
                partiality = float(parts[4])
            except ValueError:
                continue

            stats["rows_parsed"] += 1
            if (h, k, l) not in target_hkls:
                stats["excluded_not_target_hkl"] += 1
                continue

            reflection_flag = " ".join(parts[5:]).strip()
            flag_l = reflection_flag.lower()

            if current_crystal_flagged:
                stats["excluded_flagged_crystal"] += 1
                continue
            if "partiality_too_small" in flag_l:
                stats["excluded_partiality_too_small"] += 1
                continue
            if "nan_esd" in flag_l:
                stats["excluded_nan_esd"] += 1
                continue
            if not np.isfinite(partiality) or partiality <= 0.0:
                stats["excluded_nonpositive_partiality"] += 1
                continue

            stats["eligible_rows"] += 1
            rows.append(
                {
                    "source": current_source,
                    "event": current_event,
                    "h": h,
                    "k": k,
                    "l": l,
                    "I_unmerged": i_unmerged,
                    "partiality": partiality,
                    "I_pr": i_unmerged * partiality,
                    "reflection_flag": reflection_flag,
                    "crystal_flagged": current_crystal_flagged,
                }
            )

    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table, stats

    table["source"] = table["source"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    table["source_norm"] = table["source"]
    table["source_basename"] = table["source"].map(source_basename)
    table["event_norm"] = table["event"]
    table[["h", "k", "l"]] = table[["h", "k", "l"]].astype("int64")
    return table, stats


def load_scores_for_targets(
    scores_path: Path,
    target_hkls: set[tuple[int, int, int]],
    chunksize: int,
) -> tuple[pd.DataFrame, str]:
    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)

    required = [source_column, "event", "h", "k", "l", *NONSELF_COMPONENT_COLUMNS]
    missing = [c for c in required if c not in header]
    if missing:
        raise SystemExit(f"Scores file missing required column(s): {missing}")

    keep_columns = list(dict.fromkeys(required))
    target_key = {f"{h},{k},{l}" for (h, k, l) in target_hkls}

    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(scores_path, usecols=keep_columns, chunksize=int(chunksize))
    for chunk in reader:
        chunk["h"] = pd.to_numeric(chunk["h"], errors="coerce")
        chunk["k"] = pd.to_numeric(chunk["k"], errors="coerce")
        chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
        chunk = chunk.dropna(subset=["h", "k", "l"])
        chunk[["h", "k", "l"]] = chunk[["h", "k", "l"]].astype("int64")

        keys = chunk["h"].astype(str) + "," + chunk["k"].astype(str) + "," + chunk["l"].astype(str)
        chunk = chunk.loc[keys.isin(target_key)].copy()
        if chunk.empty:
            continue

        chunk = chunk.rename(columns={source_column: "source"})
        chunk["source"] = chunk["source"].map(normalize_source)
        chunk["event"] = chunk["event"].map(normalize_event)
        chunk["source_norm"] = chunk["source"]
        chunk["source_basename"] = chunk["source"].map(source_basename)
        chunk["event_norm"] = chunk["event"]

        for col in NONSELF_COMPONENT_COLUMNS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk["nonself_mean"] = chunk[NONSELF_COMPONENT_COLUMNS].mean(axis=1, skipna=True)
        chunks.append(
            chunk[
                [
                    "source",
                    "event",
                    "source_norm",
                    "source_basename",
                    "event_norm",
                    "h",
                    "k",
                    "l",
                    *NONSELF_COMPONENT_COLUMNS,
                    "nonself_mean",
                ]
            ]
        )

    if not chunks:
        return pd.DataFrame(), source_column

    table = pd.concat(chunks, ignore_index=True)
    return table, source_column


def cleanup_duplicated_columns_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df


def join_unmerged_with_scores(
    unmerged: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], str]:
    key = ["source_norm", "event_norm", "h", "k", "l"]

    unmerged = cleanup_duplicated_columns_keep_first(unmerged)
    scores = cleanup_duplicated_columns_keep_first(scores)

    non_payload_cols = {"source", "event", "source_norm", "event_norm", "source_basename"}
    payload_cols = [col for col in scores.columns if col not in non_payload_cols and col not in key]
    score_subset = scores.loc[:, list(dict.fromkeys(key + payload_cols))].copy()

    merged = unmerged.merge(score_subset, on=key, how="left", indicator=True)
    merged["match_mode"] = np.where(merged["_merge"] == "both", "primary", "missing")
    merged = merged.drop(columns=["_merge"])

    stats = {
        "unmerged_rows": int(len(unmerged)),
        "score_rows_considered": int(len(scores)),
        "primary_matches": int((merged["match_mode"] == "primary").sum()),
        "basename_matches": 0,
        "missing_after_join": int((merged["match_mode"] == "missing").sum()),
    }

    note = ""
    if stats["missing_after_join"] > 0:
        basename_key = ["source_basename", "event_norm", "h", "k", "l"]
        score_base = scores.loc[:, list(dict.fromkeys(basename_key + payload_cols))].copy()

        # Only use basename fallback when basename keys are unique in score rows.
        dup = score_base.duplicated(basename_key, keep=False)
        if not bool(dup.any()):
            missing_idx = merged.index[merged["match_mode"] == "missing"]
            left = merged.loc[
                missing_idx,
                [
                    "source",
                    "event",
                    "source_norm",
                    "source_basename",
                    "event_norm",
                    "h",
                    "k",
                    "l",
                    "I_unmerged",
                    "partiality",
                    "I_pr",
                    "reflection_flag",
                    "crystal_flagged",
                ],
            ].copy()

            fallback = left.merge(score_base, on=basename_key, how="left", indicator=True)
            got = fallback["_merge"] == "both"
            if bool(got.any()):
                for col in payload_cols:
                    merged.loc[missing_idx, col] = fallback[col].values
                matched_idx = missing_idx[got.to_numpy()]
                merged.loc[matched_idx, "match_mode"] = "basename"
                stats["basename_matches"] = int(got.sum())
                stats["missing_after_join"] = int((merged["match_mode"] == "missing").sum())
                note = "basename fallback applied to unmatched primary keys"
        else:
            note = "basename fallback skipped (non-unique basename key in scores)"

    return merged, stats, note


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].corr(frame["y"], method="spearman"))


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def sign_agrees(a: float, b: float) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    if a == 0.0 or b == 0.0:
        return False
    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    reasons_rows: list[dict[str, object]] = []

    grouped = joined.groupby(["h", "k", "l"], sort=True)
    for (h, k, l), g in grouped:
        g = g.dropna(subset=["I_pr", "nonself_mean"]).copy()
        if g.empty:
            continue

        n_obs = int(len(g))
        nonself = pd.to_numeric(g["nonself_mean"], errors="coerce")
        i_pr = pd.to_numeric(g["I_pr"], errors="coerce")

        q_low = float(nonself.quantile(low_quantile))
        q_high = float(nonself.quantile(high_quantile))
        low_mask = nonself <= q_low
        high_mask = nonself >= q_high
        g_low = g.loc[low_mask]
        g_high = g.loc[high_mask]
        n_low = int(len(g_low))
        n_high = int(len(g_high))

        median_i = float(i_pr.median())
        residual = i_pr - median_i

        median_low_i_pr = float(pd.to_numeric(g_low["I_pr"], errors="coerce").median()) if n_low > 0 else np.nan
        median_high_i_pr = float(pd.to_numeric(g_high["I_pr"], errors="coerce").median()) if n_high > 0 else np.nan
        shift = float(median_high_i_pr - median_low_i_pr) if n_low > 0 and n_high > 0 else np.nan

        baseline = max(abs(median_low_i_pr), float(intensity_floor)) if np.isfinite(median_low_i_pr) else np.nan
        relative_shift_low_baseline = float(shift / baseline) if np.isfinite(shift) and np.isfinite(baseline) else np.nan

        rho = spearman_corr(nonself, residual)

        nonself_p10 = float(nonself.quantile(0.10))
        nonself_p90 = float(nonself.quantile(0.90))
        nonself_spread = float(nonself_p90 - nonself_p10)

        x_low = float(pd.to_numeric(g_low["nonself_mean"], errors="coerce").median()) if n_low > 0 else np.nan
        x_high = float(pd.to_numeric(g_high["nonself_mean"], errors="coerce").median()) if n_high > 0 else np.nan
        x_span = x_high - x_low if np.isfinite(x_high) and np.isfinite(x_low) else np.nan
        slope = float(shift / x_span) if np.isfinite(shift) and np.isfinite(x_span) and abs(x_span) > 0.0 else np.nan

        checks: list[str] = []
        if n_obs < int(min_obs):
            checks.append("n_obs_below_min")
        if n_low < int(min_tail_obs):
            checks.append("n_low_below_min_tail")
        if n_high < int(min_tail_obs):
            checks.append("n_high_below_min_tail")
        if not np.isfinite(nonself_spread) or nonself_spread < float(min_nonself_spread):
            checks.append("nonself_spread_below_min")
        if not np.isfinite(relative_shift_low_baseline) or abs(relative_shift_low_baseline) < float(min_abs_relative_shift):
            checks.append("abs_relative_shift_below_min")
        if not np.isfinite(rho) or abs(rho) < float(min_abs_rho):
            checks.append("abs_rho_below_min")
        if not sign_agrees(float(shift), float(rho)):
            checks.append("rho_shift_sign_mismatch")
        if not np.isfinite(slope):
            checks.append("invalid_slope")

        eligible = len(checks) == 0

        if eligible:
            nonself_centered = nonself - float(x_low)
            i_pr_corr = i_pr - float(lambda_value) * float(slope) * nonself_centered
        else:
            i_pr_corr = i_pr.copy()

        delta_values = i_pr_corr - i_pr
        delta_mean = float(pd.to_numeric(delta_values, errors="coerce").mean())
        delta_median = float(pd.to_numeric(delta_values, errors="coerce").median())

        if not eligible:
            delta_mean = 0.0
            delta_median = 0.0

        delta_applied = float(delta_median if delta_method == "median" else delta_mean)

        if shift > 0.0:
            correction_direction = "reduce_high_risk"
        elif shift < 0.0:
            correction_direction = "boost_high_risk"
        else:
            correction_direction = "no_shift"

        reason = "eligible" if eligible else ";".join(checks)

        rows.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "hkl": hkl_text(int(h), int(k), int(l)),
                "n_obs": n_obs,
                "n_low": n_low,
                "n_high": n_high,
                "median_low_I_pr": median_low_i_pr,
                "median_high_I_pr": median_high_i_pr,
                "shift": shift,
                "relative_shift_low_baseline": relative_shift_low_baseline,
                "rho": rho,
                "nonself_p10": nonself_p10,
                "nonself_p90": nonself_p90,
                "nonself_spread": nonself_spread,
                "x_low": x_low,
                "x_high": x_high,
                "slope": slope,
                "delta_mean": float(delta_mean),
                "delta_median": float(delta_median),
                "delta_applied": float(delta_applied),
                "eligible": bool(eligible),
                "eligibility_reason": reason,
                "correction_direction": correction_direction,
                "residual_definition": "residual = I_pr - median(I_pr within HKL)",
                "d_spacing": first_non_null(pd.to_numeric(g.get("d_spacing", pd.Series(dtype=float)), errors="coerce")),
                "shell_label": str(first_non_null(g.get("shell_label", pd.Series(dtype=str)))),
            }
        )

        reasons_rows.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "hkl": hkl_text(int(h), int(k), int(l)),
                "n_obs": n_obs,
                "n_low": n_low,
                "n_high": n_high,
                "eligible": bool(eligible),
                "eligibility_reason": reason,
                "shift": shift,
                "relative_shift_low_baseline": relative_shift_low_baseline,
                "rho": rho,
                "nonself_spread": nonself_spread,
                "delta_applied": float(delta_applied),
            }
        )

    corr = pd.DataFrame(rows)
    reasons = pd.DataFrame(reasons_rows)
    if not corr.empty:
        corr = corr.sort_values(["h", "k", "l"], ascending=[True, True, True]).reset_index(drop=True)
    if not reasons.empty:
        reasons = reasons.sort_values(["eligible", "h", "k", "l"], ascending=[True, True, True, True]).reset_index(drop=True)
    return corr, reasons


def build_lookup(table: pd.DataFrame, match_abs_hkl: bool) -> tuple[dict[tuple[int, int, int], dict[str, object]], int]:
    if table.empty:
        return {}, 0

    work = table.copy()
    if match_abs_hkl:
        work["key_h"] = work["h"].abs().astype(int)
        work["key_k"] = work["k"].abs().astype(int)
        work["key_l"] = work["l"].abs().astype(int)
    else:
        work["key_h"] = work["h"].astype(int)
        work["key_k"] = work["k"].astype(int)
        work["key_l"] = work["l"].astype(int)

    work["abs_delta_applied"] = pd.to_numeric(work["delta_applied"], errors="coerce").abs()
    work = work.sort_values(["abs_delta_applied", "n_obs"], ascending=[False, False])

    duplicate_count = int(work.duplicated(["key_h", "key_k", "key_l"], keep="first").sum())
    dedup = work.drop_duplicates(["key_h", "key_k", "key_l"], keep="first")

    lookup: dict[tuple[int, int, int], dict[str, object]] = {}
    for row in dedup.itertuples(index=False):
        lookup[(int(row.key_h), int(row.key_k), int(row.key_l))] = {
            "h": int(row.h),
            "k": int(row.k),
            "l": int(row.l),
            "hkl": str(row.hkl),
            "eligible": bool(row.eligible),
            "eligibility_reason": str(row.eligibility_reason),
            "n_obs": int(row.n_obs),
            "shift": float(row.shift) if pd.notna(row.shift) else np.nan,
            "relative_shift_low_baseline": float(row.relative_shift_low_baseline)
            if pd.notna(row.relative_shift_low_baseline)
            else np.nan,
            "rho": float(row.rho) if pd.notna(row.rho) else np.nan,
            "nonself_spread": float(row.nonself_spread) if pd.notna(row.nonself_spread) else np.nan,
            "delta_applied": float(row.delta_applied) if pd.notna(row.delta_applied) else 0.0,
            "correction_direction": str(row.correction_direction),
            "delta_method": str(row.delta_method) if hasattr(row, "delta_method") else "median",
        }

    return lookup, duplicate_count


def apply_corrections_to_hkl(
    lines: list[str],
    reflections: list[ReflectionRow],
    lookup: dict[tuple[int, int, int], dict[str, object]],
    match_abs_hkl: bool,
) -> tuple[list[str], pd.DataFrame, set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    out_lines = list(lines)
    rows: list[dict[str, object]] = []
    matched_keys: set[tuple[int, int, int]] = set()
    matched_signed: set[tuple[int, int, int]] = set()

    for ref in reflections:
        exact_key = (int(ref.h), int(ref.k), int(ref.l))
        key = (abs(ref.h), abs(ref.k), abs(ref.l)) if match_abs_hkl else exact_key

        info = lookup.get(key)
        if info is None:
            out_lines[ref.line_idx] = ref.raw_line if ref.raw_line.endswith("\n") else ref.raw_line + "\n"
            rows.append(
                {
                    "h": int(ref.h),
                    "k": int(ref.k),
                    "l": int(ref.l),
                    "hkl": hkl_text(int(ref.h), int(ref.k), int(ref.l)),
                    "matched": False,
                    "eligible": False,
                    "I_old": float(ref.intensity),
                    "I_new": float(ref.intensity),
                    "delta_applied": 0.0,
                    "sigma_old": float(ref.sigma_i),
                    "sigma_new": float(ref.sigma_i),
                    "n_obs": np.nan,
                    "shift": np.nan,
                    "relative_shift_low_baseline": np.nan,
                    "rho": np.nan,
                    "nonself_spread": np.nan,
                    "correction_direction": np.nan,
                    "source_h": np.nan,
                    "source_k": np.nan,
                    "source_l": np.nan,
                    "source_hkl": np.nan,
                }
            )
            continue

        matched_keys.add(key)
        matched_signed.add(exact_key)

        delta = float(info["delta_applied"])
        i_new = float(ref.intensity + delta)

        out_lines[ref.line_idx] = format_reflection_line(
            int(ref.h), int(ref.k), int(ref.l), i_new, str(ref.phase), float(ref.sigma_i), int(ref.nmeas)
        )

        rows.append(
            {
                "h": int(ref.h),
                "k": int(ref.k),
                "l": int(ref.l),
                "hkl": hkl_text(int(ref.h), int(ref.k), int(ref.l)),
                "matched": True,
                "eligible": bool(info["eligible"]),
                "I_old": float(ref.intensity),
                "I_new": float(i_new),
                "delta_applied": float(delta),
                "sigma_old": float(ref.sigma_i),
                "sigma_new": float(ref.sigma_i),
                "n_obs": int(info["n_obs"]),
                "shift": float(info["shift"]) if pd.notna(info["shift"]) else np.nan,
                "relative_shift_low_baseline": float(info["relative_shift_low_baseline"])
                if pd.notna(info["relative_shift_low_baseline"])
                else np.nan,
                "rho": float(info["rho"]) if pd.notna(info["rho"]) else np.nan,
                "nonself_spread": float(info["nonself_spread"]) if pd.notna(info["nonself_spread"]) else np.nan,
                "correction_direction": str(info["correction_direction"]),
                "source_h": int(info["h"]),
                "source_k": int(info["k"]),
                "source_l": int(info["l"]),
                "source_hkl": str(info["hkl"]),
            }
        )

    return out_lines, pd.DataFrame(rows), matched_keys, matched_signed


def ensure_output_layout(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "corrected_hkl": root / "crystfel_nonself_lambda05_fulljoin.hkl",
        "original_copy_hkl": root / "crystfel_original_copy.hkl",
        "correction_table_csv": root / "hkl_correction_table_fulljoin.csv",
        "applied_rows_csv": root / "applied_hkl_corrections_fulljoin.csv",
        "unmatched_target_hkls": root / "unmatched_target_hkls.csv",
        "unmatched_correction_hkls": root / "unmatched_correction_hkls.csv",
        "ineligible_hkls_csv": root / "ineligible_hkls_with_reasons.csv",
        "readme": root / "README_nonself_lambda05_fulljoin_correction.txt",
    }


def build_target_00l_diagnostics(
    target_hkls: set[tuple[int, int, int]],
    joined: pd.DataFrame,
    corr: pd.DataFrame,
    applied: pd.DataFrame,
) -> pd.DataFrame:
    if corr.empty:
        corr = pd.DataFrame(columns=["h", "k", "l", "n_obs", "nonself_spread", "shift", "delta_applied", "eligible"])
    if applied.empty:
        applied = pd.DataFrame(columns=["h", "k", "l", "delta_applied", "I_old", "I_new", "matched"])

    rows = []
    for h, k, l in TARGET_00L:
        in_target = (h, k, l) in target_hkls
        g = joined[(joined["h"] == h) & (joined["k"] == k) & (joined["l"] == l)] if not joined.empty else pd.DataFrame()
        c = corr[(corr["h"] == h) & (corr["k"] == k) & (corr["l"] == l)] if not corr.empty else pd.DataFrame()
        a = applied[(applied["h"] == h) & (applied["k"] == k) & (applied["l"] == l)] if not applied.empty else pd.DataFrame()

        delta = float(c["delta_applied"].iloc[0]) if not c.empty else np.nan
        modified = bool((not a.empty) and (abs(float(a["delta_applied"].iloc[0])) > 0.0))

        rows.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "hkl": hkl_text(h, k, l),
                "exists_in_target_hkl": bool(in_target),
                "rebuilt_observation_rows": bool(len(g) > 0),
                "n_obs": int(len(g)) if len(g) > 0 else 0,
                "nonself_spread": float(c["nonself_spread"].iloc[0]) if not c.empty else np.nan,
                "shift": float(c["shift"].iloc[0]) if not c.empty else np.nan,
                "delta_applied": delta,
                "eligible": bool(c["eligible"].iloc[0]) if not c.empty else False,
                "modified": modified,
            }
        )

    return pd.DataFrame(rows)


def write_readme(
    path: Path,
    args: argparse.Namespace,
    source_column: str,
    unmerged_stats: dict[str, int],
    join_stats: dict[str, int],
    join_note: str,
    n_target_hkls: int,
    n_rebuilt_hkls: int,
    n_eligible: int,
    n_modified: int,
    exact_match_count: int,
    mode_match_count: int,
    eligible_exact_match_count: int,
    eligible_mode_match_count: int,
    top20: pd.DataFrame,
    diag_00l: pd.DataFrame,
    duplicate_key_choices: int,
) -> None:
    lines: list[str] = []
    lines.append("Proof-of-concept nonself lambda=0.5 correction using full unmerged+scores join")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"- unmerged: {args.unmerged}")
    lines.append(f"- scores: {args.scores}")
    lines.append(f"- input_hkl: {args.input_hkl}")
    lines.append(f"- source_column_selected: {source_column}")
    lines.append(f"- residual_definition: residual = I_pr - median(I_pr within HKL)")
    lines.append("")
    lines.append("Key matching:")
    lines.append("- primary join key: source_filename/source_norm + event_norm + signed h,k,l")
    lines.append("- basename fallback: only if score basename key is unique")
    if join_note:
        lines.append(f"- join_note: {join_note}")
    lines.append("")
    lines.append("Counts:")
    lines.append(f"- target_merged_hkls: {n_target_hkls:,}")
    lines.append(f"- hkls_with_rebuilt_observation_rows: {n_rebuilt_hkls:,}")
    lines.append(f"- eligible_hkls: {n_eligible:,}")
    lines.append(f"- modified_target_hkls: {n_modified:,}")
    lines.append(f"- exact_match_count_all_target_hkls: {exact_match_count:,}")
    lines.append(f"- selected_mode_match_count_all_target_hkls ({'abs' if args.match_abs_hkl else 'exact'}): {mode_match_count:,}")
    lines.append(f"- eligible_exact_match_count: {eligible_exact_match_count:,}")
    lines.append(
        f"- eligible_selected_mode_match_count ({'abs' if args.match_abs_hkl else 'exact'}): {eligible_mode_match_count:,}"
    )

    if n_target_hkls > 0:
        lines.append(f"- exact_match_success_rate: {100.0 * exact_match_count / n_target_hkls:.2f}%")
        lines.append(
            f"- selected_mode_match_success_rate ({'abs' if args.match_abs_hkl else 'exact'}): "
            f"{100.0 * mode_match_count / n_target_hkls:.2f}%"
        )

    lines.append("")
    lines.append("Unmerged parser stats:")
    for k in [
        "rows_parsed",
        "excluded_not_target_hkl",
        "excluded_flagged_crystal",
        "excluded_partiality_too_small",
        "excluded_nan_esd",
        "excluded_nonpositive_partiality",
        "eligible_rows",
    ]:
        lines.append(f"- {k}: {int(unmerged_stats.get(k, 0)):,}")

    lines.append("")
    lines.append("Join stats:")
    for k in ["unmerged_rows", "score_rows_considered", "primary_matches", "basename_matches", "missing_after_join"]:
        lines.append(f"- {k}: {int(join_stats.get(k, 0)):,}")

    if duplicate_key_choices > 0:
        lines.append("")
        lines.append(
            "Diagnostic note: selected match mode collapsed duplicate correction keys and kept largest |delta| for "
            f"{duplicate_key_choices:,} key collisions."
        )

    lines.append("")
    lines.append("Top 20 absolute applied corrections:")
    if top20.empty:
        lines.append("(none)")
    else:
        lines.extend(top20.to_string(index=False).splitlines())

    lines.append("")
    lines.append("Seven 00l diagnostics:")
    if diag_00l.empty:
        lines.append("(none)")
    else:
        lines.extend(diag_00l.to_string(index=False).splitlines())

    lines.append("")
    lines.append("Warnings:")
    lines.append("- Sigma/ESD values were left unchanged in this proof-of-concept output.")
    lines.append("- This output is diagnostic only; not intended as a final physically validated merge.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure_output_layout(args.output_root)

    lines, _header_idx, _end_idx, reflections, target_hkls = read_crystfel_hkl(args.input_hkl)
    n_target_hkls = int(len(target_hkls))

    unmerged, unmerged_stats = parse_unmerged_observations(args.unmerged, target_hkls)
    if unmerged.empty:
        if args.stream is not None:
            raise SystemExit(
                "No eligible unmerged rows for target HKLs were parsed. "
                "Current parser did not require --stream, but no rows survived filtering."
            )
        raise SystemExit(
            "No eligible unmerged rows for target HKLs were parsed. "
            "Cannot continue fulljoin correction."
        )

    scores, source_column = load_scores_for_targets(args.scores, target_hkls, chunksize=int(args.score_chunksize))
    if scores.empty:
        raise SystemExit(
            "No score rows found for target HKLs in reflection_scores.csv. "
            "Cannot build correction table."
        )

    joined, join_stats, join_note = join_unmerged_with_scores(unmerged, scores)
    joined = joined.dropna(subset=["nonself_mean", "I_pr"]).copy()
    if joined.empty:
        raise SystemExit(
            "Joined table has no rows with both I_pr and nonself_mean after matching. "
            "Cannot build correction table."
        )

    corr, reasons = compute_hkl_correction_table(
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

    # Keep only HKLs that actually exist in target merged HKL.
    if not corr.empty:
        corr = corr[corr.apply(lambda r: (int(r["h"]), int(r["k"]), int(r["l"])) in target_hkls, axis=1)].copy()

    if corr.empty:
        raise SystemExit("Correction table is empty for target HKLs after fulljoin computation.")

    corr = corr.copy()
    corr["delta_method"] = str(args.delta_method)

    corr.to_csv(out["correction_table_csv"], index=False)

    ineligible = reasons.loc[~reasons["eligible"]].copy() if not reasons.empty else reasons
    if not ineligible.empty:
        ineligible = ineligible.sort_values(["h", "k", "l"]).reset_index(drop=True)
    ineligible.to_csv(out["ineligible_hkls_csv"], index=False)

    lookup_exact, _dup_exact = build_lookup(corr, match_abs_hkl=False)
    lookup_mode, duplicate_key_choices = build_lookup(corr, match_abs_hkl=bool(args.match_abs_hkl))

    out_lines, applied, matched_mode_keys, _matched_signed = apply_corrections_to_hkl(
        lines=lines,
        reflections=reflections,
        lookup=lookup_mode,
        match_abs_hkl=bool(args.match_abs_hkl),
    )

    # Prepare unmatched reports
    corr_exact_key = list(zip(corr["h"].astype(int), corr["k"].astype(int), corr["l"].astype(int)))
    corr_abs_key = list(zip(corr["h"].abs().astype(int), corr["k"].abs().astype(int), corr["l"].abs().astype(int)))

    if args.match_abs_hkl:
        matched_key_set = set(matched_mode_keys)
        corr["matched_in_mode"] = [k in matched_key_set for k in corr_abs_key]
    else:
        matched_key_set = set(matched_mode_keys)
        corr["matched_in_mode"] = [k in matched_key_set for k in corr_exact_key]

    unmatched_corrections = corr.loc[~corr["matched_in_mode"]].copy()
    unmatched_corrections = unmatched_corrections.drop(columns=["matched_in_mode"])
    unmatched_corrections.to_csv(out["unmatched_correction_hkls"], index=False)

    matched_target = applied.loc[applied["matched"]].copy() if not applied.empty else applied
    unmatched_target = applied.loc[~applied["matched"]].copy() if not applied.empty else applied
    unmatched_target.to_csv(out["unmatched_target_hkls"], index=False)

    # Write corrected and original copy
    shutil.copy2(args.input_hkl, out["original_copy_hkl"])
    out["corrected_hkl"].write_text("".join(out_lines), encoding="utf-8")

    # Final applied output columns requested
    applied_out = applied.copy()
    if not applied_out.empty:
        rename_cols = {
            "sigma_old": "sigma_old",
            "sigma_new": "sigma_new",
        }
        applied_out = applied_out.rename(columns=rename_cols)

        applied_cols = [
            "h",
            "k",
            "l",
            "hkl",
            "I_old",
            "I_new",
            "delta_applied",
            "sigma_old",
            "sigma_new",
            "n_obs",
            "shift",
            "relative_shift_low_baseline",
            "rho",
            "nonself_spread",
            "correction_direction",
            "matched",
            "eligible",
            "source_hkl",
        ]
        for col in applied_cols:
            if col not in applied_out.columns:
                applied_out[col] = np.nan
        applied_out = applied_out[applied_cols]

    applied_out.to_csv(out["applied_rows_csv"], index=False)

    # 00l diagnostics table for README
    diag_00l = build_target_00l_diagnostics(target_hkls, joined, corr, applied)

    # Metrics
    n_rebuilt_hkls = int(len(set(zip(joined["h"].astype(int), joined["k"].astype(int), joined["l"].astype(int)))))
    n_eligible = int((corr["eligible"] == True).sum())
    n_modified = int(((applied["matched"] == True) & (applied["delta_applied"].abs() > 0.0)).sum()) if not applied.empty else 0

    exact_match_count = int(sum(1 for hkl in target_hkls if hkl in lookup_exact))
    if args.match_abs_hkl:
        mode_match_count = int(len(matched_mode_keys))
    else:
        mode_match_count = int(sum(1 for hkl in target_hkls if hkl in lookup_mode))

    corr_exact_set = set(corr_exact_key)
    eligible_exact_match_count = int(sum((h, k, l) in target_hkls for (h, k, l) in corr_exact_set if bool(corr[(corr['h']==h)&(corr['k']==k)&(corr['l']==l)]['eligible'].iloc[0])))

    eligible_mode_match_count = int((corr["eligible"] & corr["matched_in_mode"]).sum()) if "matched_in_mode" in corr.columns else 0

    top20 = applied[(applied["matched"] == True) & (applied["delta_applied"].abs() > 0.0)].copy() if not applied.empty else pd.DataFrame()
    if not top20.empty:
        top20["abs_delta"] = top20["delta_applied"].abs()
        top20 = top20.sort_values(["abs_delta", "n_obs"], ascending=[False, False]).head(20)
        top20 = top20[["hkl", "delta_applied", "I_old", "I_new", "source_hkl", "n_obs"]]

    write_readme(
        path=out["readme"],
        args=args,
        source_column=source_column,
        unmerged_stats=unmerged_stats,
        join_stats=join_stats,
        join_note=join_note,
        n_target_hkls=n_target_hkls,
        n_rebuilt_hkls=n_rebuilt_hkls,
        n_eligible=n_eligible,
        n_modified=n_modified,
        exact_match_count=exact_match_count,
        mode_match_count=mode_match_count,
        eligible_exact_match_count=eligible_exact_match_count,
        eligible_mode_match_count=eligible_mode_match_count,
        top20=top20,
        diag_00l=diag_00l,
        duplicate_key_choices=duplicate_key_choices,
    )

    print("NONSELF_FULLJOIN_APPLY_OK")
    print(f"target_merged_hkls={n_target_hkls:,}")
    print(f"hkls_with_rebuilt_observation_rows={n_rebuilt_hkls:,}")
    print(f"eligible_hkls={n_eligible:,}")
    print(f"modified_target_hkls={n_modified:,}")
    print(f"exact_match_count={exact_match_count:,}")
    print(f"mode_match_count={'abs' if args.match_abs_hkl else 'exact'}:{mode_match_count:,}")
    if n_target_hkls > 0:
        print(f"exact_match_success_rate={100.0 * exact_match_count / n_target_hkls:.2f}%")
    print(f"corrected_hkl={out['corrected_hkl']}")
    print(f"original_copy_hkl={out['original_copy_hkl']}")
    print(f"correction_table_csv={out['correction_table_csv']}")
    print(f"applied_rows_csv={out['applied_rows_csv']}")
    print(f"unmatched_target_hkls_csv={out['unmatched_target_hkls']}")
    print(f"unmatched_correction_hkls_csv={out['unmatched_correction_hkls']}")
    print(f"ineligible_hkls_csv={out['ineligible_hkls_csv']}")
    print(f"readme={out['readme']}")


if __name__ == "__main__":
    main()

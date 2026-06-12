#!/usr/bin/env python3
"""Analyze partiality-weighted unmerged intensities versus OriDyn risk.

This script is intended for diagnostics on CrystFEL partialator --unmerged-output.
It treats the fifth numeric reflection column as partiality:

    h k l I_unmerged partiality [flags ...]

Core outputs:
- outputs/joined_selected_observations.csv
- outputs/hkl_summary_statistics.csv
- diagnostics/match_diagnostics.txt
- diagnostics/duplicate_diagnostics.csv (if duplicates are found)
- plots/*.png
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
EVENT_INT_RE = re.compile(r"-?\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join partialator --unmerged-output observations with OriDyn scores, "
            "then compute per-HKL partiality-risk diagnostics and plots."
        )
    )
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator --unmerged-output file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Dataset-specific output root containing diagnostics/ plots/ outputs/",
    )
    parser.add_argument(
        "--selected-hkls",
        type=Path,
        default=None,
        help="Text file with one HKL per line: 'h k l'. Lines can contain comments after '#'.",
    )
    parser.add_argument(
        "--risk-column",
        default="S_dyn_geom",
        help="Risk column from scores used for quantiles/correlations (default: S_dyn_geom)",
    )
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--min-hkl-observations", type=int, default=10)
    parser.add_argument("--target-d", type=float, default=None, help="Optional auto-select target d spacing in Angstrom")
    parser.add_argument("--target-d-tolerance", type=float, default=0.5, help="Absolute d tolerance in Angstrom")
    parser.add_argument(
        "--target-d-max-hkls",
        type=int,
        default=50,
        help="Max HKLs retained by auto d-selection (nearest d, then highest counts)",
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        default=None,
        help=(
            "Optional OriDyn run_metadata.json for unit-cell driven d-based auto selection. "
            "If omitted, script tries <scores_dir>/run_metadata.json."
        ),
    )
    parser.add_argument(
        "--score-chunksize",
        type=int,
        default=1_000_000,
        help="Chunk size for reading large reflection_scores.csv",
    )
    parser.add_argument(
        "--smoke-max-reflections",
        type=int,
        default=None,
        help="Stop after reading this many eligible unmerged reflections (for tiny smoke tests)",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only parse selected HKLs and initial unmerged rows; do not load scores or plot.",
    )
    return parser.parse_args()


def normalize_source(value: object) -> str:
    return str(value).strip()


def normalize_event(value: object) -> str:
    return str(value).strip()


def source_basename(value: object) -> str:
    return Path(str(value).strip()).name


def looks_like_source_filename(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(1000)
    if values.empty:
        return False
    return bool(values.str.contains(r"\.h5\b|/|\\", regex=True).any())


def choose_score_source_column(scores_path: Path, header: list[str]) -> str:
    candidates = [column for column in SOURCE_COLUMNS if column in header]
    if not candidates:
        raise SystemExit(
            "Score file does not contain a recognized source column. "
            f"Checked {list(SOURCE_COLUMNS)}. Available columns: {header}"
        )
    sample = pd.read_csv(scores_path, usecols=candidates, nrows=1000)
    for column in candidates:
        if looks_like_source_filename(sample[column]):
            return column
    # Fallback to first available source-like column.
    return candidates[0]


def parse_hkl_line(line: str) -> tuple[int, int, int] | None:
    text = line.split("#", 1)[0].strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def load_selected_hkls(path: Path) -> set[tuple[int, int, int]]:
    hkls: set[tuple[int, int, int]] = set()
    for idx, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        parsed = parse_hkl_line(raw)
        if parsed is None:
            continue
        hkls.add(parsed)
    if not hkls:
        raise SystemExit(f"No HKLs were parsed from {path}")
    return hkls


def flag_token_mask(text: str, token: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE))


def hkl_key(h: int, k: int, l: int) -> str:
    return f"{h},{k},{l}"


def load_unit_cell_from_metadata(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    unit_cell = data.get("unit_cell")
    if not isinstance(unit_cell, dict):
        return None
    required = ["a", "b", "c", "alpha", "beta", "gamma"]
    if not all(key in unit_cell for key in required):
        return None
    try:
        return {key: float(unit_cell[key]) for key in required}
    except (TypeError, ValueError):
        return None


def d_spacing_orthogonal(h: int, k: int, l: int, cell: dict[str, float]) -> float | None:
    """Return d spacing for orthogonal cells (alpha=beta=gamma=90).

    If the cell is non-orthogonal, return None and let caller report TODO.
    """

    if h == 0 and k == 0 and l == 0:
        return None
    alpha = cell["alpha"]
    beta = cell["beta"]
    gamma = cell["gamma"]
    if not (abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6):
        return None
    a = cell["a"]
    b = cell["b"]
    c = cell["c"]
    if min(a, b, c) <= 0.0:
        return None
    inv_d_sq = (h / a) ** 2 + (k / b) ** 2 + (l / c) ** 2
    if inv_d_sq <= 0.0:
        return None
    return 1.0 / math.sqrt(inv_d_sq)


def scan_unmerged_hkls(
    unmerged_path: Path,
) -> tuple[Counter[tuple[int, int, int]], dict[str, int]]:
    """Scan unmerged file and count eligible HKLs after standard exclusions."""

    counts: Counter[tuple[int, int, int]] = Counter()
    stats = {
        "rows_parsed": 0,
        "excluded_flagged_crystal": 0,
        "excluded_partiality_too_small": 0,
        "excluded_nan_esd": 0,
        "excluded_nonpositive_partiality": 0,
        "eligible_rows": 0,
    }

    current_crystal_flagged = False
    with unmerged_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Crystal "):
                current_crystal_flagged = False
                continue
            if match := UNMERGED_FLAGGED_RE.match(line):
                current_crystal_flagged = match.group(1).strip().lower() in {"yes", "y", "true", "1"}
                continue
            if line.startswith("Filename:"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                _i_unmerged = float(parts[3])
                partiality = float(parts[4])
            except ValueError:
                continue

            stats["rows_parsed"] += 1
            flags = " ".join(parts[5:]).strip() if len(parts) > 5 else ""

            if current_crystal_flagged:
                stats["excluded_flagged_crystal"] += 1
                continue
            if flag_token_mask(flags, "partiality_too_small"):
                stats["excluded_partiality_too_small"] += 1
                continue
            if flag_token_mask(flags, "nan_esd"):
                stats["excluded_nan_esd"] += 1
                continue
            if not np.isfinite(partiality) or partiality <= 0.0:
                stats["excluded_nonpositive_partiality"] += 1
                continue

            stats["eligible_rows"] += 1
            counts[(h, k, l)] += 1

    return counts, stats


def auto_select_hkls_by_d(
    unmerged_path: Path,
    target_d: float,
    tolerance: float,
    max_hkls: int,
    unit_cell: dict[str, float] | None,
) -> tuple[set[tuple[int, int, int]], list[str], dict[str, int]]:
    notes: list[str] = []
    counts, stats = scan_unmerged_hkls(unmerged_path)

    if unit_cell is None:
        notes.append(
            "TODO: automatic d-shell HKL selection requested, but unit-cell metadata is unavailable. "
            "Provide --run-metadata run_metadata.json or a selected HKL file."
        )
        return set(), notes, stats

    selected: list[tuple[tuple[int, int, int], float, int]] = []
    non_orthogonal = False
    for hkl, n_obs in counts.items():
        d = d_spacing_orthogonal(hkl[0], hkl[1], hkl[2], unit_cell)
        if d is None:
            non_orthogonal = True
            continue
        if abs(d - target_d) <= tolerance:
            selected.append((hkl, d, n_obs))

    if non_orthogonal:
        notes.append(
            "TODO: current auto d-selection supports orthogonal cells only (alpha=beta=gamma=90). "
            "For non-orthogonal cells, provide explicit --selected-hkls."
        )

    if not selected:
        notes.append(
            f"No eligible HKLs found near d={target_d:.3f} +/- {tolerance:.3f} A after exclusions."
        )
        return set(), notes, stats

    # Prefer closest d, then larger observation count.
    selected.sort(key=lambda x: (abs(x[1] - target_d), -x[2], x[0]))
    if max_hkls > 0:
        selected = selected[:max_hkls]

    chosen = {item[0] for item in selected}
    notes.append(
        f"Auto-selected {len(chosen)} HKLs near d={target_d:.3f} +/- {tolerance:.3f} A (max={max_hkls})."
    )
    return chosen, notes, stats


def parse_unmerged_observations(
    unmerged_path: Path,
    selected_hkls: set[tuple[int, int, int]] | None,
    smoke_max_reflections: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    current_source = ""
    current_event = ""
    current_crystal_flagged = False

    stats = {
        "rows_parsed": 0,
        "excluded_flagged_crystal": 0,
        "excluded_partiality_too_small": 0,
        "excluded_nan_esd": 0,
        "excluded_nonpositive_partiality": 0,
        "excluded_not_selected_hkl": 0,
        "eligible_rows": 0,
        "kept_rows": 0,
    }

    with unmerged_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Crystal "):
                current_source = ""
                current_event = ""
                current_crystal_flagged = False
                continue

            if match := UNMERGED_FILENAME_RE.match(line):
                current_source = normalize_source(match.group(1))
                current_event = normalize_event(match.group(2) or "")
                continue

            if match := UNMERGED_FLAGGED_RE.match(line):
                current_crystal_flagged = match.group(1).strip().lower() in {"yes", "y", "true", "1"}
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
            reflection_flag = " ".join(parts[5:]).strip() if len(parts) > 5 else ""

            if current_crystal_flagged:
                stats["excluded_flagged_crystal"] += 1
                continue
            if flag_token_mask(reflection_flag, "partiality_too_small"):
                stats["excluded_partiality_too_small"] += 1
                continue
            if flag_token_mask(reflection_flag, "nan_esd"):
                stats["excluded_nan_esd"] += 1
                continue
            if not np.isfinite(partiality) or partiality <= 0.0:
                stats["excluded_nonpositive_partiality"] += 1
                continue

            stats["eligible_rows"] += 1

            if selected_hkls is not None and (h, k, l) not in selected_hkls:
                stats["excluded_not_selected_hkl"] += 1
                continue

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
            stats["kept_rows"] += 1

            if smoke_max_reflections is not None and stats["kept_rows"] >= int(smoke_max_reflections):
                break

    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table, stats

    table["source"] = table["source"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    table["source_norm"] = table["source"]
    table["source_basename"] = table["source"].map(source_basename)
    table["event_norm"] = table["event"]
    table["h"] = table["h"].astype("int64")
    table["k"] = table["k"].astype("int64")
    table["l"] = table["l"].astype("int64")
    return table, stats


def duplicate_rows(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = df.duplicated(key_columns, keep=False)
    return df.loc[mask].copy()


def duplicate_summary(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=key_columns + ["n_duplicates"])
    out = (
        df.groupby(key_columns, as_index=False)
        .size()
        .rename(columns={"size": "n_duplicates"})
        .sort_values("n_duplicates", ascending=False)
    )
    return out


def write_duplicate_diagnostics(path: Path, label: str, dup_rows: pd.DataFrame, key_columns: list[str]) -> None:
    summary = duplicate_summary(dup_rows, key_columns)
    dup_rows = dup_rows.copy()
    dup_rows.insert(0, "duplicate_set", label)
    summary.insert(0, "duplicate_set", label)
    summary.insert(1, "record_type", "summary")
    dup_rows.insert(1, "record_type", "row")

    # Align columns for single CSV output.
    for col in summary.columns:
        if col not in dup_rows.columns:
            dup_rows[col] = np.nan
    for col in dup_rows.columns:
        if col not in summary.columns:
            summary[col] = np.nan

    out = pd.concat([summary[dup_rows.columns], dup_rows], ignore_index=True)
    out.to_csv(path, index=False)


def duplicated_column_names(df: pd.DataFrame) -> list[str]:
    """Return duplicate column labels in first-seen order."""

    dups: list[str] = []
    seen: set[str] = set()
    for col in df.columns[df.columns.duplicated(keep=False)]:
        name = str(col)
        if name not in seen:
            seen.add(name)
            dups.append(name)
    return dups


def cleanup_duplicated_columns_keep_first(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str]]:
    """Drop duplicate labels with keep='first' and emit a warning."""

    dup_names = duplicated_column_names(df)
    if dup_names:
        print(
            f"WARNING: duplicate column labels found in {label}: {dup_names}. "
            "Dropping duplicates with keep='first'.",
            file=sys.stderr,
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df, dup_names


def load_scores(
    scores_path: Path,
    selected_hkls: set[tuple[int, int, int]] | None,
    risk_column: str,
    chunksize: int,
) -> tuple[pd.DataFrame, str, list[str]]:
    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)

    required = [source_column, "event", "h", "k", "l", "sigma_dyn_rel", risk_column]
    missing = [col for col in required if col not in header]
    if missing:
        raise SystemExit(f"Scores file missing required column(s): {missing}")

    component_columns = [c for c in header if c.startswith("S_")]
    keep_columns = list(dict.fromkeys([source_column, "event", "h", "k", "l", "sigma_dyn_rel", risk_column, *component_columns]))

    selected_hkl_keys: set[str] | None = None
    if selected_hkls is not None:
        selected_hkl_keys = {hkl_key(h, k, l) for h, k, l in selected_hkls}

    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(scores_path, usecols=keep_columns, chunksize=chunksize)
    for chunk in reader:
        chunk["h"] = pd.to_numeric(chunk["h"], errors="coerce")
        chunk["k"] = pd.to_numeric(chunk["k"], errors="coerce")
        chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
        chunk = chunk.dropna(subset=["h", "k", "l"])
        chunk[["h", "k", "l"]] = chunk[["h", "k", "l"]].astype("int64")

        if selected_hkl_keys is not None:
            keys = chunk["h"].astype(str) + "," + chunk["k"].astype(str) + "," + chunk["l"].astype(str)
            chunk = chunk.loc[keys.isin(selected_hkl_keys)].copy()

        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(), source_column, component_columns

    table = pd.concat(chunks, ignore_index=True)
    table = table.rename(columns={source_column: "source"})
    table["source"] = table["source"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    table["source_norm"] = table["source"]
    table["source_basename"] = table["source"].map(source_basename)
    table["event_norm"] = table["event"]
    table["sigma_dyn_rel"] = pd.to_numeric(table["sigma_dyn_rel"], errors="coerce")
    table[risk_column] = pd.to_numeric(table[risk_column], errors="coerce")

    for c in component_columns:
        table[c] = pd.to_numeric(table[c], errors="coerce")

    return table, source_column, component_columns


def join_unmerged_with_scores(
    unmerged: pd.DataFrame,
    scores: pd.DataFrame,
    risk_column: str,
) -> tuple[pd.DataFrame, dict[str, int], str, dict[str, object]]:
    key = ["source_norm", "event_norm", "h", "k", "l"]
    join_debug: dict[str, object] = {}

    # Defensive cleanup: duplicate labels make pandas merge operations ambiguous.
    unmerged, dup_unmerged_cols = cleanup_duplicated_columns_keep_first(unmerged, "unmerged")
    scores, dup_score_cols = cleanup_duplicated_columns_keep_first(scores, "scores")

    non_payload_cols = {"source", "event", "source_norm", "event_norm", "source_basename"}
    payload_candidates = [col for col in scores.columns if col not in non_payload_cols]
    removed_from_payload = [col for col in payload_candidates if col in key]
    payload_cols = [col for col in payload_candidates if col not in key]
    score_subset_cols = list(dict.fromkeys(key + payload_cols))
    scores_subset = scores.loc[:, score_subset_cols].copy()

    join_debug["duplicated_score_column_names_before_cleanup"] = dup_score_cols
    join_debug["duplicated_unmerged_column_names_before_cleanup"] = dup_unmerged_cols
    join_debug["final_merge_key"] = key
    join_debug["payload_column_count"] = int(len(payload_cols))
    join_debug["hkl_removed_from_payload_cols"] = bool(any(c in {"h", "k", "l"} for c in removed_from_payload))
    join_debug["payload_cols_removed_because_in_key"] = list(dict.fromkeys(removed_from_payload))

    print(
        "JOIN_DEBUG "
        + json.dumps(
            {
                "duplicated_score_column_names_before_cleanup": dup_score_cols,
                "duplicated_unmerged_column_names_before_cleanup": dup_unmerged_cols,
                "final_merge_key": key,
                "payload_column_count": int(len(payload_cols)),
                "hkl_removed_from_payload_cols": bool(any(c in {"h", "k", "l"} for c in removed_from_payload)),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )

    # Primary join on full source path.
    merged = unmerged.merge(
        scores_subset,
        on=key,
        how="left",
        indicator=True,
    )
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
        # Optional fallback by basename when full-path source mismatch exists.
        basename_key = ["source_basename", "event_norm", "h", "k", "l"]
        score_basename_dups = duplicate_rows(scores, basename_key)
        if score_basename_dups.empty:
            missing_idx = merged.index[merged["match_mode"] == "missing"]
            if len(missing_idx) > 0:
                left = merged.loc[missing_idx, [
                    "source",
                    "event",
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
                    "source_norm",
                ]].copy()

                score_basename_cols = list(dict.fromkeys(basename_key + payload_cols))
                score_basename = scores.loc[:, score_basename_cols].copy()
                fallback = left.merge(score_basename, on=basename_key, how="left", indicator=True)
                got = fallback["_merge"] == "both"
                if got.any():
                    for col in payload_cols:
                        merged.loc[missing_idx, col] = fallback[col].values
                    matched_idx = missing_idx[got.to_numpy()]
                    merged.loc[matched_idx, "match_mode"] = "basename"
                    stats["basename_matches"] = int(got.sum())
                    stats["missing_after_join"] = int((merged["match_mode"] == "missing").sum())
                    note = "basename fallback applied to unmatched primary keys"
        else:
            note = "basename fallback skipped because basename keys are non-unique in score table"

    # Keep only meaningful sigma values for downstream weight logic.
    merged["sigma_dyn_rel"] = pd.to_numeric(merged.get("sigma_dyn_rel"), errors="coerce")
    merged[risk_column] = pd.to_numeric(merged.get(risk_column), errors="coerce")

    return merged, stats, note, join_debug


def robust_mad(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def robust_skew(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 3:
        return np.nan
    q10, q50, q90 = np.quantile(arr, [0.10, 0.50, 0.90])
    denom = q90 - q10
    if abs(denom) < 1e-12:
        return np.nan
    return float((q90 + q10 - 2.0 * q50) / denom)


def parse_event_numeric(event_value: object) -> float:
    text = str(event_value)
    match = EVENT_INT_RE.search(text)
    if not match:
        return np.nan
    try:
        return float(int(match.group(0)))
    except ValueError:
        return np.nan


def maybe_spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].corr(frame["y"], method="spearman"))


def maybe_scipy_tests(low: pd.Series, high: pd.Series) -> tuple[float, float, float, float]:
    low_v = pd.to_numeric(low, errors="coerce").dropna()
    high_v = pd.to_numeric(high, errors="coerce").dropna()
    if len(low_v) < 2 or len(high_v) < 2:
        return np.nan, np.nan, np.nan, np.nan
    try:
        from scipy.stats import ks_2samp, mannwhitneyu

        mw = mannwhitneyu(high_v, low_v, alternative="two-sided")
        ks = ks_2samp(high_v, low_v)
        return float(mw.statistic), float(mw.pvalue), float(ks.statistic), float(ks.pvalue)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def hkl_slug(h: int, k: int, l: int) -> str:
    return f"h{h:+d}_k{k:+d}_l{l:+d}".replace("+", "p").replace("-", "m")


def make_hkl_plots(
    group: pd.DataFrame,
    risk_column: str,
    low_q: float,
    high_q: float,
    plots_dir: Path,
) -> list[str]:
    h = int(group["h"].iloc[0])
    k = int(group["k"].iloc[0])
    l = int(group["l"].iloc[0])
    slug = hkl_slug(h, k, l)

    files_written: list[str] = []

    # Risk quantiles within HKL.
    q_lo = float(group[risk_column].quantile(low_q))
    q_hi = float(group[risk_column].quantile(high_q))
    low = group[group[risk_column] <= q_lo]
    high = group[group[risk_column] >= q_hi]

    # A. Histogram of I_pr.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(group["I_pr"], bins=40, color="#3a7", alpha=0.85)
    ax.set_title(f"I_pr histogram ({h} {k} {l})")
    ax.set_xlabel("I_pr = I_unmerged * partiality")
    ax.set_ylabel("count")
    out = plots_dir / f"{slug}_hist_I_pr.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    files_written.append(str(out))

    # B. Histogram split by low/high risk.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if not low.empty:
        ax.hist(low["I_pr"], bins=30, alpha=0.60, label=f"low risk <= q{int(low_q*100)}", color="#2c7fb8")
    if not high.empty:
        ax.hist(high["I_pr"], bins=30, alpha=0.60, label=f"high risk >= q{int(high_q*100)}", color="#d7191c")
    ax.set_title(f"I_pr by risk quantiles ({h} {k} {l})")
    ax.set_xlabel("I_pr")
    ax.set_ylabel("count")
    ax.legend(loc="best")
    out = plots_dir / f"{slug}_hist_I_pr_low_high_risk.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    files_written.append(str(out))

    # C. Scatter: risk vs I_pr.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(group[risk_column], group["I_pr"], s=10, alpha=0.45)
    ax.set_title(f"{risk_column} vs I_pr ({h} {k} {l})")
    ax.set_xlabel(risk_column)
    ax.set_ylabel("I_pr")
    out = plots_dir / f"{slug}_scatter_risk_vs_I_pr.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    files_written.append(str(out))

    # D. Scatter: risk vs signed residual.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(group[risk_column], group["signed_residual"], s=10, alpha=0.45)
    ax.axhline(0.0, color="gray", lw=1)
    ax.set_title(f"{risk_column} vs signed residual ({h} {k} {l})")
    ax.set_xlabel(risk_column)
    ax.set_ylabel("I_pr - median(I_pr) [within HKL]")
    out = plots_dir / f"{slug}_scatter_risk_vs_signed_residual.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    files_written.append(str(out))

    # E. Scatter: risk vs absolute relative residual.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(group[risk_column], group["abs_relative_residual"], s=10, alpha=0.45)
    ax.set_title(f"{risk_column} vs abs relative residual ({h} {k} {l})")
    ax.set_xlabel(risk_column)
    ax.set_ylabel("|relative residual| (robust denominator)")
    out = plots_dir / f"{slug}_scatter_risk_vs_abs_relative_residual.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    files_written.append(str(out))

    # F. Optional event plot if event number can be parsed.
    event_num = group["event"].map(parse_event_numeric)
    valid = event_num.notna()
    if valid.any():
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), dpi=150, sharex=True)
        sc1 = axes[0].scatter(event_num[valid], group.loc[valid, "I_pr"], c=group.loc[valid, risk_column], s=10, alpha=0.6)
        axes[0].set_ylabel("I_pr")
        axes[0].set_title(f"event vs I_pr, colored by {risk_column} ({h} {k} {l})")
        plt.colorbar(sc1, ax=axes[0], label=risk_column)

        sc2 = axes[1].scatter(
            event_num[valid],
            group.loc[valid, "signed_residual"],
            c=group.loc[valid, risk_column],
            s=10,
            alpha=0.6,
        )
        axes[1].axhline(0.0, color="gray", lw=1)
        axes[1].set_xlabel("event (numeric parse)")
        axes[1].set_ylabel("signed residual")
        plt.colorbar(sc2, ax=axes[1], label=risk_column)

        out = plots_dir / f"{slug}_event_colored_by_risk.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        files_written.append(str(out))

    return files_written


def ensure_output_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "outputs": root / "outputs",
        "diagnostics": root / "diagnostics",
        "plots": root / "plots",
        "selected_hkls": root / "selected_hkls",
        "logs": root / "logs",
        "commands": root / "commands",
    }
    for p in paths.values():
        if p is not root:
            p.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    out = ensure_output_layout(args.output_root)

    metadata_path = args.run_metadata
    if metadata_path is None:
        maybe = args.scores.parent / "run_metadata.json"
        if maybe.exists():
            metadata_path = maybe

    selected_hkls: set[tuple[int, int, int]] | None = None
    auto_select_notes: list[str] = []
    auto_select_stats: dict[str, int] | None = None

    if args.selected_hkls is not None:
        selected_hkls = load_selected_hkls(args.selected_hkls)
    elif args.target_d is not None:
        unit_cell = load_unit_cell_from_metadata(metadata_path)
        selected_hkls, auto_select_notes, auto_select_stats = auto_select_hkls_by_d(
            args.unmerged,
            target_d=float(args.target_d),
            tolerance=float(args.target_d_tolerance),
            max_hkls=int(args.target_d_max_hkls),
            unit_cell=unit_cell,
        )
        if selected_hkls:
            auto_file = out["selected_hkls"] / "auto_selected_hkls_from_target_d.txt"
            with auto_file.open("w", encoding="utf-8") as fh:
                for h, k, l in sorted(selected_hkls):
                    fh.write(f"{h:4d} {k:4d} {l:4d}\n")

    if args.smoke_only:
        table, stats = parse_unmerged_observations(
            args.unmerged,
            selected_hkls=selected_hkls,
            smoke_max_reflections=args.smoke_max_reflections,
        )
        print("SMOKE_ONLY_OK")
        print(f"rows_kept={len(table):,}")
        print(f"stats={json.dumps(stats, sort_keys=True)}")
        if selected_hkls is not None:
            print(f"selected_hkls={len(selected_hkls)}")
        return

    unmerged, unmerged_stats = parse_unmerged_observations(
        args.unmerged,
        selected_hkls=selected_hkls,
        smoke_max_reflections=args.smoke_max_reflections,
    )
    if unmerged.empty:
        raise SystemExit("No unmerged rows remained after exclusions/selection.")

    key_cols = ["source_norm", "event_norm", "h", "k", "l"]
    unmerged_dup = duplicate_rows(unmerged, key_cols)
    dup_diag_path = out["diagnostics"] / "duplicate_diagnostics.csv"
    if not unmerged_dup.empty:
        write_duplicate_diagnostics(dup_diag_path, "unmerged", unmerged_dup, key_cols)
        raise SystemExit(
            "Duplicate source+event+signed hkl rows found in unmerged observations. "
            f"Diagnostics written: {dup_diag_path}"
        )

    scores, source_col, component_columns = load_scores(
        args.scores,
        selected_hkls=selected_hkls,
        risk_column=args.risk_column,
        chunksize=args.score_chunksize,
    )
    if scores.empty:
        raise SystemExit("No score rows were loaded after HKL filtering.")

    score_dup = duplicate_rows(scores, key_cols)
    if not score_dup.empty:
        write_duplicate_diagnostics(dup_diag_path, "scores", score_dup, key_cols)
        raise SystemExit(
            "Duplicate source+event+signed hkl rows found in score table. "
            f"Diagnostics written: {dup_diag_path}"
        )

    joined, match_stats, join_note, join_debug = join_unmerged_with_scores(
        unmerged,
        scores,
        risk_column=args.risk_column,
    )

    joined["score_matched"] = joined["match_mode"].isin(["primary", "basename"])
    joined.to_csv(out["outputs"] / "joined_selected_observations.csv", index=False)

    matched = joined.loc[joined["score_matched"]].copy()
    if matched.empty:
        raise SystemExit("No matched rows between unmerged observations and scores.")

    # Per-HKL residuals.
    per_hkl_median = matched.groupby(["h", "k", "l"])["I_pr"].transform("median")
    matched["median_I_pr_within_hkl"] = per_hkl_median
    matched["signed_residual"] = matched["I_pr"] - matched["median_I_pr_within_hkl"]

    # Robust denominator per HKL: max(MAD, IQR/1.349, tiny).
    grouped = matched.groupby(["h", "k", "l"])["I_pr"]
    mad_series = grouped.transform(lambda s: robust_mad(s))
    iqr_series = grouped.transform(lambda s: float(s.quantile(0.75) - s.quantile(0.25)))
    robust_denom = np.maximum(np.maximum(mad_series.to_numpy(dtype=float), (iqr_series / 1.349).to_numpy(dtype=float)), 1e-12)
    matched["robust_denominator"] = robust_denom
    matched["relative_residual"] = matched["signed_residual"] / matched["robust_denominator"]
    matched["abs_relative_residual"] = matched["relative_residual"].abs()

    # Risk quantiles and groups.
    q_low_map = matched.groupby(["h", "k", "l"])[args.risk_column].transform(lambda s: float(s.quantile(args.low_quantile)))
    q_high_map = matched.groupby(["h", "k", "l"])[args.risk_column].transform(lambda s: float(s.quantile(args.high_quantile)))
    matched["risk_q_low"] = q_low_map
    matched["risk_q_high"] = q_high_map
    matched["risk_group"] = "mid"
    matched.loc[matched[args.risk_column] <= matched["risk_q_low"], "risk_group"] = "low"
    matched.loc[matched[args.risk_column] >= matched["risk_q_high"], "risk_group"] = "high"

    # Save enriched matched rows (still under outputs/).
    matched.to_csv(out["outputs"] / "joined_selected_observations.csv", index=False)

    # Summary statistics per HKL.
    summary_rows: list[dict[str, object]] = []
    plot_files: list[str] = []

    for (h, k, l), group in matched.groupby(["h", "k", "l"], sort=True):
        if len(group) < int(args.min_hkl_observations):
            continue

        low = group[group["risk_group"] == "low"]
        high = group[group["risk_group"] == "high"]

        mad_i = robust_mad(group["I_pr"])
        iqr_i = float(group["I_pr"].quantile(0.75) - group["I_pr"].quantile(0.25))
        skew_i = robust_skew(group["I_pr"])

        low_median = float(low["I_pr"].median()) if not low.empty else np.nan
        high_median = float(high["I_pr"].median()) if not high.empty else np.nan
        low_mad = robust_mad(low["I_pr"]) if not low.empty else np.nan
        high_mad = robust_mad(high["I_pr"]) if not high.empty else np.nan

        mw_u, mw_p, ks_d, ks_p = maybe_scipy_tests(low["I_pr"], high["I_pr"])

        summary_rows.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "n_obs": int(len(group)),
                "median_I_pr": float(group["I_pr"].median()),
                "mean_I_pr": float(group["I_pr"].mean()),
                "std_I_pr": float(group["I_pr"].std(ddof=1)) if len(group) > 1 else np.nan,
                "MAD_I_pr": mad_i,
                "IQR_I_pr": iqr_i,
                "robust_skew_I_pr": skew_i,
                "spearman_risk_vs_I_pr": maybe_spearman(group[args.risk_column], group["I_pr"]),
                "spearman_risk_vs_signed_residual": maybe_spearman(group[args.risk_column], group["signed_residual"]),
                "spearman_risk_vs_abs_relative_residual": maybe_spearman(
                    group[args.risk_column], group["abs_relative_residual"]
                ),
                "high_risk_median_I_pr": high_median,
                "low_risk_median_I_pr": low_median,
                "high_risk_MAD": high_mad,
                "low_risk_MAD": low_mad,
                "high_minus_low_median_I_pr": high_median - low_median if np.isfinite(high_median) and np.isfinite(low_median) else np.nan,
                "high_over_low_MAD_ratio": high_mad / low_mad if np.isfinite(high_mad) and np.isfinite(low_mad) and abs(low_mad) > 1e-12 else np.nan,
                "mannwhitney_u": mw_u,
                "mannwhitney_pvalue": mw_p,
                "ks_statistic": ks_d,
                "ks_pvalue": ks_p,
                "risk_column": args.risk_column,
                "risk_low_quantile": float(args.low_quantile),
                "risk_high_quantile": float(args.high_quantile),
            }
        )

        plot_files.extend(
            make_hkl_plots(
                group,
                risk_column=args.risk_column,
                low_q=float(args.low_quantile),
                high_q=float(args.high_quantile),
                plots_dir=out["plots"],
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out["outputs"] / "hkl_summary_statistics.csv", index=False)

    # Diagnostics text summary.
    match_diag_lines = []
    match_diag_lines.append("Partiality-risk match diagnostics")
    match_diag_lines.append("")
    match_diag_lines.append(f"unmerged_file: {args.unmerged}")
    match_diag_lines.append(f"score_file: {args.scores}")
    match_diag_lines.append(f"score_source_column: {source_col}")
    match_diag_lines.append(f"risk_column: {args.risk_column}")
    match_diag_lines.append("")
    match_diag_lines.append("unmerged_filter_stats:")
    for k, v in unmerged_stats.items():
        match_diag_lines.append(f"  {k}: {v}")

    if auto_select_stats is not None:
        match_diag_lines.append("")
        match_diag_lines.append("auto_selection_scan_stats:")
        for k, v in auto_select_stats.items():
            match_diag_lines.append(f"  {k}: {v}")

    if auto_select_notes:
        match_diag_lines.append("")
        match_diag_lines.append("auto_selection_notes:")
        for line in auto_select_notes:
            match_diag_lines.append(f"  - {line}")

    match_diag_lines.append("")
    match_diag_lines.append("join_stats:")
    for k, v in match_stats.items():
        match_diag_lines.append(f"  {k}: {v}")
    if join_note:
        match_diag_lines.append(f"  note: {join_note}")

    match_diag_lines.append("")
    match_diag_lines.append("join_debug:")
    match_diag_lines.append(
        "  duplicated_score_column_names_before_cleanup: "
        f"{join_debug.get('duplicated_score_column_names_before_cleanup', [])}"
    )
    match_diag_lines.append(
        "  duplicated_unmerged_column_names_before_cleanup: "
        f"{join_debug.get('duplicated_unmerged_column_names_before_cleanup', [])}"
    )
    match_diag_lines.append(f"  final_merge_key: {join_debug.get('final_merge_key', [])}")
    match_diag_lines.append(f"  payload_column_count: {join_debug.get('payload_column_count', 0)}")
    match_diag_lines.append(
        "  hkl_removed_from_payload_cols: "
        f"{join_debug.get('hkl_removed_from_payload_cols', False)}"
    )
    match_diag_lines.append(
        "  payload_cols_removed_because_in_key: "
        f"{join_debug.get('payload_cols_removed_because_in_key', [])}"
    )

    match_diag_lines.append("")
    match_diag_lines.append(f"selected_hkls_count: {len(selected_hkls) if selected_hkls is not None else 'all'}")
    match_diag_lines.append(f"matched_rows: {len(matched)}")
    match_diag_lines.append(f"summary_hkls_written: {len(summary)}")
    match_diag_lines.append(f"plot_files_written: {len(plot_files)}")

    (out["diagnostics"] / "match_diagnostics.txt").write_text("\n".join(match_diag_lines) + "\n", encoding="utf-8")

    # Optional unmatched sample.
    unmatched = joined.loc[~joined["score_matched"]].copy()
    if not unmatched.empty:
        unmatched.head(500).to_csv(out["diagnostics"] / "unmatched_first500.csv", index=False)

    print("DONE")
    print(f"joined_rows={len(joined):,}")
    print(f"matched_rows={len(matched):,}")
    print(f"summary_hkls={len(summary):,}")
    print(f"plots_written={len(plot_files):,}")


if __name__ == "__main__":
    main()

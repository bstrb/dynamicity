#!/usr/bin/env python3
"""Curated strong/weak HKL comparison plots for OriDyn partiality-risk diagnostics.

This tool reuses the validated partialator-unmerged + OriDyn score join logic:
source_filename + event + signed h,k,l.

It computes I_pr = I_unmerged * partiality, classifies HKLs by strength within
resolution shells, and creates curated shell-wise figures and per-HKL figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

DEFAULT_SHELL_EDGES = [1.8, 1.4, 1.1, 0.9, 0.7, 0.55, 0.4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create curated weak-vs-strong HKL comparison plots across resolution "
            "shells using OriDyn risk metrics and partiality-weighted unmerged data."
        )
    )
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator --unmerged-output file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder for curated comparisons")
    parser.add_argument(
        "--run-metadata",
        type=Path,
        default=None,
        help="Optional run_metadata.json. If omitted, tries <scores_dir>/run_metadata.json",
    )
    parser.add_argument(
        "--cell",
        nargs=6,
        type=float,
        metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"),
        default=None,
        help="Fallback unit cell if metadata does not include one",
    )
    parser.add_argument("--risk-column", default="S_dyn_geom", help="Risk column used for risk-grouping")
    parser.add_argument(
        "--color-risk-column",
        default="sigma_dyn_rel",
        help="Optional risk-like column used for spread ranking and color maps",
    )
    parser.add_argument("--score-chunksize", type=int, default=1_000_000)

    parser.add_argument(
        "--shell-edges",
        nargs="+",
        type=float,
        default=DEFAULT_SHELL_EDGES,
        help="Descending shell edges in Angstrom (used if --shell-centers not given)",
    )
    parser.add_argument(
        "--shell-centers",
        nargs="+",
        type=float,
        default=None,
        help="Optional shell centers in Angstrom (alternative to --shell-edges)",
    )
    parser.add_argument(
        "--shell-half-width",
        type=float,
        default=0.12,
        help="Half-width in Angstrom for center-based shell assignment",
    )

    parser.add_argument("--min-obs-per-hkl", type=int, default=50)
    parser.add_argument(
        "--prefer-high-nobs",
        dest="prefer_high_nobs",
        action="store_true",
        default=True,
        help="Prefer candidates with many observations during ranking (default: true)",
    )
    parser.add_argument(
        "--no-prefer-high-nobs",
        dest="prefer_high_nobs",
        action="store_false",
        help="Disable observation-count preference in ranking",
    )
    parser.add_argument(
        "--nobs-score-weight",
        type=float,
        default=0.5,
        help="Exponent weight for log1p(n_obs) contribution in candidate scoring",
    )

    parser.add_argument(
        "--explicit-hkls",
        type=Path,
        default=None,
        help="Optional text file with explicit HKLs (h k l), one per line; these are forced into plotting",
    )

    parser.add_argument("--weak-q-low", type=float, default=0.10)
    parser.add_argument("--weak-q-high", type=float, default=0.30)
    parser.add_argument("--strong-q-low", type=float, default=0.80)
    parser.add_argument("--strong-q-high", type=float, default=0.95)

    parser.add_argument("--n-weak-per-shell", type=int, default=3)
    parser.add_argument("--n-strong-per-shell", type=int, default=3)
    parser.add_argument("--n-control-per-shell", type=int, default=1)

    parser.add_argument("--risk-low-quantile", type=float, default=0.25)
    parser.add_argument("--risk-high-quantile", type=float, default=0.75)
    parser.add_argument("--eps", type=float, default=1e-12)

    parser.add_argument(
        "--smoke-max-reflections",
        type=int,
        default=None,
        help="Read at most this many eligible unmerged observations",
    )
    parser.add_argument(
        "--smoke-max-score-rows",
        type=int,
        default=None,
        help="Read at most this many rows from reflection_scores.csv",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Stop after join and print key stats",
    )

    args = parser.parse_args()

    if args.min_obs_per_hkl <= 0:
        raise SystemExit("--min-obs-per-hkl must be > 0")
    if args.nobs_score_weight < 0.0:
        raise SystemExit("--nobs-score-weight must be >= 0")
    if args.eps <= 0.0:
        raise SystemExit("--eps must be > 0")
    if args.shell_half_width <= 0.0:
        raise SystemExit("--shell-half-width must be > 0")
    if args.n_weak_per_shell < 0 or args.n_strong_per_shell < 0 or args.n_control_per_shell < 0:
        raise SystemExit("selection counts must be >= 0")

    for name in (
        "weak_q_low",
        "weak_q_high",
        "strong_q_low",
        "strong_q_high",
        "risk_low_quantile",
        "risk_high_quantile",
    ):
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise SystemExit(f"{name} must be in [0, 1]")

    if not (args.weak_q_low <= args.weak_q_high < args.strong_q_low <= args.strong_q_high):
        raise SystemExit("Expected weak_q_low <= weak_q_high < strong_q_low <= strong_q_high")
    if not (args.risk_low_quantile < args.risk_high_quantile):
        raise SystemExit("Expected risk_low_quantile < risk_high_quantile")

    if args.shell_centers is None:
        if len(args.shell_edges) < 2:
            raise SystemExit("--shell-edges must contain at least two values")
        if not all(args.shell_edges[i] > args.shell_edges[i + 1] for i in range(len(args.shell_edges) - 1)):
            raise SystemExit("--shell-edges must be strictly descending")

    return args


def normalize_source(value: object) -> str:
    return str(value).strip()


def normalize_event(value: object) -> str:
    return str(value).strip()


def source_basename(value: object) -> str:
    return Path(str(value).strip()).name


def hkl_key(h: int, k: int, l: int) -> str:
    return f"{int(h)},{int(k)},{int(l)}"


def hkl_slug(h: int, k: int, l: int) -> str:
    return f"h{h:+d}_k{k:+d}_l{l:+d}".replace("+", "p").replace("-", "m")


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


def load_explicit_hkls(path: Path | None) -> set[tuple[int, int, int]]:
    if path is None:
        return set()
    hkls: set[tuple[int, int, int]] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = parse_hkl_line(raw)
        if parsed is not None:
            hkls.add(parsed)
    if not hkls:
        raise SystemExit(f"No HKLs were parsed from --explicit-hkls file: {path}")
    return hkls


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

    for col in summary.columns:
        if col not in dup_rows.columns:
            dup_rows[col] = np.nan
    for col in dup_rows.columns:
        if col not in summary.columns:
            summary[col] = np.nan

    out = pd.concat([summary[dup_rows.columns], dup_rows], ignore_index=True)
    out.to_csv(path, index=False)


def duplicated_column_names(df: pd.DataFrame) -> list[str]:
    dups: list[str] = []
    seen: set[str] = set()
    for col in df.columns[df.columns.duplicated(keep=False)]:
        name = str(col)
        if name not in seen:
            seen.add(name)
            dups.append(name)
    return dups


def cleanup_duplicated_columns_keep_first(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str]]:
    dup_names = duplicated_column_names(df)
    if dup_names:
        print(
            f"WARNING: duplicate column labels found in {label}: {dup_names}. "
            "Dropping duplicates with keep='first'.",
            file=sys.stderr,
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df, dup_names


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
    return candidates[0]


def parse_unmerged_observations(
    unmerged_path: Path,
    smoke_max_reflections: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []

    stats = {
        "rows_parsed": 0,
        "excluded_flagged_crystal": 0,
        "excluded_partiality_too_small": 0,
        "excluded_nan_esd": 0,
        "excluded_nonpositive_partiality": 0,
        "eligible_rows": 0,
        "kept_rows": 0,
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
            reflection_flag = " ".join(parts[5:]).strip()

            if current_crystal_flagged:
                stats["excluded_flagged_crystal"] += 1
                continue
            if "partiality_too_small" in reflection_flag.lower():
                stats["excluded_partiality_too_small"] += 1
                continue
            if "nan_esd" in reflection_flag.lower():
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
    table[["h", "k", "l"]] = table[["h", "k", "l"]].astype("int64")
    return table, stats


def load_scores(
    scores_path: Path,
    risk_column: str,
    color_risk_column: str,
    chunksize: int,
    max_rows: int | None,
) -> tuple[pd.DataFrame, str, bool]:
    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)

    required = [source_column, "event", "h", "k", "l", risk_column]
    missing = [col for col in required if col not in header]
    if missing:
        raise SystemExit(f"Scores file missing required column(s): {missing}")

    has_color_column = color_risk_column in header
    keep_columns = [source_column, "event", "h", "k", "l", risk_column]
    if has_color_column and color_risk_column not in keep_columns:
        keep_columns.append(color_risk_column)

    chunks: list[pd.DataFrame] = []
    loaded_rows = 0

    reader = pd.read_csv(scores_path, usecols=keep_columns, chunksize=chunksize)
    for chunk in reader:
        chunk["h"] = pd.to_numeric(chunk["h"], errors="coerce")
        chunk["k"] = pd.to_numeric(chunk["k"], errors="coerce")
        chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
        chunk = chunk.dropna(subset=["h", "k", "l"])
        chunk[["h", "k", "l"]] = chunk[["h", "k", "l"]].astype("int64")

        if max_rows is not None:
            remaining = int(max_rows) - loaded_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        if chunk.empty:
            continue

        chunks.append(chunk)
        loaded_rows += int(len(chunk))

        if max_rows is not None and loaded_rows >= int(max_rows):
            break

    if not chunks:
        return pd.DataFrame(), source_column, has_color_column

    table = pd.concat(chunks, ignore_index=True)
    table = table.rename(columns={source_column: "source"})
    table["source"] = table["source"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    table["source_norm"] = table["source"]
    table["source_basename"] = table["source"].map(source_basename)
    table["event_norm"] = table["event"]

    table[risk_column] = pd.to_numeric(table[risk_column], errors="coerce")
    if has_color_column:
        table[color_risk_column] = pd.to_numeric(table[color_risk_column], errors="coerce")
    else:
        table[color_risk_column] = np.nan

    return table, source_column, has_color_column


def join_unmerged_with_scores(
    unmerged: pd.DataFrame,
    scores: pd.DataFrame,
    risk_column: str,
) -> tuple[pd.DataFrame, dict[str, int], str, dict[str, object]]:
    key = ["source_norm", "event_norm", "h", "k", "l"]
    join_debug: dict[str, object] = {}

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

    merged = unmerged.merge(scores_subset, on=key, how="left", indicator=True)
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
        score_basename_dups = duplicate_rows(scores, basename_key)
        if score_basename_dups.empty:
            missing_idx = merged.index[merged["match_mode"] == "missing"]
            if len(missing_idx) > 0:
                left = merged.loc[
                    missing_idx,
                    [
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
                    ],
                ].copy()

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

    merged[risk_column] = pd.to_numeric(merged.get(risk_column), errors="coerce")
    return merged, stats, note, join_debug


def coerce_cell_from_mapping(mapping: object) -> tuple[float, float, float, float, float, float] | None:
    if not isinstance(mapping, dict):
        return None

    required = ["a", "b", "c", "alpha", "beta", "gamma"]
    if not all(key in mapping for key in required):
        return None

    try:
        return (
            float(mapping["a"]),
            float(mapping["b"]),
            float(mapping["c"]),
            float(mapping["alpha"]),
            float(mapping["beta"]),
            float(mapping["gamma"]),
        )
    except (TypeError, ValueError):
        return None


def extract_cell_recursive(node: object, depth: int = 0, max_depth: int = 8) -> tuple[float, float, float, float, float, float] | None:
    if depth > max_depth:
        return None

    direct = coerce_cell_from_mapping(node)
    if direct is not None:
        return direct

    if isinstance(node, dict):
        for key in ("unit_cell", "cell", "unitCell", "unit_cell_parameters"):
            if key in node:
                parsed = extract_cell_recursive(node[key], depth + 1, max_depth)
                if parsed is not None:
                    return parsed
        for value in node.values():
            parsed = extract_cell_recursive(value, depth + 1, max_depth)
            if parsed is not None:
                return parsed

    if isinstance(node, list):
        for value in node:
            parsed = extract_cell_recursive(value, depth + 1, max_depth)
            if parsed is not None:
                return parsed

    return None


def load_cell_from_metadata(path: Path | None) -> tuple[float, float, float, float, float, float] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return extract_cell_recursive(data)


def resolve_cell(cli_cell: list[float] | None, metadata_path: Path | None, scores_path: Path) -> tuple[tuple[float, float, float, float, float, float], str]:
    if cli_cell is not None:
        return tuple(float(v) for v in cli_cell), "cli"

    cell = load_cell_from_metadata(metadata_path)
    if cell is not None:
        return cell, f"metadata:{metadata_path}"

    auto_meta = scores_path.parent / "run_metadata.json"
    if metadata_path is None and auto_meta.exists():
        auto_cell = load_cell_from_metadata(auto_meta)
        if auto_cell is not None:
            return auto_cell, f"metadata:{auto_meta}"

    raise SystemExit(
        "Could not extract unit-cell parameters from metadata. "
        "Provide --cell a b c alpha beta gamma."
    )


def reciprocal_metric_tensor(cell: tuple[float, float, float, float, float, float]) -> np.ndarray:
    a, b, c, alpha_deg, beta_deg, gamma_deg = cell
    if min(a, b, c) <= 0.0:
        raise SystemExit("Unit-cell lengths must be positive")

    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)

    g = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ],
        dtype=float,
    )

    det = float(np.linalg.det(g))
    if not np.isfinite(det) or det <= 0.0:
        raise SystemExit("Invalid unit-cell metric tensor; determinant must be positive")

    return np.linalg.inv(g)


def d_spacing_from_hkl(h: int, k: int, l: int, g_star: np.ndarray) -> float:
    if h == 0 and k == 0 and l == 0:
        return np.nan
    vec = np.array([float(h), float(k), float(l)], dtype=float)
    inv_d_sq = float(vec @ g_star @ vec)
    if not np.isfinite(inv_d_sq) or inv_d_sq <= 0.0:
        return np.nan
    return float(1.0 / math.sqrt(inv_d_sq))


def maybe_spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].corr(frame["y"], method="spearman"))


def robust_mad(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def make_shell_records(shell_edges: list[float], shell_centers: list[float] | None, shell_half_width: float) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []

    if shell_centers is not None and len(shell_centers) > 0:
        centers = sorted([float(x) for x in shell_centers], reverse=True)
        for center in centers:
            high = center + float(shell_half_width)
            low = max(center - float(shell_half_width), 1e-6)
            label = f"{high:.3f}-{low:.3f}A"
            records.append(
                {
                    "mode": "center",
                    "center": center,
                    "high": high,
                    "low": low,
                    "label": label,
                }
            )
        return records

    for idx in range(len(shell_edges) - 1):
        high = float(shell_edges[idx])
        low = float(shell_edges[idx + 1])
        label = f"{high:.3f}-{low:.3f}A"
        records.append(
            {
                "mode": "edge",
                "index": idx,
                "high": high,
                "low": low,
                "label": label,
            }
        )
    return records


def assign_shell(d_spacing: float, records: list[dict[str, float | str]]) -> tuple[str | None, float, float]:
    if not np.isfinite(d_spacing):
        return None, np.nan, np.nan

    if not records:
        return None, np.nan, np.nan

    mode = str(records[0]["mode"])

    if mode == "center":
        candidates: list[tuple[float, dict[str, float | str]]] = []
        for rec in records:
            high = float(rec["high"])
            low = float(rec["low"])
            if d_spacing <= high and d_spacing >= low:
                center = float(rec["center"])
                candidates.append((abs(d_spacing - center), rec))
        if not candidates:
            return None, np.nan, np.nan
        rec = sorted(candidates, key=lambda t: t[0])[0][1]
        return str(rec["label"]), float(rec["high"]), float(rec["low"])

    for idx, rec in enumerate(records):
        high = float(rec["high"])
        low = float(rec["low"])
        if idx < len(records) - 1:
            in_shell = d_spacing <= high and d_spacing > low
        else:
            in_shell = d_spacing <= high and d_spacing >= low
        if in_shell:
            return str(rec["label"]), high, low

    return None, np.nan, np.nan


def compute_hkl_basics(
    matched: pd.DataFrame,
    g_star: np.ndarray,
    shell_records: list[dict[str, float | str]],
) -> pd.DataFrame:
    grouped = matched.groupby(["h", "k", "l"], as_index=False)
    hkl_table = grouped.agg(
        n_obs=("I_pr", "size"),
        median_I_pr=("I_pr", "median"),
    )

    mad_table = matched.groupby(["h", "k", "l"])["I_pr"].apply(robust_mad).reset_index(name="MAD_I_pr")
    hkl_table = hkl_table.merge(mad_table, on=["h", "k", "l"], how="left")

    hkl_table["d_spacing"] = hkl_table.apply(
        lambda row: d_spacing_from_hkl(int(row["h"]), int(row["k"]), int(row["l"]), g_star),
        axis=1,
    )

    assigned = hkl_table["d_spacing"].map(lambda d: assign_shell(float(d), shell_records))
    hkl_table["shell_label"] = assigned.map(lambda x: x[0])
    hkl_table["shell_d_high"] = assigned.map(lambda x: x[1])
    hkl_table["shell_d_low"] = assigned.map(lambda x: x[2])

    hkl_table = hkl_table.loc[hkl_table["shell_label"].notna()].copy()

    return hkl_table


def add_observation_residuals(
    matched: pd.DataFrame,
    hkl_table: pd.DataFrame,
    risk_column: str,
    color_risk_column: str,
    risk_low_q: float,
    risk_high_q: float,
    eps: float,
) -> pd.DataFrame:
    keep_cols = ["h", "k", "l", "n_obs", "median_I_pr", "MAD_I_pr", "d_spacing", "shell_label", "shell_d_high", "shell_d_low"]
    obs = matched.merge(hkl_table[keep_cols], on=["h", "k", "l"], how="inner")
    if obs.empty:
        return obs

    obs[risk_column] = pd.to_numeric(obs[risk_column], errors="coerce")
    obs[color_risk_column] = pd.to_numeric(obs[color_risk_column], errors="coerce")

    obs["residual"] = obs["I_pr"] - obs["median_I_pr"]
    obs["abs_residual"] = obs["residual"].abs()

    denom_rel = np.maximum(np.abs(obs["median_I_pr"].to_numpy(dtype=float)), float(eps))
    denom_rob = np.maximum(obs["MAD_I_pr"].to_numpy(dtype=float), float(eps))
    obs["relative_residual"] = obs["residual"].to_numpy(dtype=float) / denom_rel
    obs["robust_residual"] = obs["residual"].to_numpy(dtype=float) / denom_rob

    q_low = obs.groupby(["h", "k", "l"])[risk_column].transform(lambda s: float(s.quantile(risk_low_q)))
    q_high = obs.groupby(["h", "k", "l"])[risk_column].transform(lambda s: float(s.quantile(risk_high_q)))
    obs["risk_q_low"] = q_low
    obs["risk_q_high"] = q_high

    obs["risk_group"] = "mid"
    obs.loc[obs[risk_column] <= obs["risk_q_low"], "risk_group"] = "low"
    obs.loc[obs[risk_column] >= obs["risk_q_high"], "risk_group"] = "high"

    return obs


def summarize_hkls(
    obs: pd.DataFrame,
    risk_column: str,
    color_risk_column: str,
) -> pd.DataFrame:
    keys = ["h", "k", "l"]

    def summarize_one(group: pd.DataFrame) -> pd.Series:
        risk_vals = pd.to_numeric(group[risk_column], errors="coerce")
        color_vals = pd.to_numeric(group[color_risk_column], errors="coerce")

        low = group.loc[group["risk_group"] == "low"]
        high = group.loc[group["risk_group"] == "high"]

        med_low = float(low["I_pr"].median()) if not low.empty else np.nan
        med_high = float(high["I_pr"].median()) if not high.empty else np.nan
        med_low_rob = float(low["robust_residual"].median()) if not low.empty else np.nan
        med_high_rob = float(high["robust_residual"].median()) if not high.empty else np.nan

        return pd.Series(
            {
                "n_obs": int(len(group)),
                "d_spacing": float(group["d_spacing"].iloc[0]),
                "shell_label": str(group["shell_label"].iloc[0]),
                "shell_d_high": float(group["shell_d_high"].iloc[0]),
                "shell_d_low": float(group["shell_d_low"].iloc[0]),
                "median_I_pr": float(group["median_I_pr"].iloc[0]),
                "MAD_I_pr": float(group["MAD_I_pr"].iloc[0]),
                "risk_p05": float(risk_vals.quantile(0.05)) if risk_vals.notna().any() else np.nan,
                "risk_p50": float(risk_vals.quantile(0.50)) if risk_vals.notna().any() else np.nan,
                "risk_p95": float(risk_vals.quantile(0.95)) if risk_vals.notna().any() else np.nan,
                "color_p05": float(color_vals.quantile(0.05)) if color_vals.notna().any() else np.nan,
                "color_p50": float(color_vals.quantile(0.50)) if color_vals.notna().any() else np.nan,
                "color_p95": float(color_vals.quantile(0.95)) if color_vals.notna().any() else np.nan,
                "median_low_risk_I_pr": med_low,
                "median_high_risk_I_pr": med_high,
                "high_minus_low_median_shift": med_high - med_low if np.isfinite(med_high) and np.isfinite(med_low) else np.nan,
                "median_low_risk_robust_residual": med_low_rob,
                "median_high_risk_robust_residual": med_high_rob,
                "high_minus_low_median_shift_robust": (
                    med_high_rob - med_low_rob if np.isfinite(med_high_rob) and np.isfinite(med_low_rob) else np.nan
                ),
                "spearman_risk_vs_I_pr_residual": maybe_spearman(group[risk_column], group["residual"]),
                "spearman_risk_vs_abs_residual": maybe_spearman(group[risk_column], group["abs_residual"]),
            }
        )

    try:
        summary = obs.groupby(keys).apply(summarize_one, include_groups=False).reset_index()
    except TypeError:
        # Fallback for older pandas without include_groups.
        summary = obs.groupby(keys).apply(
            lambda g: summarize_one(g.drop(columns=keys, errors="ignore"))
        ).reset_index()

    if "risk_p95" in summary.columns and "risk_p05" in summary.columns:
        summary["risk_spread"] = summary["risk_p95"] - summary["risk_p05"]
    else:
        summary["risk_spread"] = np.nan

    if "color_p95" in summary.columns and "color_p05" in summary.columns:
        summary["color_spread"] = summary["color_p95"] - summary["color_p05"]
    else:
        summary["color_spread"] = np.nan

    summary["preference_spread"] = np.where(summary["color_spread"].notna(), summary["color_spread"], summary["risk_spread"])

    return summary


def classify_and_select_hkls(
    hkl_summary: pd.DataFrame,
    weak_q_low: float,
    weak_q_high: float,
    strong_q_low: float,
    strong_q_high: float,
    n_weak_per_shell: int,
    n_strong_per_shell: int,
    n_control_per_shell: int,
    min_obs_per_hkl: int,
    prefer_high_nobs: bool,
    nobs_score_weight: float,
    explicit_hkls: set[tuple[int, int, int]],
) -> pd.DataFrame:
    out = hkl_summary.copy()
    out["strength_class"] = "other"
    out["selected_role"] = "none"
    out["selection_rank"] = np.nan
    out["candidate_score"] = np.nan

    out["median_I_pr_rank_within_shell"] = np.nan
    out["strength_percentile"] = np.nan

    out["nobs_below_threshold"] = out["n_obs"] < int(min_obs_per_hkl)
    out["nobs_warning"] = np.where(out["nobs_below_threshold"], f"LOW_NOBS(<{int(min_obs_per_hkl)})", "")

    out["explicit_requested"] = False
    if explicit_hkls:
        tuples = list(zip(out["h"].astype(int), out["k"].astype(int), out["l"].astype(int)))
        out["explicit_requested"] = [t in explicit_hkls for t in tuples]

    out["weak_q_low_shell"] = np.nan
    out["weak_q_high_shell"] = np.nan
    out["strong_q_low_shell"] = np.nan
    out["strong_q_high_shell"] = np.nan

    def build_candidate_score(pool: pd.DataFrame, expect_negative: bool | None) -> pd.Series:
        if pool.empty:
            return pd.Series(dtype=float)

        robust_shift = pd.to_numeric(pool["high_minus_low_median_shift_robust"], errors="coerce")
        raw_shift = pd.to_numeric(pool["high_minus_low_median_shift"], errors="coerce")
        effect = robust_shift.abs()
        effect = np.where(np.isfinite(effect), effect, np.abs(raw_shift))
        effect = pd.Series(effect, index=pool.index, dtype=float).fillna(0.0)

        spread = pd.to_numeric(pool["preference_spread"], errors="coerce")
        spread = spread.where(np.isfinite(spread), pd.to_numeric(pool["risk_spread"], errors="coerce"))
        spread = spread.fillna(0.0).clip(lower=0.0)

        if prefer_high_nobs:
            nobs_term = np.power(np.log1p(pd.to_numeric(pool["n_obs"], errors="coerce").fillna(0.0).clip(lower=0.0)), float(nobs_score_weight))
        else:
            nobs_term = pd.Series(1.0, index=pool.index)

        sign_factor = pd.Series(1.0, index=pool.index)
        if expect_negative is True:
            sign_factor = np.where(pd.to_numeric(pool["high_minus_low_median_shift_robust"], errors="coerce") < 0.0, 1.25, 0.75)
            sign_factor = pd.Series(sign_factor, index=pool.index)
        elif expect_negative is False:
            sign_factor = np.where(pd.to_numeric(pool["high_minus_low_median_shift_robust"], errors="coerce") > 0.0, 1.25, 0.75)
            sign_factor = pd.Series(sign_factor, index=pool.index)

        score = effect * spread * nobs_term * sign_factor
        return pd.Series(score, index=pool.index).fillna(0.0)

    for shell_label, shell_group in out.groupby("shell_label"):
        idx = shell_group.index

        rank_desc = shell_group["median_I_pr"].rank(method="min", ascending=False)
        strength_pct = shell_group["median_I_pr"].rank(method="average", pct=True)
        out.loc[idx, "median_I_pr_rank_within_shell"] = rank_desc.to_numpy(dtype=float)
        out.loc[idx, "strength_percentile"] = strength_pct.to_numpy(dtype=float)

        quant_source = shell_group.loc[shell_group["n_obs"] >= int(min_obs_per_hkl)]
        if quant_source.empty:
            quant_source = shell_group

        med = quant_source["median_I_pr"]

        q_w_lo = float(med.quantile(weak_q_low))
        q_w_hi = float(med.quantile(weak_q_high))
        q_s_lo = float(med.quantile(strong_q_low))
        q_s_hi = float(med.quantile(strong_q_high))

        out.loc[idx, "weak_q_low_shell"] = q_w_lo
        out.loc[idx, "weak_q_high_shell"] = q_w_hi
        out.loc[idx, "strong_q_low_shell"] = q_s_lo
        out.loc[idx, "strong_q_high_shell"] = q_s_hi

        shell_idx = out.index.isin(idx)
        weak_mask = shell_idx & out["median_I_pr"].between(q_w_lo, q_w_hi, inclusive="both")
        strong_mask = shell_idx & out["median_I_pr"].between(q_s_lo, q_s_hi, inclusive="both")

        out.loc[weak_mask, "strength_class"] = "weak"
        out.loc[strong_mask, "strength_class"] = "strong"

        shell_slice = out.loc[idx].copy()
        auto_pool = shell_slice.loc[shell_slice["n_obs"] >= int(min_obs_per_hkl)].copy()
        if auto_pool.empty:
            auto_pool = shell_slice.copy()

        strong_pool = auto_pool.loc[auto_pool["strength_class"] == "strong"].copy()
        strong_pool["candidate_score"] = build_candidate_score(strong_pool, expect_negative=True)
        out.loc[strong_pool.index, "candidate_score"] = strong_pool["candidate_score"]
        strong_pool = strong_pool.sort_values(
            ["candidate_score", "preference_spread", "n_obs", "median_I_pr"],
            ascending=[False, False, False, False],
        )

        weak_pool = auto_pool.loc[auto_pool["strength_class"] == "weak"].copy()
        weak_pool["candidate_score"] = build_candidate_score(weak_pool, expect_negative=False)
        out.loc[weak_pool.index, "candidate_score"] = weak_pool["candidate_score"]
        weak_pool = weak_pool.sort_values(
            ["candidate_score", "preference_spread", "n_obs", "median_I_pr"],
            ascending=[False, False, False, True],
        )

        selected_strong = strong_pool.head(int(n_strong_per_shell))
        selected_weak = weak_pool.head(int(n_weak_per_shell))

        if not selected_strong.empty:
            out.loc[selected_strong.index, "selected_role"] = "selected_strong"
            out.loc[selected_strong.index, "selection_rank"] = np.arange(1, len(selected_strong) + 1, dtype=float)

        if not selected_weak.empty:
            out.loc[selected_weak.index, "selected_role"] = "selected_weak"
            out.loc[selected_weak.index, "selection_rank"] = np.arange(1, len(selected_weak) + 1, dtype=float)

        if int(n_control_per_shell) > 0:
            used = set(selected_strong.index.tolist() + selected_weak.index.tolist())
            control_pool = auto_pool.loc[~auto_pool.index.isin(used)].copy()
            control_pool["candidate_score"] = build_candidate_score(control_pool, expect_negative=None)
            out.loc[control_pool.index, "candidate_score"] = control_pool["candidate_score"]
            control_pool["abs_shift_robust"] = pd.to_numeric(control_pool["high_minus_low_median_shift_robust"], errors="coerce").abs()
            control_pool = control_pool.sort_values(
                ["abs_shift_robust", "n_obs", "preference_spread", "candidate_score"],
                ascending=[True, False, False, False],
            )
            selected_control = control_pool.head(int(n_control_per_shell))
            if not selected_control.empty:
                out.loc[selected_control.index, "selected_role"] = "control"
                out.loc[selected_control.index, "selection_rank"] = np.arange(1, len(selected_control) + 1, dtype=float)

        if explicit_hkls:
            shell_explicit = shell_slice.loc[shell_slice["explicit_requested"]].copy()
            if not shell_explicit.empty:
                shell_explicit["candidate_score"] = build_candidate_score(shell_explicit, expect_negative=None)
                out.loc[shell_explicit.index, "candidate_score"] = shell_explicit["candidate_score"]

                to_add = shell_explicit.loc[out.loc[shell_explicit.index, "selected_role"] == "none"]
                if not to_add.empty:
                    out.loc[to_add.index, "selected_role"] = "explicit"
                    out.loc[to_add.index, "selection_rank"] = np.arange(1, len(to_add) + 1, dtype=float)

    return out


def role_sort_key(role: str) -> int:
    return {
        "selected_weak": 0,
        "selected_strong": 1,
        "explicit": 2,
        "control": 3,
    }.get(role, 99)


def plot_shell_curated_comparison(
    shell_label: str,
    shell_summary: pd.DataFrame,
    shell_obs: pd.DataFrame,
    risk_column: str,
    color_risk_column: str,
    out_path: Path,
) -> bool:
    if shell_summary.empty or shell_obs.empty:
        return False

    order = shell_summary.copy()
    order["role_order"] = order["selected_role"].map(role_sort_key)
    order = order.sort_values(["role_order", "selection_rank", "n_obs"], ascending=[True, True, False])

    ncols = int(len(order))
    if ncols == 0:
        return False

    fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 8.0), dpi=170, squeeze=False)

    for col_idx, (_, row) in enumerate(order.iterrows()):
        h = int(row["h"])
        k = int(row["k"])
        l = int(row["l"])

        g = shell_obs.loc[(shell_obs["h"] == h) & (shell_obs["k"] == k) & (shell_obs["l"] == l)].copy()
        low = g.loc[g["risk_group"] == "low"]
        high = g.loc[g["risk_group"] == "high"]

        ax_hist = axes[0, col_idx]
        if not low.empty:
            ax_hist.hist(low["I_pr"], bins=45, alpha=0.55, color="#2b8cbe", label=f"low (n={len(low)})")
        if not high.empty:
            ax_hist.hist(high["I_pr"], bins=45, alpha=0.55, color="#de2d26", label=f"high (n={len(high)})")
        ax_hist.set_xlabel("I_pr")
        ax_hist.set_ylabel("count")
        if col_idx == 0:
            ax_hist.legend(loc="best", fontsize=8)

        rho = row.get("spearman_risk_vs_I_pr_residual", np.nan)
        warning_text = str(row.get("nobs_warning", "")).strip()
        shift_raw = row.get("high_minus_low_median_shift", np.nan)
        shift_rob = row.get("high_minus_low_median_shift_robust", np.nan)
        title = (
            f"({h},{k},{l}) | d={row['d_spacing']:.3f}A\n"
            f"n={int(row['n_obs'])}, med={row['median_I_pr']:.2f}\n"
            f"shift={shift_raw:.3f}, shift_rob={shift_rob:.3f}, rho={rho:.3f}"
        )
        if warning_text:
            title += f"\n{warning_text}"
        ax_hist.set_title(title, fontsize=9)

        ax_scatter = axes[1, col_idx]
        x = pd.to_numeric(g[risk_column], errors="coerce")
        y = pd.to_numeric(g["residual"], errors="coerce")
        c = pd.to_numeric(g[color_risk_column], errors="coerce")

        valid = x.notna() & y.notna()
        if valid.any():
            if c.notna().any():
                sc = ax_scatter.scatter(
                    x[valid],
                    y[valid],
                    c=c[valid],
                    cmap="viridis",
                    s=10,
                    alpha=0.7,
                    linewidths=0,
                )
                if col_idx == ncols - 1:
                    cbar = fig.colorbar(sc, ax=ax_scatter, fraction=0.05, pad=0.02)
                    cbar.set_label(color_risk_column)
            else:
                ax_scatter.scatter(x[valid], y[valid], s=10, alpha=0.7, color="#444444", linewidths=0)
        ax_scatter.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
        ax_scatter.set_xlabel(risk_column)
        ax_scatter.set_ylabel("residual")

    fig.suptitle(f"Curated HKL comparison for shell {shell_label}", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_per_hkl(
    row: pd.Series,
    hkl_obs: pd.DataFrame,
    risk_column: str,
    color_risk_column: str,
    out_path: Path,
) -> bool:
    if hkl_obs.empty:
        return False

    h = int(row["h"])
    k = int(row["k"])
    l = int(row["l"])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), dpi=170)

    low = hkl_obs.loc[hkl_obs["risk_group"] == "low"]
    high = hkl_obs.loc[hkl_obs["risk_group"] == "high"]

    ax = axes[0, 0]
    if not low.empty:
        ax.hist(low["I_pr"], bins=45, alpha=0.55, color="#2b8cbe", label=f"low (n={len(low)})")
    if not high.empty:
        ax.hist(high["I_pr"], bins=45, alpha=0.55, color="#de2d26", label=f"high (n={len(high)})")
    ax.set_title("Raw I_pr histogram by risk group")
    ax.set_xlabel("I_pr")
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    x_strip = pd.to_numeric(hkl_obs["I_pr"], errors="coerce")
    c_strip = pd.to_numeric(hkl_obs[color_risk_column], errors="coerce")
    jitter = np.random.default_rng(0).normal(loc=0.0, scale=0.02, size=len(hkl_obs))
    valid_strip = x_strip.notna()
    if valid_strip.any():
        if c_strip.notna().any():
            sc = ax.scatter(x_strip[valid_strip], jitter[valid_strip.to_numpy()], c=c_strip[valid_strip], cmap="plasma", s=15, alpha=0.75)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
            cbar.set_label(color_risk_column)
        else:
            ax.scatter(x_strip[valid_strip], jitter[valid_strip.to_numpy()], s=15, color="#666666", alpha=0.75)
    ax.set_title("I_pr strip (colored by color-risk column)")
    ax.set_xlabel("I_pr")
    ax.set_yticks([])

    ax = axes[1, 0]
    xr = pd.to_numeric(hkl_obs[risk_column], errors="coerce")
    yr = pd.to_numeric(hkl_obs["residual"], errors="coerce")
    valid_r = xr.notna() & yr.notna()
    if valid_r.any():
        colors = hkl_obs.loc[valid_r, "risk_group"].map({"low": "#2b8cbe", "high": "#de2d26", "mid": "#999999"}).fillna("#999999")
        ax.scatter(xr[valid_r], yr[valid_r], c=colors, s=15, alpha=0.8, linewidths=0)
    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    ax.set_title(f"{risk_column} vs residual")
    ax.set_xlabel(risk_column)
    ax.set_ylabel("residual")

    ax = axes[1, 1]
    xc = pd.to_numeric(hkl_obs[color_risk_column], errors="coerce")
    yc = pd.to_numeric(hkl_obs["residual"], errors="coerce")
    valid_c = xc.notna() & yc.notna()
    if valid_c.any():
        ax.scatter(xc[valid_c], yc[valid_c], s=15, color="#444444", alpha=0.75, linewidths=0)
    else:
        ax.text(0.5, 0.5, f"No numeric {color_risk_column}", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    ax.set_title(f"{color_risk_column} vs residual")
    ax.set_xlabel(color_risk_column)
    ax.set_ylabel("residual")

    rho = row.get("spearman_risk_vs_I_pr_residual", np.nan)
    warning_text = str(row.get("nobs_warning", "")).strip()
    fig.suptitle(
        (
            f"HKL ({h},{k},{l}) | d={row['d_spacing']:.3f}A | n={int(row['n_obs'])} | "
            f"med={row['median_I_pr']:.3f} | shift={row['high_minus_low_median_shift']:.3f} | rho={rho:.3f}"
            + (f" | {warning_text}" if warning_text else "")
        ),
        fontsize=12,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def ensure_output_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "plots": root / "plots",
        "per_hkl": root / "plots" / "per_hkl",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_readme(
    root: Path,
    args: argparse.Namespace,
    cell: tuple[float, float, float, float, float, float],
    cell_source: str,
    source_col: str,
    has_color_col: bool,
    unmerged_stats: dict[str, int],
    join_stats: dict[str, int],
    join_note: str,
    join_debug: dict[str, object],
    selected_count: int,
    shell_plot_count: int,
    per_hkl_plot_count: int,
) -> None:
    lines: list[str] = []
    lines.append("Curated HKL strength-pair plotting diagnostics")
    lines.append("")
    lines.append("Goal:")
    lines.append("  Compare strong vs weak HKLs at similar resolution and inspect whether high-risk observations")
    lines.append("  shift/skew strong reflections downward and weak reflections upward.")
    lines.append("")
    lines.append(f"unmerged_file: {args.unmerged}")
    lines.append(f"scores_file: {args.scores}")
    lines.append(f"score_source_column: {source_col}")
    lines.append(f"risk_column: {args.risk_column}")
    lines.append(f"color_risk_column: {args.color_risk_column} (present={has_color_col})")
    lines.append(f"cell_source: {cell_source}")
    lines.append("cell(a b c alpha beta gamma): " + " ".join(f"{x:.6g}" for x in cell))
    lines.append("")

    if args.shell_centers:
        lines.append("shell_mode: centers")
        lines.append("shell_centers_A: " + " ".join(str(x) for x in args.shell_centers))
        lines.append(f"shell_half_width_A: {args.shell_half_width}")
    else:
        lines.append("shell_mode: edges")
        lines.append("shell_edges_A_desc: " + " ".join(str(x) for x in args.shell_edges))

    lines.append(f"min_obs_per_hkl: {args.min_obs_per_hkl}")
    lines.append(f"prefer_high_nobs: {args.prefer_high_nobs}")
    lines.append(f"nobs_score_weight: {args.nobs_score_weight}")
    lines.append(f"weak_quantiles: [{args.weak_q_low}, {args.weak_q_high}]")
    lines.append(f"strong_quantiles: [{args.strong_q_low}, {args.strong_q_high}]")
    lines.append(f"risk_quantiles_within_hkl: [{args.risk_low_quantile}, {args.risk_high_quantile}]")
    lines.append(f"selection_counts_per_shell: weak={args.n_weak_per_shell}, strong={args.n_strong_per_shell}, control={args.n_control_per_shell}")
    lines.append(f"explicit_hkls_file: {args.explicit_hkls}")
    lines.append("")
    lines.append("Plot logic:")
    lines.append("  - Within each shell, classify HKLs by median I_pr quantiles into weak/strong.")
    lines.append("  - Rank candidates by effect relevance, spread and n_obs preference.")
    lines.append("  - Candidate score uses |high_minus_low_median_shift_robust| * spread * log1p(n_obs)^weight.")
    lines.append("  - Spread uses color-risk spread when available, else risk spread.")
    lines.append("  - Select strong HKLs favoring negative high-minus-low shift and high median_I_pr.")
    lines.append("  - Select weak HKLs favoring positive high-minus-low shift and low/non-extreme median_I_pr.")
    lines.append("  - Explicit HKLs are always plotted; low-n explicit HKLs are flagged with LOW_NOBS warnings.")
    lines.append("  - Optionally add control HKLs with small shift magnitude.")
    lines.append("  - Shell figure rows: (A) low-vs-high risk I_pr histogram, (B) risk-vs-residual scatter.")
    lines.append("  - Per-HKL figure: I_pr histogram, I_pr strip colored by color-risk, risk-vs-residual, color-risk-vs-residual.")
    lines.append("")
    lines.append("unmerged_filter_stats:")
    for key, value in unmerged_stats.items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("join_stats:")
    for key, value in join_stats.items():
        lines.append(f"  {key}: {value}")
    if join_note:
        lines.append(f"  note: {join_note}")

    lines.append("")
    lines.append("join_debug:")
    lines.append(
        "  duplicated_score_column_names_before_cleanup: "
        + str(join_debug.get("duplicated_score_column_names_before_cleanup", []))
    )
    lines.append(
        "  duplicated_unmerged_column_names_before_cleanup: "
        + str(join_debug.get("duplicated_unmerged_column_names_before_cleanup", []))
    )
    lines.append("  final_merge_key: " + str(join_debug.get("final_merge_key", [])))
    lines.append("  payload_column_count: " + str(join_debug.get("payload_column_count", 0)))
    lines.append("  hkl_removed_from_payload_cols: " + str(join_debug.get("hkl_removed_from_payload_cols", False)))

    lines.append("")
    lines.append(f"selected_hkls: {selected_count}")
    lines.append(f"shell_plot_files: {shell_plot_count}")
    lines.append(f"per_hkl_plot_files: {per_hkl_plot_count}")

    (root / "README_curated_hkl_comparisons.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure_output_layout(args.output_root)

    metadata_path = args.run_metadata
    if metadata_path is None:
        maybe = args.scores.parent / "run_metadata.json"
        if maybe.exists():
            metadata_path = maybe

    cell, cell_source = resolve_cell(args.cell, metadata_path, args.scores)
    g_star = reciprocal_metric_tensor(cell)

    shell_records = make_shell_records(args.shell_edges, args.shell_centers, args.shell_half_width)
    explicit_hkls = load_explicit_hkls(args.explicit_hkls)

    unmerged, unmerged_stats = parse_unmerged_observations(
        args.unmerged,
        smoke_max_reflections=args.smoke_max_reflections,
    )
    if unmerged.empty:
        raise SystemExit("No eligible unmerged observations were parsed")

    key_cols = ["source_norm", "event_norm", "h", "k", "l"]
    dup_diag_path = out["root"] / "duplicate_diagnostics.csv"

    unmerged_dup = duplicate_rows(unmerged, key_cols)
    if not unmerged_dup.empty:
        write_duplicate_diagnostics(dup_diag_path, "unmerged", unmerged_dup, key_cols)
        raise SystemExit(
            "Duplicate source+event+signed hkl rows found in unmerged observations. "
            f"Diagnostics written: {dup_diag_path}"
        )

    scores, source_col, has_color_col = load_scores(
        args.scores,
        risk_column=args.risk_column,
        color_risk_column=args.color_risk_column,
        chunksize=args.score_chunksize,
        max_rows=args.smoke_max_score_rows,
    )
    if scores.empty:
        raise SystemExit("No score rows were loaded")

    score_dup = duplicate_rows(scores, key_cols)
    if not score_dup.empty:
        write_duplicate_diagnostics(dup_diag_path, "scores", score_dup, key_cols)
        raise SystemExit(
            "Duplicate source+event+signed hkl rows found in score table. "
            f"Diagnostics written: {dup_diag_path}"
        )

    joined, join_stats, join_note, join_debug = join_unmerged_with_scores(
        unmerged,
        scores,
        risk_column=args.risk_column,
    )
    joined["score_matched"] = joined["match_mode"].isin(["primary", "basename"])
    matched = joined.loc[joined["score_matched"]].copy()

    if args.smoke_only:
        print("SMOKE_ONLY_OK")
        print(f"unmerged_rows={len(unmerged):,}")
        print(f"score_rows={len(scores):,}")
        print(f"joined_rows={len(joined):,}")
        print(f"matched_rows={len(matched):,}")
        print("join_debug=" + json.dumps(join_debug, sort_keys=True))
        return

    if matched.empty:
        raise SystemExit("No matched rows between unmerged observations and scores")

    hkl_basics = compute_hkl_basics(
        matched,
        g_star=g_star,
        shell_records=shell_records,
    )
    if hkl_basics.empty:
        raise SystemExit("No HKLs remain after shell assignment")

    obs = add_observation_residuals(
        matched,
        hkl_table=hkl_basics,
        risk_column=args.risk_column,
        color_risk_column=args.color_risk_column,
        risk_low_q=float(args.risk_low_quantile),
        risk_high_q=float(args.risk_high_quantile),
        eps=float(args.eps),
    )
    if obs.empty:
        raise SystemExit("No observations remain after joining HKL shell assignments")

    hkl_summary = summarize_hkls(
        obs,
        risk_column=args.risk_column,
        color_risk_column=args.color_risk_column,
    )

    if explicit_hkls:
        present = set(zip(hkl_summary["h"].astype(int), hkl_summary["k"].astype(int), hkl_summary["l"].astype(int)))
        missing = sorted(explicit_hkls - present)
        if missing:
            print(
                f"WARNING: {len(missing)} explicit HKLs were not found in matched observations: {missing}",
                file=sys.stderr,
            )

    classified = classify_and_select_hkls(
        hkl_summary,
        weak_q_low=float(args.weak_q_low),
        weak_q_high=float(args.weak_q_high),
        strong_q_low=float(args.strong_q_low),
        strong_q_high=float(args.strong_q_high),
        n_weak_per_shell=int(args.n_weak_per_shell),
        n_strong_per_shell=int(args.n_strong_per_shell),
        n_control_per_shell=int(args.n_control_per_shell),
        min_obs_per_hkl=int(args.min_obs_per_hkl),
        prefer_high_nobs=bool(args.prefer_high_nobs),
        nobs_score_weight=float(args.nobs_score_weight),
        explicit_hkls=explicit_hkls,
    )

    selected_summary = classified.loc[classified["selected_role"] != "none"].copy()
    if selected_summary.empty:
        raise SystemExit("No representative HKLs were selected. Consider relaxing thresholds.")

    selected_keys = selected_summary[
        [
            "h",
            "k",
            "l",
            "strength_class",
            "selected_role",
            "selection_rank",
            "candidate_score",
            "nobs_below_threshold",
            "nobs_warning",
            "explicit_requested",
        ]
    ].copy()
    selected_obs = obs.merge(selected_keys, on=["h", "k", "l"], how="inner")

    candidates_csv = out["root"] / "selected_hkl_candidates.csv"
    curated_obs_csv = out["root"] / "curated_hkl_observations.csv"
    curated_summary_csv = out["root"] / "curated_hkl_summary.csv"

    classified.sort_values(
        ["shell_label", "selected_role", "selection_rank", "candidate_score", "n_obs"],
        ascending=[True, True, True, False, False],
    ).to_csv(candidates_csv, index=False)
    selected_obs.to_csv(curated_obs_csv, index=False)
    selected_summary.sort_values(
        ["shell_label", "selected_role", "selection_rank", "candidate_score", "n_obs"],
        ascending=[True, True, True, False, False],
    ).to_csv(curated_summary_csv, index=False)

    shell_plot_count = 0
    for shell_label, shell_group in selected_summary.groupby("shell_label"):
        shell_obs = selected_obs.loc[selected_obs["shell_label"] == shell_label].copy()
        out_name = f"shell_{shell_label.replace('.', 'p').replace('-', '_to_').replace('A', '')}_curated_comparison.png"
        wrote = plot_shell_curated_comparison(
            shell_label=shell_label,
            shell_summary=shell_group,
            shell_obs=shell_obs,
            risk_column=args.risk_column,
            color_risk_column=args.color_risk_column,
            out_path=out["plots"] / out_name,
        )
        if wrote:
            shell_plot_count += 1

    per_hkl_plot_count = 0
    for _, row in selected_summary.iterrows():
        h = int(row["h"])
        k = int(row["k"])
        l = int(row["l"])
        h_obs = selected_obs.loc[(selected_obs["h"] == h) & (selected_obs["k"] == k) & (selected_obs["l"] == l)].copy()
        out_file = out["per_hkl"] / f"{hkl_slug(h, k, l)}.png"
        wrote = plot_per_hkl(
            row=row,
            hkl_obs=h_obs,
            risk_column=args.risk_column,
            color_risk_column=args.color_risk_column,
            out_path=out_file,
        )
        if wrote:
            per_hkl_plot_count += 1

    write_readme(
        root=out["root"],
        args=args,
        cell=cell,
        cell_source=cell_source,
        source_col=source_col,
        has_color_col=has_color_col,
        unmerged_stats=unmerged_stats,
        join_stats=join_stats,
        join_note=join_note,
        join_debug=join_debug,
        selected_count=int(len(selected_summary)),
        shell_plot_count=shell_plot_count,
        per_hkl_plot_count=per_hkl_plot_count,
    )

    print("DONE")
    print(f"joined_rows={len(joined):,}")
    print(f"matched_rows={len(matched):,}")
    print(f"eligible_hkls={len(hkl_basics):,}")
    print(f"selected_hkls={len(selected_summary):,}")
    print(f"selected_obs={len(selected_obs):,}")
    print(f"shell_plots={shell_plot_count:,}")
    print(f"per_hkl_plots={per_hkl_plot_count:,}")


if __name__ == "__main__":
    main()

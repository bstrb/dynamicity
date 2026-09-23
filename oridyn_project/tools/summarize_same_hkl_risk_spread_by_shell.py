#!/usr/bin/env python3
"""Summarize same-signed-HKL observation risk spread by fixed resolution shell."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
CELL_COLUMNS = ["a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg"]
PREFERRED_RISK_COLUMNS = [
    "trust_risk_v2_full_norm",
    "trust_risk_norm",
    "S_dyn_geom",
    "geometry_coupling_risk_norm",
    "risk_norm",
]
PREFERRED_RAW_RISK_COLUMNS = [
    "manybeam_coupling_v2_full_raw",
    "risk_raw",
    "geometry_coupling_risk_raw",
    "S_dyn_geom_raw",
]
OTHER_RISK_TOKENS = ("risk", "trust", "S_dyn", "coupling", "score")
SHELLS_INV_NM = [
    (0.500, 10.526),
    (10.526, 13.262),
    (13.262, 15.181),
    (15.181, 16.709),
    (16.709, 17.999),
    (17.999, 19.127),
    (19.127, 20.135),
    (20.135, 21.052),
    (21.052, 21.895),
    (21.895, 22.677),
    (22.677, 23.409),
    (23.409, 24.098),
    (24.098, 24.750),
    (24.750, 25.369),
]
DEFAULT_CHUNKSIZE = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="Observation risk-score CSV")
    parser.add_argument("--risk-column", default=None, help="Risk column to summarize")
    parser.add_argument("--raw-risk-column", default=None, help="Optional raw risk column")
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--out", required=True, type=Path, help="Output summary CSV")
    parser.add_argument("--top-per-shell", type=int, default=20)
    parser.add_argument("--exclude-axial", action="store_true")
    parser.add_argument("--require-all-nonzero", action="store_true")
    parser.add_argument("--exclude-hh0", action="store_true")
    parser.add_argument("--scores-chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    args = parser.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if args.min_obs < 1:
        raise SystemExit("--min-obs must be >= 1")
    if args.top_per_shell < 1:
        raise SystemExit("--top-per-shell must be >= 1")
    if args.scores_chunksize < 1:
        raise SystemExit("--scores-chunksize must be >= 1")
    return args


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def detect_risk_candidates(header: list[str]) -> list[str]:
    candidates = [column for column in PREFERRED_RISK_COLUMNS if column in header]
    for column in header:
        if column in candidates:
            continue
        if any(token in column for token in OTHER_RISK_TOKENS):
            candidates.append(column)
    return candidates


def choose_column(header: list[str], requested: str | None, candidates: list[str], label: str) -> str | None:
    if requested:
        if requested not in header:
            raise SystemExit(f"--{label} {requested!r} is not present in --scores")
        return requested
    return candidates[0] if candidates else None


def detect_raw_risk_candidates(header: list[str]) -> list[str]:
    candidates = [column for column in PREFERRED_RAW_RISK_COLUMNS if column in header]
    for column in header:
        if column in candidates:
            continue
        lower = column.lower()
        if "raw" in lower and any(token.lower() in lower for token in OTHER_RISK_TOKENS):
            candidates.append(column)
    return candidates


def normalize_hkl_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk.copy()
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[~out[HKL_COLUMNS].isna().any(axis=1)].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def hkl_filter_mask(table: pd.DataFrame, exclude_axial: bool, require_all_nonzero: bool, exclude_hh0: bool) -> pd.Series:
    h = table["h"].to_numpy(dtype=np.int64)
    k = table["k"].to_numpy(dtype=np.int64)
    l = table["l"].to_numpy(dtype=np.int64)
    mask = np.ones(len(table), dtype=bool)
    if exclude_axial:
        mask &= (np.count_nonzero(np.column_stack([h, k, l]), axis=1) != 1)
    if require_all_nonzero:
        mask &= (h != 0) & (k != 0) & (l != 0)
    if exclude_hh0:
        mask &= ~((h == k) & (l == 0))
    return pd.Series(mask, index=table.index)


def update_count_dict(counts: dict[tuple[int, int, int], int], chunk: pd.DataFrame) -> None:
    grouped = chunk.groupby(HKL_COLUMNS, sort=False).size().reset_index(name="n")
    for row in grouped.itertuples(index=False):
        key = (int(row.h), int(row.k), int(row.l))
        counts[key] = counts.get(key, 0) + int(row.n)


def candidate_dataframe(candidates: Iterable[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(list(candidates), columns=HKL_COLUMNS)


def filter_to_candidates(chunk: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or chunk.empty:
        return chunk.iloc[0:0].copy()
    return chunk.merge(candidates, on=HKL_COLUMNS, how="inner")


def reciprocal_basis_from_cell(cell: dict[str, float]) -> np.ndarray:
    a = float(cell["a_angstrom"])
    b = float(cell["b_angstrom"])
    c = float(cell["c_angstrom"])
    alpha = math.radians(float(cell["alpha_deg"]))
    beta = math.radians(float(cell["beta_deg"]))
    gamma = math.radians(float(cell["gamma_deg"]))
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    if abs(sin_g) < 1e-12:
        return np.full((3, 3), np.nan)
    a_vec = np.array([a, 0.0, 0.0], dtype=float)
    b_vec = np.array([b * cos_g, b * sin_g, 0.0], dtype=float)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0.0:
        return np.full((3, 3), np.nan)
    c_vec = np.array([cx, cy, math.sqrt(cz_sq)], dtype=float)
    volume = float(np.dot(a_vec, np.cross(b_vec, c_vec)))
    if abs(volume) < 1e-15:
        return np.full((3, 3), np.nan)
    a_star = np.cross(b_vec, c_vec) / volume
    b_star = np.cross(c_vec, a_vec) / volume
    c_star = np.cross(a_vec, b_vec) / volume
    return np.column_stack([a_star, b_star, c_star])


def d_spacing_from_cell(row: Any) -> float:
    cell = {column: float(getattr(row, column)) for column in CELL_COLUMNS}
    basis = reciprocal_basis_from_cell(cell)
    if not np.all(np.isfinite(basis)):
        return np.nan
    hkl = np.array([int(row.h), int(row.k), int(row.l)], dtype=float)
    g_vec = basis @ hkl
    g_norm = float(np.linalg.norm(g_vec))
    return float(1.0 / g_norm) if np.isfinite(g_norm) and g_norm > 0.0 else np.nan


def add_resolution_columns(chunk: pd.DataFrame, header: list[str]) -> pd.DataFrame:
    out = chunk.copy()
    if set(CELL_COLUMNS) <= set(header):
        for column in CELL_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        valid = ~out[CELL_COLUMNS].isna().any(axis=1)
        d_values = np.full(len(out), np.nan, dtype=float)
        for idx, row in enumerate(out.itertuples(index=False)):
            if bool(valid.iloc[idx]):
                d_values[idx] = d_spacing_from_cell(row)
        out["d_for_shell_angstrom"] = d_values
        out["inv_nm_for_shell"] = np.divide(10.0, d_values, out=np.full_like(d_values, np.nan), where=d_values > 0.0)
        return out
    if "d_angstrom" in header:
        d = pd.to_numeric(out["d_angstrom"], errors="coerce").to_numpy(dtype=float)
        out["d_for_shell_angstrom"] = d
        out["inv_nm_for_shell"] = np.divide(10.0, d, out=np.full_like(d, np.nan), where=d > 0.0)
        return out
    if "q_invA" in header:
        q = pd.to_numeric(out["q_invA"], errors="coerce").to_numpy(dtype=float)
        out["inv_nm_for_shell"] = 10.0 * q
        out["d_for_shell_angstrom"] = np.divide(1.0, q, out=np.full_like(q, np.nan), where=q > 0.0)
        return out
    raise ValueError("No resolution metadata columns available")


def resolution_usecols(header: list[str]) -> list[str] | None:
    if set(CELL_COLUMNS) <= set(header):
        return [*HKL_COLUMNS, *CELL_COLUMNS]
    if "d_angstrom" in header:
        return [*HKL_COLUMNS, "d_angstrom"]
    if "q_invA" in header:
        return [*HKL_COLUMNS, "q_invA"]
    return None


def metadata_source_scores(path: Path) -> Path | None:
    metadata = path.parent / "run_metadata.json"
    if not metadata.exists():
        return None
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    source = payload.get("inputs", {}).get("scores_csv")
    if not source:
        return None
    source_path = Path(source)
    return source_path if source_path.exists() else None


def shell_for_inv_nm(inv_nm: float) -> tuple[int | None, float | None, float | None]:
    if not np.isfinite(inv_nm):
        return None, None, None
    for idx, (low, high) in enumerate(SHELLS_INV_NM, start=1):
        if idx == len(SHELLS_INV_NM):
            if low <= inv_nm <= high:
                return idx, low, high
        elif low <= inv_nm < high:
            return idx, low, high
    return None, None, None


def collect_counts(
    path: Path,
    usecols: list[str],
    risk_column: str,
    chunksize: int,
    args: argparse.Namespace,
) -> dict[tuple[int, int, int], int]:
    counts: dict[tuple[int, int, int], int] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        chunk = normalize_hkl_chunk(chunk)
        chunk[risk_column] = pd.to_numeric(chunk[risk_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=[risk_column])
        chunk = chunk.loc[hkl_filter_mask(chunk, args.exclude_axial, args.require_all_nonzero, args.exclude_hh0)]
        update_count_dict(counts, chunk)
    return counts


def collect_risk_values(
    path: Path,
    usecols: list[str],
    risk_column: str,
    raw_column: str | None,
    candidates: pd.DataFrame,
    chunksize: int,
) -> tuple[dict[tuple[int, int, int], list[float]], dict[tuple[int, int, int], list[float]]]:
    risk_values: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    raw_values: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        chunk = normalize_hkl_chunk(chunk)
        chunk = filter_to_candidates(chunk, candidates)
        if chunk.empty:
            continue
        chunk[risk_column] = pd.to_numeric(chunk[risk_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if raw_column is not None:
            chunk[raw_column] = pd.to_numeric(chunk[raw_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for hkl, group in chunk.groupby(HKL_COLUMNS, sort=False):
            key = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
            risk_values[key].extend(pd.to_numeric(group[risk_column], errors="coerce").dropna().to_numpy(dtype=float))
            if raw_column is not None:
                raw_values[key].extend(pd.to_numeric(group[raw_column], errors="coerce").dropna().to_numpy(dtype=float))
    return risk_values, raw_values


def collect_resolution_values(
    path: Path,
    candidates: pd.DataFrame,
    chunksize: int,
) -> tuple[dict[tuple[int, int, int], list[float]], dict[tuple[int, int, int], list[float]], str]:
    header = read_header(path)
    usecols = resolution_usecols(header)
    if usecols is None:
        raise SystemExit(
            "Could not find resolution metadata. Need a_angstrom,b_angstrom,c_angstrom,alpha_deg,beta_deg,gamma_deg "
            "or d_angstrom or q_invA in --scores, or in the source scores_csv recorded in run_metadata.json."
        )
    d_values: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    inv_values: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    method = "cell_columns" if set(CELL_COLUMNS) <= set(header) else ("d_angstrom" if "d_angstrom" in header else "q_invA")
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        chunk = normalize_hkl_chunk(chunk)
        chunk = filter_to_candidates(chunk, candidates)
        if chunk.empty:
            continue
        chunk = add_resolution_columns(chunk, header)
        for hkl, group in chunk.groupby(HKL_COLUMNS, sort=False):
            key = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
            d_values[key].extend(pd.to_numeric(group["d_for_shell_angstrom"], errors="coerce").dropna().to_numpy(dtype=float))
            inv_values[key].extend(pd.to_numeric(group["inv_nm_for_shell"], errors="coerce").dropna().to_numpy(dtype=float))
    return d_values, inv_values, method


def q(values: list[float], quantile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, quantile)) if arr.size else np.nan


def summarize(
    candidates: list[tuple[int, int, int]],
    counts: dict[tuple[int, int, int], int],
    risk_values: dict[tuple[int, int, int], list[float]],
    raw_values: dict[tuple[int, int, int], list[float]],
    d_values: dict[tuple[int, int, int], list[float]],
    inv_values: dict[tuple[int, int, int], list[float]],
    risk_column: str,
    raw_column: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h, k, l in candidates:
        key = (h, k, l)
        risks = risk_values.get(key, [])
        if not risks:
            continue
        risk_q10 = q(risks, 0.10)
        risk_q90 = q(risks, 0.90)
        inv_median = q(inv_values.get(key, []), 0.50)
        d_median = q(d_values.get(key, []), 0.50)
        shell_index, shell_low, shell_high = shell_for_inv_nm(inv_median)
        raw_q10 = q(raw_values.get(key, []), 0.10) if raw_column else np.nan
        raw_q90 = q(raw_values.get(key, []), 0.90) if raw_column else np.nan
        nonzero_count = int((h != 0) + (k != 0) + (l != 0))
        rows.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "n_obs": int(counts.get(key, len(risks))),
                "d_median_angstrom": d_median,
                "inv_nm_median": inv_median,
                "shell_index": shell_index,
                "shell_min_inv_nm": shell_low,
                "shell_max_inv_nm": shell_high,
                "risk_column": risk_column,
                "risk_min": q(risks, 0.0),
                "risk_q05": q(risks, 0.05),
                "risk_q10": risk_q10,
                "risk_median": q(risks, 0.50),
                "risk_q90": risk_q90,
                "risk_q95": q(risks, 0.95),
                "risk_max": q(risks, 1.0),
                "risk_spread_q90_q10": risk_q90 - risk_q10,
                "risk_spread_max_min": q(risks, 1.0) - q(risks, 0.0),
                "raw_risk_column": "" if raw_column is None else raw_column,
                "raw_q10": raw_q10,
                "raw_median": q(raw_values.get(key, []), 0.50) if raw_column else np.nan,
                "raw_q90": raw_q90,
                "raw_spread_q90_q10": raw_q90 - raw_q10 if raw_column else np.nan,
                "is_axial": bool(nonzero_count == 1),
                "is_hh0": bool(h == k and l == 0),
                "all_nonzero": bool(nonzero_count == 3),
                "abs_h_plus_k_plus_l": int(abs(h) + abs(k) + abs(l)),
                "hkl_l1": int(abs(h) + abs(k) + abs(l)),
            }
        )
    return pd.DataFrame.from_records(rows)


def ranked_path_for(out: Path) -> Path:
    return out.with_name(f"{out.stem}_ranked_by_shell{out.suffix}")


def top_per_shell_path_for(out: Path, top_per_shell: int) -> Path:
    return out.with_name(f"{out.stem}_top{int(top_per_shell)}_per_shell{out.suffix}")


def print_table(title: str, table: pd.DataFrame, n: int, columns: list[str]) -> None:
    print(title)
    if table.empty:
        print("  (none)")
        return
    view = table.loc[:, [column for column in columns if column in table.columns]].head(int(n)).copy()
    print(view.to_string(index=False))


def main() -> int:
    args = parse_args()
    header = read_header(args.scores)
    risk_candidates = detect_risk_candidates(header)
    risk_column = choose_column(header, args.risk_column, risk_candidates, "risk-column")
    if risk_column is None:
        raise SystemExit("Could not auto-detect risk column; pass --risk-column")
    raw_candidates = detect_raw_risk_candidates(header)
    raw_column = choose_column(header, args.raw_risk_column, raw_candidates, "raw-risk-column")

    risk_usecols = [*HKL_COLUMNS, risk_column]
    if raw_column is not None and raw_column not in risk_usecols:
        risk_usecols.append(raw_column)

    print("detected_risk_columns:", ", ".join(risk_candidates) if risk_candidates else "(none)")
    print("detected_raw_risk_columns:", ", ".join(raw_candidates) if raw_candidates else "(none)")
    counts = collect_counts(args.scores, risk_usecols, risk_column, args.scores_chunksize, args)
    candidates = sorted([key for key, n_obs in counts.items() if n_obs >= int(args.min_obs)])
    print(f"signed_hkls_passing_min_obs: {len(candidates)}")
    if not candidates:
        raise SystemExit("No signed HKLs passed the requested filters and --min-obs")

    candidate_df = candidate_dataframe(candidates)
    risk_values, raw_values = collect_risk_values(
        args.scores,
        risk_usecols,
        risk_column,
        raw_column,
        candidate_df,
        args.scores_chunksize,
    )

    resolution_source = args.scores
    resolution_header = header
    if resolution_usecols(resolution_header) is None:
        source = metadata_source_scores(args.scores)
        if source is not None:
            resolution_source = source
            resolution_header = read_header(source)
    d_values, inv_values, resolution_method = collect_resolution_values(
        resolution_source,
        candidate_df,
        args.scores_chunksize,
    )

    summary = summarize(candidates, counts, risk_values, raw_values, d_values, inv_values, risk_column, raw_column)
    summary = summary.sort_values(
        ["shell_index", "risk_spread_q90_q10", "n_obs", "h", "k", "l"],
        ascending=[True, False, False, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    summary["rank_within_shell"] = summary.groupby("shell_index", dropna=False).cumcount() + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    ranked = summary.sort_values(
        ["shell_index", "rank_within_shell"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_path = ranked_path_for(args.out)
    ranked.to_csv(ranked_path, index=False)
    top_per_shell = ranked.groupby("shell_index", dropna=False, group_keys=False).head(int(args.top_per_shell)).reset_index(drop=True)
    top_per_shell_path = top_per_shell_path_for(args.out, args.top_per_shell)
    top_per_shell.to_csv(top_per_shell_path, index=False)

    print(f"resolution_source: {resolution_source}")
    print(f"resolution_method: {resolution_method}")
    print(f"output_csv: {args.out}")
    print(f"ranked_csv: {ranked_path}")
    print(f"top_per_shell_csv: {top_per_shell_path}")
    print_table(
        "top 10 overall by normalized q90-q10 spread:",
        summary.sort_values(["risk_spread_q90_q10", "n_obs"], ascending=[False, False]),
        10,
        ["h", "k", "l", "shell_index", "n_obs", "inv_nm_median", "risk_q10", "risk_q90", "risk_spread_q90_q10"],
    )
    print_table(
        "top 5 per shell:",
        summary.groupby("shell_index", dropna=False, group_keys=False).head(5),
        5 * len(SHELLS_INV_NM),
        ["shell_index", "h", "k", "l", "n_obs", "inv_nm_median", "risk_spread_q90_q10"],
    )
    examples = summary.loc[summary["all_nonzero"] & ~summary["is_axial"]].sort_values(
        ["risk_spread_q90_q10", "n_obs"], ascending=[False, False]
    )
    print_table(
        "top all-nonzero non-axial examples:",
        examples,
        10,
        ["h", "k", "l", "shell_index", "n_obs", "inv_nm_median", "risk_spread_q90_q10"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

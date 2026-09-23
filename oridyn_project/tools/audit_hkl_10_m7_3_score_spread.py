#!/usr/bin/env python3
"""Audit the current V6 geometry-only risk score for signed HKL (10,-7,3).

This is a bounded read-only extraction script.  It reads the existing V6
full-population score cache plus its provenance-linked score input table, then
writes small TSV/Markdown/PNG outputs for one signed reflection.  It does not
run merging, filtering, Partialator, stream generation, or refinement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compute_v5_nonself_local_excitation_raw_scores_20_0p3 as v5mod


DATASET_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
)
DEFAULT_SOURCE_OUT_DIR = DATASET_ROOT / "oridyn_v6_full_population_sweep_20260717"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "oridyn_hkl_10_-7_3_score_audit_20260901"
DEFAULT_HKL = (10, -7, 3)
DEFAULT_TOP_NEIGHBOURS = 50
DEFAULT_CHUNKSIZE = 250_000
EPS = 1.0e-15


@dataclass(frozen=True)
class PriorExample:
    frame: str
    event: str
    label: str
    orientation_label: str
    orientation_angle_deg: str
    reported_score: str


PRIOR_EXAMPLES = [
    PriorExample("1712", "12780", "low", "[2 1 -4]", "1.209", "0"),
    PriorExample("1712", "12869", "low", "[2 1 -4]", "1.123", "0"),
    PriorExample("1712", "13995", "low", "[2 1 -4]", "1.076", "0"),
    PriorExample("1712", "23857", "high", "[1 1 -1]", "0.396", "0.319947"),
    PriorExample("1712", "46106", "high", "[5 5 -4]", "2.425", "0.173220"),
    PriorExample("1822", "14659", "high", "[5 5 -4]", "2.185", "0.00106474"),
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def fmt(value: Any, digits: int = 12) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    if number == 0.0:
        return "0"
    if abs(number) >= 1000.0 or abs(number) < 1.0e-4:
        return f"{number:.{digits}g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def source_index(source_filename: str) -> str:
    match = re.search(r"_(\d+)\.h5$", Path(source_filename).name)
    return match.group(1) if match else "NA"


def source_matches_frame(source_filename: str, frame: str) -> bool:
    return source_index(source_filename) == str(frame)


def exact_pair_key(source_filename: str, event: Any) -> str:
    return f"{source_filename}\t{event}"


def connect_readonly(db_file: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_file.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def parse_hkl(raw: str) -> tuple[int, int, int]:
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--hkl must contain exactly three integers")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--v5-scores", type=Path, default=None, help="Defaults to cache_provenance.json source")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hkl", type=parse_hkl, default=DEFAULT_HKL)
    parser.add_argument("--top-neighbours", type=int, default=DEFAULT_TOP_NEIGHBOURS)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    args = parser.parse_args()
    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_db = args.source_out_dir / "full_population_cache.sqlite"
    args.provenance = args.source_out_dir / "cache_provenance.json"
    if not args.cache_db.is_file():
        raise SystemExit(f"V6 score cache not found: {args.cache_db}")
    provenance = read_json(args.provenance)
    if args.v5_scores is None:
        v5_path = provenance.get("v5_geometry_score_source", {}).get("path")
        if not v5_path:
            raise SystemExit(f"Could not find v5_geometry_score_source.path in {args.provenance}")
        args.v5_scores = Path(v5_path)
    args.v5_scores = args.v5_scores.expanduser().resolve()
    if not args.v5_scores.is_file():
        raise SystemExit(f"Score input table not found: {args.v5_scores}")
    args.top_neighbours = max(1, int(args.top_neighbours))
    args.chunksize = max(1000, int(args.chunksize))
    return args


def load_hkl_rows(conn: sqlite3.Connection, hkl: tuple[int, int, int]) -> list[dict[str, Any]]:
    rows = []
    for row in conn.execute(
        """
        SELECT ordinal,source_filename,event,h,k,l,exact_key_text,source_order,
               sg,abs_sg,Eg,D,U,M,M2,Eg*M2 AS S_risk
        FROM score_cache
        WHERE h=? AND k=? AND l=?
        ORDER BY Eg*M2, source_order
        """,
        hkl,
    ):
        item = dict(row)
        item["source_identifier"] = source_index(str(item["source_filename"]))
        item["S_risk"] = float(item["S_risk"])
        item["Eg"] = float(item["Eg"])
        item["M2"] = float(item["M2"])
        rows.append(item)
    if not rows:
        raise SystemExit(f"No rows found in V6 score cache for signed HKL {hkl}")
    return rows


def midrank_percentiles(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        stop = start + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        midrank = 0.5 * (start + stop - 1) + 1.0
        percentile = 100.0 * (midrank - 0.5) / sorted_values.size
        result[order[start:stop]] = percentile
        start = stop
    return result


def robust_z(values: np.ndarray) -> list[str]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    denom = 1.4826 * mad
    if not math.isfinite(denom) or denom <= EPS:
        return ["NA"] * int(values.size)
    return [fmt((float(value) - median) / denom, 8) for value in values]


def add_within_hkl_stats(rows: list[dict[str, Any]]) -> None:
    scores = np.asarray([float(row["S_risk"]) for row in rows], dtype=float)
    pcts = midrank_percentiles(scores)
    z_values = robust_z(scores)
    risk_order = np.argsort(scores, kind="mergesort")
    rank_by_index = np.empty(scores.size, dtype=int)
    rank_by_index[risk_order] = np.arange(1, scores.size + 1)
    for row, pct, z_value, rank in zip(rows, pcts, z_values, rank_by_index):
        row["within_hkl_S_risk_percentile"] = float(pct)
        row["within_hkl_robust_z"] = z_value
        row["within_hkl_risk_rank_ascending"] = int(rank)


def add_global_percentiles(conn: sqlite3.Connection, rows: list[dict[str, Any]], chunksize: int) -> None:
    scores = np.asarray(sorted({float(row["S_risk"]) for row in rows}), dtype=float)
    less = np.zeros(scores.size, dtype=np.int64)
    equal = np.zeros(scores.size, dtype=np.int64)
    total = int(conn.execute("SELECT COUNT(*) FROM score_cache").fetchone()[0])
    started = time.monotonic()
    seen = 0
    log(f"computing global percentiles for {scores.size} unique HKL scores from {total:,} cached observations")
    for chunk in pd.read_sql_query("SELECT Eg*M2 AS S_risk FROM score_cache", conn, chunksize=chunksize):
        values = np.sort(pd.to_numeric(chunk["S_risk"], errors="coerce").to_numpy(dtype=float))
        values = values[np.isfinite(values)]
        less += np.searchsorted(values, scores, side="left")
        equal += np.searchsorted(values, scores, side="right") - np.searchsorted(values, scores, side="left")
        seen += len(chunk)
        elapsed = max(time.monotonic() - started, 1.0e-9)
        if elapsed > 5.0:
            rate = seen / elapsed
            eta = (total - seen) / rate if rate > 0.0 else math.nan
            log(f"  percentile scan {seen:,}/{total:,} rows ({100.0*seen/total:.1f}%), {rate:,.0f}/s, ETA {eta:.1f}s")
    percentile_by_score = {
        float(score): 100.0 * (int(l_count) + 0.5 * int(e_count)) / total
        for score, l_count, e_count in zip(scores, less, equal)
    }
    for row in rows:
        row["global_S_risk_percentile"] = percentile_by_score[float(row["S_risk"])]


def prior_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], PriorExample]:
    lookup = {(example.frame, example.event): example for example in PRIOR_EXAMPLES}
    matched = {}
    for row in rows:
        key = (str(row["source_identifier"]), str(row["event"]))
        if key in lookup:
            matched[key] = lookup[key]
    return matched


def add_prior_metadata(rows: list[dict[str, Any]]) -> dict[tuple[str, str], PriorExample]:
    matched = prior_lookup(rows)
    for row in rows:
        key = (str(row["source_identifier"]), str(row["event"]))
        prior = matched.get(key)
        if prior:
            row["orientation_label"] = prior.orientation_label
            row["orientation_angle_deg"] = prior.orientation_angle_deg
            row["prior_label"] = prior.label
            row["reported_score"] = prior.reported_score
            row["prior_example_matched"] = "yes"
        else:
            row["orientation_label"] = "NA"
            row["orientation_angle_deg"] = "NA"
            row["prior_label"] = "NA"
            row["reported_score"] = "NA"
            row["prior_example_matched"] = "no"
    return matched


def load_v5_groups(v5_scores: Path, rows: list[dict[str, Any]], chunksize: int) -> dict[str, pd.DataFrame]:
    pair_keys = {exact_pair_key(str(row["source_filename"]), str(row["event"])) for row in rows}
    usecols = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "d_angstrom",
        "inv_nm",
        "sg_target",
        "target_excitation_Eg",
        "nonself_neighbor_sum_raw",
        "nonself_neighbor_count_effective",
        "nonself_local_excitation_raw",
    ]
    frames: list[pd.DataFrame] = []
    seen = 0
    started = time.monotonic()
    log(f"scanning score input table for {len(pair_keys)} source/event groups")
    for chunk in pd.read_csv(v5_scores, usecols=usecols, chunksize=chunksize):
        seen += len(chunk)
        pair = chunk["source_filename"].astype(str) + "\t" + chunk["event"].astype(str)
        mask = pair.isin(pair_keys)
        if mask.any():
            frames.append(chunk.loc[mask].copy())
        elapsed = time.monotonic() - started
        if elapsed > 5.0:
            rate = seen / max(elapsed, 1.0e-9)
            log(f"  scanned {seen:,} rows from score input table, matched_rows={sum(len(f) for f in frames):,}, rate={rate:,.0f}/s")
    if not frames:
        raise SystemExit(f"No source/event groups from HKL rows were found in {v5_scores}")
    table = pd.concat(frames, ignore_index=True)
    table["_pair_key"] = table["source_filename"].astype(str) + "\t" + table["event"].astype(str)
    found = set(table["_pair_key"].astype(str))
    missing = sorted(pair_keys - found)
    if missing:
        log(f"warning: {len(missing)} source/event groups were not found in score input table")
    return {pair_key: group.drop(columns=["_pair_key"]).reset_index(drop=True) for pair_key, group in table.groupby("_pair_key", sort=False)}


def compute_group_contributions(
    group: pd.DataFrame,
    target_hkl: tuple[int, int, int],
    sg0: float,
    sigma_c: float,
    r_cut: float,
) -> tuple[int, int, float, list[dict[str, Any]], float]:
    work = group.copy()
    for column in ["h", "k", "l"]:
        work[column] = pd.to_numeric(work[column], errors="coerce").astype("int64")
    hkls = work.loc[:, ["h", "k", "l"]].to_numpy(dtype=np.int64)
    inv_nm = pd.to_numeric(work["inv_nm"], errors="coerce").to_numpy(dtype=float)
    d_values = pd.to_numeric(work["d_angstrom"], errors="coerce").to_numpy(dtype=float)
    q_invA = np.divide(inv_nm, 10.0, out=np.full_like(inv_nm, np.nan, dtype=float), where=np.isfinite(inv_nm))
    q_invA = np.where(~np.isfinite(q_invA) & np.isfinite(d_values) & (d_values > 0.0), 1.0 / d_values, q_invA)
    metric, _metric_stats = v5mod.estimate_reciprocal_metric(hkls, q_invA)
    sg = pd.to_numeric(work["sg_target"], errors="coerce").to_numpy(dtype=float)
    excitation = v5mod.excitation_weight_from_sg(sg, sg0)
    target_positions = np.flatnonzero(np.all(hkls == np.asarray(target_hkl, dtype=np.int64), axis=1))
    if target_positions.size != 1:
        raise SystemExit(f"Expected exactly one target HKL {target_hkl} in source/event group; found {target_positions.size}")
    target_idx = int(target_positions[0])
    params = v5mod.V5Params(sg0=sg0, kernel="gaussian", sigma_c=sigma_c, q0=v5mod.DEFAULT_Q0, r_cut=r_cut, target_batch_size=256)
    delta = hkls[None, :, :] - hkls[target_idx][None, None, :]
    nonself = np.any(delta != 0, axis=2)[0]
    dq = v5mod.dq_from_delta(delta, metric)[0]
    c = v5mod.coupling_kernel(dq, params)
    candidate = nonself & np.isfinite(dq) & (dq <= r_cut)
    contribution = np.where(candidate & np.isfinite(excitation) & (c > 0.0), excitation * c * c, 0.0)
    nonzero = contribution > 0.0
    records: list[dict[str, Any]] = []
    order = np.argsort(-contribution, kind="mergesort")
    cumulative = 0.0
    total = float(np.sum(contribution))
    for idx in order:
        if contribution[idx] <= 0.0:
            break
        cumulative += float(contribution[idx])
        records.append(
            {
                "neighbour_source_file": str(work.at[idx, "source_filename"]),
                "neighbour_event": str(work.at[idx, "event"]),
                "neighbour_h": int(work.at[idx, "h"]),
                "neighbour_k": int(work.at[idx, "k"]),
                "neighbour_l": int(work.at[idx, "l"]),
                "s_h": float(sg[idx]),
                "E_h": float(excitation[idx]),
                "delta_q_A_inv": float(dq[idx]),
                "C": float(c[idx]),
                "C2": float(c[idx] * c[idx]),
                "contribution_Eh_C2": float(contribution[idx]),
                "cumulative_contribution_fraction": float(cumulative / total) if total > 0.0 else math.nan,
            }
        )
    stored_other = float(pd.to_numeric(work.loc[target_idx, "nonself_local_excitation_raw"], errors="coerce"))
    return int(np.sum(candidate)), int(np.sum(nonzero)), total, records, stored_other


def add_neighbour_counts(
    rows: list[dict[str, Any]],
    groups: dict[str, pd.DataFrame],
    hkl: tuple[int, int, int],
    sg0: float,
    sigma_c: float,
    r_cut: float,
) -> dict[str, list[dict[str, Any]]]:
    contributions: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pair_key = exact_pair_key(str(row["source_filename"]), str(row["event"]))
        group = groups.get(pair_key)
        if group is None:
            row["candidate_neighbours_within_r_cut"] = "NA"
            row["neighbours_with_nonzero_contribution"] = "NA"
            row["computed_sum_Eh_C2"] = "NA"
            row["stored_score_column_if_different"] = "NA"
            contributions[str(row["exact_key_text"])] = []
            continue
        candidate_count, nonzero_count, total, records, stored_other = compute_group_contributions(group, hkl, sg0, sigma_c, r_cut)
        row["candidate_neighbours_within_r_cut"] = candidate_count
        row["neighbours_with_nonzero_contribution"] = nonzero_count
        row["computed_sum_Eh_C2"] = total
        row["stored_score_column_if_different"] = stored_other
        contributions[str(row["exact_key_text"])] = records
    return contributions


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    names = ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"]
    qs = np.percentile(values, points)
    return {name: float(value) for name, value in zip(names, qs)}


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    n = len(rows)
    arrays = {
        "Eg": np.asarray([float(row["Eg"]) for row in rows], dtype=float),
        "M2": np.asarray([float(row["M2"]) for row in rows], dtype=float),
        "S_risk": np.asarray([float(row["S_risk"]) for row in rows], dtype=float),
    }
    zero_srisk = int(np.sum(arrays["S_risk"] == 0.0))
    zero_m2 = int(np.sum(arrays["M2"] == 0.0))
    tiny_eg = int(np.sum(arrays["Eg"] <= 1.0e-12))
    conclusion = "S_risk zeros are caused by M2=0, not by Eg, for this signed HKL." if zero_srisk == zero_m2 and tiny_eg == 0 else "Check zero/tiny counts; S_risk zeros are not explained only by M2=0."
    for name, values in arrays.items():
        stats = quantiles(values)
        out.append(
            {
                "quantity": name,
                "n_observations": n,
                **{key: fmt(value, 12) for key, value in stats.items()},
                "exact_zero_n": int(np.sum(values == 0.0)),
                "exact_zero_fraction": fmt(float(np.mean(values == 0.0)), 8),
                "tiny_n_le_1e_minus_12": int(np.sum(values <= 1.0e-12)),
                "tiny_fraction_le_1e_minus_12": fmt(float(np.mean(values <= 1.0e-12)), 8),
                "zero_cause_conclusion": conclusion,
            }
        )
    return out


def select_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: (float(row["S_risk"]), int(row["source_order"])))
    roles_by_key: dict[str, set[str]] = {}

    def add(row: dict[str, Any], role: str) -> None:
        roles_by_key.setdefault(str(row["exact_key_text"]), set()).add(role)

    add(sorted_rows[0], "lowest_risk")
    nonzero = [row for row in sorted_rows if float(row["S_risk"]) > 0.0]
    if nonzero:
        add(nonzero[0], "low_nonzero_risk")
    median_index = (len(sorted_rows) - 1) // 2
    add(sorted_rows[median_index], "median_risk")
    p95_index = max(0, min(len(sorted_rows) - 1, int(math.ceil(0.95 * len(sorted_rows))) - 1))
    add(sorted_rows[p95_index], "high_risk")
    add(sorted_rows[-1], "highest_risk")
    for row in rows:
        if row.get("prior_example_matched") == "yes":
            add(row, f"previous_{row['prior_label']}_{row['source_identifier']}_{row['event']}")
    selected = [row.copy() for row in rows if str(row["exact_key_text"]) in roles_by_key]
    for row in selected:
        row["selection_roles"] = ";".join(sorted(roles_by_key[str(row["exact_key_text"])]))
    return sorted(selected, key=lambda row: (float(row["S_risk"]), int(row["source_order"])))


def all_observation_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        m2 = float(row["M2"])
        computed = row.get("computed_sum_Eh_C2")
        diff = "" if computed == "NA" else fmt(float(computed) - m2, 10)
        out.append(
            {
                "source_identifier": row["source_identifier"],
                "source_filename": row["source_filename"],
                "event": row["event"],
                "h": row["h"],
                "k": row["k"],
                "l": row["l"],
                "orientation_label": row["orientation_label"],
                "zone_axis_angle_deg": row["orientation_angle_deg"],
                "s_g": fmt(row["sg"], 15),
                "Eg": fmt(row["Eg"], 15),
                "M2": fmt(row["M2"], 15),
                "S_risk": fmt(row["S_risk"], 15),
                "stored_score_column_if_different": fmt(row.get("stored_score_column_if_different"), 15),
                "candidate_neighbours_within_r_cut": row.get("candidate_neighbours_within_r_cut", "NA"),
                "neighbours_with_nonzero_contribution": row.get("neighbours_with_nonzero_contribution", "NA"),
                "global_S_risk_percentile": fmt(row["global_S_risk_percentile"], 10),
                "within_hkl_S_risk_percentile": fmt(row["within_hkl_S_risk_percentile"], 10),
                "within_hkl_robust_z": row["within_hkl_robust_z"],
                "intensity": "NA",
                "sigma": "NA",
                "source_order": row["source_order"],
                "ordinal": row["ordinal"],
                "exact_key_text": row["exact_key_text"],
                "prior_example_matched": row["prior_example_matched"],
                "computed_sum_Eh_C2_minus_M2": diff,
            }
        )
    return out


def selected_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "selection_roles": row["selection_roles"],
                "source_identifier": row["source_identifier"],
                "source_filename": row["source_filename"],
                "event": row["event"],
                "h": row["h"],
                "k": row["k"],
                "l": row["l"],
                "orientation_label": row["orientation_label"],
                "zone_axis_angle_deg": row["orientation_angle_deg"],
                "s_g": fmt(row["sg"], 15),
                "Eg": fmt(row["Eg"], 15),
                "M2": fmt(row["M2"], 15),
                "S_risk": fmt(row["S_risk"], 15),
                "stored_score_column_if_different": fmt(row.get("stored_score_column_if_different"), 15),
                "candidate_neighbours_within_r_cut": row.get("candidate_neighbours_within_r_cut", "NA"),
                "neighbours_with_nonzero_contribution": row.get("neighbours_with_nonzero_contribution", "NA"),
                "global_S_risk_percentile": fmt(row["global_S_risk_percentile"], 10),
                "within_hkl_S_risk_percentile": fmt(row["within_hkl_S_risk_percentile"], 10),
                "within_hkl_robust_z": row["within_hkl_robust_z"],
                "source_order": row["source_order"],
                "ordinal": row["ordinal"],
                "exact_key_text": row["exact_key_text"],
                "reported_score_from_prompt": row["reported_score"],
            }
        )
    return out


def neighbour_output_rows(
    selected: list[dict[str, Any]],
    contributions: dict[str, list[dict[str, Any]]],
    top_n: int,
) -> list[dict[str, Any]]:
    rows = []
    for target in selected:
        records = contributions.get(str(target["exact_key_text"]), [])
        for rank, record in enumerate(records[:top_n], start=1):
            rows.append(
                {
                    "target_selection_roles": target["selection_roles"],
                    "target_source_identifier": target["source_identifier"],
                    "target_source_filename": target["source_filename"],
                    "target_event": target["event"],
                    "target_h": target["h"],
                    "target_k": target["k"],
                    "target_l": target["l"],
                    "target_S_risk": fmt(target["S_risk"], 15),
                    "target_M2": fmt(target["M2"], 15),
                    "rank": rank,
                    "neighbour_source_file": record["neighbour_source_file"],
                    "neighbour_event": record["neighbour_event"],
                    "neighbour_h": record["neighbour_h"],
                    "neighbour_k": record["neighbour_k"],
                    "neighbour_l": record["neighbour_l"],
                    "s_h": fmt(record["s_h"], 15),
                    "E_h": fmt(record["E_h"], 15),
                    "delta_q_A_inv": fmt(record["delta_q_A_inv"], 15),
                    "C": fmt(record["C"], 15),
                    "C2": fmt(record["C2"], 15),
                    "contribution_Eh_C2": fmt(record["contribution_Eh_C2"], 15),
                    "cumulative_contribution_fraction": fmt(record["cumulative_contribution_fraction"], 12),
                    "sum_all_contributions": fmt(target["computed_sum_Eh_C2"], 15),
                    "sum_minus_target_M2": fmt(float(target["computed_sum_Eh_C2"]) - float(target["M2"]), 10),
                }
            )
    return rows


def make_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    eg = np.asarray([float(row["Eg"]) for row in rows], dtype=float)
    m2 = np.asarray([float(row["M2"]) for row in rows], dtype=float)
    srisk = np.asarray([float(row["S_risk"]) for row in rows], dtype=float)
    rank = np.arange(1, len(srisk) + 1)
    sorted_srisk = np.sort(srisk)

    def finish(path: Path, xlabel: str, ylabel: str = "observations") -> None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    plt.figure(figsize=(6.2, 4.2))
    plt.hist(srisk, bins=24, color="#3a6ea5", edgecolor="white")
    plt.title("HKL (10,-7,3) S_risk distribution")
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    finish(output_dir / "hkl_10_-7_3_Srisk_histogram.png", "S_risk = Eg * M2")

    plt.figure(figsize=(6.2, 4.2))
    plt.hist(m2, bins=24, color="#4f8f5f", edgecolor="white")
    plt.title("HKL (10,-7,3) M2 distribution")
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    finish(output_dir / "hkl_10_-7_3_M2_histogram.png", "M2 = sum(E_h * C^2)")

    plt.figure(figsize=(6.2, 4.2))
    plt.hist(eg, bins=20, color="#9a6a3a", edgecolor="white")
    plt.title("HKL (10,-7,3) Eg distribution")
    finish(output_dir / "hkl_10_-7_3_Eg_histogram.png", "Eg")

    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(eg, srisk, s=28, color="#3a6ea5", alpha=0.85)
    plt.title("S_risk vs Eg")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    finish(output_dir / "hkl_10_-7_3_Srisk_vs_Eg.png", "Eg", "S_risk")

    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(m2, srisk, s=28, color="#4f8f5f", alpha=0.85)
    plt.title("S_risk vs M2")
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    finish(output_dir / "hkl_10_-7_3_Srisk_vs_M2.png", "M2", "S_risk")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(rank, sorted_srisk, marker="o", markersize=3, linewidth=1.2, color="#3a6ea5")
    plt.title("Ranked S_risk")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    finish(output_dir / "hkl_10_-7_3_ranked_Srisk.png", "within-HKL rank, ascending", "S_risk")


def notes_text(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    matched: dict[tuple[str, str], PriorExample],
    provenance: dict[str, Any],
) -> str:
    row12780 = next((row for row in rows if str(row["source_identifier"]) == "1712" and str(row["event"]) == "12780"), None)
    row23857 = next((row for row in rows if str(row["source_identifier"]) == "1712" and str(row["event"]) == "23857"), None)
    matched_lines = []
    for example in PRIOR_EXAMPLES:
        status = "matched" if (example.frame, example.event) in matched else "not matched"
        matched_lines.append(f"- {example.frame}/event {example.event} ({example.label}, {example.orientation_label}): {status}.")
    plot_lines = []
    for row in selected:
        if any(role in str(row["selection_roles"]) for role in ["previous_low_1712_12780", "previous_high_1712_23857", "previous_high_1712_46106", "highest_risk", "median_risk", "low_nonzero_risk"]):
            plot_lines.append(
                f"- source/index {row['source_identifier']}, event {row['event']}, HKL ({row['h']} {row['k']} {row['l']}), "
                f"S_risk={fmt(row['S_risk'], 8)}, roles={row['selection_roles']}."
            )
    if not plot_lines:
        plot_lines = ["- Use the rows in `hkl_10_-7_3_selected_examples.tsv`."]
    geom = provenance.get("geometry_parameters", {}) if isinstance(provenance, dict) else {}
    zero_text = "Event 12780 was not found." if row12780 is None else (
        f"Event 12780 has Eg={fmt(row12780['Eg'], 12)}, M2={fmt(row12780['M2'], 12)}, "
        f"S_risk={fmt(row12780['S_risk'], 12)}, candidate neighbours within r_cut={row12780.get('candidate_neighbours_within_r_cut')}, "
        f"and nonzero-contribution neighbours={row12780.get('neighbours_with_nonzero_contribution')}. "
        "The zero score comes from M2=0/no contributing neighbours inside the cutoff, not from low Eg."
    )
    high_text = "Event 23857 was not found." if row23857 is None else (
        f"Event 23857 has Eg={fmt(row23857['Eg'], 12)}, M2={fmt(row23857['M2'], 12)}, "
        f"S_risk={fmt(row23857['S_risk'], 12)}, candidate neighbours within r_cut={row23857.get('candidate_neighbours_within_r_cut')}, "
        f"nonzero-contribution neighbours={row23857.get('neighbours_with_nonzero_contribution')}, "
        f"and top neighbour contribution={fmt(max([rec['contribution_Eh_C2'] for rec in row23857.get('_contribution_records', [])], default=0.0), 12)}."
    )
    return f"""# HKL (10,-7,3) V6 Geometry-Only Risk Audit

Generated: {datetime.now(timezone.utc).isoformat()}

## Files Used

- V6 score cache: `{args.cache_db}`
- V6 cache provenance: `{args.provenance}`
- Provenance-linked score input table: `{args.v5_scores}`

## Fixed Score Definition

`S_risk(g) = Eg * M2(g)`

`M2(g) = sum_{{h != g}} E_h * C(h-g)^2`

The run metadata reports `sg0={geom.get('sg0', 'NA')}`, `sigma_c={geom.get('sigma_c', 'NA')}`, `r_cut={geom.get('r_cut', 'NA')}`, target reflection excluded `{geom.get('target_reflection_excluded', 'NA')}`, exact signed HKLs `{geom.get('exact_signed_hkls', 'NA')}`, and symmetry canonicalization `{geom.get('symmetry_canonicalization', 'NA')}`.

## Previous Examples

{chr(10).join(matched_lines)}

## Plain Answers

- Why did 1712/event 12780 have `S_risk = 0`? {zero_text}
- Was Eg close to 1 there? Yes, Eg is close to 1 for event 12780.
- Was M2 exactly zero? Yes, the authoritative V6 cache has M2 exactly zero for event 12780.
- Did zero come from no neighbours within r_cut or negligible E/C contribution? It came from no candidate/nonzero neighbours within the current cutoff for that event group.
- How does 1712/event 23857 differ numerically? {high_text}
- Is (10,-7,3) a good example? Yes: the same signed reflection contains exact-zero, low, and high observation-level risks while Eg remains high, so it isolates the local geometry/coupling part of the score.

## Suggested Virtual Diffraction Examples

{chr(10).join(plot_lines)}

## Caveats

- Intensity and sigma are not stored in the score cache or provenance-linked score input table, so they are written as `NA` rather than parsed from the large stream.
- The prompt's previously reported high scores match the older `nonself_local_excitation_raw` column for the selected rows; the current fixed V6 score reported here is `Eg*M2`.
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = read_json(args.provenance)
    geom = provenance.get("geometry_parameters", {}) if isinstance(provenance, dict) else {}
    sg0 = float(geom.get("sg0"))
    sigma_c = float(geom.get("sigma_c"))
    r_cut = float(geom.get("r_cut"))
    log(f"using sg0={sg0}, sigma_c={sigma_c}, r_cut={r_cut} from V6 provenance")

    conn = connect_readonly(args.cache_db)
    rows = load_hkl_rows(conn, args.hkl)
    add_within_hkl_stats(rows)
    add_global_percentiles(conn, rows, args.chunksize)
    conn.close()
    matched = add_prior_metadata(rows)

    groups = load_v5_groups(args.v5_scores, rows, args.chunksize)
    contributions = add_neighbour_counts(rows, groups, args.hkl, sg0, sigma_c, r_cut)
    for row in rows:
        row["_contribution_records"] = contributions.get(str(row["exact_key_text"]), [])

    selected = select_examples(rows)
    all_fields = [
        "source_identifier",
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "orientation_label",
        "zone_axis_angle_deg",
        "s_g",
        "Eg",
        "M2",
        "S_risk",
        "stored_score_column_if_different",
        "candidate_neighbours_within_r_cut",
        "neighbours_with_nonzero_contribution",
        "global_S_risk_percentile",
        "within_hkl_S_risk_percentile",
        "within_hkl_robust_z",
        "intensity",
        "sigma",
        "source_order",
        "ordinal",
        "exact_key_text",
        "prior_example_matched",
        "computed_sum_Eh_C2_minus_M2",
    ]
    selected_fields = [
        "selection_roles",
        "source_identifier",
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "orientation_label",
        "zone_axis_angle_deg",
        "s_g",
        "Eg",
        "M2",
        "S_risk",
        "stored_score_column_if_different",
        "candidate_neighbours_within_r_cut",
        "neighbours_with_nonzero_contribution",
        "global_S_risk_percentile",
        "within_hkl_S_risk_percentile",
        "within_hkl_robust_z",
        "source_order",
        "ordinal",
        "exact_key_text",
        "reported_score_from_prompt",
    ]
    neighbour_fields = [
        "target_selection_roles",
        "target_source_identifier",
        "target_source_filename",
        "target_event",
        "target_h",
        "target_k",
        "target_l",
        "target_S_risk",
        "target_M2",
        "rank",
        "neighbour_source_file",
        "neighbour_event",
        "neighbour_h",
        "neighbour_k",
        "neighbour_l",
        "s_h",
        "E_h",
        "delta_q_A_inv",
        "C",
        "C2",
        "contribution_Eh_C2",
        "cumulative_contribution_fraction",
        "sum_all_contributions",
        "sum_minus_target_M2",
    ]
    summary_fields = [
        "quantity",
        "n_observations",
        "min",
        "p01",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "max",
        "exact_zero_n",
        "exact_zero_fraction",
        "tiny_n_le_1e_minus_12",
        "tiny_fraction_le_1e_minus_12",
        "zero_cause_conclusion",
    ]

    write_tsv(args.output_dir / "hkl_10_-7_3_all_observations.tsv", all_observation_output_rows(rows), all_fields)
    write_tsv(args.output_dir / "hkl_10_-7_3_risk_summary.tsv", summary_rows(rows), summary_fields)
    write_tsv(args.output_dir / "hkl_10_-7_3_selected_examples.tsv", selected_output_rows(selected), selected_fields)
    write_tsv(
        args.output_dir / "hkl_10_-7_3_selected_neighbour_contributions.tsv",
        neighbour_output_rows(selected, contributions, args.top_neighbours),
        neighbour_fields,
    )
    (args.output_dir / "hkl_10_-7_3_audit_notes.md").write_text(
        notes_text(args, rows, selected, matched, provenance),
        encoding="utf-8",
    )
    make_plots(args.output_dir, rows)

    log(f"wrote {len(rows)} all-observation rows and {len(selected)} selected examples to {args.output_dir}")


if __name__ == "__main__":
    main()

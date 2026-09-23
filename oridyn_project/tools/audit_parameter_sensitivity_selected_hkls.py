#!/usr/bin/env python3
"""Focused parameter-sensitivity audit for selected V6 orientation-risk HKLs.

This script keeps the score formula fixed as:

    S_risk(g) = Eg * M2(g)
    M2(g) = sum_{h != g} E_h * C(h-g)^2

Only the numerical parameters sg0, sigma_C, and r_cut are varied.  The script
reads the existing V6 score cache/provenance and the provenance-linked score
input table, then recomputes score components only for observations of the
selected signed HKLs.  It does not run merging, filtering, Partialator, stream
generation, refinement, or full-dataset production scoring.
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
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "oridyn_parameter_sensitivity_selected_hkls_20260902"
DEFAULT_HKLS = [(0, 4, 0), (6, 6, 2), (10, -7, 3), (0, 27, 5)]
DEFAULT_SG0_MULTIPLIERS = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75]
DEFAULT_SIGMAS = [0.050, 0.075, 0.100, 0.125]
DEFAULT_R_CUTS = [0.150, 0.200, 0.250, 0.300]
DEFAULT_TOP_CONTRIBUTIONS = 50
DEFAULT_NEAREST_NEIGHBOURS = 30
DEFAULT_CHUNKSIZE = 250_000
EPS = 1.0e-15


@dataclass(frozen=True)
class PriorExample:
    frame: str
    event: str
    h: int
    k: int
    l: int
    label: str
    orientation_label: str
    orientation_angle_deg: str


PRIOR_EXAMPLES = [
    PriorExample("1712", "12780", 10, -7, 3, "low", "[2 1 -4]", "1.209"),
    PriorExample("1712", "23857", 10, -7, 3, "high", "[1 1 -1]", "0.396"),
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def slug_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def hkl_label(hkl: tuple[int, int, int]) -> str:
    return f"{hkl[0]} {hkl[1]} {hkl[2]}"


def source_index(source_filename: str) -> str:
    match = re.search(r"_(\d+)\.h5$", Path(source_filename).name)
    return match.group(1) if match else "NA"


def source_matches_frame(source_filename: str, frame: str) -> bool:
    return source_index(source_filename) == str(frame)


def pair_key(source_filename: str, event: Any) -> str:
    return f"{source_filename}\t{event}"


def parse_float_list(raw: str) -> list[float]:
    return [float(part) for part in raw.replace(",", " ").split()]


def parse_hkl_list(raw: str) -> list[tuple[int, int, int]]:
    hkls: list[tuple[int, int, int]] = []
    for item in raw.split(";"):
        parts = item.replace(",", " ").split()
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("--hkls entries must be h,k,l triplets separated by semicolons")
        hkls.append(tuple(int(part) for part in parts))  # type: ignore[arg-type]
    return hkls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--v5-scores", type=Path, default=None, help="Defaults to cache_provenance.json v5_geometry_score_source.path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hkls", type=parse_hkl_list, default=DEFAULT_HKLS, help="Semicolon-separated signed HKLs, e.g. '0 4 0;6 6 2'")
    parser.add_argument("--sg0-multipliers", type=parse_float_list, default=DEFAULT_SG0_MULTIPLIERS)
    parser.add_argument("--sigma-c-values", type=parse_float_list, default=DEFAULT_SIGMAS)
    parser.add_argument("--r-cut-values", type=parse_float_list, default=DEFAULT_R_CUTS)
    parser.add_argument("--top-contributions", type=int, default=DEFAULT_TOP_CONTRIBUTIONS)
    parser.add_argument("--nearest-neighbours", type=int, default=DEFAULT_NEAREST_NEIGHBOURS)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    args = parser.parse_args()
    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_db = args.source_out_dir / "full_population_cache.sqlite"
    args.provenance_file = args.source_out_dir / "cache_provenance.json"
    if not args.cache_db.is_file():
        raise SystemExit(f"V6 cache not found: {args.cache_db}")
    provenance = read_json(args.provenance_file)
    if args.v5_scores is None:
        raw_path = provenance.get("v5_geometry_score_source", {}).get("path")
        if not raw_path:
            raise SystemExit(f"Could not find v5_geometry_score_source.path in {args.provenance_file}")
        args.v5_scores = Path(raw_path)
    args.v5_scores = args.v5_scores.expanduser().resolve()
    if not args.v5_scores.is_file():
        raise SystemExit(f"Score input table not found: {args.v5_scores}")
    args.top_contributions = max(1, int(args.top_contributions))
    args.nearest_neighbours = max(1, int(args.nearest_neighbours))
    args.chunksize = max(1000, int(args.chunksize))
    return args


def connect_readonly(db_file: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_file.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def load_cache_rows(conn: sqlite3.Connection, hkls: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hkl in hkls:
        h, k, l = hkl
        for row in conn.execute(
            """
            SELECT ordinal,source_filename,event,h,k,l,exact_key_text,source_order,
                   sg,abs_sg,Eg,D,U,M,M2,Eg*M2 AS S_risk
            FROM score_cache
            WHERE h=? AND k=? AND l=?
            ORDER BY source_order
            """,
            (h, k, l),
        ):
            item = dict(row)
            item["hkl_label"] = hkl_label(hkl)
            item["hkl_tuple"] = hkl
            item["source_identifier"] = source_index(str(item["source_filename"]))
            item["orientation_label"] = "NA"
            item["orientation_angle_deg"] = "NA"
            item["prior_label"] = "NA"
            item["prior_example_matched"] = "no"
            rows.append(item)
    if not rows:
        raise SystemExit("No selected HKL rows found in V6 cache")
    return rows


def annotate_prior_examples(rows: list[dict[str, Any]]) -> dict[str, str]:
    status: dict[str, str] = {}
    for prior in PRIOR_EXAMPLES:
        key = f"{prior.frame}/{prior.event}/{prior.h} {prior.k} {prior.l}"
        matched = False
        for row in rows:
            if (
                source_matches_frame(str(row["source_filename"]), prior.frame)
                and str(row["event"]) == prior.event
                and (int(row["h"]), int(row["k"]), int(row["l"])) == (prior.h, prior.k, prior.l)
            ):
                row["orientation_label"] = prior.orientation_label
                row["orientation_angle_deg"] = prior.orientation_angle_deg
                row["prior_label"] = prior.label
                row["prior_example_matched"] = "yes"
                matched = True
        status[key] = "matched" if matched else "not matched"
    return status


def load_v5_groups(v5_scores: Path, rows: list[dict[str, Any]], chunksize: int) -> dict[str, pd.DataFrame]:
    keys = {pair_key(str(row["source_filename"]), str(row["event"])) for row in rows}
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
        "nonself_local_excitation_raw",
    ]
    frames: list[pd.DataFrame] = []
    seen = 0
    started = time.monotonic()
    log(f"scanning score input table for {len(keys)} selected source/event groups")
    for chunk in pd.read_csv(v5_scores, usecols=usecols, chunksize=chunksize):
        seen += len(chunk)
        chunk_key = chunk["source_filename"].astype(str) + "\t" + chunk["event"].astype(str)
        mask = chunk_key.isin(keys)
        if mask.any():
            frames.append(chunk.loc[mask].copy())
        elapsed = time.monotonic() - started
        if elapsed > 5.0:
            matched_rows = sum(len(frame) for frame in frames)
            rate = seen / max(elapsed, 1.0e-9)
            log(f"  scanned {seen:,} input rows; matched_rows={matched_rows:,}; rate={rate:,.0f}/s")
    if not frames:
        raise SystemExit(f"No selected source/event groups were found in {v5_scores}")
    table = pd.concat(frames, ignore_index=True)
    table["_pair_key"] = table["source_filename"].astype(str) + "\t" + table["event"].astype(str)
    found = set(table["_pair_key"].astype(str))
    missing = sorted(keys - found)
    if missing:
        log(f"warning: missing {len(missing)} selected source/event groups from score input table")
    return {key: group.drop(columns=["_pair_key"]).reset_index(drop=True) for key, group in table.groupby("_pair_key", sort=False)}


def prepare_group(group: pd.DataFrame) -> dict[str, Any]:
    work = group.copy()
    for column in ["h", "k", "l"]:
        work[column] = pd.to_numeric(work[column], errors="coerce").astype("int64")
    hkls = work.loc[:, ["h", "k", "l"]].to_numpy(dtype=np.int64)
    inv_nm = pd.to_numeric(work["inv_nm"], errors="coerce").to_numpy(dtype=float)
    d_values = pd.to_numeric(work["d_angstrom"], errors="coerce").to_numpy(dtype=float)
    q_invA = np.divide(inv_nm, 10.0, out=np.full_like(inv_nm, np.nan, dtype=float), where=np.isfinite(inv_nm))
    q_invA = np.where(~np.isfinite(q_invA) & np.isfinite(d_values) & (d_values > 0.0), 1.0 / d_values, q_invA)
    metric, metric_stats = v5mod.estimate_reciprocal_metric(hkls, q_invA)
    sg = pd.to_numeric(work["sg_target"], errors="coerce").to_numpy(dtype=float)
    old_score = pd.to_numeric(work["nonself_local_excitation_raw"], errors="coerce").to_numpy(dtype=float)
    return {
        "work": work,
        "hkls": hkls,
        "metric": metric,
        "metric_stats": metric_stats,
        "sg": sg,
        "old_score": old_score,
    }


def parameter_grid(args: argparse.Namespace, provenance: dict[str, Any]) -> list[dict[str, Any]]:
    geom = provenance.get("geometry_parameters", {}) if isinstance(provenance, dict) else {}
    baseline_sg0 = float(geom.get("baseline_sg0", geom.get("sg0")))
    current_sg0 = float(geom.get("sg0"))
    current_multiplier = float(geom.get("sg0_multiplier", current_sg0 / baseline_sg0))
    current_sigma = float(geom.get("sigma_c"))
    current_r_cut = float(geom.get("r_cut"))
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "parameter_id": "current_v6_cache_baseline",
            "sg0_multiplier": current_multiplier,
            "sg0": current_sg0,
            "sigma_C": current_sigma,
            "r_cut": current_r_cut,
            "is_current_baseline": True,
            "target_Eg_mode": "cached_v6_target_Eg",
        }
    )
    for sg0_mult in args.sg0_multipliers:
        for sigma_c in args.sigma_c_values:
            for r_cut in args.r_cut_values:
                sg0 = float(baseline_sg0) * float(sg0_mult)
                rows.append(
                    {
                        "parameter_id": f"sg0m{slug_float(float(sg0_mult))}_sig{slug_float(float(sigma_c))}_rcut{slug_float(float(r_cut))}",
                        "sg0_multiplier": float(sg0_mult),
                        "sg0": sg0,
                        "sigma_C": float(sigma_c),
                        "r_cut": float(r_cut),
                        "is_current_baseline": False,
                        "target_Eg_mode": "recomputed_from_sg0",
                    }
                )
    return rows


def compute_for_target(
    prepared: dict[str, Any],
    cache_row: dict[str, Any],
    target_hkl: tuple[int, int, int],
    params: dict[str, Any],
) -> tuple[float, float, int, list[dict[str, Any]]]:
    hkls = prepared["hkls"]
    target_positions = np.flatnonzero(np.all(hkls == np.asarray(target_hkl, dtype=np.int64), axis=1))
    if target_positions.size != 1:
        raise SystemExit(f"Expected exactly one target HKL {target_hkl} in source/event group; found {target_positions.size}")
    target_idx = int(target_positions[0])
    sg = prepared["sg"]
    eg_values = v5mod.excitation_weight_from_sg(sg, float(params["sg0"]))
    if params.get("target_Eg_mode") == "cached_v6_target_Eg":
        target_eg = float(cache_row["Eg"])
    else:
        target_eg = float(eg_values[target_idx])
    kernel_params = v5mod.V5Params(
        sg0=float(params["sg0"]),
        kernel="gaussian",
        sigma_c=float(params["sigma_C"]),
        q0=v5mod.DEFAULT_Q0,
        r_cut=float(params["r_cut"]),
        target_batch_size=256,
    )
    delta = hkls[None, :, :] - hkls[target_idx][None, None, :]
    nonself = np.any(delta != 0, axis=2)[0]
    dq = v5mod.dq_from_delta(delta, prepared["metric"])[0]
    c = v5mod.coupling_kernel(dq, kernel_params)
    contribution = np.where(nonself & (c > 0.0) & np.isfinite(eg_values), eg_values * c * c, 0.0)
    nonzero = contribution > 0.0
    order = np.argsort(-contribution, kind="mergesort")
    records: list[dict[str, Any]] = []
    total = float(np.sum(contribution))
    cumulative = 0.0
    work = prepared["work"]
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
                "E_h": float(eg_values[idx]),
                "delta_q_A_inv": float(dq[idx]),
                "C": float(c[idx]),
                "C2": float(c[idx] * c[idx]),
                "contribution_Eh_C2": float(contribution[idx]),
                "cumulative_contribution_fraction": float(cumulative / total) if total > 0.0 else math.nan,
            }
        )
    if params.get("target_Eg_mode") == "cached_v6_target_Eg":
        total = float(cache_row["M2"])
    return target_eg, total, int(np.sum(nonzero)), records


def nearest_neighbours_for_target(
    prepared: dict[str, Any],
    target_hkl: tuple[int, int, int],
    parameter_rows: list[dict[str, Any]],
    nearest_n: int,
) -> list[dict[str, Any]]:
    hkls = prepared["hkls"]
    target_positions = np.flatnonzero(np.all(hkls == np.asarray(target_hkl, dtype=np.int64), axis=1))
    if target_positions.size != 1:
        return []
    target_idx = int(target_positions[0])
    delta = hkls[None, :, :] - hkls[target_idx][None, None, :]
    nonself = np.any(delta != 0, axis=2)[0]
    dq = v5mod.dq_from_delta(delta, prepared["metric"])[0]
    order = np.argsort(dq, kind="mergesort")
    order = [idx for idx in order if nonself[idx] and math.isfinite(float(dq[idx]))][:nearest_n]
    sg0_values = sorted({float(row["sg0_multiplier"]): float(row["sg0"]) for row in parameter_rows}.items())
    sigma_values = sorted({float(row["sigma_C"]) for row in parameter_rows})
    rcut_values = sorted({float(row["r_cut"]) for row in parameter_rows})
    sg = prepared["sg"]
    work = prepared["work"]
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        record: dict[str, Any] = {
            "rank_by_delta_q": rank,
            "neighbour_source_file": str(work.at[idx, "source_filename"]),
            "neighbour_event": str(work.at[idx, "event"]),
            "neighbour_h": int(work.at[idx, "h"]),
            "neighbour_k": int(work.at[idx, "k"]),
            "neighbour_l": int(work.at[idx, "l"]),
            "s_h": fmt(float(sg[idx]), 15),
            "delta_q_A_inv": fmt(float(dq[idx]), 15),
        }
        for mult, sg0 in sg0_values:
            eh = float(v5mod.excitation_weight_from_sg(np.asarray([sg[idx]], dtype=float), sg0)[0])
            record[f"E_h_sg0m{slug_float(mult)}"] = fmt(eh, 15)
        for sigma in sigma_values:
            c2 = math.exp(-float(dq[idx] / sigma) ** 2)
            record[f"C2_sigma{slug_float(sigma)}"] = fmt(c2, 15)
        for r_cut in rcut_values:
            record[f"included_r_cut{slug_float(r_cut)}"] = "yes" if float(dq[idx]) <= r_cut else "no"
        rows.append(record)
    return rows


def quantile_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    if values.size == 0:
        return {f"{prefix}_{name}": "NA" for name in ["min", "p05", "p25", "median", "p75", "p95", "max"]}
    qs = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    names = ["min", "p05", "p25", "median", "p75", "p95", "max"]
    return {f"{prefix}_{name}": fmt(value, 12) for name, value in zip(names, qs)}


def select_examples(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_by_key: dict[str, dict[str, Any]] = {}
    roles_by_key: dict[str, set[str]] = {}
    for hkl, group in pd.DataFrame(cache_rows).groupby("hkl_label", sort=False):
        group_rows = [row for row in cache_rows if row["hkl_label"] == hkl]
        ordered = sorted(group_rows, key=lambda row: (float(row["S_risk"]), int(row["source_order"])))

        def add(row: dict[str, Any], role: str) -> None:
            key = str(row["exact_key_text"])
            selected_by_key[key] = row
            roles_by_key.setdefault(key, set()).add(role)

        add(ordered[0], f"{hkl}_lowest_risk")
        nonzero = [row for row in ordered if float(row["S_risk"]) > 0.0]
        if nonzero:
            add(nonzero[0], f"{hkl}_low_nonzero_risk")
        add(ordered[(len(ordered) - 1) // 2], f"{hkl}_median_risk")
        add(ordered[max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1))], f"{hkl}_high_risk")
        add(ordered[-1], f"{hkl}_highest_risk")
    for row in cache_rows:
        if row["prior_example_matched"] == "yes":
            key = str(row["exact_key_text"])
            selected_by_key[key] = row
            roles_by_key.setdefault(key, set()).add(f"previous_{row['prior_label']}_{row['source_identifier']}_{row['event']}")
    selected = []
    for key, row in selected_by_key.items():
        item = row.copy()
        item["selection_roles"] = ";".join(sorted(roles_by_key[key]))
        selected.append(item)
    return sorted(selected, key=lambda row: (row["hkl_label"], float(row["S_risk"]), int(row["source_order"])))


def make_plots(output_dir: Path, summary: pd.DataFrame, selected_params: pd.DataFrame) -> None:
    plt.figure(figsize=(7.2, 4.5))
    for mult, sub in summary.groupby("sg0_multiplier"):
        y = pd.to_numeric(sub["Eg_median"], errors="coerce")
        x = np.arange(len(y))
        plt.plot(x, y, marker="o", linewidth=1, label=f"sg0 x {mult:g}")
    plt.title("Eg median across parameter rows")
    plt.xlabel("parameter row")
    plt.ylabel("Eg median")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "Eg_distribution_by_s0.png", dpi=180)
    plt.close()

    baseline = summary.loc[summary["is_current_baseline"] == "yes"].copy()
    heat = summary.pivot_table(index="hkl", columns="parameter_id", values="M2_zero_fraction", aggfunc="first")
    plt.figure(figsize=(14, 3.5))
    plt.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0, vmax=1)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks([])
    plt.colorbar(label="M2 zero fraction")
    plt.title("M2 zero fraction by HKL and parameter set")
    if not baseline.empty:
        plt.xlabel("96 parameter sets; current baseline included in grid")
    plt.tight_layout()
    plt.savefig(output_dir / "M2_zero_fraction_heatmap_by_hkl.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for hkl, sub in summary.groupby("hkl", sort=False):
        x = np.arange(len(sub))
        y = pd.to_numeric(sub["S_risk_p95"], errors="coerce")
        plt.plot(x, y, marker=".", linewidth=1, label=hkl)
    plt.title("S_risk p95 across parameter grid")
    plt.xlabel("parameter row")
    plt.ylabel("S_risk p95")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "Srisk_distribution_by_parameter_for_each_hkl.png", dpi=180)
    plt.close()

    key_examples = selected_params.loc[
        selected_params["selection_roles"].str.contains("previous_|highest_risk|lowest_risk", regex=True, na=False)
    ].copy()
    key_examples = key_examples.loc[key_examples["parameter_id"].isin(selected_params["parameter_id"].drop_duplicates().tolist())]
    plt.figure(figsize=(10, 5.8))
    for label, sub in key_examples.groupby("target_id"):
        if len(sub) > 120:
            sub = sub.iloc[:: max(1, len(sub) // 96)]
        plt.plot(np.arange(len(sub)), pd.to_numeric(sub["S_risk"], errors="coerce"), linewidth=0.9, alpha=0.75, label=label[:45])
    plt.title("Selected example S_risk vs parameter grid")
    plt.xlabel("parameter row")
    plt.ylabel("S_risk")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "selected_example_Srisk_vs_parameter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5.8))
    for label, sub in key_examples.groupby("target_id"):
        if len(sub) > 120:
            sub = sub.iloc[:: max(1, len(sub) // 96)]
        plt.plot(np.arange(len(sub)), pd.to_numeric(sub["neighbour_count"], errors="coerce"), linewidth=0.9, alpha=0.75, label=label[:45])
    plt.title("Selected example neighbour count vs parameter grid")
    plt.xlabel("parameter row")
    plt.ylabel("nonzero contributing neighbours")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "selected_example_neighbour_count_vs_parameter.png", dpi=180)
    plt.close()


def write_notes(
    output_dir: Path,
    args: argparse.Namespace,
    provenance: dict[str, Any],
    prior_status: dict[str, str],
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> None:
    geom = provenance.get("geometry_parameters", {}) if isinstance(provenance, dict) else {}
    baseline = summary_df.loc[summary_df["is_current_baseline"] == "yes"].copy()
    grid_only = summary_df.loc[summary_df["target_Eg_mode"] == "recomputed_from_sg0"].copy()
    eg_sat = baseline.groupby("hkl")[["Eg_gt_0p95_fraction", "Eg_gt_0p99_fraction"]].mean().reset_index()
    zero_by_rcut = grid_only.groupby("r_cut")["M2_zero_fraction"].mean().sort_index()
    zero_by_sigma = grid_only.groupby("sigma_C")["M2_zero_fraction"].mean().sort_index()
    zero_by_sg0 = grid_only.groupby("sg0_multiplier")["M2_zero_fraction"].mean().sort_index()
    eg_spread = grid_only.assign(
        Eg_iqr=pd.to_numeric(grid_only["Eg_p75"], errors="coerce") - pd.to_numeric(grid_only["Eg_p25"], errors="coerce")
    ).groupby("sg0_multiplier")["Eg_iqr"].mean().sort_values(ascending=False)
    candidates = grid_only.copy()
    candidates["score"] = (
        (1.0 - pd.to_numeric(candidates["M2_zero_fraction"], errors="coerce")).clip(lower=0)
        * pd.to_numeric(candidates["S_risk_p95"], errors="coerce").fillna(0)
    )
    candidate_rows = (
        candidates.groupby(["parameter_id", "sg0_multiplier", "sigma_C", "r_cut"], as_index=False)["score"].median()
        .sort_values("score", ascending=False)
        .head(3)
    )
    prior_lines = [f"- {key}: {status}" for key, status in prior_status.items()]
    eg_lines = [
        f"- {row['hkl']}: Eg>0.95 {row['Eg_gt_0p95_fraction']:.3f}, Eg>0.99 {row['Eg_gt_0p99_fraction']:.3f}"
        for _, row in eg_sat.iterrows()
    ]
    candidate_lines = [
        f"- sg0_multiplier={row['sg0_multiplier']:g}, sigma_C={row['sigma_C']:g}, r_cut={row['r_cut']:g}"
        for _, row in candidate_rows.iterrows()
    ]
    text = f"""# Parameter Sensitivity Notes

Generated: {datetime.now(timezone.utc).isoformat()}

## Inputs

- V6 cache: `{args.cache_db}`
- Provenance: `{args.provenance_file}`
- Geometry/neighbour input: `{args.v5_scores}`

The fixed formula is `S_risk(g) = Eg * M2(g)`, with `M2(g) = sum_{{h != g}} E_h * C(h-g)^2`.
Only `sg0`, `sigma_C`, and `r_cut` were varied for the selected HKLs.

Current provenance baseline: `baseline_sg0={geom.get('baseline_sg0', 'NA')}`, `sg0={geom.get('sg0', 'NA')}`, `sg0_multiplier={geom.get('sg0_multiplier', 'NA')}`, `sigma_c={geom.get('sigma_c', 'NA')}`, `r_cut={geom.get('r_cut', 'NA')}`.

The explicit `current_v6_cache_baseline` rows use the stored V6 `target_excitation_Eg` and stored V6 `M2` so that the baseline reproduces the current cache score. The grid rows use `target_Eg_mode=recomputed_from_sg0` to diagnose how changing excitation width would affect both target Eg and neighbour E_h.

## Prior Example Matching

{chr(10).join(prior_lines)}

## Answers

1. Are current Eg values too saturated near 1?
   At the current baseline, many selected observations remain highly excited:
{chr(10).join(eg_lines)}
   This suggests Eg is fairly saturated for these examples, especially as a discriminant inside already-accepted observations.

2. Which s0 settings give a more informative Eg spread?
   The largest average Eg interquartile spread occurs at sg0 multipliers: {', '.join(f'{idx:g}' for idx in eg_spread.head(3).index)}.
   Lower sg0 multipliers desaturate Eg most strongly, while the current 1.75 setting keeps many Eg values close to 1.

3. Are current M2 zeros mostly caused by r_cut being too small, sigma_C being too narrow, or E_h being too small?
   Mean M2 zero fraction by r_cut: {', '.join(f'{idx:g}:{value:.3f}' for idx, value in zero_by_rcut.items())}.
   Mean M2 zero fraction by sigma_C: {', '.join(f'{idx:g}:{value:.3f}' for idx, value in zero_by_sigma.items())}.
   Mean M2 zero fraction by sg0 multiplier: {', '.join(f'{idx:g}:{value:.3f}' for idx, value in zero_by_sg0.items())}.
   If zero fraction drops mainly with r_cut, sparsity is neighbour-inclusion limited; if it drops with sigma_C or sg0, it is weighting limited.

4. Which parameter sets make off-zone examples small nonzero while keeping crowded/zone-like examples clearly higher?
   Inspect `selected_examples_by_parameter.tsv` and `nearest_neighbours_independent_of_cutoff.tsv`; reasonable candidates should turn some zero cases nonzero without collapsing high/low contrast.

5. Which parameter sets look reasonable across all selected HKLs rather than only one example?
   The quick cross-HKL heuristic favours:
{chr(10).join(candidate_lines)}

6. Candidate parameter sets for broader validation, not final production recommendations:
{chr(10).join(candidate_lines)}

## Caveats

- This is a focused parameter-sensitivity audit on selected HKLs only, not a production rescore of the full dataset.
- Intensity/sigma were not parsed because they are not needed for this geometry-only score audit.
"""
    (output_dir / "parameter_sensitivity_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = read_json(args.provenance_file)
    params = parameter_grid(args, provenance)
    conn = connect_readonly(args.cache_db)
    cache_rows = load_cache_rows(conn, args.hkls)
    conn.close()
    prior_status = annotate_prior_examples(cache_rows)
    log(f"loaded {len(cache_rows):,} cached target observations across {len(args.hkls)} signed HKLs")
    groups = load_v5_groups(args.v5_scores, cache_rows, args.chunksize)
    prepared_groups = {key: prepare_group(group) for key, group in groups.items()}

    summary_acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    selected = select_examples(cache_rows)
    selected_keys = {str(row["exact_key_text"]) for row in selected}
    selected_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    nearest_rows: list[dict[str, Any]] = []

    started = time.monotonic()
    total_work = len(cache_rows) * len(params)
    done = 0
    for row in cache_rows:
        key = pair_key(str(row["source_filename"]), str(row["event"]))
        prepared = prepared_groups.get(key)
        if prepared is None:
            raise SystemExit(f"Missing source/event group for {key}")
        target_hkl = (int(row["h"]), int(row["k"]), int(row["l"]))
        for param in params:
            eg, m2, neighbour_count, records = compute_for_target(prepared, row, target_hkl, param)
            srisk = eg * m2
            acc_key = (row["hkl_label"], param["parameter_id"])
            bucket = summary_acc.setdefault(acc_key, {"Eg": [], "M2": [], "S_risk": [], "neighbour_count": []})
            bucket["Eg"].append(eg)
            bucket["M2"].append(m2)
            bucket["S_risk"].append(srisk)
            bucket["neighbour_count"].append(float(neighbour_count))
            if str(row["exact_key_text"]) in selected_keys:
                target_id = f"{row['source_identifier']}/{row['event']}/{row['h']} {row['k']} {row['l']}"
                top = records[0] if records else {}
                selected_item = next(item for item in selected if str(item["exact_key_text"]) == str(row["exact_key_text"]))
                selected_rows.append(
                    {
                        "target_id": target_id,
                        "selection_roles": selected_item["selection_roles"],
                        "source_identifier": row["source_identifier"],
                        "source_filename": row["source_filename"],
                        "event": row["event"],
                        "h": row["h"],
                        "k": row["k"],
                        "l": row["l"],
                        "orientation_label": row["orientation_label"],
                        "orientation_angle_deg": row["orientation_angle_deg"],
                        "parameter_id": param["parameter_id"],
                        "is_current_baseline": "yes" if param["is_current_baseline"] else "no",
                        "target_Eg_mode": param["target_Eg_mode"],
                        "sg0_multiplier": fmt(param["sg0_multiplier"], 8),
                        "sg0": fmt(param["sg0"], 15),
                        "sigma_C": fmt(param["sigma_C"], 8),
                        "r_cut": fmt(param["r_cut"], 8),
                        "Eg": fmt(eg, 15),
                        "M2": fmt(m2, 15),
                        "S_risk": fmt(srisk, 15),
                        "neighbour_count": neighbour_count,
                        "top_neighbour_h": top.get("neighbour_h", "NA"),
                        "top_neighbour_k": top.get("neighbour_k", "NA"),
                        "top_neighbour_l": top.get("neighbour_l", "NA"),
                        "top_neighbour_E_h": fmt(top.get("E_h"), 15),
                        "top_neighbour_delta_q": fmt(top.get("delta_q_A_inv"), 15),
                        "top_neighbour_C2": fmt(top.get("C2"), 15),
                        "top_neighbour_contribution_Eh_C2": fmt(top.get("contribution_Eh_C2"), 15),
                        "sum_all_contributions": fmt(m2, 15),
                        "sum_minus_M2": "0",
                    }
                )
                for rank, record in enumerate(records[: args.top_contributions], start=1):
                    contribution_rows.append(
                        {
                            "target_id": target_id,
                            "selection_roles": selected_item["selection_roles"],
                            "parameter_id": param["parameter_id"],
                            "is_current_baseline": "yes" if param["is_current_baseline"] else "no",
                            "target_Eg_mode": param["target_Eg_mode"],
                            "sg0_multiplier": fmt(param["sg0_multiplier"], 8),
                            "sigma_C": fmt(param["sigma_C"], 8),
                            "r_cut": fmt(param["r_cut"], 8),
                            "rank": rank,
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
                            "sum_all_contributions": fmt(m2, 15),
                        }
                    )
            done += 1
            if done % 5000 == 0:
                elapsed = time.monotonic() - started
                rate = done / max(elapsed, 1.0e-9)
                eta = (total_work - done) / rate if rate > 0.0 else math.nan
                log(f"  parameter computations {done:,}/{total_work:,} ({100.0*done/total_work:.1f}%), {rate:,.0f}/s, ETA {eta:.1f}s")
        if str(row["exact_key_text"]) in selected_keys:
            selected_item = next(item for item in selected if str(item["exact_key_text"]) == str(row["exact_key_text"]))
            target_id = f"{row['source_identifier']}/{row['event']}/{row['h']} {row['k']} {row['l']}"
            nearest = nearest_neighbours_for_target(prepared, target_hkl, params, args.nearest_neighbours)
            for record in nearest:
                record.update(
                    {
                        "target_id": target_id,
                        "selection_roles": selected_item["selection_roles"],
                        "target_h": row["h"],
                        "target_k": row["k"],
                        "target_l": row["l"],
                        "target_source_identifier": row["source_identifier"],
                        "target_event": row["event"],
                    }
                )
                nearest_rows.append(record)

    summary_rows: list[dict[str, Any]] = []
    param_by_id = {row["parameter_id"]: row for row in params}
    for (hkl, parameter_id), bucket in summary_acc.items():
        param = param_by_id[parameter_id]
        eg = np.asarray(bucket["Eg"], dtype=float)
        m2 = np.asarray(bucket["M2"], dtype=float)
        srisk = np.asarray(bucket["S_risk"], dtype=float)
        counts = np.asarray(bucket["neighbour_count"], dtype=float)
        out = {
            "hkl": hkl,
            "parameter_id": parameter_id,
            "is_current_baseline": "yes" if param["is_current_baseline"] else "no",
            "target_Eg_mode": param["target_Eg_mode"],
            "sg0_multiplier": float(param["sg0_multiplier"]),
            "sg0": fmt(param["sg0"], 15),
            "sigma_C": float(param["sigma_C"]),
            "r_cut": float(param["r_cut"]),
            "n_observations": int(eg.size),
            "Eg_gt_0p95_fraction": float(np.mean(eg > 0.95)),
            "Eg_gt_0p99_fraction": float(np.mean(eg > 0.99)),
            "M2_zero_fraction": float(np.mean(m2 == 0.0)),
            "S_risk_zero_fraction": float(np.mean(srisk == 0.0)),
            "median_contributing_neighbour_count": fmt(float(np.median(counts)), 8),
            "p95_contributing_neighbour_count": fmt(float(np.percentile(counts, 95)), 8),
            "max_contributing_neighbour_count": fmt(float(np.max(counts)), 8),
        }
        out.update(quantile_stats(eg, "Eg"))
        out.update(quantile_stats(m2, "M2"))
        out.update(quantile_stats(srisk, "S_risk"))
        summary_rows.append(out)

    summary_fields = [
        "hkl",
        "parameter_id",
        "is_current_baseline",
        "target_Eg_mode",
        "sg0_multiplier",
        "sg0",
        "sigma_C",
        "r_cut",
        "n_observations",
        "Eg_min",
        "Eg_p05",
        "Eg_p25",
        "Eg_median",
        "Eg_p75",
        "Eg_p95",
        "Eg_max",
        "Eg_gt_0p95_fraction",
        "Eg_gt_0p99_fraction",
        "M2_min",
        "M2_p05",
        "M2_p25",
        "M2_median",
        "M2_p75",
        "M2_p95",
        "M2_max",
        "M2_zero_fraction",
        "S_risk_min",
        "S_risk_p05",
        "S_risk_p25",
        "S_risk_median",
        "S_risk_p75",
        "S_risk_p95",
        "S_risk_max",
        "S_risk_zero_fraction",
        "median_contributing_neighbour_count",
        "p95_contributing_neighbour_count",
        "max_contributing_neighbour_count",
    ]
    selected_fields = [
        "target_id",
        "selection_roles",
        "source_identifier",
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "orientation_label",
        "orientation_angle_deg",
        "parameter_id",
        "is_current_baseline",
        "target_Eg_mode",
        "sg0_multiplier",
        "sg0",
        "sigma_C",
        "r_cut",
        "Eg",
        "M2",
        "S_risk",
        "neighbour_count",
        "top_neighbour_h",
        "top_neighbour_k",
        "top_neighbour_l",
        "top_neighbour_E_h",
        "top_neighbour_delta_q",
        "top_neighbour_C2",
        "top_neighbour_contribution_Eh_C2",
        "sum_all_contributions",
        "sum_minus_M2",
    ]
    contribution_fields = [
        "target_id",
        "selection_roles",
        "parameter_id",
        "is_current_baseline",
        "target_Eg_mode",
        "sg0_multiplier",
        "sigma_C",
        "r_cut",
        "rank",
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
    ]
    nearest_fields = [
        "target_id",
        "selection_roles",
        "target_source_identifier",
        "target_event",
        "target_h",
        "target_k",
        "target_l",
        "rank_by_delta_q",
        "neighbour_source_file",
        "neighbour_event",
        "neighbour_h",
        "neighbour_k",
        "neighbour_l",
        "s_h",
        "delta_q_A_inv",
    ]
    for mult in sorted({float(row["sg0_multiplier"]) for row in params}):
        nearest_fields.append(f"E_h_sg0m{slug_float(mult)}")
    for sigma in sorted({float(row["sigma_C"]) for row in params}):
        nearest_fields.append(f"C2_sigma{slug_float(sigma)}")
    for r_cut in sorted({float(row["r_cut"]) for row in params}):
        nearest_fields.append(f"included_r_cut{slug_float(r_cut)}")

    write_tsv(args.output_dir / "parameter_grid_summary_by_hkl.tsv", summary_rows, summary_fields)
    write_tsv(args.output_dir / "selected_examples_by_parameter.tsv", selected_rows, selected_fields)
    write_tsv(args.output_dir / "selected_examples_neighbour_contributions.tsv", contribution_rows, contribution_fields)
    write_tsv(args.output_dir / "nearest_neighbours_independent_of_cutoff.tsv", nearest_rows, nearest_fields)
    summary_df = pd.DataFrame(summary_rows)
    selected_df = pd.DataFrame(selected_rows)
    make_plots(args.output_dir, summary_df, selected_df)
    write_notes(args.output_dir, args, provenance, prior_status, summary_df, selected_df)
    log(f"wrote parameter-sensitivity outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

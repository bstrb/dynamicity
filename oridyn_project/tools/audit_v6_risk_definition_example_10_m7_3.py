#!/usr/bin/env python3
"""Audit the V6 geometry-only observation-level risk score for HKL (10,-7,3).

This script reads existing V6 score/cache files and writes a compact
definition/example package.  It does not run merging, filtering, Partialator,
stream generation, or production processing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import compute_v5_nonself_local_excitation_raw_scores_20_0p3 as v5mod
from oridyn.coupling_exposure_v2 import estimate_reciprocal_metric


DEFAULT_SOURCE_OUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_full_population_sweep_20260717"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_risk_definition_audit_10_-7_3_20260831"
)
DEFAULT_V5_SCORES = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v5_nonself_local_excitation_raw_20_0p3_20260705/"
    "geometry_coupling_v5_nonself_local_excitation_raw_scores.csv"
)

HKL = (10, -7, 3)
DEFAULT_SOURCE_FRAME = "1712"
DEFAULT_TARGETS = [
    {
        "case_label": "low",
        "event": "12780",
        "zone_axis": "[2 1 -4]",
        "zone_angle_deg": "1.209",
        "expected_srisk_approx": "0",
    },
    {
        "case_label": "high",
        "event": "23857",
        "zone_axis": "[1 1 -1]",
        "zone_angle_deg": "0.396",
        "expected_srisk_approx": "0.319947",
    },
    {
        "case_label": "high",
        "event": "46106",
        "zone_axis": "[5 5 -4]",
        "zone_angle_deg": "2.425",
        "expected_srisk_approx": "0.173220",
    },
]
SIGMA_C = 0.050
R_CUT = 0.150
SG0 = 0.002306535801437913
EPS = 1.0e-12
CSV_CHUNKSIZE = 250_000
TOP_N = 10


@dataclass(frozen=True)
class Target:
    case_label: str
    event: str
    zone_axis: str
    zone_angle_deg: str
    expected_srisk_approx: str


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def connect_readonly(db_file: Path) -> sqlite3.Connection:
    uri = f"file:{db_file.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def normalize_source(value: Any) -> str:
    return "" if value is None else str(value).strip()


def source_matches_frame(source_filename: str, frame: str) -> bool:
    base = Path(source_filename).name
    return bool(re.search(rf"_{re.escape(str(frame))}\.h5$", base))


def parse_targets(raw: str | None) -> list[Target]:
    if not raw:
        return [Target(**row) for row in DEFAULT_TARGETS]
    targets: list[Target] = []
    for item in raw.split(","):
        parts = item.split(":")
        if len(parts) != 5:
            raise SystemExit("--targets entries must be label:event:zone_axis:angle:expected_srisk")
        targets.append(
            Target(
                case_label=parts[0],
                event=parts[1],
                zone_axis=parts[2],
                zone_angle_deg=parts[3],
                expected_srisk_approx=parts[4],
            )
        )
    return targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--v5-scores", type=Path, default=None, help="Defaults to cache_provenance.json source, then known V5 score table")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-frame", default=DEFAULT_SOURCE_FRAME)
    parser.add_argument("--targets", default=None, help="Optional comma list of label:event:zone_axis:angle:expected_srisk")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--chunksize", type=int, default=CSV_CHUNKSIZE)
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.targets = parse_targets(args.targets)
    args.top_n = max(1, int(args.top_n))
    args.chunksize = max(1, int(args.chunksize))
    if not args.source_out_dir.is_dir():
        raise SystemExit(f"--source-out-dir not found: {args.source_out_dir}")
    args.cache_db = args.source_out_dir / "full_population_cache.sqlite"
    if not args.cache_db.is_file():
        raise SystemExit(f"full_population_cache.sqlite not found: {args.cache_db}")
    if args.v5_scores is None:
        provenance = read_json(args.source_out_dir / "cache_provenance.json")
        raw_path = provenance.get("v5_geometry_score_source", {}).get("path") if isinstance(provenance, dict) else None
        args.v5_scores = Path(raw_path) if raw_path else DEFAULT_V5_SCORES
    args.v5_scores = args.v5_scores.expanduser().resolve()
    if not args.v5_scores.is_file():
        raise SystemExit(f"--v5-scores not found: {args.v5_scores}")
    return args


def query_target_rows(conn: sqlite3.Connection, frame: str, targets: list[Target]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        matches = conn.execute(
            """
            SELECT ordinal,source_filename,event,h,k,l,exact_key_text,source_order,sg,abs_sg,Eg,D,U,M,M2,Eg*M2 AS srisk
            FROM score_cache
            WHERE h=? AND k=? AND l=? AND event=?
            ORDER BY source_filename
            """,
            (*HKL, target.event),
        ).fetchall()
        filtered = [row for row in matches if source_matches_frame(str(row[1]), frame)]
        if len(filtered) != 1:
            raise SystemExit(f"Expected one cache row for frame {frame} event {target.event} HKL {HKL}, found {len(filtered)}")
        row = filtered[0]
        rows.append(
            {
                "case_label": target.case_label,
                "expected_srisk_approx": target.expected_srisk_approx,
                "nearest_low_index_zone_axis": target.zone_axis,
                "nearest_zone_axis_angle_deg": target.zone_angle_deg,
                "orientation_metadata_source": "user-provided target list",
                "ordinal": int(row[0]),
                "source_file": str(row[1]),
                "frame": frame,
                "event": str(row[2]),
                "h": int(row[3]),
                "k": int(row[4]),
                "l": int(row[5]),
                "exact_key_text": str(row[6]),
                "source_order": int(row[7]),
                "s_g": float(row[8]),
                "abs_s_g": float(row[9]),
                "Eg": float(row[10]),
                "D": float(row[11]),
                "U": float(row[12]),
                "M": float(row[13]),
                "M2": float(row[14]),
                "S_risk": float(row[15]),
            }
        )
    return rows


def hkl_distribution(conn: sqlite3.Connection) -> list[float]:
    return [
        float(row[0])
        for row in conn.execute(
            "SELECT Eg*M2 AS srisk FROM score_cache WHERE h=? AND k=? AND l=? ORDER BY srisk",
            HKL,
        )
    ]


def percentile(values: list[float], score: float) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    less = sum(1 for value in values if value < score)
    equal = sum(1 for value in values if value == score)
    return 100.0 * (less + 0.5 * equal) / n


def robust_z(values: list[float], score: float) -> float | None:
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size < 2:
        return None
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    denom = 1.4826 * mad
    if not math.isfinite(denom) or denom <= EPS:
        return None
    return float((score - med) / denom)


def add_percentiles(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    total = int(conn.execute("SELECT COUNT(*) FROM score_cache").fetchone()[0])
    hkl_values = hkl_distribution(conn)
    hkl_n = len(hkl_values)
    hkl_median = float(np.median(hkl_values)) if hkl_values else float("nan")
    hkl_mad = float(np.median(np.abs(np.asarray(hkl_values) - hkl_median))) if hkl_values else float("nan")
    for row in rows:
        score = float(row["S_risk"])
        less, equal = conn.execute(
            """
            SELECT
              SUM(CASE WHEN Eg*M2 < ? THEN 1 ELSE 0 END),
              SUM(CASE WHEN Eg*M2 = ? THEN 1 ELSE 0 END)
            FROM score_cache
            """,
            (score, score),
        ).fetchone()
        row["global_srisk_percentile_midrank"] = 100.0 * (int(less or 0) + 0.5 * int(equal or 0)) / total
        row["within_signed_hkl_srisk_percentile_midrank"] = percentile(hkl_values, score)
        z = robust_z(hkl_values, score)
        row["within_signed_hkl_robust_z"] = "" if z is None else z
        row["within_signed_hkl_n"] = hkl_n
        row["within_signed_hkl_median_srisk"] = hkl_median
        row["within_signed_hkl_mad_srisk"] = hkl_mad


def load_v5_rows_for_frame(v5_scores: Path, frame: str, needed_events: set[str], chunksize: int) -> pd.DataFrame:
    usecols = ["source_filename", "event", "h", "k", "l", "d_angstrom", "inv_nm", "sg_target", "target_excitation_Eg"]
    frames: list[pd.DataFrame] = []
    started = time.monotonic()
    rows_scanned = 0
    last_log = started
    log(f"scanning V5 score table for source frame {frame}")
    for chunk in pd.read_csv(v5_scores, usecols=usecols, chunksize=chunksize):
        rows_scanned += len(chunk)
        source = chunk["source_filename"].astype(str)
        event = chunk["event"].astype(str)
        mask = source.map(lambda value: source_matches_frame(value, frame))
        if mask.any():
            frames.append(chunk.loc[mask].copy())
        now = time.monotonic()
        if now - last_log > 5.0:
            rate = rows_scanned / max(1.0e-9, now - started)
            log(f"  scanned {rows_scanned:,} rows; rate={rate:,.0f}/s; matched_chunks={len(frames)}")
            last_log = now
    if not frames:
        raise SystemExit(f"No V5 rows found for frame {frame} in {v5_scores}")
    table = pd.concat(frames, ignore_index=True)
    found_events = set(table.loc[(table["h"] == HKL[0]) & (table["k"] == HKL[1]) & (table["l"] == HKL[2]), "event"].astype(str))
    missing_events = sorted(needed_events - found_events)
    if missing_events:
        raise SystemExit(f"Target events not found in V5 table for HKL {HKL}: {missing_events}")
    log(f"loaded {len(table):,} V5 rows for frame {frame}")
    return table


def compute_neighbours(frame_table: pd.DataFrame, target_rows: list[dict[str, Any]], top_n: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    work = frame_table.reset_index(drop=True).copy()
    for column in ["h", "k", "l"]:
        work[column] = pd.to_numeric(work[column], errors="coerce").astype("int64")
    hkls = work.loc[:, ["h", "k", "l"]].to_numpy(dtype=np.int64)
    inv_nm = pd.to_numeric(work["inv_nm"], errors="coerce").to_numpy(dtype=float)
    d_values = pd.to_numeric(work["d_angstrom"], errors="coerce").to_numpy(dtype=float)
    q_invA = np.divide(inv_nm, 10.0, out=np.full_like(inv_nm, np.nan, dtype=float), where=np.isfinite(inv_nm))
    q_invA = np.where(~np.isfinite(q_invA) & np.isfinite(d_values) & (d_values > 0.0), 1.0 / d_values, q_invA)
    metric, _metric_stats = estimate_reciprocal_metric(hkls, q_invA)
    sg = pd.to_numeric(work["sg_target"], errors="coerce").to_numpy(dtype=float)
    eq = v5mod.excitation_weight_from_sg(sg, SG0)
    params = v5mod.V5Params(sg0=SG0, kernel="gaussian", sigma_c=SIGMA_C, q0=v5mod.DEFAULT_Q0, r_cut=R_CUT, target_batch_size=256)
    by_event = {str(row.event): idx for idx, row in work.iterrows()}
    counts: dict[str, int] = {}
    neighbour_rows: list[dict[str, Any]] = []
    for target in target_rows:
        event = str(target["event"])
        target_idx = by_event.get(event)
        if target_idx is None:
            counts[event] = 0
            continue
        delta = hkls - hkls[target_idx][None, :]
        nonself = np.any(delta != 0, axis=1)
        dq = v5mod.dq_from_delta(delta[None, :, :], metric)[0]
        c = v5mod.coupling_kernel(dq, params)
        contribution = np.where(nonself & (c > 0.0) & np.isfinite(eq), eq * c * c, 0.0)
        contributing_idx = np.nonzero(contribution > 0.0)[0]
        counts[event] = int(contributing_idx.size)
        order = contributing_idx[np.lexsort((work.loc[contributing_idx, "event"].astype(str).to_numpy(), -contribution[contributing_idx]))]
        for rank, idx in enumerate(order[:top_n], start=1):
            neighbour_rows.append(
                {
                    "target_event": event,
                    "target_case_label": target["case_label"],
                    "target_h": target["h"],
                    "target_k": target["k"],
                    "target_l": target["l"],
                    "rank": rank,
                    "neighbour_source_file": work.at[idx, "source_filename"],
                    "neighbour_event": str(work.at[idx, "event"]),
                    "neighbour_h": int(work.at[idx, "h"]),
                    "neighbour_k": int(work.at[idx, "k"]),
                    "neighbour_l": int(work.at[idx, "l"]),
                    "s_h": float(sg[idx]),
                    "E_h": float(eq[idx]),
                    "dq_A_inv": float(dq[idx]),
                    "C": float(c[idx]),
                    "E_h_C2": float(contribution[idx]),
                }
            )
    return counts, neighbour_rows


def normalization_rows() -> list[dict[str, Any]]:
    return [
        {
            "normalization": "raw S_risk",
            "formula": "S_risk(g) = Eg * M2(g); M2(g) = sum_{h != g} E_h * C(h-g)^2",
            "scope": "single observation",
            "interpretation": "absolute geometry-only local coupling burden for that observation",
        },
        {
            "normalization": "global percentile/rank",
            "formula": "midrank percentile of S_risk among all score_cache observations",
            "scope": "all accepted scoreable observations",
            "interpretation": "diagnostic whole-dataset high-risk tail position",
        },
        {
            "normalization": "within-HKL percentile",
            "formula": "midrank percentile of S_risk among rows with the same exact signed h,k,l",
            "scope": "same signed reflection only",
            "interpretation": "which measurements of this reflection are unusually risky",
        },
        {
            "normalization": "within-HKL robust z-score",
            "formula": "(S_risk - median_hkl(S_risk)) / (1.4826 * MAD_hkl(S_risk))",
            "scope": "same signed reflection only",
            "interpretation": "robust same-reflection outlier score; NA when same-HKL spread is zero or too small",
        },
    ]


def notes_text(args: argparse.Namespace, example_rows: list[dict[str, Any]]) -> str:
    source_lines = [
        "- Cache columns `sg`, `Eg`, `D`, `U`, `M`, `M2`: `tools/build_v6_full_population_sweep.py:447-462`.",
        "- V6 cache fills `Eg` from `target_excitation_Eg` and `M2` from the V5/coarse aggregate: `tools/build_v6_full_population_sweep.py:688-692`, inserted at `:700-701`.",
        "- Geometry metadata records `sigma_c=0.050`, `r_cut=0.150`, target reflection excluded, exact signed HKLs, and no symmetry canonicalization: `tools/build_v6_full_population_sweep.py:726-737`.",
        "- V6 score registry defines `eg_m2` as `Eg * M2`: `tools/build_v6_score_target_filter_map.py:394-398`; scores are evaluated from `Eg,D,U,M,M2` at `:800-808`.",
        "- Coarse scorer defines `M2` column name as `excitation_coupling_sq_sum_*`: `tools/build_v5_coarse_score_sweep_streams.py:1042-1050`.",
        "- Nonself frame geometry uses `delta = hkls[None,:,:] - target_hkl[:,None,:]`, excludes self with `np.any(delta != 0)`, applies kernel, then stores `M2 = sum(Eq*C^2)`: `tools/build_v5_coarse_score_sweep_streams.py:1460-1492`.",
        "- Excitation weight is `E = exp[-(|s|/sg0)^2]`: `tools/compute_v5_nonself_local_excitation_raw_scores_20_0p3.py:326-330`.",
        "- Gaussian kernel is `exp[-0.5*(dq/sigma_c)^2]`, zeroed when `dq > r_cut`: `tools/compute_v5_nonself_local_excitation_raw_scores_20_0p3.py:333-343`; `dq = sqrt(delta^T G* delta)` at `:346-351`.",
    ]
    rows = "\n".join(
        f"- event {row['event']}: S_risk={float(row['S_risk']):.12g}, Eg={float(row['Eg']):.12g}, M2={float(row['M2']):.12g}, "
        f"global percentile={float(row['global_srisk_percentile_midrank']):.4f}, same-HKL percentile={float(row['within_signed_hkl_srisk_percentile_midrank']):.4f}."
        for row in example_rows
    )
    return f"""# V6 Geometry-Only Risk Definition Audit: HKL (10,-7,3)

Generated: {datetime.now(timezone.utc).isoformat()}

## Exact Definition

For one observation `g`, the current geometry-only risk score used here is:

```text
S_risk(g) = Eg * M2(g)
M2(g) = sum_{{h != g}} E_h * C(h-g)^2
E_h = exp[-(|s_h| / sg0)^2]
C(h-g) = exp[-0.5 * (dq / sigma_c)^2] for dq <= r_cut, otherwise 0
dq = sqrt((h-g)^T G* (h-g))
```

The current V6 cache metadata records `sg0 = 0.002306535801437913 A^-1`, `sigma_c = 0.050 A^-1`, and `r_cut = 0.150 A^-1`. The target reflection itself is excluded (`h != g`). Signed HKLs are used exactly; there is no symmetry canonicalization.

## Source Code Lines

{chr(10).join(source_lines)}

## Example Rows

{rows}

## Normalization Choices

- Raw `S_risk` is observation-level, not frame-level.
- Global ranking is useful diagnostically because it identifies the highest-risk tail across the whole accepted population.
- Within-HKL normalization asks a different question: which observations of the same exact signed reflection are unusually risky relative to their peers?
- Weighting should be calibrated from residual scatter, not chosen as an arbitrary intensity scale factor.

## Caveats

- The SQLite cache stores aggregate `M2`, not every neighbour edge. The neighbour table in this audit recomputes the top edges for the target frame using the same score-table rows, reciprocal metric estimation, `E_h`, `C`, and nonself rule.
- Zone-axis labels and angles for these three observations are carried from the target example metadata unless an external orientation table is supplied in a later analysis.
- Percentiles are midrank percentiles: `100 * (n_less + 0.5*n_equal) / n`.

## Inputs

- source_out_dir: `{args.source_out_dir}`
- score_cache: `{args.cache_db}`
- v5_scores: `{args.v5_scores}`
- output_dir: `{args.output_dir}`
"""


def run_audit(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in [
        "risk_definition_example_10_-7_3.tsv",
        "risk_definition_top_neighbours_10_-7_3.tsv",
        "risk_normalization_summary.tsv",
        "risk_definition_notes.md",
    ]):
        raise SystemExit(f"Refusing to overwrite existing audit outputs in {args.output_dir}")
    with connect_readonly(args.cache_db) as conn:
        target_rows = query_target_rows(conn, args.source_frame, args.targets)
        add_percentiles(conn, target_rows)
    frame_table = load_v5_rows_for_frame(args.v5_scores, args.source_frame, {t.event for t in args.targets}, args.chunksize)
    neighbour_counts, neighbour_rows = compute_neighbours(frame_table, target_rows, int(args.top_n))
    for row in target_rows:
        row["contributing_neighbours_inside_cutoff"] = neighbour_counts.get(str(row["event"]), 0)
    example_fields = [
        "case_label",
        "source_file",
        "frame",
        "event",
        "h",
        "k",
        "l",
        "nearest_low_index_zone_axis",
        "nearest_zone_axis_angle_deg",
        "orientation_metadata_source",
        "s_g",
        "Eg",
        "M2",
        "S_risk",
        "global_srisk_percentile_midrank",
        "within_signed_hkl_srisk_percentile_midrank",
        "within_signed_hkl_robust_z",
        "within_signed_hkl_n",
        "within_signed_hkl_median_srisk",
        "within_signed_hkl_mad_srisk",
        "contributing_neighbours_inside_cutoff",
        "D",
        "U",
        "M",
        "source_order",
        "ordinal",
        "exact_key_text",
        "expected_srisk_approx",
    ]
    write_rows(args.output_dir / "risk_definition_example_10_-7_3.tsv", target_rows, example_fields)
    write_rows(args.output_dir / "risk_definition_top_neighbours_10_-7_3.tsv", neighbour_rows)
    write_rows(args.output_dir / "risk_normalization_summary.tsv", normalization_rows())
    (args.output_dir / "risk_definition_notes.md").write_text(notes_text(args, target_rows), encoding="utf-8")
    log(f"wrote audit outputs to {args.output_dir}")


def run_self_test() -> int:
    root = PROJECT_ROOT / ".codex_smoke" / "risk_definition_example"
    if root.exists():
        shutil.rmtree(root)
    source_out = root / "source_out"
    output_dir = root / "out"
    source_out.mkdir(parents=True)
    db_file = source_out / "full_population_cache.sqlite"
    v5_csv = root / "v5.csv"
    try:
        conn = sqlite3.connect(db_file)
        conn.execute(
            """
            CREATE TABLE score_cache(
                ordinal INTEGER PRIMARY KEY,
                source_filename TEXT,
                event TEXT,
                h INTEGER,
                k INTEGER,
                l INTEGER,
                exact_key_text TEXT,
                source_order INTEGER,
                sg REAL,
                abs_sg REAL,
                Eg REAL,
                D REAL,
                U REAL,
                M REAL,
                M2 REAL
            )
            """
        )
        source = str(root / "synthetic_1712.h5")
        rows = [
            (0, source, "12780", 10, -7, 3, f"{source}|12780|10|-7|3", 0, 0.001, 0.001, 0.9, 0.0, 0.0, 0.0, 0.0),
            (1, source, "23857", 10, -7, 3, f"{source}|23857|10|-7|3", 1, 0.001, 0.001, 0.9, 1.0, 2.0, 0.5, 0.2),
            (2, source, "46106", 10, -7, 3, f"{source}|46106|10|-7|3", 2, 0.001, 0.001, 0.9, 1.0, 2.0, 0.5, 0.1),
            (3, source, "1", 11, -7, 3, f"{source}|1|11|-7|3", 3, 0.001, 0.001, 0.9, 1.0, 2.0, 0.5, 0.1),
        ]
        conn.executemany("INSERT INTO score_cache VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        with v5_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_filename", "event", "h", "k", "l", "d_angstrom", "inv_nm", "sg_target", "target_excitation_Eg"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"source_filename": source, "event": row[2], "h": row[3], "k": row[4], "l": row[5], "d_angstrom": 1.0, "inv_nm": 10.0, "sg_target": row[8], "target_excitation_Eg": row[10]})
        args = argparse.Namespace(
            source_out_dir=source_out.resolve(),
            cache_db=db_file.resolve(),
            v5_scores=v5_csv.resolve(),
            output_dir=output_dir.resolve(),
            source_frame="1712",
            targets=[Target(**row) for row in DEFAULT_TARGETS],
            top_n=3,
            chunksize=2,
        )
        run_audit(args)
        for name in ["risk_definition_example_10_-7_3.tsv", "risk_definition_top_neighbours_10_-7_3.tsv", "risk_normalization_summary.tsv", "risk_definition_notes.md"]:
            assert (output_dir / name).is_file(), name
        print("self-test passed: wrote example rows, neighbour table, normalization summary, and notes")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

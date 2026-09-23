#!/usr/bin/env python3
"""Extract same-signed-HKL low/high OriDyn risk examples with orientation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oridyn.stream_parser import (  # noqa: E402
    STREAM_AVERAGE_CLEN_RE,
    STREAM_CELL_RE,
    STREAM_DET_SHIFT_X_RE,
    STREAM_DET_SHIFT_Y_RE,
    STREAM_EVENT_RE,
    STREAM_IMAGE_FILENAME_RE,
    STREAM_MATRIX_COLUMNS,
    STREAM_SERIAL_RE,
    STREAM_VECTOR_RE,
)


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
PREFERRED_RISK_COLUMNS = [
    "trust_risk_v2_full_norm",
    "trust_risk_norm",
    "S_dyn_geom",
    "geometry_coupling_risk_norm",
    "risk_norm",
]
OTHER_RISK_TOKENS = ("risk", "trust", "S_dyn", "coupling", "score")
U_MATRIX_COLUMNS = tuple(f"U{i}{j}" for i in range(1, 4) for j in range(1, 4))
UB_MATRIX_COLUMNS = STREAM_MATRIX_COLUMNS
ORIENTATION_FIELD_CANDIDATES = [
    "frame",
    "frame_number",
    "chunk_id",
    "crystal_in_chunk",
    "image_serial",
    "distance_mm",
    "det_shift_x_mm",
    "det_shift_y_mm",
    "a_angstrom",
    "b_angstrom",
    "c_angstrom",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
    *UB_MATRIX_COLUMNS,
    *U_MATRIX_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="Risk-score CSV")
    parser.add_argument("--stream", type=Path, default=None, help="CrystFEL stream used to recover orientation matrices")
    parser.add_argument("--risk-column", default=None, help="Risk column to rank by")
    parser.add_argument("--hkl", nargs=3, type=int, metavar=("H", "K", "L"), help="Signed HKL to extract")
    parser.add_argument("--n-low", type=int, default=3, help="Number of low-risk observations to select")
    parser.add_argument("--n-high", type=int, default=3, help="Number of high-risk observations to select")
    parser.add_argument("--min-obs", type=int, default=20, help="Minimum observations for automatic HKL choice")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional output JSON")
    parser.add_argument("--scores-chunksize", type=int, default=500_000, help="Rows per score-CSV chunk")
    parser.add_argument("--survivor-mask", type=Path, default=None, help="Exact-key partialator survivor mask CSV")
    parser.add_argument("--survivors-only", action="store_true", help="Require --survivor-mask and select survivor rows only")
    parser.add_argument("--min-partiality", type=float, default=None, help="Minimum partialator partiality from --survivor-mask")
    args = parser.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if args.stream is not None and not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if args.n_low < 1:
        raise SystemExit("--n-low must be >= 1")
    if args.n_high < 1:
        raise SystemExit("--n-high must be >= 1")
    if args.min_obs < 1:
        raise SystemExit("--min-obs must be >= 1")
    if args.scores_chunksize < 1:
        raise SystemExit("--scores-chunksize must be >= 1")
    if args.survivor_mask is not None and not args.survivor_mask.exists():
        raise SystemExit(f"--survivor-mask not found: {args.survivor_mask}")
    if args.survivors_only and args.survivor_mask is None:
        raise SystemExit("--survivors-only requires --survivor-mask")
    if args.min_partiality is not None:
        if args.survivor_mask is None:
            raise SystemExit("--min-partiality requires --survivor-mask")
        if not np.isfinite(float(args.min_partiality)):
            raise SystemExit("--min-partiality must be finite")
    return args


def normalize_source(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_event(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    return text


def normalize_key_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    bad_hkl = out[HKL_COLUMNS].isna().any(axis=1)
    out = out.loc[~bad_hkl].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_survivor_mask(
    path: Path,
    min_partiality: float | None,
    chunksize: int,
    hkl: tuple[int, int, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = read_header(path)
    missing = [column for column in KEY_COLUMNS if column not in header]
    if missing:
        raise SystemExit(f"--survivor-mask is missing required exact-key column(s): {missing}")
    usecols = [*KEY_COLUMNS]
    payload_columns = []
    for optional in ["partialator_survived", "partiality", "I_unmerged", "sigma_unmerged"]:
        if optional in header:
            usecols.append(optional)
            payload_columns.append(optional)
    if min_partiality is not None and "partiality" not in payload_columns:
        raise SystemExit("--min-partiality was supplied but --survivor-mask has no partiality column")

    h, k, l = hkl
    chunks = []
    rows_read = 0
    rows_for_hkl = 0
    rows_kept = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        rows_read += int(len(chunk))
        work = normalize_key_columns(chunk)
        work = work.loc[(work["h"] == h) & (work["k"] == k) & (work["l"] == l)].copy()
        rows_for_hkl += int(len(work))
        if "partiality" in work.columns:
            work["partiality"] = pd.to_numeric(work["partiality"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "I_unmerged" in work.columns:
            work["I_unmerged"] = pd.to_numeric(work["I_unmerged"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "sigma_unmerged" in work.columns:
            work["sigma_unmerged"] = pd.to_numeric(work["sigma_unmerged"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if min_partiality is not None:
            work = work.loc[work["partiality"] >= float(min_partiality)].copy()
        if not work.empty:
            chunks.append(work.loc[:, [*KEY_COLUMNS, *payload_columns]].copy())
            rows_kept += int(len(work))

    if chunks:
        keys = pd.concat(chunks, ignore_index=True).drop_duplicates(KEY_COLUMNS, keep="first")
    else:
        keys = pd.DataFrame(columns=[*KEY_COLUMNS, *payload_columns])
    stats = {
        "survivor_mask_rows_read": int(rows_read),
        "survivor_mask_rows_for_hkl": int(rows_for_hkl),
        "survivor_mask_rows_after_partiality_filter": int(rows_kept),
        "survivor_mask_unique_keys_for_hkl": int(len(keys)),
        "survivor_mask_payload_columns": payload_columns,
    }
    return keys, stats


def apply_survivor_filter(scores: pd.DataFrame, survivor_keys: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or survivor_keys.empty:
        return pd.DataFrame(columns=[*scores.columns, *[column for column in survivor_keys.columns if column not in KEY_COLUMNS]])
    return scores.merge(survivor_keys, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")


def score_range_stats(scores: pd.DataFrame, hkl: tuple[int, int, int], risk_column: str) -> dict[str, Any]:
    h, k, l = hkl
    if scores.empty:
        return {
            "h": h,
            "k": k,
            "l": l,
            "n_observations": 0,
            "risk_min": None,
            "risk_max": None,
            "risk_range": None,
            "risk_median": None,
        }
    values = scores[risk_column].dropna()
    if values.empty:
        return {
            "h": h,
            "k": k,
            "l": l,
            "n_observations": int(len(scores)),
            "risk_min": None,
            "risk_max": None,
            "risk_range": None,
            "risk_median": None,
        }
    return {
        "h": h,
        "k": k,
        "l": l,
        "n_observations": int(len(scores)),
        "risk_min": float(values.min()),
        "risk_max": float(values.max()),
        "risk_range": float(values.max() - values.min()),
        "risk_median": float(values.median()),
    }


def format_range(stats: dict[str, Any]) -> str:
    if stats["risk_min"] is None or stats["risk_max"] is None:
        return "none"
    return f"{float(stats['risk_min']):.8g} .. {float(stats['risk_max']):.8g}"


def detect_risk_candidates(header: list[str]) -> list[str]:
    candidates = [column for column in PREFERRED_RISK_COLUMNS if column in header]
    for column in header:
        if column in candidates:
            continue
        if any(token in column for token in OTHER_RISK_TOKENS):
            candidates.append(column)
    return candidates


def choose_risk_column(header: list[str], requested: str | None) -> tuple[str, list[str]]:
    candidates = detect_risk_candidates(header)
    if requested:
        if requested not in header:
            raise SystemExit(f"--risk-column {requested!r} is not present in --scores")
        return requested, candidates
    if not candidates:
        raise SystemExit(
            "Could not auto-detect a risk column. Pass --risk-column explicitly. "
            f"Available columns: {', '.join(header)}"
        )
    return candidates[0], candidates


def detect_orientation_columns(header: list[str]) -> tuple[str | None, list[str]]:
    if all(column in header for column in UB_MATRIX_COLUMNS):
        return "csv_UB", list(UB_MATRIX_COLUMNS)
    if all(column in header for column in U_MATRIX_COLUMNS):
        return "csv_U", list(U_MATRIX_COLUMNS)
    return None, []


def score_usecols(header: list[str], risk_column: str, orientation_columns: list[str]) -> list[str]:
    optional = [
        column
        for column in [
            "frame",
            "frame_number",
            "chunk_id",
            "crystal_in_chunk",
            "image_serial",
            "d_angstrom",
            "q_invA",
            "sg",
            *orientation_columns,
        ]
        if column in header
    ]
    return list(dict.fromkeys([*KEY_COLUMNS, risk_column, *optional]))


def clean_score_chunk(table: pd.DataFrame, risk_column: str, row_offset: int) -> pd.DataFrame:
    out = table.copy()
    out["_score_row"] = np.arange(int(row_offset), int(row_offset) + len(out), dtype=np.int64)
    out = normalize_key_columns(out)
    out[risk_column] = pd.to_numeric(out[risk_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[risk_column]).copy()
    return out


def iter_score_chunks(path: Path, risk_column: str, usecols: list[str], chunksize: int):
    row_offset = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        clean = clean_score_chunk(chunk, risk_column, row_offset)
        row_offset += len(chunk)
        if not clean.empty:
            yield clean


def select_hkl_automatically(path: Path, risk_column: str, min_obs: int, chunksize: int) -> tuple[int, int, int]:
    usecols = [*KEY_COLUMNS, risk_column]
    aggregates: dict[tuple[int, int, int], list[float]] = {}
    rows_seen = 0
    for chunk in iter_score_chunks(path, risk_column, usecols, chunksize):
        rows_seen += len(chunk)
        grouped = chunk.groupby(HKL_COLUMNS, sort=False)[risk_column].agg(["size", "min", "max"]).reset_index()
        for row in grouped.itertuples(index=False):
            key = (int(row.h), int(row.k), int(row.l))
            current = aggregates.get(key)
            if current is None:
                aggregates[key] = [int(row.size), float(row.min), float(row.max)]
            else:
                current[0] += int(row.size)
                current[1] = min(float(current[1]), float(row.min))
                current[2] = max(float(current[2]), float(row.max))
    if rows_seen == 0:
        raise SystemExit(f"No finite values found in risk column {risk_column!r}")

    summary = pd.DataFrame.from_records(
        [
            {"h": h, "k": k, "l": l, "n_obs": int(values[0]), "risk_min": values[1], "risk_max": values[2]}
            for (h, k, l), values in aggregates.items()
        ]
    )
    summary = summary.loc[summary["n_obs"] >= int(min_obs)].copy()
    if summary.empty:
        raise SystemExit(f"No signed HKL has at least --min-obs {min_obs} finite-risk observations")
    summary["risk_spread"] = summary["risk_max"] - summary["risk_min"]
    summary = summary.sort_values(
        ["risk_spread", "n_obs", "risk_max", "h", "k", "l"],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    )
    top = summary.iloc[0]
    return int(top.h), int(top.k), int(top.l)


def load_hkl_scores(
    path: Path,
    risk_column: str,
    usecols: list[str],
    hkl: tuple[int, int, int],
    chunksize: int,
) -> pd.DataFrame:
    h, k, l = hkl
    matches: list[pd.DataFrame] = []
    rows_seen = 0
    for chunk in iter_score_chunks(path, risk_column, usecols, chunksize):
        rows_seen += len(chunk)
        subset = chunk.loc[(chunk["h"] == h) & (chunk["k"] == k) & (chunk["l"] == l)].copy()
        if not subset.empty:
            matches.append(subset)
    if rows_seen == 0:
        raise SystemExit(f"No finite values found in risk column {risk_column!r}")
    return pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=[*usecols, "_score_row"])


def select_tail_examples(
    scores: pd.DataFrame,
    hkl: tuple[int, int, int],
    risk_column: str,
    n_low: int,
    n_high: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    h, k, l = hkl
    subset = scores.loc[(scores["h"] == h) & (scores["k"] == k) & (scores["l"] == l)].copy()
    if subset.empty:
        raise SystemExit(f"No finite-risk observations found for signed HKL {h} {k} {l}")
    subset = subset.sort_values([risk_column, "source_filename", "event", "_score_row"], kind="mergesort").reset_index(
        drop=True
    )
    subset["rank_within_hkl"] = np.arange(1, len(subset) + 1, dtype=np.int64)
    subset["quantile_within_hkl"] = (
        0.0 if len(subset) == 1 else (subset["rank_within_hkl"].astype(float) - 1.0) / float(len(subset) - 1)
    )
    low = subset.head(int(n_low)).copy()
    high = subset.tail(int(n_high)).copy()
    low["risk_tail"] = "low"
    high["risk_tail"] = "high"
    selected = pd.concat([low, high], ignore_index=True)
    stats = {
        "h": h,
        "k": k,
        "l": l,
        "n_observations": int(len(subset)),
        "risk_min": float(subset[risk_column].min()),
        "risk_max": float(subset[risk_column].max()),
        "risk_range": float(subset[risk_column].max() - subset[risk_column].min()),
        "risk_median": float(subset[risk_column].median()),
    }
    return selected, stats


def parse_stream_reflection_hkl(line: str) -> tuple[int, int, int] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        float(parts[3])
        float(parts[4])
    except ValueError:
        return None
    return h, k, l


def make_stream_orientation_row(
    source_filename: str,
    event: str,
    image_serial: int | None,
    current_clen_m: float | None,
    det_shift_x_mm: float,
    det_shift_y_mm: float,
    current_cell: dict[str, float] | None,
    current_vectors: dict[str, np.ndarray],
    frame: int,
    chunk_id: int,
    crystal_in_chunk: int,
) -> dict[str, Any]:
    missing = [axis for axis in ("a", "b", "c") if axis not in current_vectors]
    if missing:
        raise ValueError(f"Selected crystal in chunk {chunk_id} is missing reciprocal vectors: {missing}")
    reciprocal = np.column_stack([current_vectors["a"], current_vectors["b"], current_vectors["c"]]) / 10.0
    row: dict[str, Any] = {
        "frame": int(frame),
        "frame_number": int(frame) + 1,
        "chunk_id": int(chunk_id),
        "crystal_in_chunk": int(crystal_in_chunk),
        "source_filename": normalize_source(source_filename),
        "event": normalize_event(event),
        "image_serial": -1 if image_serial is None else int(image_serial),
        "distance_mm": np.nan if current_clen_m is None else 1000.0 * float(current_clen_m),
        "det_shift_x_mm": float(det_shift_x_mm),
        "det_shift_y_mm": float(det_shift_y_mm),
    }
    if current_cell is not None:
        row.update(current_cell)
    for idx, value in enumerate(reciprocal.reshape(-1)):
        row[STREAM_MATRIX_COLUMNS[idx]] = float(value)
    return row


def load_stream_orientation_for_selected(stream_path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    target_keys = {
        (
            normalize_source(row.source_filename),
            normalize_event(row.event),
            int(row.h),
            int(row.k),
            int(row.l),
        )
        for row in selected.loc[:, KEY_COLUMNS].itertuples(index=False)
    }
    found: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    chunk_id = -1
    crystal_in_chunk = 0
    frame = 0
    in_chunk = False
    in_crystal = False
    in_reflections = False
    current_source = ""
    current_event = ""
    current_serial: int | None = None
    current_clen_m: float | None = None
    current_det_shift_x_mm = 0.0
    current_det_shift_y_mm = 0.0
    current_vectors: dict[str, np.ndarray] = {}
    current_cell: dict[str, float] | None = None
    pending_keys: set[tuple[str, str, int, int, int]] = set()

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith("----- Begin chunk -----"):
                in_chunk = True
                in_crystal = False
                in_reflections = False
                chunk_id += 1
                crystal_in_chunk = 0
                current_source = ""
                current_event = ""
                current_serial = None
                current_det_shift_x_mm = 0.0
                current_det_shift_y_mm = 0.0
                continue
            if raw_line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                continue
            if not in_chunk:
                continue

            if match := STREAM_IMAGE_FILENAME_RE.match(raw_line):
                current_source = normalize_source(match.group(1))
                continue
            if match := STREAM_EVENT_RE.match(raw_line):
                current_event = normalize_event(match.group(1))
                continue
            if match := STREAM_SERIAL_RE.match(raw_line):
                current_serial = int(match.group(1))
                continue
            if match := STREAM_AVERAGE_CLEN_RE.match(raw_line):
                current_clen_m = float(match.group(1))
                continue
            if match := STREAM_DET_SHIFT_X_RE.match(raw_line):
                current_det_shift_x_mm = float(match.group(1))
                continue
            if match := STREAM_DET_SHIFT_Y_RE.match(raw_line):
                current_det_shift_y_mm = float(match.group(1))
                continue

            if raw_line.startswith("--- Begin crystal"):
                in_crystal = True
                in_reflections = False
                crystal_in_chunk += 1
                current_vectors = {}
                current_cell = None
                pending_keys = set()
                continue

            if raw_line.startswith("--- End crystal"):
                if pending_keys:
                    event = current_event or f"chunk{chunk_id}_crystal{crystal_in_chunk}"
                    base_row = make_stream_orientation_row(
                        current_source,
                        event,
                        current_serial,
                        current_clen_m,
                        current_det_shift_x_mm,
                        current_det_shift_y_mm,
                        current_cell,
                        current_vectors,
                        frame,
                        chunk_id,
                        crystal_in_chunk,
                    )
                    for key in pending_keys:
                        found[key] = {**base_row, "h": key[2], "k": key[3], "l": key[4]}
                    if len(found) >= len(target_keys):
                        break
                frame += 1
                in_crystal = False
                in_reflections = False
                continue

            if not in_crystal:
                continue
            if match := STREAM_CELL_RE.match(raw_line):
                current_cell = {
                    "a_angstrom": 10.0 * float(match.group(1)),
                    "b_angstrom": 10.0 * float(match.group(2)),
                    "c_angstrom": 10.0 * float(match.group(3)),
                    "alpha_deg": float(match.group(4)),
                    "beta_deg": float(match.group(5)),
                    "gamma_deg": float(match.group(6)),
                }
                continue
            if match := STREAM_VECTOR_RE.match(raw_line):
                current_vectors[match.group(1)] = np.asarray(
                    [float(match.group(2)), float(match.group(3)), float(match.group(4))], dtype=float
                )
                continue
            if "Reflections measured after indexing" in raw_line:
                in_reflections = True
                continue
            if "End of reflections" in raw_line:
                in_reflections = False
                continue
            if in_reflections:
                hkl = parse_stream_reflection_hkl(raw_line)
                if hkl is None:
                    continue
                event = current_event or f"chunk{chunk_id}_crystal{crystal_in_chunk}"
                key = (current_source, event, int(hkl[0]), int(hkl[1]), int(hkl[2]))
                if key in target_keys:
                    pending_keys.add(key)

    rows = [found[key] for key in target_keys if key in found]
    return pd.DataFrame.from_records(rows)


def add_orientation(
    selected: pd.DataFrame,
    score_orientation_columns: list[str],
    stream_path: Path | None,
) -> tuple[pd.DataFrame, str, list[str], dict[str, Any]]:
    if score_orientation_columns:
        orientation_columns = [
            column for column in ORIENTATION_FIELD_CANDIDATES if column in selected.columns and column not in KEY_COLUMNS
        ]
        return selected.copy(), "scores_csv", orientation_columns, {
            "orientation_rows_matched": int(len(selected)),
            "orientation_rows_missing": 0,
        }

    if stream_path is None:
        raise SystemExit(
            "The score CSV does not contain a complete U11..U33 or UB11..UB33 matrix. "
            "Pass --stream to recover UB orientation matrices from the CrystFEL stream."
        )

    orientation = load_stream_orientation_for_selected(stream_path, selected)
    if orientation.empty:
        raise SystemExit("No selected observation keys were found in --stream; cannot recover orientation matrices")
    orientation_columns = [
        column for column in ORIENTATION_FIELD_CANDIDATES if column in orientation.columns and column not in KEY_COLUMNS
    ]
    merge_orientation_columns = [column for column in orientation_columns if column not in selected.columns]
    joined = selected.merge(
        orientation.loc[:, [*KEY_COLUMNS, *merge_orientation_columns]],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    has_matrix = joined.loc[:, [column for column in UB_MATRIX_COLUMNS if column in joined.columns]].notna().all(axis=1)
    missing = int((~has_matrix).sum()) if len(has_matrix) else int(len(joined))
    if missing:
        raise SystemExit(
            f"Missing orientation matrices for {missing} selected observation(s); "
            "check that --stream matches --scores exactly."
        )
    return joined, "stream_exact_key", orientation_columns, {
        "orientation_rows_matched": int(len(joined) - missing),
        "orientation_rows_missing": missing,
    }


def output_columns(table: pd.DataFrame, risk_column: str, orientation_columns: list[str]) -> list[str]:
    base = [
        "h",
        "k",
        "l",
        "source_filename",
        "event",
        "frame",
        "frame_number",
        "chunk_id",
        "crystal_in_chunk",
        "image_serial",
        "risk_column",
        "risk_value",
        "risk_tail",
        "rank_within_hkl",
        "quantile_within_hkl",
    ]
    columns = [column for column in base if column in table.columns]
    columns.extend(column for column in orientation_columns if column in table.columns and column not in columns)
    extras = [
        column
        for column in ["d_angstrom", "q_invA", "sg", "partialator_survived", "partiality", "I_unmerged", "sigma_unmerged", "_score_row"]
        if column in table.columns and column not in columns
    ]
    columns.extend(extras)
    return columns


def write_json(path: Path, records: pd.DataFrame, metadata: dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "records": records.where(pd.notna(records), None).to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_selection(label: str, selected: pd.DataFrame, risk_column: str) -> None:
    view = selected.loc[selected["risk_tail"] == label].copy()
    print(f"{label}-risk selected observations:")
    for row in view.itertuples(index=False):
        print(
            "  "
            f"{row.source_filename} event={row.event} "
            f"hkl=({int(row.h)},{int(row.k)},{int(row.l)}) "
            f"{risk_column}={float(getattr(row, 'risk_value')):.8g}"
        )


def main() -> int:
    args = parse_args()
    header = read_header(args.scores)
    risk_column, risk_candidates = choose_risk_column(header, args.risk_column)
    orientation_source, score_orientation_columns = detect_orientation_columns(header)

    hkl = (
        tuple(args.hkl)
        if args.hkl is not None
        else select_hkl_automatically(args.scores, risk_column, args.min_obs, args.scores_chunksize)
    )
    usecols = score_usecols(header, risk_column, score_orientation_columns)
    scores = load_hkl_scores(args.scores, risk_column, usecols, hkl, args.scores_chunksize)
    hkl_stats_before_survivor_filter = score_range_stats(scores, hkl, risk_column)
    survivor_filter_applied = args.survivor_mask is not None
    survivor_mask_stats: dict[str, Any] = {}
    if survivor_filter_applied:
        survivor_keys, survivor_mask_stats = load_survivor_mask(
            args.survivor_mask,
            args.min_partiality,
            args.scores_chunksize,
            hkl,
        )
        scores_for_selection = apply_survivor_filter(scores, survivor_keys)
    else:
        scores_for_selection = scores
    hkl_stats_after_survivor_filter = score_range_stats(scores_for_selection, hkl, risk_column)
    selected, hkl_stats = select_tail_examples(scores_for_selection, hkl, risk_column, args.n_low, args.n_high)
    selected["risk_column"] = risk_column
    selected["risk_value"] = selected[risk_column]

    selected, orientation_source_used, orientation_columns, orientation_stats = add_orientation(
        selected,
        score_orientation_columns,
        args.stream,
    )
    if orientation_source is not None:
        orientation_source_used = orientation_source

    out_columns = output_columns(selected, risk_column, orientation_columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected.loc[:, out_columns].to_csv(args.out, index=False)

    metadata = {
        "scores": str(args.scores),
        "stream": None if args.stream is None else str(args.stream),
        "risk_column": risk_column,
        "risk_column_candidates": risk_candidates,
        "chosen_hkl": {"h": hkl[0], "k": hkl[1], "l": hkl[2]},
        "hkl_stats": hkl_stats,
        "hkl_stats_before_survivor_filter": hkl_stats_before_survivor_filter,
        "hkl_stats_after_survivor_filter": hkl_stats_after_survivor_filter,
        "survivor_mask": None if args.survivor_mask is None else str(args.survivor_mask),
        "survivor_filter_applied": bool(survivor_filter_applied),
        "min_partiality": None if args.min_partiality is None else float(args.min_partiality),
        **survivor_mask_stats,
        "orientation_source": orientation_source_used,
        "orientation_columns": orientation_columns,
        **orientation_stats,
        "n_low": int(args.n_low),
        "n_high": int(args.n_high),
        "min_obs": int(args.min_obs),
        "output_csv": str(args.out),
        "output_json": None if args.out_json is None else str(args.out_json),
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out_json, selected.loc[:, out_columns], metadata)

    print(f"chosen_hkl: {hkl[0]} {hkl[1]} {hkl[2]}")
    print(f"total_v4_observations_for_hkl: {hkl_stats_before_survivor_filter['n_observations']}")
    print(f"observations_after_survivor_filter: {hkl_stats_after_survivor_filter['n_observations']}")
    print(f"risk_column: {risk_column}")
    print(f"risk_range_before_survivor_filter: {format_range(hkl_stats_before_survivor_filter)}")
    print(f"risk_range_after_survivor_filter: {format_range(hkl_stats_after_survivor_filter)}")
    print(f"orientation_source: {orientation_source_used}")
    print_selection("low", selected, risk_column)
    print_selection("high", selected, risk_column)
    print(f"output_csv: {args.out}")
    if args.out_json is not None:
        print(f"output_json: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

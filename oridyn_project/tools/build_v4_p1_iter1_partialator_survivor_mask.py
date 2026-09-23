#!/usr/bin/env python3
"""Build a v4 exact-key survivor mask from a P1 iter1 partialator unmerged HKL file."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
UNMERGED_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+(\S+))?\s*$")
UNMERGED_FLAGGED_RE = re.compile(r"^\s*Flagged:\s*(\S+)\s*$", re.IGNORECASE)
KEYS_CSV = "v4_p1_iter1_partialator_survivor_keys.csv"
SURVIVOR_SCORES_CSV = "v4_p1_iter1_partialator_survivors_only_scores.csv"
SUMMARY_MD = "v4_p1_iter1_partialator_survivor_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-scores", required=True, type=Path, help="V4 raw local-crowding score CSV")
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator --unmerged-output HKL file")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory for mask and survivor scores")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Rows per v4 score chunk")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    if not args.v4_scores.exists():
        raise SystemExit(f"--v4-scores not found: {args.v4_scores}")
    if not args.unmerged.exists():
        raise SystemExit(f"--unmerged not found: {args.unmerged}")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def output_paths(outdir: Path) -> dict[str, Path]:
    return {
        "keys": outdir / KEYS_CSV,
        "survivor_scores": outdir / SURVIVOR_SCORES_CSV,
        "summary": outdir / SUMMARY_MD,
    }


def ensure_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    blocked = [path for path in paths.values() if path.exists()]
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing output file(s):\n{formatted}\nUse --overwrite if intended.")
    paths["keys"].parent.mkdir(parents=True, exist_ok=True)


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
    out = out.loc[~out[HKL_COLUMNS].isna().any(axis=1)].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def parse_flagged_value(value: str) -> bool:
    return value.strip().lower() in {"yes", "y", "true", "t", "1"}


def parse_unmerged(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns: dict[str, list[Any]] = {
        "source_filename": [],
        "event": [],
        "h": [],
        "k": [],
        "l": [],
        "partialator_survived": [],
        "partiality": [],
        "I_unmerged": [],
    }
    current_source = ""
    current_event = ""
    current_flagged = False
    current_crystal_id: int | str | None = None
    flag_counts: Counter[str] = Counter()
    stats: dict[str, Any] = {
        "crystal_blocks_seen": 0,
        "flagged_crystal_blocks": 0,
        "unmerged_reflection_rows_seen": 0,
        "malformed_unmerged_rows": 0,
        "nonfinite_intensity_rows": 0,
        "nonfinite_partiality_rows": 0,
        "rows_from_flagged_crystals": 0,
        "excluded_flagged_crystal_rows": 0,
        "excluded_row_flag_rows": 0,
        "excluded_partiality_too_small_rows": 0,
        "excluded_other_row_flag_rows": 0,
        "sigma_unmerged_available": False,
    }

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Crystal "):
                stats["crystal_blocks_seen"] += 1
                token = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else ""
                try:
                    current_crystal_id = int(token)
                except ValueError:
                    current_crystal_id = token
                current_source = ""
                current_event = ""
                current_flagged = False
                continue

            filename_match = UNMERGED_FILENAME_RE.match(line)
            if filename_match:
                current_source = normalize_source(filename_match.group(1))
                current_event = normalize_event(filename_match.group(2) or "")
                continue

            flagged_match = UNMERGED_FLAGGED_RE.match(line)
            if flagged_match:
                current_flagged = parse_flagged_value(flagged_match.group(1))
                if current_flagged:
                    stats["flagged_crystal_blocks"] += 1
                continue

            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                intensity = float(parts[3])
                partiality = float(parts[4])
            except ValueError:
                stats["malformed_unmerged_rows"] += 1
                continue

            stats["unmerged_reflection_rows_seen"] += 1
            if not np.isfinite(intensity):
                stats["nonfinite_intensity_rows"] += 1
                continue
            if not np.isfinite(partiality):
                stats["nonfinite_partiality_rows"] += 1
                continue

            row_flags = " ".join(parts[5:]).strip() or "<none>"
            flag_counts[row_flags] += 1
            has_row_flags = row_flags != "<none>"
            if current_flagged:
                stats["rows_from_flagged_crystals"] += 1
                stats["excluded_flagged_crystal_rows"] += 1
                continue
            if has_row_flags:
                stats["excluded_row_flag_rows"] += 1
                if "partiality_too_small" in row_flags.lower():
                    stats["excluded_partiality_too_small_rows"] += 1
                else:
                    stats["excluded_other_row_flag_rows"] += 1
                continue
            columns["source_filename"].append(current_source)
            columns["event"].append(current_event)
            columns["h"].append(h)
            columns["k"].append(k)
            columns["l"].append(l)
            columns["partialator_survived"].append(1)
            columns["partiality"].append(float(partiality))
            columns["I_unmerged"].append(float(intensity))

            if stats["unmerged_reflection_rows_seen"] == 1 or stats["unmerged_reflection_rows_seen"] % 1_000_000 == 0:
                log(
                    "Parsed unmerged reflection rows="
                    f"{stats['unmerged_reflection_rows_seen']:,} current_crystal={current_crystal_id} line={line_number:,}"
                )

    table = pd.DataFrame(columns)
    table = normalize_key_columns(table)
    table[HKL_COLUMNS] = table[HKL_COLUMNS].astype("int32")
    table["partialator_survived"] = table["partialator_survived"].astype("int8")
    stats["unmerged_reflection_rows_kept"] = int(len(table))
    stats["row_flag_counts"] = dict(flag_counts)
    return table, stats


def compact_mask(table: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    stats["survivor_key_rows_before_dedup"] = int(len(table))
    duplicate_mask = table.duplicated(KEY_COLUMNS, keep=False)
    stats["duplicate_survivor_key_rows"] = int(duplicate_mask.sum())
    stats["duplicate_survivor_keys"] = int(table.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_mask.any() else 0
    if duplicate_mask.any():
        table = table.sort_values(
            [*KEY_COLUMNS, "partiality"],
            ascending=[True, True, True, True, True, False],
            kind="mergesort",
        ).drop_duplicates(KEY_COLUMNS, keep="first")
    table = table.reset_index(drop=True)
    table["_mask_row_id"] = np.arange(len(table), dtype=np.int64)
    stats["survivor_unique_keys"] = int(len(table))
    return table


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def require_v4_columns(header: list[str]) -> None:
    missing = [column for column in KEY_COLUMNS if column not in header]
    if missing:
        raise SystemExit(f"--v4-scores is missing required exact-key column(s): {missing}")


def key_payload_columns(mask: pd.DataFrame) -> list[str]:
    columns = [*KEY_COLUMNS, "partialator_survived"]
    for optional in ["partiality", "I_unmerged", "sigma_unmerged"]:
        if optional in mask.columns:
            columns.append(optional)
    return columns


def update_unique_hkls(target: set[tuple[int, int, int]], table: pd.DataFrame) -> None:
    if table.empty:
        return
    target.update((int(h), int(k), int(l)) for h, k, l in table.loc[:, HKL_COLUMNS].itertuples(index=False, name=None))


def stream_survivor_scores(
    v4_scores: Path,
    mask: pd.DataFrame,
    out_path: Path,
    chunksize: int,
    overwrite: bool,
) -> dict[str, Any]:
    if out_path.exists() and overwrite:
        out_path.unlink()

    payload_columns = [*key_payload_columns(mask), "_mask_row_id"]
    mask_payload = mask.loc[:, payload_columns].copy()
    matched_mask_rows = np.zeros(len(mask_payload), dtype=bool)
    unique_hkls_v4: set[tuple[int, int, int]] = set()
    unique_hkls_survivors: set[tuple[int, int, int]] = set()
    total_v4 = 0
    valid_v4_key_rows = 0
    bad_v4_key_rows = 0
    survivor_rows = 0
    header_written = False

    for chunk_index, chunk in enumerate(pd.read_csv(v4_scores, chunksize=int(chunksize)), start=1):
        total_v4 += int(len(chunk))
        clean = normalize_key_columns(chunk)
        valid_v4_key_rows += int(len(clean))
        bad_v4_key_rows += int(len(chunk) - len(clean))
        update_unique_hkls(unique_hkls_v4, clean)
        joined = clean.merge(mask_payload, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
        if not joined.empty:
            matched_ids = joined["_mask_row_id"].to_numpy(dtype=np.int64)
            matched_mask_rows[matched_ids] = True
            survivor_rows += int(len(joined))
            update_unique_hkls(unique_hkls_survivors, joined)
            output = joined.drop(columns=["_mask_row_id"])
            output.to_csv(out_path, index=False, mode="a", header=not header_written)
            header_written = True
        if chunk_index == 1 or chunk_index % 5 == 0:
            log(
                f"Matched v4 chunks={chunk_index:,} rows_read={total_v4:,} "
                f"survivor_rows={survivor_rows:,}"
            )

    if not header_written:
        empty_columns = [*read_header(v4_scores), *[column for column in key_payload_columns(mask) if column not in KEY_COLUMNS]]
        pd.DataFrame(columns=empty_columns).to_csv(out_path, index=False)

    matched_unique_keys = int(matched_mask_rows.sum())
    unmatched_unmerged_keys = int(len(mask_payload) - matched_unique_keys)
    return {
        "total_v4_observations": int(total_v4),
        "valid_v4_key_rows": int(valid_v4_key_rows),
        "bad_v4_key_rows": int(bad_v4_key_rows),
        "survivor_observations": int(survivor_rows),
        "survivor_fraction": float(survivor_rows / total_v4) if total_v4 else np.nan,
        "unique_signed_hkls_in_v4": int(len(unique_hkls_v4)),
        "unique_signed_hkls_among_survivors": int(len(unique_hkls_survivors)),
        "exact_match_count": int(survivor_rows),
        "matched_unique_unmerged_keys": matched_unique_keys,
        "unmatched_unmerged_keys": unmatched_unmerged_keys,
        "v4_rows_without_survivor_match": int(total_v4 - survivor_rows),
        "matched_mask_rows": matched_mask_rows,
    }


def format_number(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.8g}"
    return str(value)


def partiality_quantiles(mask: pd.DataFrame, matched_mask_rows: np.ndarray) -> dict[str, float]:
    if "partiality" not in mask.columns:
        return {}
    values = pd.to_numeric(mask.loc[matched_mask_rows, "partiality"], errors="coerce").dropna()
    if values.empty:
        return {}
    qs = values.quantile([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    return {f"q{int(round(q * 100)):02d}": float(value) for q, value in qs.items()}


def write_summary(
    path: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
    parse_stats: dict[str, Any],
    match_stats: dict[str, Any],
    mask: pd.DataFrame,
) -> None:
    quantiles = partiality_quantiles(mask, match_stats["matched_mask_rows"])
    lines = [
        "# V4 P1 Iter1 Partialator Survivor Mask Summary",
        "",
        f"- v4 score CSV: `{args.v4_scores}`",
        f"- partialator unmerged HKL: `{args.unmerged}`",
        "- partialator context: symmetry 1 / P1, iterations 1, unmerged output, no PR; diagnostic survivor/partiality mask only",
        "- exact key: `source_filename + normalized event + signed h,k,l`",
        "- symmetry handling: signed HKLs preserved; no 4/mmm canonicalization",
        "- v4 score handling: raw scores only; no normalization or log1p transform",
        "- stream handling: no stream filtering or stream writing",
        "",
        "## Detected Unmerged Format",
        "- crystal header: `Crystal <id>`",
        "- source/event header: `Filename: <source_filename> //<event>`",
        "- reflection row columns: `h k l I_unmerged partiality [row_flags]`",
        "- sigma/esd column: not present in this unmerged output",
        "- accepted survivor rule: `Flagged: no` crystal block and no row flags",
        "- rejected rows: `Flagged: yes` crystal blocks and rows flagged `partiality_too_small` or any other row flag",
        "",
        "## Outputs",
        f"- survivor keys CSV: `{paths['keys']}`",
        f"- survivors-only v4 scores CSV: `{paths['survivor_scores']}`",
        f"- summary markdown: `{paths['summary']}`",
        "",
        "## Match Summary",
        f"- total v4 observations: {match_stats['total_v4_observations']:,}",
        f"- survivor observations: {match_stats['survivor_observations']:,}",
        f"- survivor fraction: {format_number(match_stats['survivor_fraction'])}",
        f"- unique signed HKLs in v4: {match_stats['unique_signed_hkls_in_v4']:,}",
        f"- unique signed HKLs among survivors: {match_stats['unique_signed_hkls_among_survivors']:,}",
        f"- exact match count: {match_stats['exact_match_count']:,}",
        f"- unmatched unmerged keys: {match_stats['unmatched_unmerged_keys']:,}",
        f"- v4 rows without survivor match: {match_stats['v4_rows_without_survivor_match']:,}",
        f"- matched unique unmerged keys: {match_stats['matched_unique_unmerged_keys']:,}",
        f"- valid v4 key rows: {match_stats['valid_v4_key_rows']:,}",
        f"- bad v4 key rows: {match_stats['bad_v4_key_rows']:,}",
        "",
        "## Unmerged Parse Summary",
        f"- crystal blocks seen: {parse_stats['crystal_blocks_seen']:,}",
        f"- flagged crystal blocks: {parse_stats['flagged_crystal_blocks']:,}",
        f"- unmerged reflection rows seen: {parse_stats['unmerged_reflection_rows_seen']:,}",
        f"- unmerged reflection rows kept: {parse_stats['unmerged_reflection_rows_kept']:,}",
        f"- survivor unique keys: {parse_stats['survivor_unique_keys']:,}",
        f"- duplicate survivor key rows: {parse_stats['duplicate_survivor_key_rows']:,}",
        f"- duplicate survivor keys: {parse_stats['duplicate_survivor_keys']:,}",
        f"- rows from flagged crystals: {parse_stats['rows_from_flagged_crystals']:,}",
        f"- excluded flagged-crystal rows: {parse_stats['excluded_flagged_crystal_rows']:,}",
        f"- excluded row-flag rows: {parse_stats['excluded_row_flag_rows']:,}",
        f"- excluded partiality_too_small rows: {parse_stats['excluded_partiality_too_small_rows']:,}",
        f"- excluded other row-flag rows: {parse_stats['excluded_other_row_flag_rows']:,}",
        f"- malformed unmerged rows: {parse_stats['malformed_unmerged_rows']:,}",
        f"- nonfinite intensity rows: {parse_stats['nonfinite_intensity_rows']:,}",
        f"- nonfinite partiality rows: {parse_stats['nonfinite_partiality_rows']:,}",
    ]
    if quantiles:
        lines.extend(["", "## Partiality Quantiles", "| quantile | partiality |", "| --- | --- |"])
        for name, value in quantiles.items():
            lines.append(f"| {name} | {value:.8g} |")
    row_flag_counts = parse_stats.get("row_flag_counts", {})
    if row_flag_counts:
        lines.extend(["", "## Unmerged Row Flags", "| row flags | rows |", "| --- | ---: |"])
        for flag, count in sorted(row_flag_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{flag}` | {int(count):,} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = output_paths(args.outdir)
    ensure_outputs(paths, bool(args.overwrite))
    header = read_header(args.v4_scores)
    require_v4_columns(header)

    log("Parsing partialator unmerged HKL")
    mask, parse_stats = parse_unmerged(args.unmerged)
    mask = compact_mask(mask, parse_stats)
    key_columns = key_payload_columns(mask)
    log(f"Writing survivor key mask: {paths['keys']}")
    mask.loc[:, key_columns].to_csv(paths["keys"], index=False)

    log("Streaming v4 score table and writing survivors-only score CSV")
    match_stats = stream_survivor_scores(args.v4_scores, mask, paths["survivor_scores"], int(args.chunksize), bool(args.overwrite))

    log(f"Writing summary: {paths['summary']}")
    write_summary(paths["summary"], args, paths, parse_stats, match_stats, mask)
    log("Done")
    print(f"survivor_keys_csv: {paths['keys']}")
    print(f"survivors_only_scores_csv: {paths['survivor_scores']}")
    print(f"summary_md: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
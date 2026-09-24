"""XDS file parsing for the cRED validation workflow."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


INPUT_FILENAMES = (
    "XDS.INP",
    "XPARM.XDS",
    "GXPARM.XDS",
    "INTEGRATE.HKL",
    "INTEGRATE.LP",
    "CORRECT.LP",
    "XDS_ASCII.HKL",
)


@dataclass(frozen=True)
class XdsTableHeader:
    """Header metadata for an XDS reflection table."""

    path: Path
    lines: list[str]
    columns: list[str]
    metadata: dict[str, Any]


def read_bang_header(path: Path) -> list[str]:
    """Read XDS bang-prefixed header lines through END_OF_HEADER."""

    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!"):
                break
            clean = line.rstrip("\n")
            lines.append(clean)
            if clean.strip() == "!END_OF_HEADER":
                break
    return lines


def parse_key_value_header(lines: list[str]) -> dict[str, Any]:
    """Parse simple XDS header key/value records."""

    metadata: dict[str, Any] = {}
    for raw in lines:
        text = raw[1:].strip() if raw.startswith("!") else raw.strip()
        if "=" not in text:
            continue
        left, right = text.split("=", 1)
        key = left.strip()
        value = right.strip()
        if not key:
            continue
        first_key = key
        first_value = value
        if " " in key:
            parts = key.split()
            first_key = parts[0]
            first_value = " ".join(parts[1:] + [value])
        metadata[first_key] = _coerce_value(first_value)
        for match in re.finditer(r"([A-Z0-9_().'/-]+)\s*=\s*([^=]+?)(?=\s+[A-Z0-9_().'/-]+\s*=|$)", text):
            metadata[match.group(1)] = _coerce_value(match.group(2).strip())
    return metadata


def _coerce_value(value: str) -> Any:
    """Coerce a short XDS value to numbers when that is unambiguous."""

    clean = value.split("!")[0].strip()
    if not clean:
        return ""
    parts = clean.split()
    converted: list[Any] = []
    numeric = True
    for part in parts:
        try:
            number = float(part)
        except ValueError:
            numeric = False
            break
        converted.append(int(number) if number.is_integer() else number)
    if numeric:
        return converted[0] if len(converted) == 1 else converted
    return clean


def parse_integrate_header(path: Path) -> XdsTableHeader:
    """Parse the self-describing INTEGRATE.HKL header."""

    lines = read_bang_header(path)
    metadata = parse_key_value_header(lines)
    n_items = int(metadata.get("NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD", 0))
    item_columns = _columns_from_item_records(lines)
    columns = item_columns or _columns_from_comma_header(lines, n_items)
    if n_items and len(columns) != n_items:
        raise ValueError(f"{path} declares {n_items} items but {len(columns)} columns were parsed.")
    return XdsTableHeader(path=path, lines=lines, columns=columns, metadata=metadata)


def parse_xds_ascii_header(path: Path) -> XdsTableHeader:
    """Parse XDS_ASCII.HKL header, preferring ITEM_* records."""

    lines = read_bang_header(path)
    metadata = parse_key_value_header(lines)
    n_items = int(metadata.get("NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD", 0))
    columns = _columns_from_item_records(lines) or _columns_from_comma_header(lines, n_items)
    if n_items and len(columns) != n_items:
        raise ValueError(f"{path} declares {n_items} items but {len(columns)} columns were parsed.")
    return XdsTableHeader(path=path, lines=lines, columns=columns, metadata=metadata)


def _columns_from_item_records(lines: list[str]) -> list[str]:
    """Return columns from ITEM_NAME=index header records."""

    numbered: dict[int, str] = {}
    for raw in lines:
        text = raw[1:].strip() if raw.startswith("!") else raw.strip()
        match = re.match(r"ITEM_([^=]+)\s*=\s*(\d+)", text)
        if match:
            numbered[int(match.group(2))] = _sanitize_column_name(match.group(1))
    return [numbered[idx] for idx in sorted(numbered)] if numbered else []


def _columns_from_comma_header(lines: list[str], n_items: int) -> list[str]:
    """Return columns from INTEGRATE.HKL comma-separated header records."""

    columns: list[str] = []
    collecting = False
    for raw in lines:
        text = raw[1:].strip() if raw.startswith("!") else raw.strip()
        if text.startswith("H,K,L"):
            collecting = True
        if not collecting:
            continue
        if text.startswith("Items are") or text == "END_OF_HEADER":
            break
        parts = [part.strip() for part in text.split(",") if part.strip()]
        columns.extend(_sanitize_column_name(part) for part in parts)
        if n_items and len(columns) >= n_items:
            break
    return columns


def _sanitize_column_name(name: str) -> str:
    """Normalize XDS column names to stable identifiers."""

    clean = name.strip().replace("SIGMA(IOBS)", "SIGMA")
    clean = clean.replace("/", "_over_")
    clean = clean.replace("(", "").replace(")", "")
    return clean


def count_integrate_records(path: Path, max_observations: int | None = None) -> int:
    """Count data records in INTEGRATE.HKL without loading them."""

    count = 0
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not in_data:
                if line.startswith("!END_OF_HEADER"):
                    in_data = True
                continue
            if not line.strip() or line.startswith("!"):
                continue
            count += 1
            if max_observations is not None and count >= max_observations:
                break
    return count


def iter_integrate_chunks(
    path: Path,
    columns: list[str],
    chunk_size: int,
    max_observations: int | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield finite-IOBS observation chunks and exclusion rows."""

    rows: list[list[float]] = []
    excluded: list[dict[str, Any]] = []
    observation_id = 0
    seen = 0
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not in_data:
                if line.startswith("!END_OF_HEADER"):
                    in_data = True
                continue
            if not line.strip() or line.startswith("!"):
                continue
            if max_observations is not None and seen >= max_observations:
                break
            seen += 1
            parts = line.split()
            if len(parts) != len(columns):
                excluded.append(
                    {
                        "observation_id": observation_id,
                        "line_number": line_number,
                        "exclusion_reason": f"malformed_record_{len(parts)}_fields",
                        "raw_record": line.strip(),
                    }
                )
                observation_id += 1
                continue
            try:
                values = [float(part) for part in parts]
            except ValueError:
                excluded.append(
                    {
                        "observation_id": observation_id,
                        "line_number": line_number,
                        "exclusion_reason": "non_numeric_record",
                        "raw_record": line.strip(),
                    }
                )
                observation_id += 1
                continue
            iobs_idx = columns.index("IOBS")
            if not np.isfinite(values[iobs_idx]):
                excluded.append(
                    {
                        "observation_id": observation_id,
                        "line_number": line_number,
                        "exclusion_reason": "nonfinite_IOBS",
                        "raw_record": line.strip(),
                    }
                )
                observation_id += 1
                continue
            rows.append([observation_id, *values])
            observation_id += 1
            if len(rows) >= chunk_size:
                yield _chunk_dataframe(rows, columns), pd.DataFrame.from_records(excluded)
                rows = []
                excluded = []
    if rows or excluded:
        yield _chunk_dataframe(rows, columns), pd.DataFrame.from_records(excluded)


def read_integrate_subset(path: Path, columns: list[str], limit: int | None) -> pd.DataFrame:
    """Read a deterministic finite-IOBS subset from INTEGRATE.HKL."""

    chunks: list[pd.DataFrame] = []
    read_count = 0
    chunk_size = max(1, min(limit or 1000, 1000))
    for chunk, _excluded in iter_integrate_chunks(path, columns, chunk_size, limit):
        chunks.append(chunk)
        read_count += len(chunk)
        if limit is not None and read_count >= limit:
            break
    if not chunks:
        return pd.DataFrame(columns=["observation_id", *columns])
    return pd.concat(chunks, ignore_index=True).head(limit)


def _chunk_dataframe(rows: list[list[float]], columns: list[str]) -> pd.DataFrame:
    """Build a typed observation DataFrame from parsed rows."""

    df = pd.DataFrame(rows, columns=["observation_id", *columns])
    for col in ("observation_id", "H", "K", "L", "ISEG"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def parse_xds_inp(path: Path) -> dict[str, Any]:
    """Parse simple KEY=VALUE records from XDS.INP."""

    out: dict[str, Any] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = raw.split("!")[0].strip()
            if "=" not in text:
                continue
            for match in re.finditer(r"([A-Z0-9_().'/-]+)\s*=\s*([^=]+?)(?=\s+[A-Z0-9_().'/-]+\s*=|$)", text):
                out[match.group(1)] = _coerce_value(match.group(2).strip())
    return out


def parse_resolution_shells(correct_lp: Path, include_range: tuple[float, float] | None = None) -> pd.DataFrame:
    """Parse final XDS resolution shell limits from CORRECT.LP."""

    lines = correct_lp.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [idx for idx, line in enumerate(lines) if "SUBSET OF INTENSITY DATA" in line and "AS FUNCTION OF RESOLUTION" in line]
    if not starts:
        return pd.DataFrame()
    start = starts[-1]
    limits: list[float] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("total"):
            break
        parts = stripped.replace("*", "").replace("%", "").split()
        if len(parts) >= 2:
            try:
                limit = float(parts[0])
                observed = int(float(parts[1]))
            except ValueError:
                continue
            if observed >= 0:
                limits.append(limit)
    if not limits:
        return pd.DataFrame()
    low_outer = include_range[0] if include_range else float("inf")
    records: list[dict[str, Any]] = []
    previous_outer = low_outer
    for idx, inner in enumerate(limits, start=1):
        records.append(
            {
                "shell": idx,
                "d_outer_angstrom": previous_outer,
                "d_inner_angstrom": inner,
                "source": "CORRECT.LP final resolution-limit table",
            }
        )
        previous_outer = inner
    return pd.DataFrame.from_records(records)


def assign_resolution_shell(resolution: float, shells: pd.DataFrame) -> int | float:
    """Return XDS shell index for a resolution value, or NaN when unavailable."""

    if shells.empty or not np.isfinite(resolution):
        return np.nan
    for row in shells.itertuples(index=False):
        outer = float(row.d_outer_angstrom)
        inner = float(row.d_inner_angstrom)
        if resolution <= outer and resolution >= inner:
            return int(row.shell)
    return np.nan


def parse_xds_ascii_matches(path: Path) -> tuple[dict[tuple[int, int, int, float], dict[str, Any]], dict[str, Any]]:
    """Parse XDS_ASCII.HKL rows for conservative matching diagnostics."""

    header = parse_xds_ascii_header(path)
    if not {"H", "K", "L"} <= set(header.columns):
        return {}, {"matched_status_available": False, "reason": "missing H/K/L columns"}
    z_column = "ZD" if "ZD" in header.columns else "ZCAL" if "ZCAL" in header.columns else None
    if z_column is None:
        return {}, {"matched_status_available": False, "reason": "missing ZD/ZCAL column"}
    records: dict[tuple[int, int, int, float], dict[str, Any]] = {}
    duplicates: set[tuple[int, int, int, float]] = set()
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not in_data:
                if line.startswith("!END_OF_HEADER"):
                    in_data = True
                continue
            if not line.strip() or line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) != len(header.columns):
                continue
            row = {col: float(value) for col, value in zip(header.columns, parts, strict=True)}
            key = (int(row["H"]), int(row["K"]), int(row["L"]), round(float(row[z_column]), 1))
            if key in records:
                duplicates.add(key)
            else:
                records[key] = {
                    "correct_status": "accepted_in_XDS_ASCII",
                    "corrected_IOBS": row.get("IOBS"),
                    "corrected_SIGMA": row.get("SIGMA"),
                }
    for key in duplicates:
        records.pop(key, None)
    return records, {
        "matched_status_available": True,
        "match_key": f"H,K,L,round({z_column},1)",
        "unique_match_count": len(records),
        "duplicate_match_keys": len(duplicates),
    }


def input_file_manifest(input_dir: Path) -> dict[str, Any]:
    """Return paths, sizes, and hashes for important XDS inputs."""

    manifest: dict[str, Any] = {}
    for filename in INPUT_FILENAMES:
        path = input_dir / filename
        if path.exists():
            manifest[filename] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            manifest[filename] = {"path": str(path), "missing": True}
    return manifest


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Return SHA256 hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_input_manifest(path: Path, input_dir: Path) -> None:
    """Write an input manifest JSON file."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(input_file_manifest(input_dir), handle, indent=2, sort_keys=True)
        handle.write("\n")

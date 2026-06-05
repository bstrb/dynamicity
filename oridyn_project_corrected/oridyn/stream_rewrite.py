"""Rewrite CrystFEL stream reflection sigmas from OriDyn scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd


STREAM_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)")
STREAM_SERIAL_RE = re.compile(r"^\s*Image serial number:\s*(\d+)")


@dataclass(frozen=True)
class RewriteStats:
    """Summary of stream sigma rewriting."""

    chunks_seen: int
    crystals_seen: int
    reflection_rows_seen: int
    reflection_rows_rewritten: int
    missing_score_rows: int
    duplicate_score_keys: int
    sigma_dyn_rel_cap: float | None
    backup_path: str | None


def rewrite_stream_sigmas(
    stream_path: str | Path,
    scores_path: str | Path,
    output_path: str | Path,
    sigma_column: str = "sigma_dyn",
    sigma_dyn_rel_cap: float | None = None,
    make_backup: bool = False,
) -> RewriteStats:
    """Write a stream copy with reflection sigma values replaced by score output."""

    stream = Path(stream_path)
    scores = pd.read_csv(scores_path)
    score_lookup, duplicate_count = _build_score_lookup(scores, sigma_column, sigma_dyn_rel_cap)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if make_backup and output.exists():
        backup = output.with_suffix(output.suffix + ".bak")
        shutil.copy2(output, backup)
        backup_path = str(backup)

    stats = _rewrite_lines(stream, output, score_lookup)
    return RewriteStats(
        chunks_seen=stats["chunks_seen"],
        crystals_seen=stats["crystals_seen"],
        reflection_rows_seen=stats["reflection_rows_seen"],
        reflection_rows_rewritten=stats["reflection_rows_rewritten"],
        missing_score_rows=stats["missing_score_rows"],
        duplicate_score_keys=duplicate_count,
        sigma_dyn_rel_cap=sigma_dyn_rel_cap,
        backup_path=backup_path,
    )


def _build_score_lookup(
    scores: pd.DataFrame,
    sigma_column: str,
    sigma_dyn_rel_cap: float | None,
) -> tuple[dict[tuple[int, int, int, int], float], int]:
    required = {"frame", "h", "k", "l", sigma_column}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Scores table is missing required columns: {missing}")
    table = scores.copy()
    table["_sigma_rewrite"] = pd.to_numeric(table[sigma_column], errors="coerce")
    if sigma_dyn_rel_cap is not None:
        if "sigma" not in table.columns or "sigma_dyn_rel" not in table.columns:
            raise ValueError("--sigma-dyn-rel-cap requires score columns 'sigma' and 'sigma_dyn_rel'.")
        rel = pd.to_numeric(table["sigma_dyn_rel"], errors="coerce").clip(upper=float(sigma_dyn_rel_cap))
        sigma = pd.to_numeric(table["sigma"], errors="coerce")
        table["_sigma_rewrite"] = sigma * rel
    table = table.dropna(subset=["frame", "h", "k", "l", "_sigma_rewrite"])
    duplicate_count = int(table.duplicated(["frame", "h", "k", "l"], keep=False).sum())
    table = table.drop_duplicates(["frame", "h", "k", "l"], keep="first")
    lookup: dict[tuple[int, int, int, int], float] = {}
    for row in table[["frame", "h", "k", "l", "_sigma_rewrite"]].itertuples(index=False, name=None):
        frame, h, k, l, sigma_value = row
        value = float(sigma_value)
        if not np.isfinite(value) or value <= 0.0:
            continue
        lookup[(int(frame), int(h), int(k), int(l))] = value
    return lookup, duplicate_count


def _rewrite_lines(
    stream_path: Path,
    output_path: Path,
    score_lookup: dict[tuple[int, int, int, int], float],
) -> dict[str, int]:
    in_chunk = False
    in_crystal = False
    in_reflections = False
    chunk_id = -1
    crystal_counter = 0
    reflection_rows_seen = 0
    reflection_rows_rewritten = 0
    missing_score_rows = 0
    out_lines: list[str] = []

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("----- Begin chunk -----"):
                in_chunk = True
                in_crystal = False
                in_reflections = False
                chunk_id += 1
                out_lines.append(line)
                continue
            if line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                out_lines.append(line)
                continue
            if line.startswith("--- Begin crystal"):
                in_crystal = True
                in_reflections = False
                out_lines.append(line)
                continue
            if line.startswith("--- End crystal"):
                if in_crystal:
                    crystal_counter += 1
                in_crystal = False
                in_reflections = False
                out_lines.append(line)
                continue
            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                out_lines.append(line)
                continue
            if in_reflections and "End of reflections" in line:
                in_reflections = False
                out_lines.append(line)
                continue
            if in_chunk and in_crystal and in_reflections:
                rewritten = _rewrite_reflection_line(line, crystal_counter, score_lookup)
                if rewritten is not None:
                    reflection_rows_seen += 1
                    if rewritten != line:
                        reflection_rows_rewritten += 1
                    else:
                        missing_score_rows += 1
                    out_lines.append(rewritten)
                    continue
            out_lines.append(line)

    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {
        "chunks_seen": chunk_id + 1,
        "crystals_seen": crystal_counter,
        "reflection_rows_seen": reflection_rows_seen,
        "reflection_rows_rewritten": reflection_rows_rewritten,
        "missing_score_rows": missing_score_rows,
    }


def _rewrite_reflection_line(
    line: str,
    frame: int,
    score_lookup: dict[tuple[int, int, int, int], float],
) -> str | None:
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
    value = score_lookup.get((int(frame), h, k, l))
    if value is None:
        return line
    parts[4] = _format_sigma(value)
    return _format_reflection_parts(parts)


def _format_sigma(value: float) -> str:
    return f"{float(value):.6g}"


def _format_reflection_parts(parts: list[str]) -> str:
    if len(parts) >= 10:
        return (
            f"{int(parts[0]):4d} {int(parts[1]):4d} {int(parts[2]):4d} "
            f"{float(parts[3]):12.6g} {parts[4]:>12s} "
            f"{float(parts[5]):10.6g} {float(parts[6]):10.6g} "
            f"{float(parts[7]):10.4f} {float(parts[8]):10.4f} {parts[9]}"
        )
    return " ".join(parts)

"""Extraction of XDS scale and correction diagnostics."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd


def parse_integrate_scales(path: Path) -> pd.DataFrame:
    """Parse image scaling rows from INTEGRATE.LP."""

    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_table = False
    for line in lines:
        if line.strip().startswith("IMAGE IER") and "SCALE" in line:
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped:
            in_table = False
            continue
        parts = stripped.split()
        if len(parts) < 9:
            continue
        try:
            image = int(parts[0])
            ier = int(parts[1])
        except ValueError:
            in_table = False
            continue
        records.append(
            {
                "source": "INTEGRATE.LP",
                "image": image,
                "ier": ier,
                "scale": float(parts[2]),
                "nbkg": int(parts[3]),
                "novl": int(parts[4]),
                "newald": int(parts[5]),
                "nstrong": int(parts[6]),
                "nrej": int(parts[7]),
                "sigmab": float(parts[8]),
                "sigmar": float(parts[9]) if len(parts) > 9 else None,
                "interpretation": "INTEGRATE image scaling factor; kept separate from CORRECT corrections.",
            }
        )
    return pd.DataFrame.from_records(records)


def parse_correct_scales(correct_lp: Path, input_dir: Path) -> pd.DataFrame:
    """Parse reliable CORRECT scale/correction metadata and factor tables."""

    records: list[dict[str, Any]] = []
    lines = correct_lp.read_text(encoding="utf-8", errors="replace").splitlines()
    records.extend(_parse_spindle_position_factors(lines))
    records.extend(_parse_correction_sections(lines))
    for filename in ("DECAY.cbf", "MODPIX.cbf", "ABSORP.cbf"):
        path = input_dir / filename
        if path.exists():
            records.append(_parse_cbf_correction_header(path))
    return pd.DataFrame.from_records(records)


def _parse_spindle_position_factors(lines: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    in_table = False
    for line in lines:
        if line.strip().startswith("INTERVAL") and "FACTOR" in line:
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped:
            if records:
                break
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        try:
            interval_start = float(parts[0])
            interval_end = float(parts[1])
            number = int(parts[2])
            intensity = float(parts[3])
            factor = float(parts[4])
        except ValueError:
            continue
        records.append(
            {
                "source": "CORRECT.LP",
                "scale_type": "spindle_position_within_image",
                "interval_start": interval_start,
                "interval_end": interval_end,
                "number": number,
                "intensity": intensity,
                "factor": factor,
                "interpretation": "CORRECT shutter-position diagnostic factor; not treated as INTEGRATE image scale.",
            }
        )
    return records


def _parse_correction_sections(lines: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if "CORRECTION FACTORS AS FUNCTION" not in line:
            continue
        section = line.strip()
        window = "\n".join(lines[idx : idx + 24])
        source_file = _regex_value(r"visual inspection by XDS-Viewer\s+(\S+)", window)
        total = _regex_value(r"TOTAL NUMBER OF CORRECTION FACTORS DEFINED\s+(\d+)", window)
        chi2 = _regex_value(r"CHI\^2-VALUE OF FIT OF CORRECTION FACTORS\s+([0-9.Ee+-]+)", window)
        records.append(
            {
                "source": "CORRECT.LP",
                "scale_type": "correction_grid_metadata",
                "correction_function": section,
                "source_file": source_file,
                "total_factors": int(total) if total else None,
                "chi2_fit": float(chi2) if chi2 else None,
                "interpretation": (
                    "CORRECT image/detector correction metadata. Binary CBF grid values are not decoded "
                    "in this first workflow."
                ),
            }
        )
    return records


def _parse_cbf_correction_header(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    text = data[:2048].decode("utf-8", errors="replace")
    header = text.split("_array_data.data", 1)[0]
    row: dict[str, Any] = {
        "source": path.name,
        "scale_type": "cbf_correction_grid_header",
        "source_file": path.name,
        "interpretation": "XDS correction-factor CBF header only; binary byte-offset grid values not decoded.",
    }
    for key in ("INPUT_FILE", "XMIN", "XMAX", "YMIN", "YMAX", "NXBIN", "NYBIN"):
        value = _regex_value(rf"{key}=\s*([^\n]+)", header)
        if value:
            row[key.lower()] = value.strip()
    for key in ("X-Binary-Number-of-Elements", "X-Binary-Size-Fastest-Dimension", "X-Binary-Size-Second-Dimension"):
        value = _regex_value(rf"{key}:\s*([0-9]+)", text)
        if value:
            row[key.lower().replace("-", "_")] = int(value)
    if "REC. CORRECTION FACTORS" in header:
        row["correction_function"] = _regex_value(r"REC\. CORRECTION FACTORS AS FUNCTION OF ([^\n]+)", header)
    return row


def _regex_value(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None

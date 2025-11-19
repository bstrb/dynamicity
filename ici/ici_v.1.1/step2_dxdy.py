#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_dxdy.py — STRICT Step-2 proposer that ONLY uses
'per_frame_dx_dy.csv' in the event directory.

Behavior:
- Read event_dir/per_frame_dx_dy.csv
- Expect columns 'dx' and 'dy' (case-insensitive)
- Use the LAST row (dx, dy) verbatim as the next proposal
- If anything is missing/invalid, return (None, None, <reason>)
- Absolutely NO fallbacks, NO clamping, NO spacing repair.

Contract:
    propose_step2_dxdy(successes_w, failures, tried, R, min_spacing_mm, event_dir, cfg)
        -> (dx_mm, dy_mm, reason)

The extra arguments are ignored (kept for API symmetry).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import csv
import math


PER_FRAME_DXDY = "per_frame_dx_dy.csv"


@dataclass
class Step2DxDyConfig:
    # expected column names (case-insensitive)
    col_dx: str = "dx"
    col_dy: str = "dy"


def _lower_list(xs: List[str]) -> List[str]:
    return [x.strip().lower() for x in xs]


def _parse_last_dxdy(csv_path: str, col_dx: str, col_dy: str) -> Tuple[Optional[float], Optional[float], str]:
    """Return (dx, dy, reason). On success, reason is 'dxdy_refined_from(per_frame_dx_dy.csv)'. Otherwise (None,None,<reason>)."""
    if not os.path.isfile(csv_path):
        return None, None, "dxdy_missing_csv"

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception:
        return None, None, "dxdy_parse_error"

    if not rows:
        return None, None, "dxdy_empty_csv"

    # Detect header (case-insensitive presence of col names)
    header = _lower_list(rows[0])
    has_header = (any(c.isalpha() for c in rows[0])) and (col_dx.lower() in header and col_dy.lower() in header)

    if not has_header:
        return None, None, f"dxdy_missing_columns({col_dx},{col_dy})"

    i_dx = header.index(col_dx.lower())
    i_dy = header.index(col_dy.lower())

    if len(rows) <= 1:
        return None, None, "dxdy_empty_csv"

    last = rows[-1]
    if max(i_dx, i_dy) >= len(last):
        return None, None, f"dxdy_missing_columns({col_dx},{col_dy})"

    try:
        dx = float(last[i_dx]); dy = float(last[i_dy])
    except Exception:
        return None, None, "dxdy_parse_error"

    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None, None, "dxdy_nonfinite(dx,dy)"

    return 1e3*float(dx), 1e3*float(dy), f"dxdy_refined_from({PER_FRAME_DXDY})"


def propose_step2_dxdy(
    successes_w,              # unused
    failures,                 # unused
    tried,                    # unused
    R,                        # unused
    min_spacing_mm,           # unused
    event_dir: str,
    cfg: Step2DxDyConfig,
    **kwargs,                 # accept extra args without breaking
) -> Tuple[Optional[float], Optional[float], str]:
    """
    STRICT: Only read per_frame_dx_dy.csv and return its last (dx, dy).
    No modifications, no fallbacks.
    """
    csv_path = os.path.join(event_dir, PER_FRAME_DXDY)
    return _parse_last_dxdy(csv_path, cfg.col_dx, cfg.col_dy)

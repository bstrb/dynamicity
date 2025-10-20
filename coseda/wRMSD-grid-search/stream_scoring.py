#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stream_scoring.py
Single-chunk scoring utilities.

Preferred path:
- Import your existing lowest_wrmsd module and call a helper function
  wrmsd_for_single_chunk(stream_path_or_text) which returns a dict.

If that helper is not present, we fall back to a regex-based parse of
a CrystFEL .stream chunk to extract wRMSD and a few metrics. The fallback
is conservative and may need small adjustments depending on your stream flavor.
"""

from __future__ import annotations
import os
import re
from typing import Dict, Optional

# Try to import your existing scorer if present
_WRMSD_HELPER = None
try:
    # assuming lowest_wrmsd.py is importable on PYTHONPATH
    import lowest_wrmsd as _lw

    if hasattr(_lw, "wrmsd_for_single_chunk"):
        _WRMSD_HELPER = _lw.wrmsd_for_single_chunk
except Exception:
    _WRMSD_HELPER = None


_WRMSD_PATTERNS = [
    # try common phrasings; adjust if needed to your exact output wording
    re.compile(r"\bweighted\s+RMSD\b[^0-9]*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"\bwRMSD\b[^0-9]*([0-9]*\.?[0-9]+)", re.IGNORECASE),
]

_NREFL_PATTERNS = [
    re.compile(r"\breflections\b[^0-9]*([0-9]+)", re.IGNORECASE),
    re.compile(r"\bn_reflections\b[^0-9]*([0-9]+)", re.IGNORECASE),
]

_NPEAKS_PATTERNS = [
    re.compile(r"\bn_peaks\b[^0-9]*([0-9]+)", re.IGNORECASE),
    re.compile(r"\bpeaks\b[^0-9]*([0-9]+)", re.IGNORECASE),
]

_CELLDEV_PATTERNS = [
    re.compile(r"\bcell\s*deviation\b[^0-9\-+]*([\-+]?[0-9]*\.?[0-9]+)\s*%?", re.IGNORECASE),
    re.compile(r"\bcell_dev_pct\b[^0-9\-+]*([\-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
]


def _regex_find_first(patterns, text: str) -> Optional[float]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


def score_single_chunk_text(chunk_text: str) -> Dict[str, float]:
    """
    Regex fallback scoring from a single chunk's text.
    Returns dict with at least {"wrmsd": float}; other fields optional.
    """
    wr = _regex_find_first(_WRMSD_PATTERNS, chunk_text)
    if wr is None:
        raise ValueError("Could not find wRMSD in stream chunk text.")
    nrefl = _regex_find_first(_NREFL_PATTERNS, chunk_text)
    npeaks = _regex_find_first(_NPEAKS_PATTERNS, chunk_text)
    celldev = _regex_find_first(_CELLDEV_PATTERNS, chunk_text)

    out = {"wrmsd": float(wr)}
    if nrefl is not None:
        out["n_reflections"] = int(nrefl)
    if npeaks is not None:
        out["n_peaks"] = int(npeaks)
    if celldev is not None:
        out["cell_dev_pct"] = float(celldev)
    return out


def score_single_chunk_stream_file(stream_path: str) -> Dict[str, float]:
    """
    Prefer lowest_wrmsd.wrmsd_for_single_chunk if available.
    Else parse the last chunk in the file (caller should pass a 1-chunk file
    or pre-sliced text). We'll add a text-slice variant in stream_extract.py.
    """
    if _WRMSD_HELPER is not None:
        return _WRMSD_HELPER(stream_path)

    with open(stream_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # naive split by chunk delimiter (CrystFEL-style)
    parts = re.split(r"-{5,}\s*Begin\s+chunk\s*-{5,}", text, flags=re.IGNORECASE)
    if len(parts) <= 1:
        # maybe the whole file is one chunk without explicit separators
        return score_single_chunk_text(text)

    # take the last chunk text (after the last 'Begin chunk')
    chunk_text = parts[-1]
    return score_single_chunk_text(chunk_text)

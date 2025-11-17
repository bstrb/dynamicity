#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lowest_wrmsd.py  (new)
Single-chunk scoring helper for CrystFEL .stream content.

This module provides:
  - wrmsd_for_single_chunk(stream_path_or_text)
      -> {"wrmsd": float,
          "n_reflections": Optional[int],
          "n_peaks": Optional[int],
          "cell_dev_pct": Optional[float]}

Behavior:
- If given a file path, it will:
    * Read the file text
    * If multiple chunks are present, it uses the **last** chunk
      (the caller should pass a 1-chunk winner file or a specific slice).
- If given a text blob (not a path), it treats the blob as a **single chunk**
  and parses it directly.

Notes:
- We parse metrics from the chunk text using robust regex patterns that match
  common CrystFEL stream phrasings (wRMSD, reflections, peaks, cell deviation).
- If wRMSD cannot be found, a ValueError is raised.

You can swap this implementation with a more physics-faithful scorer later if
you’d like to compute wRMSD from reflection lists directly; the public API
is kept minimal for that purpose.
"""

from __future__ import annotations
import os
import re
from typing import Dict, Optional, Union


# --------- Chunk boundary detection (for file inputs) ---------

BEGIN_CHUNK_RE = re.compile(r"-{3,}\s*Begin\s+chunk\s*-{3,}", re.IGNORECASE)
END_CHUNK_RE   = re.compile(r"-{3,}\s*End\s+chunk\s*-{3,}", re.IGNORECASE)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_header_and_body(text: str):
    m = BEGIN_CHUNK_RE.search(text)
    if not m:
        return text, ""
    return text[:m.start()], text[m.start():]


def _iter_chunks_with_spans(text: str):
    pos = 0
    L = len(text)
    while True:
        m_begin = BEGIN_CHUNK_RE.search(text, pos)
        if not m_begin:
            return
        m_end = END_CHUNK_RE.search(text, m_begin.end())
        if not m_end:
            yield (m_begin.start(), L)
            return
        yield (m_begin.start(), m_end.end())
        pos = m_end.end()


# --------- Metric regex (robust to wording) ---------

WRMSD_PATTERNS = [
    re.compile(r"\bweighted\s+RMSD\b[^0-9]*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"\bwRMSD\b[^0-9]*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"\bRMSD\b[^0-9]*([0-9]*\.?[0-9]+)", re.IGNORECASE),  # fallback (may catch plain RMSD)
]

NREFL_PATTERNS = [
    re.compile(r"\bn[_\s-]*reflections\b[^0-9]*([0-9]+)", re.IGNORECASE),
    re.compile(r"\breflections\b[^0-9]*([0-9]+)", re.IGNORECASE),
]

NPEAKS_PATTERNS = [
    re.compile(r"\bn[_\s-]*peaks\b[^0-9]*([0-9]+)", re.IGNORECASE),
    re.compile(r"\bpeaks\b[^0-9]*([0-9]+)", re.IGNORECASE),
]

CELLDEV_PATTERNS = [
    re.compile(r"\bcell[_\s-]*dev(?:iation)?(?:\s*percent|\s*%|_pct)?\b[^0-9\-+]*([\-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
]


def _find_first_float(patterns, text: str) -> Optional[float]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


def _find_first_int(patterns, text: str) -> Optional[int]:
    val = _find_first_float(patterns, text)
    return int(val) if val is not None else None


def _score_chunk_text(chunk_text: str) -> Dict[str, float]:
    wr = _find_first_float(WRMSD_PATTERNS, chunk_text)
    if wr is None:
        raise ValueError("wRMSD not found in stream chunk.")
    nrefl = _find_first_int(NREFL_PATTERNS, chunk_text)
    npeaks = _find_first_int(NPEAKS_PATTERNS, chunk_text)
    celldev = _find_first_float(CELLDEV_PATTERNS, chunk_text)

    out: Dict[str, float] = {"wrmsd": float(wr)}
    if nrefl is not None:
        out["n_reflections"] = int(nrefl)
    if npeaks is not None:
        out["n_peaks"] = int(npeaks)
    if celldev is not None:
        out["cell_dev_pct"] = float(celldev)
    return out


def wrmsd_for_single_chunk(stream_path_or_text: Union[str, bytes]) -> Dict[str, float]:
    """
    Public API.

    If 'stream_path_or_text' is a path to an existing file:
        - Read it, find chunks; if multiple, use the **last** chunk.
    Else:
        - Treat it as the raw chunk text.

    Returns:
        dict with {"wrmsd": float, "n_reflections": int|None, "n_peaks": int|None, "cell_dev_pct": float|None}
    """
    # Heuristic: if argument is a path to an existing file, read it; otherwise treat as text
    if isinstance(stream_path_or_text, (str, bytes)) and isinstance(stream_path_or_text, str) and os.path.exists(stream_path_or_text):
        text = _read_text(stream_path_or_text)
        # If file has no explicit chunk bounds, score whole text
        header, body = _split_header_and_body(text)
        if not body:
            return _score_chunk_text(text)
        # Take last chunk
        last_span = None
        for span in _iter_chunks_with_spans(text):
            last_span = span
        if last_span is None:
            return _score_chunk_text(text)
        s, e = last_span
        return _score_chunk_text(text[s:e])

    # Treat as text
    if isinstance(stream_path_or_text, bytes):
        text = stream_path_or_text.decode("utf-8", errors="ignore")
    else:
        text = str(stream_path_or_text)

    return _score_chunk_text(text)

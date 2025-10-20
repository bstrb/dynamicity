#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stream_extract.py

Utilities to:
- Iterate and parse CrystFEL .stream files into chunks
- Identify a specific image's chunk (by overlay HDF5 path and image index)
- Extract a single chunk + header into a per-image winner .stream
- Merge many per-image winner streams into a single merged .stream

Assumptions (tunable via regex):
- Chunks are delimited by lines like:
    ----- Begin chunk -----
    ...
    ----- End chunk -----
- Each chunk contains metadata lines including:
    Image filename: /abs/path/to/overlays/<src>.overlay.h5
    Event: <int>            (often the per-image index)
  Some stream variants also include a "Image index:" or "Image number:" field;
  we include regex for common alternatives.

Robustness:
- If your stream variant differs, adjust the regex patterns near the top.
- All functions use UTF-8 with 'ignore' errors to be resilient to odd bytes.
"""

from __future__ import annotations
import io
import os
import re
from typing import Iterator, Tuple, Optional, List


# ---------- Regex patterns (adjust if your stream differs) ----------

BEGIN_CHUNK_RE = re.compile(r"-{3,}\s*Begin\s+chunk\s*-{3,}", re.IGNORECASE)
END_CHUNK_RE   = re.compile(r"-{3,}\s*End\s+chunk\s*-{3,}", re.IGNORECASE)

# Common field for file path:
IMAGE_FILE_RE = re.compile(r"^\s*Image\s+filename\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# Common fields for event/index (we try several):
EVENT_RE       = re.compile(r"^\s*Event\s*:\s*([0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)
IMG_INDEX_RE   = re.compile(r"^\s*Image\s+(?:index|number)\s*:\s*([0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)
SEL_IDX_ORDER  = [EVENT_RE, IMG_INDEX_RE]  # try Event first; fall back to Image index/number


# ---------- Low-level: header + chunk iteration ----------

def split_header_and_body(text: str) -> Tuple[str, str]:
    """
    Return (header_text, body_text) where header_text is everything before the first
    'Begin chunk' delimiter, and body_text is the rest (possibly empty).
    """
    m = BEGIN_CHUNK_RE.search(text)
    if not m:
        # If no explicit chunk delimiter, treat whole text as 'header' (rare)
        return text, ""
    return text[:m.start()], text[m.start():]


def iter_chunks_with_spans(text: str) -> Iterator[Tuple[int, int]]:
    """
    Yield (start_offset, end_offset) byte offsets (character offsets in Python str)
    for each chunk including its Begin/End delimiters.
    """
    pos = 0
    L = len(text)
    while True:
        m_begin = BEGIN_CHUNK_RE.search(text, pos)
        if not m_begin:
            return
        m_end = END_CHUNK_RE.search(text, m_begin.end())
        if not m_end:
            # If no matching end, take until end-of-file
            yield (m_begin.start(), L)
            return
        # Include entire line of the end marker
        yield (m_begin.start(), m_end.end())
        pos = m_end.end()


def read_file_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------- Metadata extraction per chunk ----------

def _find_first_int(patterns: List[re.Pattern], text: str) -> Optional[int]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def parse_chunk_metadata(chunk_text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Attempt to extract (image_filename, image_index) from a single chunk's text.
    Returns (filename or None, index or None).
    """
    file_match = IMAGE_FILE_RE.search(chunk_text)
    img_path = file_match.group(1).strip() if file_match else None
    idx = _find_first_int(SEL_IDX_ORDER, chunk_text)
    return img_path, idx


# ---------- Find a chunk by (overlay_path, image_idx) ----------

def _normalize_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    try:
        return os.path.abspath(p)
    except Exception:
        return p


def find_chunk_span_for_image(stream_path: str, overlay_path: str, image_idx: int) -> Optional[Tuple[int, int]]:
    """
    Return (start, end) character offsets for the chunk corresponding to
    (overlay_path, image_idx) if found; else None.
    """
    overlay_path = _normalize_path(overlay_path)
    text = read_file_text(stream_path)
    # quick split to skip header
    _, body = split_header_and_body(text)

    for (s, e) in iter_chunks_with_spans(text):
        chunk = text[s:e]
        img_path, idx = parse_chunk_metadata(chunk)
        if img_path is None or idx is None:
            continue
        if _normalize_path(img_path) == overlay_path and idx == image_idx:
            return (s, e)
    return None


# ---------- Extract a single chunk into a 1-chunk .stream ----------

def extract_winner_stream(stream_in: str, out_stream: str, chunk_span: Tuple[int, int]) -> None:
    """
    Write a new .stream file containing the global header from stream_in
    and exactly one chunk defined by chunk_span (start,end) in stream_in.
    """
    text = read_file_text(stream_in)
    header, _ = split_header_and_body(text)

    s, e = chunk_span
    chunk = text[s:e]

    # Ensure parent folder exists
    os.makedirs(os.path.dirname(os.path.abspath(out_stream)), exist_ok=True)
    with open(out_stream, "w", encoding="utf-8") as f:
        # Write header first (as-is)
        f.write(header)
        if not header.endswith("\n"):
            f.write("\n")
        # Then the single chunk
        f.write(chunk)
        if not chunk.endswith("\n"):
            f.write("\n")


# ---------- Helper: get header + first chunk offset of a .stream ----------

def get_stream_header_and_first_chunk_span(stream_path: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Return (header_text, (start,end) for the first chunk) or (header_text, None) if no chunk.
    """
    text = read_file_text(stream_path)
    header, body = split_header_and_body(text)
    it = iter(iter_chunks_with_spans(text))
    try:
        span = next(it)
    except StopIteration:
        span = None
    return header, span


# ---------- Merge many 1-chunk streams into a merged .stream ----------

def merge_winner_streams(winner_paths: List[str], out_stream: str) -> None:
    """
    Merge N per-image winner streams (each with its own header + one chunk)
    into a single merged .stream:
      - Copy header from the first file.
      - Append only the chunk blocks from each file (skip their headers).
    """
    if not winner_paths:
        raise ValueError("No winner streams to merge.")

    os.makedirs(os.path.dirname(os.path.abspath(out_stream)), exist_ok=True)

    # Read header and chunk from first file
    first = winner_paths[0]
    header, span = get_stream_header_and_first_chunk_span(first)
    if span is None:
        raise ValueError(f"First winner stream has no chunk: {first}")

    with open(out_stream, "w", encoding="utf-8") as out_f:
        out_f.write(header)
        if not header.endswith("\n"):
            out_f.write("\n")

        # Append first chunk
        t = read_file_text(first)
        out_f.write(t[span[0]:span[1]])
        if not t[span[0]:span[1]].endswith("\n"):
            out_f.write("\n")

        # Append remaining chunks
        for path in winner_paths[1:]:
            ht, sp = get_stream_header_and_first_chunk_span(path)
            if sp is None:
                continue  # skip empty streams safely
            tx = read_file_text(path)
            out_f.write(tx[sp[0]:sp[1]])
            if not tx[sp[0]:sp[1]].endswith("\n"):
                out_f.write("\n")

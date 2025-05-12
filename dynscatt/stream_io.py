
# =====================================================================
# file: stream_io.py  ── CrystFEL stream‑file parser (single‑chunk‑fast)
# =====================================================================
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd

__all__ = ["load_stream"]

# Regular expressions --------------------------------------------------
_HEADER_CELL = re.compile(r"^a\s*=\s*([0-9.]+).*?b\s*=\s*([0-9.]+).*?c\s*=\s*([0-9.]+).*?al\s*=\s*([0-9.]+).*?be\s*=\s*([0-9.]+).*?ga\s*=\s*([0-9.]+)", re.MULTILINE)
_CHUNK_START = re.compile(r"^----- Begin chunk -----", re.MULTILINE)
_REF_START = re.compile(r"^Reflections measured after indexing", re.MULTILINE)
_REF_LINE = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+([-0-9.]+)\s+([0-9.]+)")


DREF_DTYPE = np.dtype([
    ("h", "i4"), ("k", "i4"), ("l", "i4"),
    ("I", "f8"), ("sigI", "f8")
])


def load_stream(path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse a CrystFEL stream file *containing exactly one chunk* and return
    *(df, meta)* compatible with the rest of the dynscatt pipeline."""
    text = Path(path).read_text()
    meta: Dict[str, Any] = {"source": str(path), "format": "stream"}

    # -- unit cell -----------------------------------------------------
    m_cell = _HEADER_CELL.search(text)
    if m_cell:
        meta["cell"] = tuple(float(x) for x in m_cell.groups())

    # -- locate reflections table -------------------------------------
    m_ref = _REF_START.search(text)
    if not m_ref:
        raise RuntimeError("No 'Reflections measured after indexing' section found")
    lines: List[str] = []
    for line in text[m_ref.end():].splitlines():
        if line.strip().startswith("End of reflections"):
            break
        if _REF_LINE.match(line):
            lines.append(line)

    # parse table lines ------------------------------------------------
    arr = np.zeros(len(lines), dtype=DREF_DTYPE)
    for i, ln in enumerate(lines):
        mh, mk, ml, mi, msig = _REF_LINE.match(ln).groups()
        arr[i] = (int(mh), int(mk), int(ml), float(mi), float(msig))
    df = pd.DataFrame.from_records(arr)

    return df, meta

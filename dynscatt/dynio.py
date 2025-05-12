
# =====================================================================
# file: dynio.py  ── *dispatch*: choose correct backend loader
# =====================================================================
from __future__ import annotations

import h5py
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd

from stream_io import load_stream

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_frame(path: str | Path, * , refl_path: str | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a single‑pattern data set from either an HDF5 (X‑gandalf/NanoEDT) **or** a
    CrystFEL *.stream* file.  Returns *(df, meta)* where *df* has at least
    columns `h k l I` (plus others) and *meta* holds misc. information.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".h5" or suffix == ".hdf5":
        return _load_h5(path, refl_path=refl_path)
    elif suffix == ".stream":
        return load_stream(path)  # ⇐ delegated to stream_io.py
    else:
        raise RuntimeError(f"Unsupported file type: {path}")

# ---------------------------------------------------------------------
# Internal: HDF5 loader (unchanged except factored out)
# ---------------------------------------------------------------------
import numpy as np

_REFLECT_DTYPE = np.dtype([
    ("h", "i4"), ("k", "i4"), ("l", "i4"),
    ("I", "f8"), ("sigI", "f8"), ("peak", "f8"), ("bkg", "f8"),
    ("fs_px", "f8"), ("ss_px", "f8"), ("panel", "S2")
])


def _find_reflection_table(h5: h5py.File):
    for name, ds in h5["/"].items():
        if isinstance(ds, h5py.Dataset) and {"h", "k", "l"}.issubset(ds.dtype.names or {}):
            return ds
    raise RuntimeError("No reflection table found – please point me to it with --refl-path")


def _load_h5(path: Path, refl_path: str | None = None):
    meta: Dict[str, Any] = {"source": str(path), "format": "hdf5"}
    with h5py.File(path, "r") as h5:
        # 1) reflections -------------------------------------------------
        ds = h5[refl_path] if refl_path else _find_reflection_table(h5)
        df = pd.DataFrame.from_records(ds[...].astype(_REFLECT_DTYPE))
        # 2) grab cell if available ------------------------------------
        for key in ("/crystal/cell", "/entry/crystal_1/cell"):
            if key in h5:
                meta["cell"] = tuple(float(x) for x in h5[key][...])
                break
    return df, meta
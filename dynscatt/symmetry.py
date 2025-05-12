
# =====================================================================
# file: symmetry.py  ── cctbx‑preferred symmetry operations
# =====================================================================
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Dict

import pandas as pd

try:
    from cctbx import sgtbx  # type: ignore
    _USE_CCTBX = True
except ImportError:
    import gemmi
    _USE_CCTBX = False

# Helper to apply 3×3 integer rotation to hkl --------------------------

def _apply(rot: Tuple[int,...], h:int,k:int,l:int) -> Tuple[int,int,int]:
    return (rot[0]*h+rot[1]*k+rot[2]*l,
            rot[3]*h+rot[4]*k+rot[5]*l,
            rot[6]*h+rot[7]*k+rot[8]*l)

# Generate symmetry ops ------------------------------------------------
@lru_cache(maxsize=64)
def _ops(sg: str) -> List[Tuple[int,...]]:
    if _USE_CCTBX:
        g = sgtbx.space_group_info(sg).group()
        rots: List[Tuple[int,...]] = []
        for op in g.all_ops():
            # op.r().as_double() returns a *flat* tuple (r11,r12,...,r33)
            flat = tuple(int(round(v)) for v in op.r().as_double())
            rots.append(flat)
        return rots
    # Gemmi fallback ---------------------------------------------------
    return [tuple(op.rot.flatten()) for op in gemmi.SpaceGroup(sg).operations()]

# Canonical representative --------------------------------------------

def _canonical(h:int,k:int,l:int, ops:List[Tuple[int,...]]) -> Tuple[int,int,int]:
    return min(_apply(r,h,k,l) for r in ops)

# Public API -----------------------------------------------------------

def assign_equiv_classes(df: pd.DataFrame, sg: str) -> pd.DataFrame:
    ops = _ops(sg)
    mapping: Dict[Tuple[int,int,int], int] = {}
    equiv_ids: List[int] = []
    next_id = 0
    for h,k,l in df[["h","k","l"]].itertuples(index=False, name=None):
        key = _canonical(h,k,l,ops)
        if key not in mapping:
            mapping[key] = next_id; next_id += 1
        equiv_ids.append(mapping[key])
    out = df.copy(); out["equiv_id"] = equiv_ids; return out

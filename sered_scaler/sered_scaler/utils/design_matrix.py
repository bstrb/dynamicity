"""
Sparse one-hot construction helpers for millions-row regression.
Keep this module standalone—no imports from other sered_scaler sub-packages.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from pandas import Series

__all__ = ["one_hot", "stack"]


def one_hot(key: Series, n_levels: int) -> sparse.csr_matrix:  # noqa: N802
    """Return CSR one-hot matrix (N × n_levels) for integer *key*."""
    row = np.arange(len(key))
    return sparse.csr_matrix(
        (np.ones_like(row), (row, key.values)), shape=(len(key), n_levels)
    )


# simple re-export so callers can do:  X = stack([A, B])
stack = sparse.hstack

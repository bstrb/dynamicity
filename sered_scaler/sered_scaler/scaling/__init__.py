
# ================================================================
# file: sered_scaler/scaling/__init__.py
# ================================================================

"""Public re‑exports for scaling helpers."""
from .provisional import provisional_scale  # noqa: F401
from .kinematic_filter import zscore_filter  # noqa: F401
from .final import weighted_merge            # noqa: F401

# Bayesian mixture (optional – requires ``sered_scaler[bayes]`` extras)
try:
    from .bayesian_filter import mixture_filter  # noqa: F401
except ModuleNotFoundError:  # PyMC not installed
    mixture_filter = None  # type: ignore

__all__ = [
    "provisional_scale",
    "zscore_filter",
    "mixture_filter",
    "weighted_merge",
]

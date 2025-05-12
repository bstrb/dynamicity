# ───────────────────────── file: sered_scaler/__init__.py (unchanged) ─────────────────────────
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # package not installed
    __version__ = "0.0.dev0"

from .io.stream_to_dataframe import stream_to_dfs   # convenience re‑export
from .scaling.provisional import provisional_scale  # noqa
from .scaling.kinematic_filter import zscore_filter # noqa
from .scaling.final import weighted_merge           # noqa

__all__ = [
    "stream_to_dfs",
    "provisional_scale",
    "zscore_filter",
    "weighted_merge",
]
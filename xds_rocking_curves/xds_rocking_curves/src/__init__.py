"""Top-level package for XDS-based rocking-curve reconstruction."""

from .parsers import GXPARMData, IntegrateData, SpotData, UnitCell, XDSInputData
from .pipeline import AnalysisConfig, analyze_single_reflection_dataset

__all__ = [
    "GXPARMData",
    "IntegrateData",
    "SpotData",
    "UnitCell",
    "XDSInputData",
    "AnalysisConfig",
    "analyze_single_reflection_dataset",
]

__version__ = "0.1.0"

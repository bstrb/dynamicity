"""Top-level package for orientation-aware SerialED weighting."""

from .geometry import UnitCell
from .pipeline import (
    PipelineConfig,
    PipelineResults,
    export_results,
    run_pipeline_from_tables,
    run_serialed_csv_pipeline,
    run_xds_pipeline,
)

__all__ = [
    "UnitCell",
    "PipelineConfig",
    "PipelineResults",
    "run_pipeline_from_tables",
    "run_serialed_csv_pipeline",
    "run_xds_pipeline",
    "export_results",
]

__version__ = "0.1.0"

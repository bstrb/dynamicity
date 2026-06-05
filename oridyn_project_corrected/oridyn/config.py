"""Configuration dataclasses for OriDyn."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


KernelName = Literal["gaussian", "lorentzian"]
NormalizationName = Literal["median_mad", "percentile", "rank", "none"]


@dataclass(frozen=True)
class ScoreWeights:
    """Weights used to combine normalized geometry-only risk terms."""

    self: float = 1.0
    graph: float = 1.0
    zone: float = 0.75
    row: float = 0.75
    frame: float = 0.75
    interaction: float = 0.25

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class OridynConfig:
    """Runtime configuration for the first-version OriDyn workflow.

    The score path intentionally excludes detector coordinates, observed
    intensities, peak/background quantities, and downstream residuals.
    """

    dmin: float = 0.6
    dmax: float = 20.0
    hkl_limit: int | None = None
    max_candidates: int | None = 200_000
    centering: str | None = None
    uvw_max: int = 5
    max_problematic_axes: int | None = 50
    axis_score_min: float = 0.0
    axis_sigma_deg: float = 2.0
    sg0: float = 0.01
    excitation_kernel: KernelName = "gaussian"
    excitation_lorentzian_power: float = 2.0
    low_order_g0_invA: float = 0.40
    low_order_power: float = 1.5
    neighbor_excitation_min: float = 0.05
    neighbor_hkl_radius: int = 3
    max_neighbors_per_reflection: int = 64
    max_excited_nodes_per_frame: int = 2000
    row_direction_limit: int = 5
    row_max_steps: int = 12
    normalization: NormalizationName = "median_mad"
    frame_normalization: NormalizationName | None = None
    resolution_shells: int = 10
    normalization_clip: float = 6.0
    alpha: float = 1.0
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    export_candidates: bool = False
    workers: int = 1
    chunk_size: int = 50
    seed: int = 0
    progress: bool = True
    beam_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    exposure_samples: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["weights"] = self.weights.to_dict()
        return payload

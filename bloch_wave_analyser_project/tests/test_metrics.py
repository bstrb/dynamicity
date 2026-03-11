from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import (
    effective_coupling_multiplicity,
    thickness_sensitivity_metrics,
    two_beam_metric,
)


def test_two_beam_metric_sanity() -> None:
    sg = np.asarray([0.0, 0.01])
    xi = np.asarray([100.0, 100.0])
    values = two_beam_metric(sg, xi)
    assert np.isclose(values[0], 1.0)
    assert values[1] < values[0]


def test_effective_coupling_multiplicity_identity_modes() -> None:
    eigenvectors = np.eye(3)
    n_eff = effective_coupling_multiplicity(eigenvectors, beam_index=1)
    assert np.isclose(n_eff, 1.0)


def test_thickness_sensitivity_metrics() -> None:
    metrics = thickness_sensitivity_metrics(np.asarray([1.0, 2.0, 4.0]))
    assert metrics["thickness_std"] > 0.0
    assert metrics["thickness_cv"] > 0.0
    assert np.isclose(metrics["thickness_max_min_ratio"], 4.0)
    assert metrics["thickness_normalized_range"] > 0.0

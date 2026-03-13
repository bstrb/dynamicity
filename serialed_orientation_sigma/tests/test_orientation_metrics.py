from __future__ import annotations

import numpy as np
import pandas as pd

from src.geometry import UnitCell, cell_to_reciprocal_matrix, rotation_matrix_from_axis_angle
from src.orientation_metrics import OrientationMetricConfig, compute_orientation_metrics


def _orientation_table() -> pd.DataFrame:
    cell = UnitCell(a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    ub0 = cell_to_reciprocal_matrix(cell)
    ub1 = rotation_matrix_from_axis_angle([0.0, 1.0, 0.0], 20.0) @ ub0
    return pd.DataFrame(
        {
            "frame": [1, 2],
            "phi": [0.0, 20.0],
            "UB11": [ub0[0, 0], ub1[0, 0]],
            "UB12": [ub0[0, 1], ub1[0, 1]],
            "UB13": [ub0[0, 2], ub1[0, 2]],
            "UB21": [ub0[1, 0], ub1[1, 0]],
            "UB22": [ub0[1, 1], ub1[1, 1]],
            "UB23": [ub0[1, 2], ub1[1, 2]],
            "UB31": [ub0[2, 0], ub1[2, 0]],
            "UB32": [ub0[2, 1], ub1[2, 1]],
            "UB33": [ub0[2, 2], ub1[2, 2]],
        }
    )


def test_compute_orientation_metrics_returns_expected_columns() -> None:
    cell = UnitCell(a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    summary = compute_orientation_metrics(
        _orientation_table(),
        cell,
        OrientationMetricConfig(dmin=5.0, dmax=20.0, sg_threshold=0.05, zone_axis_limit=2, hkl_limit=3),
    )
    assert list(summary["frame"]) == [1, 2]
    assert {"nearest_zone_axis", "zone_axis_angle", "N_excited", "excitation_density"}.issubset(summary.columns)
    assert np.all(summary["N_excited"] >= 0)
    assert np.all((summary["excitation_density"] >= 0.0) & (summary["excitation_density"] <= 1.0))


def test_identity_orientation_has_small_zone_axis_angle() -> None:
    cell = UnitCell(a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    summary = compute_orientation_metrics(
        _orientation_table().iloc[[0]].copy(),
        cell,
        OrientationMetricConfig(dmin=5.0, dmax=20.0, sg_threshold=0.05, zone_axis_limit=2, hkl_limit=3),
    )
    assert np.isclose(summary.iloc[0]["zone_axis_angle"], 0.0, atol=1e-8)

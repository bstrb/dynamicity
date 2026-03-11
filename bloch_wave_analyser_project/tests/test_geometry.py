from __future__ import annotations

import numpy as np

from src.geometry import build_zone_axes, nearest_zone_axis, rodrigues_rotation_matrix


def test_rodrigues_rotation_matrix_z_90() -> None:
    rotation = rodrigues_rotation_matrix([0.0, 0.0, 1.0], 90.0)
    vector = rotation @ np.asarray([1.0, 0.0, 0.0])
    assert np.allclose(vector, [0.0, 1.0, 0.0], atol=1e-12)


def test_nearest_zone_axis_identity() -> None:
    zone_axes = build_zone_axes(limit=2)
    real_space_reference = np.eye(3)
    match = nearest_zone_axis(zone_axes, real_space_reference, np.eye(3))
    assert match.axis == (0, 0, 1)
    assert np.isclose(match.angle_deg, 0.0)

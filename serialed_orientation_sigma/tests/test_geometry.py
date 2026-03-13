from __future__ import annotations

import numpy as np

from src.geometry import (
    UnitCell,
    cell_to_direct_matrix,
    cell_to_reciprocal_matrix,
    generate_hkl_grid,
    nearest_zone_axis,
    rotation_matrix_from_axis_angle,
)


def test_direct_and_reciprocal_matrices_are_consistent() -> None:
    cell = UnitCell(a=10.0, b=12.0, c=15.0, alpha=90.0, beta=90.0, gamma=90.0)
    direct = cell_to_direct_matrix(cell)
    reciprocal = cell_to_reciprocal_matrix(cell)
    assert np.allclose(direct @ reciprocal.T, np.eye(3), atol=1e-12)


def test_nearest_zone_axis_for_identity_orientation() -> None:
    cell = UnitCell(a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    ub = cell_to_reciprocal_matrix(cell)
    zone, angle = nearest_zone_axis(ub, zone_axis_limit=2)
    assert zone == (0, 0, 1)
    assert np.isclose(angle, 0.0, atol=1e-8)


def test_rotation_matrix_from_axis_angle() -> None:
    rot = rotation_matrix_from_axis_angle([0.0, 1.0, 0.0], 90.0)
    rotated = rot @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, np.array([0.0, 0.0, -1.0]), atol=1e-8)


def test_generate_hkl_grid_contains_low_order_reflections() -> None:
    cell = UnitCell(a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    hkls, d = generate_hkl_grid(cell, dmin=5.0, dmax=20.0, hkl_limit=3)
    assert hkls.shape[0] > 0
    assert any(np.array_equal(hkl, np.array([1, 0, 0])) for hkl in hkls)
    assert np.all(d >= 5.0)

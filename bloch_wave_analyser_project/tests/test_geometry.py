from __future__ import annotations

import numpy as np
import pandas as pd

from src.geometry import (
    ReciprocalMatrixOrientationModel,
    build_zone_axes,
    nearest_zone_axis,
    nearest_zone_axis_from_reciprocal_matrix,
    reciprocal_lookup_from_table,
    rodrigues_rotation_matrix,
)


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


def test_nearest_zone_axis_from_reciprocal_matrix_matches_real_space_convention() -> None:
    zone_axes = build_zone_axes(limit=2)
    real_space_reference = np.diag([10.0, 11.0, 12.0])
    reciprocal_reference = np.linalg.inv(real_space_reference)
    rotation = rodrigues_rotation_matrix([0.0, 1.0, 0.0], 90.0)
    reciprocal_frame = rotation @ reciprocal_reference

    from_rotation = nearest_zone_axis(zone_axes, real_space_reference, rotation)
    from_reciprocal = nearest_zone_axis_from_reciprocal_matrix(zone_axes, reciprocal_frame)

    assert from_rotation.axis == (1, 0, 0)
    assert from_reciprocal.axis == from_rotation.axis
    assert np.isclose(from_reciprocal.angle_deg, from_rotation.angle_deg, atol=1e-12)


def test_reciprocal_orientation_model_from_table() -> None:
    rot = rodrigues_rotation_matrix([0.0, 0.0, 1.0], 90.0)
    table = pd.DataFrame(
        {
            "frame": [1, 2],
            "UB11": [1.0, rot[0, 0]],
            "UB12": [0.0, rot[0, 1]],
            "UB13": [0.0, rot[0, 2]],
            "UB21": [0.0, rot[1, 0]],
            "UB22": [1.0, rot[1, 1]],
            "UB23": [0.0, rot[1, 2]],
            "UB31": [0.0, rot[2, 0]],
            "UB32": [0.0, rot[2, 1]],
            "UB33": [1.0, rot[2, 2]],
        }
    )
    lookup = reciprocal_lookup_from_table(table)
    model = ReciprocalMatrixOrientationModel(
        reciprocal_by_frame=lookup,
        reciprocal_reference=np.eye(3),
    )
    frame1_rot = model.rotation_matrix(frame_index=1)
    assert np.allclose(frame1_rot, rot, atol=1e-12)

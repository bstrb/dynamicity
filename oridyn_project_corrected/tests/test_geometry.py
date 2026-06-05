import numpy as np

from oridyn.geometry import (
    axis_angle_deg,
    direct_matrix_from_cell,
    excitation_error,
    hkl_lab_vectors,
    reciprocal_matrix_from_cell,
)
from oridyn.stream_parser import UnitCell


def test_cubic_cell_reciprocal_mapping():
    cell = UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    direct = direct_matrix_from_cell(cell)
    reciprocal = reciprocal_matrix_from_cell(cell)

    assert np.allclose(np.diag(direct), [10.0, 10.0, 10.0])
    assert np.allclose(np.diag(reciprocal), [0.1, 0.1, 0.1])
    assert np.allclose(hkl_lab_vectors((1, 0, 0), reciprocal)[0], [0.1, 0.0, 0.0])


def test_axis_angle_and_excitation_error():
    reciprocal = np.eye(3)

    assert axis_angle_deg(reciprocal, (0, 0, 1)) == 0.0
    assert np.isclose(axis_angle_deg(reciprocal, (1, 0, 0)), 90.0)
    assert np.isclose(excitation_error([[0.0, 0.0, 0.0]], 1.0)[0], 0.0)

from __future__ import annotations

import numpy as np

from src.parsers import (
    STREAM_MATRIX_COLUMNS,
    crystfel_stream_to_analysis_inputs,
    parse_composition,
    parse_crystfel_stream_text,
    parse_gxparm_text,
    parse_integrate_text,
    parse_xds_inp_text,
)


GXPARM_TEXT = """\
XPARM.XDS generated for testing
1 10.0 0.5 0 1 0
0.0251 0 0 1
1 10.0 11.0 12.0 90.0 90.0 90.0
10.0 0.0 0.0
0.0 11.0 0.0
0.0 0.0 12.0
1 2048 2048 0.055 0.055
1024.0 1024.0 200.0
"""

INTEGRATE_TEXT = """\
!FORMAT=XDS_ASCII    MERGE=FALSE
!END_OF_HEADER
1 0 0 1000.0 10.0 0 0 0.2 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 500.0 5.0 0 0 1.8 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 1 250.0 4.0 0 0 3.1 0 0 0 0 0 0 0 0 0 0 0 0 0
"""

STREAM_TEXT = """\
CrystFEL stream format 2.3
----- Begin geometry file -----
wavelength  = 0.019687 A
clen = 0.485 m
res = 17857.14285714286
p0/min_ss = 0
p0/max_ss = 1023
p0/min_fs = 0
p0/max_fs = 1023
p0/corner_x = -512
p0/corner_y = -512
----- End geometry file -----
----- Begin unit cell -----
lattice_type = tetragonal
unique_axis = c
centering = I
a = 15.12 A
b = 15.12 A
c = 12.08 A
al = 90.00 deg
be = 90.00 deg
ga = 90.00 deg
----- End unit cell -----
----- Begin chunk -----
Event: //41-1
Image serial number: 42
average_camera_length = 0.485000 m
--- Begin crystal
Cell parameters 1.48440 1.54798 1.20092 nm, 89.67116 89.75841 89.95805 deg
astar = +0.4582728 +0.2214239 -0.4413611 nm^-1
bstar = -0.4063016 -0.1283573 -0.4855668 nm^-1
cstar = -0.3134916 +0.7686662 +0.0654781 nm^-1
Reflections measured after indexing
 -41   38  -19     -28.72      25.73      10.00      -0.37   21.2   22.8 p0
End of reflections
--- End crystal
----- End chunk -----
----- Begin chunk -----
Event: //113-1
Image serial number: 114
average_camera_length = 0.485000 m
--- Begin crystal
Cell parameters 1.54756 1.47279 1.21759 nm, 89.05374 92.02214 90.53556 deg
astar = -0.1314639 -0.5014330 +0.3864957 nm^-1
bstar = -0.3840104 -0.2806239 -0.4847301 nm^-1
cstar = +0.6592710 -0.4136830 -0.2641426 nm^-1
Reflections measured after indexing
 -44  -37    9       7.18      24.20      10.00       0.32  948.1 1002.5 p0
End of reflections
--- End crystal
----- End chunk -----
"""


def test_parse_gxparm_text() -> None:
    parsed = parse_gxparm_text(GXPARM_TEXT)
    assert parsed.phi0_deg == 10.0
    assert parsed.dphi_deg == 0.5
    assert np.allclose(parsed.rotation_axis, [0.0, 1.0, 0.0])
    assert parsed.detector_nx == 2048
    assert np.allclose(parsed.reciprocal_reference, np.diag([0.1, 1.0 / 11.0, 1.0 / 12.0]))


def test_parse_integrate_text() -> None:
    parsed = parse_integrate_text(INTEGRATE_TEXT)
    assert parsed.observations.shape[0] == 3
    assert parsed.estimated_n_frames == 4
    assert list(parsed.observations.columns) == ["h", "k", "l", "I", "sigma", "z_cal", "frame_est"]
    assert parsed.observations.iloc[1]["frame_est"] == 1


def test_parse_xds_inp_text() -> None:
    parsed = parse_xds_inp_text("UNTRUSTED_RECTANGLE= 10 20 30 40\nDATA_RANGE= 1 90\n")
    assert parsed.untrusted_rectangles == [(10.0, 20.0, 30.0, 40.0)]
    assert parsed.data_range == (1, 90)


def test_parse_composition() -> None:
    composition = parse_composition("4 C, 8 H, 2 O")
    assert len(composition.entries) == 3
    expected = 4 * 1.69**2 + 8 * 0.529**2 + 2 * 2.26**2
    assert np.isclose(composition.sum_fj2, expected)


def test_parse_crystfel_stream_text() -> None:
    parsed = parse_crystfel_stream_text(STREAM_TEXT)
    assert np.isclose(parsed.wavelength_angstrom, 0.019687)
    assert np.isclose(parsed.distance_mm, 485.0)
    assert np.isclose(parsed.pixel_x_mm, 0.056)
    assert parsed.detector_nx == 1024
    assert parsed.detector_ny == 1024
    assert np.isclose(parsed.orgx_px, 512.0)
    assert np.isclose(parsed.orgy_px, 512.0)
    assert parsed.crystal_table.shape[0] == 2
    assert parsed.reflections.shape[0] == 2
    assert np.isclose(float(parsed.crystal_table.iloc[0]["distance_mm"]), 485.0)
    assert np.isclose(float(parsed.crystal_table.iloc[0]["det_shift_x_mm"]), 0.0)
    assert np.isclose(float(parsed.crystal_table.iloc[0]["det_shift_y_mm"]), 0.0)
    ub = parsed.crystal_table.iloc[0][list(STREAM_MATRIX_COLUMNS)].to_numpy(dtype=float).reshape(3, 3)
    expected = np.asarray(
        [
            [0.04582728, -0.04063016, -0.03134916],
            [0.02214239, -0.01283573, 0.07686662],
            [-0.04413611, -0.04855668, 0.00654781],
        ]
    )
    assert np.allclose(ub, expected)


def test_crystfel_stream_to_analysis_inputs() -> None:
    parsed = parse_crystfel_stream_text(STREAM_TEXT)
    gxparm, integrate, reciprocal_by_frame = crystfel_stream_to_analysis_inputs(parsed)
    assert gxparm.detector_nx == 1024
    assert gxparm.detector_ny == 1024
    assert integrate.observations.shape[0] == 2
    assert integrate.estimated_n_frames == 2
    assert set(reciprocal_by_frame.keys()) == {0, 1}
    assert reciprocal_by_frame[0].shape == (3, 3)

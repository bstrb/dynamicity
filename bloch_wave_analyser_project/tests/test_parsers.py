from __future__ import annotations

import numpy as np

from src.parsers import parse_composition, parse_gxparm_text, parse_integrate_text, parse_xds_inp_text


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
    assert composition.sum_fj2 > 0.0
    expected = 4 * 1.69**2 + 8 * 0.529**2 + 2 * 2.26**2
    assert np.isclose(composition.sum_fj2, expected)

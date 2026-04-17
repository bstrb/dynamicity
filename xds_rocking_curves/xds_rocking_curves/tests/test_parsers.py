from __future__ import annotations

from pathlib import Path

import numpy as np

from src.parsers import parse_gxparm_text, parse_integrate_text, parse_spot_xds_text, parse_xds_inp_text


def test_parse_gxparm_text_extracts_geometry() -> None:
    text = "\n".join(
        [
            "GXPARM.XDS",
            "1 -2.0 0.1 0.0 1.0 0.0",
            "0.025 0.0 0.0 1.0",
            "221 15 15 15 90 90 90",
            "15 0 0",
            "0 15 0",
            "0 0 15",
            "1 128 128 0.055 0.055",
            "64.0 64.0 200.0",
            "1.0 0.0 0.0",
            "0.0 1.0 0.0",
            "0.0 0.0 1.0",
        ]
    )
    gx = parse_gxparm_text(text)
    assert gx.starting_frame == 1
    assert np.allclose(gx.rotation_axis, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(gx.reciprocal_reference @ np.array([1.0, 0.0, 0.0]), np.array([1 / 15, 0.0, 0.0]))
    assert gx.detector_nx == 128


def test_parse_integrate_text_reads_expected_fields() -> None:
    text = "\n".join(
        [
            "!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=21",
            "!END_OF_HEADER",
            " 1 0 0 100.0 4.0 64.2 62.8 10.4 1.0 99.0 98.0 500 64.1 62.7 10.5 0 0 0 0 0 1",
            "!END_OF_DATA",
        ]
    )
    data = parse_integrate_text(text)
    assert len(data.observations) == 1
    row = data.observations.iloc[0]
    assert row["frame_est"] == 10
    assert np.isclose(row["x_cal"], 64.2)


def test_parse_spot_xds_text_supports_indexed_lines() -> None:
    data = parse_spot_xds_text("64.2 62.8 10.0 500.0 1 0 0\n")
    row = data.spots.iloc[0]
    assert bool(row["indexed"]) is True
    assert int(row["h"]) == 1


def test_parse_xds_inp_text_extracts_template_and_range() -> None:
    inp = parse_xds_inp_text(
        "NAME_TEMPLATE_OF_DATA_FRAMES= /tmp/frame_????.tif\nDATA_RANGE= 1 40\nROTATION_AXIS= 0 1 0\n"
    )
    assert inp.name_template == "/tmp/frame_????.tif"
    assert inp.data_range == (1, 40)
    assert np.allclose(inp.rotation_axis, np.array([0.0, 1.0, 0.0]))

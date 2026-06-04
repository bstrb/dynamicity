from __future__ import annotations

import numpy as np
import pytest

from src.geometry import ReciprocalMatrixOrientationModel
from src.parsers import parse_composition, parse_gxparm_text, parse_integrate_text
from src.pipeline import AnalysisConfig, run_analysis


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


def _example_inputs():
    gxparm = parse_gxparm_text(GXPARM_TEXT)
    integrate = parse_integrate_text(INTEGRATE_TEXT)
    composition = parse_composition("4 C, 8 H, 2 O")
    return gxparm, integrate, composition


def test_run_analysis_orientation_only_skips_wilson_and_coupling() -> None:
    gxparm, integrate, composition = _example_inputs()
    config = AnalysisConfig(
        mode="proxy",
        orientation_only=True,
        dmin_angstrom=0.6,
        dmax_angstrom=20.0,
    )
    result = run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        config=config,
    )

    assert result.wilson is None
    assert result.frame_summary.shape[0] == integrate.estimated_n_frames
    assert np.allclose(result.frame_summary["S_2beam"].to_numpy(dtype=float), 0.0)
    assert np.allclose(result.frame_summary["S_MB"].to_numpy(dtype=float), 0.0)
    assert np.allclose(result.frame_summary["mean_N_eff"].to_numpy(dtype=float), 0.0)
    assert np.allclose(result.frame_summary["max_N_eff"].to_numpy(dtype=float), 0.0)
    assert np.allclose(result.frame_summary["eigenvalue_spread_invA"].to_numpy(dtype=float), 0.0)

    if not result.reflections_long.empty:
        s_orient = result.reflections_long["S_orient"].to_numpy(dtype=float)
        s_dyn = result.reflections_long["S_dyn"].to_numpy(dtype=float)
        assert np.allclose(s_orient, s_dyn)
        assert np.all(s_dyn >= 0.0)
        for column in (
            "strength_proxy",
            "visibility_gate",
            "ZOLZ_zone_axis_risk",
            "neighbor_density",
            "row_coupling",
            "pathway_risk",
            "local_crowding",
        ):
            assert column in result.reflections_long.columns
        assert np.allclose(result.reflections_long["S_comb"].to_numpy(dtype=float), 0.0)
        assert np.allclose(result.reflections_long["N_eff"].to_numpy(dtype=float), 0.0)


def test_run_analysis_orientation_only_rejects_thickness_mode() -> None:
    gxparm, integrate, composition = _example_inputs()
    config = AnalysisConfig(
        mode="thickness",
        thickness_nm=100.0,
        orientation_only=True,
    )
    with pytest.raises(ValueError, match="does not support thickness"):
        run_analysis(
            gxparm=gxparm,
            integrate=integrate,
            composition=composition,
            config=config,
        )


def test_run_analysis_uses_direct_stream_reciprocal_matrix_and_frame_geometry() -> None:
    gxparm, integrate, composition = _example_inputs()
    direct_reciprocal = np.asarray(
        [
            [0.2, 0.0, 0.0],
            [0.0, 1.0 / 11.0, 0.0],
            [0.0, 0.0, 1.0 / 12.0],
        ],
        dtype=float,
    )
    orientation_model = ReciprocalMatrixOrientationModel(
        reciprocal_by_frame={0: direct_reciprocal},
        reciprocal_reference=gxparm.reciprocal_reference,
        use_direct_reciprocal_vectors=True,
        detector_center_by_frame={0: (1034.0, 1030.0)},
        distance_by_frame={0: 250.0},
    )
    config = AnalysisConfig(
        mode="proxy",
        orientation_only=True,
        dmin_angstrom=5.0,
        dmax_angstrom=20.0,
        stream_mirror_x_axis=False,
        frame_numbers=[1],
    )

    result = run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        config=config,
        orientation_model=orientation_model,
    )

    frame_table = result.frame_table(0)
    target = frame_table[(frame_table["h"] == 1) & (frame_table["k"] == 0) & (frame_table["l"] == 0)]
    assert not target.empty

    expected_g = np.asarray([0.2, 0.0, 0.0], dtype=float)
    beam = np.asarray([0.0, 0.0, 1.0 / gxparm.wavelength_angstrom], dtype=float)
    shifted = expected_g + beam
    expected_x = 1034.0 + (shifted[0] / shifted[2]) * (250.0 / gxparm.pixel_x_mm)
    expected_y = 1030.0 + (shifted[1] / shifted[2]) * (250.0 / gxparm.pixel_y_mm)

    assert np.isclose(float(target.iloc[0]["x_px"]), expected_x, atol=1e-6)
    assert np.isclose(float(target.iloc[0]["y_px"]), expected_y, atol=1e-6)


def test_run_analysis_mirrors_direct_stream_projection_about_detector_x_axis() -> None:
    gxparm, integrate, composition = _example_inputs()
    direct_reciprocal = np.asarray(
        [
            [0.2, 0.0, 0.0],
            [0.0, 1.0 / 11.0, 0.0],
            [0.0, 0.0, 1.0 / 12.0],
        ],
        dtype=float,
    )
    orientation_model = ReciprocalMatrixOrientationModel(
        reciprocal_by_frame={0: direct_reciprocal},
        reciprocal_reference=gxparm.reciprocal_reference,
        use_direct_reciprocal_vectors=True,
        detector_center_by_frame={0: (1034.0, 1030.0)},
        distance_by_frame={0: 250.0},
    )
    config = AnalysisConfig(
        mode="proxy",
        orientation_only=True,
        dmin_angstrom=5.0,
        dmax_angstrom=20.0,
        stream_mirror_x_axis=True,
        frame_numbers=[1],
    )

    result = run_analysis(
        gxparm=gxparm,
        integrate=integrate,
        composition=composition,
        config=config,
        orientation_model=orientation_model,
    )

    frame_table = result.frame_table(0)
    target = frame_table[(frame_table["h"] == 0) & (frame_table["k"] == 1) & (frame_table["l"] == 0)]
    assert not target.empty

    expected_g = np.asarray([0.0, 1.0 / 11.0, 0.0], dtype=float)
    beam = np.asarray([0.0, 0.0, 1.0 / gxparm.wavelength_angstrom], dtype=float)
    shifted = expected_g + beam
    projected_y = 1030.0 + (shifted[1] / shifted[2]) * (250.0 / gxparm.pixel_y_mm)
    expected_y = 2.0 * 1030.0 - projected_y

    assert np.isclose(float(target.iloc[0]["y_px"]), expected_y, atol=1e-6)

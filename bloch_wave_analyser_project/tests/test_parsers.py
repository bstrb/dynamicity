from __future__ import annotations

import numpy as np

from src.parsers import (
    PETS_DEFAULT_WAVELENGTH_ANGSTROM,
    PETS_TO_PIPELINE_LAB_TRANSFORM,
    crystfel_stream_to_analysis_inputs,
    parse_composition,
    parse_crystfel_stream_text,
    parse_gxparm_text,
    parse_integrate_text,
    parse_pets_project,
    parse_rprofall_text,
    parse_xds_inp_text,
    pets_project_to_analysis_inputs,
    rprofall_to_integrate_data,
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

PETS_PTS_TEXT = """\
version 2.3
lambda 0.01969000
aperpixel 0.00628000
omega 25.0000
delta 7.5000
center 244.38 249.80
badpixels
 12 18 500 501
endbadpixels
imagelistheader imgname alpha beta domega alphaorig betaorig domegaorig xcenter ycenter intscale diffbfac magcorr elliamp elliph paraamp paraph useforcalc dataset
imagelist
"image/000000.tiff" -44.7000 0.1700 0.0700 -44.7000 0.0000 0.0000 244.0 250.0 1.0 0.0 -0.16 0.02 -60.0 0.0 0.0 1 1
"image/000001.tiff" -44.6500 0.1680 0.0710 -44.6500 0.0000 0.0000 244.2 249.9 1.0 0.0 -0.16 0.02 -60.2 0.0 0.0 1 1
"image/000002.tiff" -44.6000 0.1660 0.0720 -44.6000 0.0000 0.0000 244.4 250.1 1.0 0.0 -0.16 0.02 -60.4 0.0 0.0 1 1
endimagelist
celllist
cellItem active
ubmatrix
 -0.025942  0.032425 -0.071809
  0.056396 -0.045152 -0.040763
 -0.055021 -0.061569 -0.007924
cell 12.0553 12.0553 12.0553 90.000 90.000 90.000
endCellItem
endcelllist
"""

PETS_PTSOPT_TEXT = """\
version 2.3
lambda 0.01969000
aperpixel 0.00628000
omega 25.0000
delta 7.5000
center 280.00 285.00
imagelistheader imgname alpha beta domega alphaorig betaorig domegaorig xcenter ycenter intscale diffbfac magcorr elliamp elliph paraamp paraph useforcalc dataset
imagelist
"image/000000.tiff" -42.5000 0.1100 0.0100 -44.7000 0.0000 0.0000 280.0 285.0 1.0 0.0 -0.16 0.02 -60.0 0.0 0.0 1 1
"image/000001.tiff" -41.9000 0.1250 0.0120 -44.6500 0.0000 0.0000 280.5 285.5 1.0 0.0 -0.16 0.02 -60.2 0.0 0.0 1 1
"image/000002.tiff" -41.3000 0.1400 0.0140 -44.6000 0.0000 0.0000 281.0 286.0 1.0 0.0 -0.16 0.02 -60.4 0.0 0.0 1 1
endimagelist
celllist
cellItem active
ubmatrix
 -0.025942  0.032425 -0.071809
  0.056396 -0.045152 -0.040763
 -0.055021 -0.061569 -0.007924
cell 12.0553 12.0553 12.0553 90.000 90.000 90.000
endCellItem
endcelllist
"""

PETS_PTSOPTLIST_TEXT = """\
frame alpha beta domega
1 -42.0000 0.2000 0.0500
2 -41.4000 0.2600 0.0800
3 -40.8000 0.3200 0.1100
"""

PETS_CENLOCOPT_TEXT = """\
frame xcen ycen
1 310.0 320.0
2 311.0 321.0
3 312.0 322.0
"""

PETS_REALSTYLE_PTSOPT_TEXT = """\
xcenter   ycenter   alpha     beta      omega     magcorr   elliamp   elliphase paraamp   paraphase undisxcen undisycen RCwidth   mosaicity
image/000000.tiff  100.0  120.0  -10.0   0.10   0.01    0.5    0.0    0.0    0.0  100.0  120.0    0.0001    0.0200 |   50.0
image/000001.tiff  102.0  122.0   -9.5   0.20   0.02    0.6    0.0    0.0    0.0  102.0  122.0    0.0001    0.0300 |   51.0
"""

PETS_REALSTYLE_LOGINDEX_TEXT = """\
#########################################
# Find unit cell and orientation matrix #
#########################################

 Cell 1 (Active cell):
  Cell parameters:   12.0553  12.0553  12.0553  90.000  90.000  90.000
  Orientation matrix:
       -0.025942   0.032425  -0.071809
        0.056396  -0.045152  -0.040763
       -0.055021  -0.061569  -0.007924
"""

PETS_REALSTYLE_LOGPS_TEXT = """\
###############
# Peak search #
###############

 Details:
 Frame nr.     xcenter     ycenter      deltax      deltay      NPeaks      maxres
         1     100.000     120.000       0.000       0.000          10        1.00
         2     102.000     122.000       0.000       0.000          11        1.00
"""

PETS_REALSTYLE_DYNTMP_TEXT = """\
   1   0   0  0.10000000E+03  0.10000000E+02    300.00    120.00    300.00    120.00    1    0   1.00000   0.00010   5.00000   0.02000
   0   1   0  0.90000000E+02  0.90000000E+01    102.00    322.00    102.00    322.00    2    0   1.00000  -0.00010   5.00000   0.02000
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


def _rprofall_row(
    h: int,
    k: int,
    l: int,
    resolution: float,
    excitation: float,
    rsg: float,
    iobs: float,
    sigma: float | str,
    icalc: float,
    frame: int,
    azimuth: float,
) -> str:
    sigma_text = f"{sigma:14.6f}" if isinstance(sigma, float) else f"{sigma:>14}"
    row = (
        f"{h:4d}{k:4d}{l:4d}"
        f"{resolution:14.6f}"
        f"{excitation:14.6f}"
        f"{rsg:14.6f}"
        f"{iobs:14.6f}"
        f"{sigma_text}"
        f"{icalc:14.6f}"
        f"{frame:4d}"
        f"{azimuth:9.3f}"
    )
    assert len(row) == 109
    return row


def _rotation_x_deg(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rotation_y_deg(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rotation_z_deg(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _pets_rotation_from_angles(alpha_deg: float, beta_deg: float, domega_deg: float) -> np.ndarray:
    return _rotation_y_deg(alpha_deg) @ _rotation_x_deg(beta_deg) @ _rotation_z_deg(domega_deg)


def _axis_angle_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis.tolist()
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
        ],
        dtype=float,
    )


def test_parse_rprofall_text_and_overflow() -> None:
    row_1 = _rprofall_row(-23, -6, 1, 1.993505, -0.000895, -1.975197, 41.978603, 48.628971, 98.512764, 237, -4.194)
    row_2 = _rprofall_row(-10, 7, 20, 1.963339, 0.000230, 0.764063, 0.0, "**************", 0.148451, 3033, -2.517)
    text = "\r\n# 1\r\n\r\n" + row_1 + "\r\n\r\n# 2\r\n" + row_2 + "\r\n"

    parsed = parse_rprofall_text(text)
    table = parsed.rows
    assert parsed.n_blocks == 2
    assert list(table.columns) == [
        "row_id",
        "block_id",
        "h",
        "k",
        "l",
        "resolution",
        "excitation",
        "rsg",
        "iobs",
        "sigma",
        "icalc",
        "frame",
        "azimuth",
    ]
    assert len(table) == 2
    assert int(table.loc[0, "block_id"]) == 1
    assert int(table.loc[1, "block_id"]) == 2
    assert np.isnan(float(table.loc[1, "sigma"]))
    assert np.isclose(float(table.loc[1, "icalc"]), 0.148451)
    assert int(table.loc[1, "frame"]) == 3033


def test_rprofall_to_integrate_data() -> None:
    row = _rprofall_row(-22, -4, 2, 1.963339, -0.003075, -6.971740, 21.208347, 32.078842, 11.907216, 31, -4.174)
    parsed = parse_rprofall_text("# 1\n\n" + row + "\n")
    integrate = rprofall_to_integrate_data(parsed)
    assert integrate.observations.shape[0] == 1
    assert list(integrate.observations.columns) == ["h", "k", "l", "I", "sigma", "z_cal", "frame_est"]
    assert np.isclose(float(integrate.observations.iloc[0]["I"]), 21.208347)
    assert np.isclose(float(integrate.observations.iloc[0]["sigma"]), 32.078842)
    assert int(integrate.estimated_n_frames) == 31


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
    first_ub11 = float(parsed.crystal_table.iloc[0]["UB11"])
    assert np.isclose(first_ub11, 0.04582728)


def test_crystfel_stream_to_analysis_inputs() -> None:
    parsed = parse_crystfel_stream_text(STREAM_TEXT)
    gxparm, integrate, reciprocal_by_frame = crystfel_stream_to_analysis_inputs(parsed)
    assert gxparm.detector_nx == 1024
    assert gxparm.detector_ny == 1024
    assert integrate.observations.shape[0] == 2
    assert integrate.estimated_n_frames == 2
    assert set(reciprocal_by_frame.keys()) == {0, 1}
    assert reciprocal_by_frame[0].shape == (3, 3)


def test_parse_pets_project_and_conversion(tmp_path) -> None:
    project_dir = tmp_path / "pets_project"
    project_dir.mkdir()
    (project_dir / "sample.pts2.backup").write_text(PETS_PTS_TEXT)
    (project_dir / "sample.ptsopt").write_text(PETS_PTSOPT_TEXT)
    (project_dir / "sample.ptsoptlist").write_text(PETS_PTSOPTLIST_TEXT)
    (project_dir / "sample.cenlocopt").write_text(PETS_CENLOCOPT_TEXT)
    rprofall_text = (
        "# 1\n"
        + _rprofall_row(-1, 0, 1, 1.5, 0.001, 0.1, 120.0, 12.0, 100.0, 1, 0.0)
        + "\n# 2\n"
        + _rprofall_row(1, 0, -1, 1.5, -0.001, -0.1, 80.0, 10.0, 90.0, 2, 10.0)
        + "\n"
    )
    (project_dir / "sample.rprofall").write_text(rprofall_text)

    pets = parse_pets_project(project_dir)
    assert pets.pts_path.name == "sample.pts2.backup"
    assert np.isclose(pets.wavelength_angstrom, 0.01969)
    assert np.isclose(pets.aperpixel_invA_per_px, 0.00628)
    assert pets.imagelist.shape[0] == 3
    assert pets.frame_geometry.shape[0] == 3
    assert np.isclose(float(pets.frame_geometry.iloc[0]["alpha"]), -42.0)
    assert np.isclose(float(pets.frame_geometry.iloc[1]["beta"]), 0.26)
    assert np.isclose(float(pets.frame_geometry.iloc[2]["domega"]), 0.11)
    assert np.isclose(pets.orgx_px, 311.0)
    assert np.isclose(pets.orgy_px, 321.0)
    assert pets.detector_nx >= 501
    assert pets.detector_ny >= 502

    gxparm, integrate, reciprocal_by_frame = pets_project_to_analysis_inputs(pets)
    assert np.isclose(gxparm.wavelength_angstrom, 0.01969)
    assert np.isclose(gxparm.orgx_px, 311.0)
    assert np.isclose(gxparm.orgy_px, 321.0)
    assert integrate.observations.shape[0] == 2
    assert integrate.estimated_n_frames == 2
    assert set(reciprocal_by_frame.keys()) == {0, 1}
    assert not np.allclose(reciprocal_by_frame[0], reciprocal_by_frame[1])

    reciprocal_reference = np.asarray(pets.reciprocal_reference, dtype=float)
    transformed_reference = PETS_TO_PIPELINE_LAB_TRANSFORM @ reciprocal_reference
    assert np.allclose(gxparm.reciprocal_reference, transformed_reference)
    omega_rad = np.deg2rad(25.0)
    delta_rad = np.deg2rad(7.5)
    axis = np.asarray(
        [
            np.cos(delta_rad) * np.cos(omega_rad),
            -np.cos(delta_rad) * np.sin(omega_rad),
            np.sin(delta_rad),
        ],
        dtype=float,
    )
    first_rotation = _axis_angle_rotation(axis, -42.0)
    second_rotation = _axis_angle_rotation(axis, -41.4)
    expected_frame_1 = PETS_TO_PIPELINE_LAB_TRANSFORM @ (
        np.linalg.inv(second_rotation) @ first_rotation @ reciprocal_reference
    )
    assert np.allclose(reciprocal_by_frame[1], expected_frame_1)

    old_euler_expected = PETS_TO_PIPELINE_LAB_TRANSFORM @ (
        np.linalg.inv(_pets_rotation_from_angles(-41.4, 0.26, 0.08))
        @ _pets_rotation_from_angles(-42.0, 0.2, 0.05)
        @ reciprocal_reference
    )
    assert not np.allclose(reciprocal_by_frame[1], old_euler_expected)


def test_parse_pets_project_from_realstyle_folder_without_pts2(tmp_path) -> None:
    project_dir = tmp_path / "pets_realstyle"
    logs_dir = project_dir / "logs"
    project_dir.mkdir()
    logs_dir.mkdir()
    (project_dir / "sample.ptsopt").write_text(PETS_REALSTYLE_PTSOPT_TEXT)
    (logs_dir / "sample.logindex").write_text(PETS_REALSTYLE_LOGINDEX_TEXT)
    (logs_dir / "sample.logps").write_text(PETS_REALSTYLE_LOGPS_TEXT)
    (project_dir / "sample.dyntmp").write_text(PETS_REALSTYLE_DYNTMP_TEXT)
    rprofall_text = (
        "# 1\n"
        + _rprofall_row(-1, 0, 1, 1.0, 0.001, 0.1, 120.0, 12.0, 100.0, 1, 0.0)
        + "\n# 2\n"
        + _rprofall_row(1, 0, -1, 1.0, -0.001, -0.1, 80.0, 10.0, 90.0, 2, 10.0)
        + "\n"
    )
    (project_dir / "sample.rprofall").write_text(rprofall_text)

    pets = parse_pets_project(project_dir)
    assert pets.pts_path.name == "sample.ptsopt"
    assert np.isclose(pets.wavelength_angstrom, PETS_DEFAULT_WAVELENGTH_ANGSTROM)
    assert np.isclose(pets.aperpixel_invA_per_px, 0.005)
    assert pets.imagelist.shape[0] == 2
    assert pets.frame_geometry.shape[0] == 2
    assert np.isclose(float(pets.frame_geometry.iloc[0]["xcenter"]), 100.0)
    assert np.isclose(float(pets.frame_geometry.iloc[1]["ycenter"]), 122.0)
    assert np.isclose(float(pets.frame_geometry.iloc[0]["alpha"]), -10.0)
    assert np.isclose(float(pets.frame_geometry.iloc[1]["beta"]), 0.20)
    assert np.isclose(float(pets.frame_geometry.iloc[1]["domega"]), 0.02)
    assert np.isclose(pets.orgx_px, 101.0)
    assert np.isclose(pets.orgy_px, 121.0)
    assert pets.detector_nx >= 512
    assert pets.detector_ny >= 512
    assert any("logindex" in note for note in pets.metadata_notes)
    assert any("Estimated PETS wavelength" in note for note in pets.metadata_notes)
    assert any("Estimated PETS reciprocal calibration" in note for note in pets.metadata_notes)

    gxparm, integrate, reciprocal_by_frame = pets_project_to_analysis_inputs(pets)
    assert np.isclose(gxparm.wavelength_angstrom, PETS_DEFAULT_WAVELENGTH_ANGSTROM)
    assert np.isclose(gxparm.orgx_px, 101.0)
    assert np.isclose(gxparm.orgy_px, 121.0)
    assert integrate.observations.shape[0] == 2
    assert integrate.estimated_n_frames == 2
    assert set(reciprocal_by_frame.keys()) == {0, 1}
    assert not np.allclose(reciprocal_by_frame[0], reciprocal_by_frame[1])
    assert np.allclose(
        gxparm.reciprocal_reference,
        PETS_TO_PIPELINE_LAB_TRANSFORM @ np.asarray(pets.reciprocal_reference, dtype=float),
    )


def test_parse_pets_project_uses_parent_pts2_for_missing_scalars(tmp_path) -> None:
    root_dir = tmp_path / "pets_root"
    project_dir = root_dir / "run"
    logs_dir = project_dir / "logs"
    root_dir.mkdir()
    project_dir.mkdir()
    logs_dir.mkdir()

    (root_dir / "sample.pts2").write_text(PETS_PTS_TEXT)
    (project_dir / "sample.ptsopt").write_text(PETS_REALSTYLE_PTSOPT_TEXT)
    (logs_dir / "sample.logindex").write_text(PETS_REALSTYLE_LOGINDEX_TEXT)
    (logs_dir / "sample.logps").write_text(PETS_REALSTYLE_LOGPS_TEXT)
    rprofall_text = (
        "# 1\n"
        + _rprofall_row(-1, 0, 1, 1.0, 0.001, 0.1, 120.0, 12.0, 100.0, 1, 0.0)
        + "\n# 2\n"
        + _rprofall_row(1, 0, -1, 1.0, -0.001, -0.1, 80.0, 10.0, 90.0, 2, 10.0)
        + "\n"
    )
    (project_dir / "sample.rprofall").write_text(rprofall_text)

    pets = parse_pets_project(project_dir)
    assert pets.pts_path.name == "sample.ptsopt"
    assert np.isclose(pets.wavelength_angstrom, 0.01969)
    assert np.isclose(pets.aperpixel_invA_per_px, 0.00628)
    assert np.isclose(float(pets.omega_deg), 25.0)
    assert np.isclose(float(pets.delta_deg), 7.5)
    assert any("scalar metadata from sample.pts2" in note for note in pets.metadata_notes)

from __future__ import annotations

import numpy as np

from src.pets2 import load_pets_model, pets_model_to_analysis_inputs


PETS_PTS2_TEXT = """\
version 2.3
lambda 0.01969000
aperpixel 0.00628000
omega 25.0000
delta 7.5000
center 245.0 250.0
badpixelx = 1 1 1 512
badpixely = 1 1 1 512
imagelistheader imgname alpha beta domega alphaorig betaorig domegaorig xcenter ycenter useforcalc dataset
imagelist
"image/000000.tiff" -44.7000 0.1700 0.0700 -44.7000 0.1700 0.0700 244.0 250.0 1 1
"image/000001.tiff" -44.6500 0.1680 0.0710 -44.6500 0.1680 0.0710 245.0 251.0 1 1
endimagelist
ubmatrix
 -0.025942  0.032425 -0.071809
  0.056396 -0.045152 -0.040763
 -0.055021 -0.061569 -0.007924
cell 12.0553 12.0553 12.0553 90.000 90.000 90.000
"""


def _rprofall_row(
    h: int,
    k: int,
    l: int,
    resolution: float,
    excitation: float,
    rsg: float,
    iobs: float,
    sigma: float,
    icalc: float,
    frame: int,
    azimuth: float,
) -> str:
    return (
        f"{h:4d}{k:4d}{l:4d}"
        f"{resolution:14.6f}"
        f"{excitation:14.6f}"
        f"{rsg:14.6f}"
        f"{iobs:14.6f}"
        f"{sigma:14.6f}"
        f"{icalc:14.6f}"
        f"{frame:4d}"
        f"{azimuth:9.3f}"
    )


def test_load_pets_model_and_convert(tmp_path) -> None:
    root = tmp_path / "LTA1_PETS"
    root.mkdir()
    petsdata = root / "LTA1_petsdata"
    petsdata.mkdir()

    (root / "LTA1.pts2").write_text(PETS_PTS2_TEXT)
    (petsdata / "LTA1.ptsopt").write_text("img x y a b o\n")
    rprofall_text = (
        "# 1\n"
        + _rprofall_row(-1, 0, 1, 1.5, 0.001, 0.1, 120.0, 12.0, 100.0, 1, 0.0)
        + "\n# 2\n"
        + _rprofall_row(1, 0, -1, 1.5, -0.001, -0.1, 80.0, 10.0, 90.0, 2, 10.0)
        + "\n"
    )
    (petsdata / "LTA1.rprofall").write_text(rprofall_text)

    model = load_pets_model(root)
    assert model.pts2_path.name == "LTA1.pts2"
    assert model.rprofall_path.name == "LTA1.rprofall"
    assert len(model.frames) == 2
    assert np.isclose(model.wavelength_angstrom, 0.01969)
    assert np.isclose(model.aperpixel_invA_per_px, 0.00628)
    assert model.detector_nx == 512
    assert model.detector_ny == 512

    gxparm, integrate, reciprocal_by_frame, frame_geometry = pets_model_to_analysis_inputs(model)
    assert integrate.observations.shape[0] == 2
    assert integrate.estimated_n_frames == 2
    assert set(reciprocal_by_frame.keys()) == {0, 1}
    assert not np.allclose(reciprocal_by_frame[0], reciprocal_by_frame[1])
    assert frame_geometry.shape[0] == 2
    assert gxparm.detector_nx == 512
    assert gxparm.detector_ny == 512


def test_load_pets_model_defaults_detector_shape_without_badpixel(tmp_path) -> None:
    root = tmp_path / "LTA1_PETS"
    root.mkdir()
    petsdata = root / "LTA1_petsdata"
    petsdata.mkdir()

    pts2_text = PETS_PTS2_TEXT.replace("badpixelx = 1 1 1 512\n", "").replace(
        "badpixely = 1 1 1 512\n", ""
    )
    (root / "LTA1.pts2").write_text(pts2_text)
    (petsdata / "LTA1.ptsopt").write_text("img x y a b o\n")
    (petsdata / "LTA1.rprofall").write_text(
        "# 1\n" + _rprofall_row(-1, 0, 1, 1.5, 0.001, 0.1, 120.0, 12.0, 100.0, 1, 0.0) + "\n"
    )

    model = load_pets_model(root)

    assert model.detector_nx == 512
    assert model.detector_ny == 512

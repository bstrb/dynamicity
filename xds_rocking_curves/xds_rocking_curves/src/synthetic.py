"""Synthetic dataset generator for tests and demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .geometry import detector_geometry_from_xds, predict_reflection_on_frame
from .parsers import GXPARMData, UnitCell, XDSInputData, parse_gxparm


def write_synthetic_dataset(
    output_dir: str | Path,
    n_frames: int = 41,
    hkl: tuple[int, int, int] = (1, 0, 0),
    starting_frame: int = 1,
    phi0_deg: float = -2.0,
    dphi_deg: float = 0.1,
    base_amplitude: float = 500.0,
    sigma_x_px: float = 1.5,
    sigma_y_px: float = 1.1,
    theta_rad: float = 0.25,
    random_seed: int = 0,
) -> dict[str, Path]:
    """Write a minimal synthetic XDS-like dataset to disk."""

    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    gxparm_text = "\n".join(
        [
            "GXPARM.XDS",
            f"{starting_frame:d} {phi0_deg:.6f} {dphi_deg:.6f} 0.000000 1.000000 0.000000",
            "0.025000 0.000000 0.000000 1.000000",
            "221 15.000000 15.000000 15.000000 90.000000 90.000000 90.000000",
            "15.000000 0.000000 0.000000",
            "0.000000 15.000000 0.000000",
            "0.000000 0.000000 15.000000",
            "1 128 128 0.055000 0.055000",
            "64.000000 64.000000 200.000000",
            "1.000000 0.000000 0.000000",
            "0.000000 1.000000 0.000000",
            "0.000000 0.000000 1.000000",
        ]
    )
    gxparm_path = out / "GXPARM.XDS"
    gxparm_path.write_text(gxparm_text + "\n")

    xds_inp_text = "\n".join(
        [
            f"NAME_TEMPLATE_OF_DATA_FRAMES= {img_dir / 'frame_????.tif'}",
            f"DATA_RANGE= {starting_frame} {starting_frame + n_frames - 1}",
            "NX= 128 NY= 128 QX= 0.055000 QY= 0.055000",
            "ORGX= 64.0",
            "ORGY= 64.0",
            "DETECTOR_DISTANCE= 200.0",
            "ROTATION_AXIS= 0.0 1.0 0.0",
            "X-RAY_WAVELENGTH= 0.025000",
            "INCIDENT_BEAM_DIRECTION= 0.0 0.0 1.0",
            "DIRECTION_OF_DETECTOR_X-AXIS= 1.0 0.0 0.0",
            "DIRECTION_OF_DETECTOR_Y-AXIS= 0.0 1.0 0.0",
        ]
    )
    xds_inp_path = out / "XDS.INP"
    xds_inp_path.write_text(xds_inp_text + "\n")

    gxparm = parse_gxparm(gxparm_path)
    detector = detector_geometry_from_xds(
        gxparm,
        XDSInputData(
            name_template=str(img_dir / "frame_????.tif"),
            data_range=(starting_frame, starting_frame + n_frames - 1),
            untrusted_rectangles=[],
            detector_nx=128,
            detector_ny=128,
            pixel_x_mm=0.055,
            pixel_y_mm=0.055,
            orgx_px=64.0,
            orgy_px=64.0,
            distance_mm=200.0,
            rotation_axis=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            wavelength_angstrom=0.025,
            incident_beam_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            detector_x_axis=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            detector_y_axis=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        ),
    )

    spot_lines: list[str] = []
    best_pred = None
    best_amp = -1.0
    for frame in range(starting_frame, starting_frame + n_frames):
        pred = predict_reflection_on_frame(gxparm, detector, hkl, frame)
        image = rng.normal(loc=10.0, scale=1.5, size=(128, 128)).astype(np.float64)
        amp = base_amplitude * np.exp(-0.5 * (pred.sg / 0.003) ** 2) if pred.on_detector else 0.0
        if amp > best_amp:
            best_amp = amp
            best_pred = pred
        if pred.on_detector and amp > 3.0:
            x0 = pred.x_pred - 1.0
            y0 = pred.y_pred - 1.0
            y, x = np.indices(image.shape, dtype=np.float64)
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)
            x_shift = x - x0
            y_shift = y - y0
            x_rot = cos_t * x_shift + sin_t * y_shift
            y_rot = -sin_t * x_shift + cos_t * y_shift
            image += amp * np.exp(-0.5 * ((x_rot / sigma_x_px) ** 2 + (y_rot / sigma_y_px) ** 2))
            spot_lines.append(f"{pred.x_pred:.3f} {pred.y_pred:.3f} {frame:.3f} {amp:.3f} {hkl[0]} {hkl[1]} {hkl[2]}")
        image = np.clip(image, 0, None)
        image_path = img_dir / f"frame_{frame:04d}.tif"
        tifffile.imwrite(image_path, image.astype(np.float32))

    spot_path = out / "SPOT.XDS"
    spot_path.write_text("\n".join(spot_lines) + "\n")

    if best_pred is None:
        best_pred = predict_reflection_on_frame(gxparm, detector, hkl, starting_frame)
    integrate_text = "\n".join(
        [
            "!OUTPUT_FILE=INTEGRATE.HKL",
            f"!NAME_TEMPLATE_OF_DATA_FRAMES={img_dir / 'frame_????.tif'} TIFF",
            f"!STARTING_FRAME={starting_frame:8d}",
            f"!STARTING_ANGLE={phi0_deg:10.3f}",
            f"!OSCILLATION_RANGE={dphi_deg:10.6f}",
            "!ROTATION_AXIS= 0.000000 1.000000 0.000000",
            "!X-RAY_WAVELENGTH= 0.025000",
            "!INCIDENT_BEAM_DIRECTION= 0.000000 0.000000 1.000000",
            "!SPACE_GROUP_NUMBER= 221",
            "!UNIT_CELL_CONSTANTS= 15.000 15.000 15.000 90.000 90.000 90.000",
            "!UNIT_CELL_A-AXIS= 15.000 0.000 0.000",
            "!UNIT_CELL_B-AXIS= 0.000 15.000 0.000",
            "!UNIT_CELL_C-AXIS= 0.000 0.000 15.000",
            "!NUMBER OF DETECTOR SEGMENTS   1",
            "!NX=   128  NY=   128    QX=  0.055000  QY=  0.055000",
            "!ORGX=     64.00  ORGY=     64.00  DETECTOR_DISTANCE=   200.000",
            "!DIRECTION_OF_DETECTOR_X-AXIS=  1.000000  0.000000  0.000000",
            "!DIRECTION_OF_DETECTOR_Y-AXIS=  0.000000  1.000000  0.000000",
            "!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=21",
            "!H,K,L,IOBS,SIGMA,XCAL,YCAL,ZCAL,RLP,PEAK,CORR,MAXC,",
            "!             XOBS,YOBS,ZOBS,ALF0,BET0,ALF1,BET1,PSI,ISEG",
            "!END_OF_HEADER",
            f" {hkl[0]:3d} {hkl[1]:3d} {hkl[2]:3d} {best_amp:10.3f} 5.000 {best_pred.x_pred:7.2f} {best_pred.y_pred:7.2f} {best_pred.frame:7.2f} 1.00000 100.0 99.0 800 {best_pred.x_pred:7.2f} {best_pred.y_pred:7.2f} {best_pred.frame:7.2f} 0.0 0.0 0.0 0.0 0.0 1",
            "!END_OF_DATA",
        ]
    )
    integrate_path = out / "INTEGRATE.HKL"
    integrate_path.write_text(integrate_text + "\n")
    return {
        "gxparm": gxparm_path,
        "xds_inp": xds_inp_path,
        "spot_xds": spot_path,
        "integrate_hkl": integrate_path,
        "image_glob": img_dir / "*.tif",
        "output_dir": out,
    }

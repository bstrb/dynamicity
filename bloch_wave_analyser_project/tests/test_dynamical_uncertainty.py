from __future__ import annotations

import numpy as np
import pandas as pd

from src.dynamical_uncertainty import (
    DynamicalUncertaintyConfig,
    canonical_observations_from_stream,
    run_dynamical_uncertainty_pipeline,
)
from src.geometry import RotationSeriesOrientationModel
from src.parsers import IntegrateData, UnitCell
from src.parsers import GXPARMData


def _example_gxparm() -> GXPARMData:
    reciprocal_reference = np.diag([0.1, 1.0 / 11.0, 1.0 / 12.0]).astype(float)
    real_space_reference = np.linalg.inv(reciprocal_reference)
    return GXPARMData(
        phi0_deg=10.0,
        dphi_deg=0.5,
        rotation_axis=np.asarray([0.0, 1.0, 0.0], dtype=float),
        wavelength_angstrom=0.0251,
        space_group=1,
        unit_cell=UnitCell(a=10.0, b=11.0, c=12.0, alpha=90.0, beta=90.0, gamma=90.0),
        real_space_reference=real_space_reference,
        reciprocal_reference=reciprocal_reference,
        detector_nx=1024,
        detector_ny=1024,
        pixel_x_mm=0.055,
        pixel_y_mm=0.055,
        orgx_px=512.0,
        orgy_px=512.0,
        distance_mm=200.0,
    )


def _integrate_with_values(scale_i: float, scale_sigma: float) -> IntegrateData:
    table = pd.DataFrame(
        {
            "h": [1, 1, 0, 0],
            "k": [0, 0, 1, 1],
            "l": [0, 0, 0, 0],
            "I": [100.0 * scale_i, 300.0 * scale_i, 80.0 * scale_i, 160.0 * scale_i],
            "sigma": [10.0 * scale_sigma, 20.0 * scale_sigma, 8.0 * scale_sigma, 16.0 * scale_sigma],
            "z_cal": [0.1, 1.1, 0.2, 1.2],
            "frame_est": [0, 1, 0, 1],
        }
    )
    return IntegrateData(observations=table, estimated_n_frames=2)


def _canonical_from_manual_inputs(scale_i: float, scale_sigma: float):
    from src.dynamical_uncertainty import _canonical_from_core_inputs

    gxparm = _example_gxparm()
    integrate = _integrate_with_values(scale_i=scale_i, scale_sigma=scale_sigma)
    orienter = RotationSeriesOrientationModel(gxparm)
    reciprocal_by_frame = {
        frame: orienter.rotation_matrix(frame, offset=0.0) @ gxparm.reciprocal_reference
        for frame in range(integrate.estimated_n_frames)
    }
    return _canonical_from_core_inputs(
        gxparm=gxparm,
        integrate=integrate,
        reciprocal_by_frame=reciprocal_by_frame,
        source="xds",
        dataset_id="toy",
    )


def test_dynamical_uncertainty_is_intensity_independent() -> None:
    cfg = DynamicalUncertaintyConfig(
        orientation_step_deg=0.08,
        orientation_n_steps=1,
        thickness_min_nm=20.0,
        thickness_max_nm=120.0,
        n_thickness_steps=9,
    )
    canonical_a = _canonical_from_manual_inputs(scale_i=1.0, scale_sigma=1.0)
    canonical_b = _canonical_from_manual_inputs(scale_i=1000.0, scale_sigma=50.0)
    result_a = run_dynamical_uncertainty_pipeline(canonical_a, config=cfg).uncertainty_table
    result_b = run_dynamical_uncertainty_pipeline(canonical_b, config=cfg).uncertainty_table

    for column in ("risk_orientation", "risk_thickness", "risk_total", "dyn_sigma_rel"):
        assert np.allclose(
            result_a[column].to_numpy(dtype=float),
            result_b[column].to_numpy(dtype=float),
            atol=1e-12,
        )


def test_dynamical_uncertainty_output_columns() -> None:
    cfg = DynamicalUncertaintyConfig(include_sigma_dyn=True)
    canonical = _canonical_from_manual_inputs(scale_i=1.0, scale_sigma=1.0)
    result = run_dynamical_uncertainty_pipeline(canonical, config=cfg).uncertainty_table

    expected = {
        "obs_id",
        "source",
        "dataset_id",
        "event_id",
        "frame_input",
        "frame_index",
        "frame_number",
        "h",
        "k",
        "l",
        "ori_mean",
        "ori_std",
        "ori_cv",
        "ori_range",
        "ori_grad_rms",
        "ori_curvature",
        "ori_multipeak_score",
        "zone_axis_proximity",
        "zone_axis_layer_density",
        "zone_axis_score",
        "thick_mean",
        "thick_std",
        "thick_cv",
        "thick_range",
        "thick_derivative_rms",
        "thick_max_min_ratio",
        "risk_orientation",
        "risk_thickness",
        "risk_total",
        "risk_total_norm",
        "dyn_sigma_rel",
        "dyn_uncertainty_rel",
        "sigma_dyn",
    }
    assert expected.issubset(set(result.columns))
    assert np.all(np.isfinite(result["dyn_sigma_rel"].to_numpy(dtype=float)))
    assert np.all(result["dyn_sigma_rel"].to_numpy(dtype=float) >= 1.0)


def test_stream_adapter_to_canonical(tmp_path) -> None:
    stream_text = """\
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
a = 15.12 A
b = 15.12 A
c = 12.08 A
al = 90.00 deg
be = 90.00 deg
ga = 90.00 deg
----- End unit cell -----
----- Begin chunk -----
Event: //1
Image serial number: 1
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
"""
    path = tmp_path / "toy.stream"
    path.write_text(stream_text)
    canonical = canonical_observations_from_stream(path)
    assert canonical.source == "stream"
    assert canonical.observations.shape[0] == 1
    assert {"UB11", "UB12", "UB13", "obs_id", "frame_index", "event_id"} <= set(canonical.observations.columns)

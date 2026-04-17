from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline import AnalysisConfig, analyze_single_reflection_dataset
from src.synthetic import write_synthetic_dataset


def test_pipeline_runs_on_synthetic_dataset(tmp_path: Path) -> None:
    dataset = write_synthetic_dataset(tmp_path / "synthetic_dataset")
    results = analyze_single_reflection_dataset(
        gxparm_path=dataset["gxparm"],
        xds_inp_path=dataset["xds_inp"],
        spot_xds_path=dataset["spot_xds"],
        integrate_hkl_path=dataset["integrate_hkl"],
        image_glob=str(dataset["image_glob"]),
        image_template=None,
        config=AnalysisConfig(
            dataset_name="synthetic",
            thickness_nm=200.0,
            hkl=(1, 0, 0),
            relevance_mode="window",
            window_half_width=4,
            patch_half_size=7,
        ),
        output_dir=tmp_path / "analysis",
    )
    assert not results.curve.empty
    assert results.curve["fit_success"].any()
    assert (tmp_path / "analysis" / "rocking_curve.csv").exists()
    assert (tmp_path / "analysis" / "rocking_curve.png").exists()
    good = results.curve[results.curve["fit_success"]]
    assert np.nanmax(good["I_fit"].to_numpy(dtype=float)) > np.nanmin(good["I_fit"].to_numpy(dtype=float))

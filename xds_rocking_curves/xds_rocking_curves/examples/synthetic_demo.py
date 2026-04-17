from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AnalysisConfig, analyze_single_reflection_dataset
from src.synthetic import write_synthetic_dataset


if __name__ == "__main__":
    demo_root = PROJECT_ROOT / "synthetic_demo_output"
    dataset = write_synthetic_dataset(demo_root / "synthetic_dataset")
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
        output_dir=demo_root / "analysis",
    )
    print(results.curve[["frame", "I_fit", "fit_success"]])
    print(f"Wrote synthetic demo to: {demo_root}")

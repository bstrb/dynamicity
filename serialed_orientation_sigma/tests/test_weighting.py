from __future__ import annotations

import numpy as np
import pandas as pd

from src.weighting import WeightingConfig, apply_orientation_aware_weighting


def test_weighting_inflates_sigma_and_downweights() -> None:
    table = pd.DataFrame(
        {
            "sigma": [1.0, 2.0],
            "S": [0.0, 1.5],
        }
    )
    weighted = apply_orientation_aware_weighting(table, WeightingConfig(alpha=0.5, filter_threshold=1.0))
    assert np.isclose(weighted.loc[0, "sigma_new"], 1.0)
    assert weighted.loc[1, "sigma_new"] > weighted.loc[1, "sigma"]
    assert weighted.loc[1, "weight_new"] < weighted.loc[1, "weight_base"]
    assert bool(weighted.loc[0, "keep"]) is True
    assert bool(weighted.loc[1, "keep"]) is False

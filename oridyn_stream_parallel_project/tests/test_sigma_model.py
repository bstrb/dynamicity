import pandas as pd

from oridyn.config import OridynConfig, ScoreWeights
from oridyn.sigma_model import add_sigma_model, combine_score_terms


def test_score_combination_and_sigma_model():
    scores = pd.DataFrame(
        {
            "self_risk_norm": [1.0],
            "graph_crowding_norm": [2.0],
            "same_laue_zone_crowding_norm": [0.5],
            "systematic_row_risk_norm": [0.25],
            "frame_axis_risk_norm": [0.1],
            "sigma": [2.0],
        }
    )
    config = OridynConfig(weights=ScoreWeights(interaction=0.0), alpha=0.5)
    combined = combine_score_terms(scores, config)
    out = add_sigma_model(combined, config)

    assert out.loc[0, "S_dyn_geom"] > 0.0
    assert out.loc[0, "sigma_dyn_rel"] > 1.0
    assert out.loc[0, "sigma_dyn"] > 2.0
    assert out.loc[0, "weight_dyn"] > 0.0

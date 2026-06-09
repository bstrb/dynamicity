import pandas as pd
import pytest

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


def test_score_combination_rescales_by_active_weight_sum():
    scores = pd.DataFrame(
        {
            "self_risk_norm": [1.0],
            "graph_crowding_norm": [1.0],
            "same_laue_zone_crowding_norm": [0.0],
            "systematic_row_risk_norm": [0.0],
            "frame_axis_risk_norm": [0.0],
        }
    )
    config = OridynConfig(
        weights=ScoreWeights(self=1.0, graph=1.0, zone=0.0, row=0.0, frame=0.0, interaction=0.0),
        score_rescale_by_weights=True,
    )

    out = combine_score_terms(scores, config)

    assert out.loc[0, "S_dyn_geom_weighted_sum"] == 2.0
    assert out.loc[0, "S_dyn_geom_weight_denominator"] == 2.0
    assert out.loc[0, "S_dyn_geom"] == 1.0


def test_sigma_model_linear_without_tail_gate():
    scores = pd.DataFrame({"S_dyn_geom": [0.0, 0.5, 1.0]})
    config = OridynConfig(alpha=2.0, sigma_map="linear", sigma_tail_quantile=0.0)

    out = add_sigma_model(scores, config)

    assert out["sigma_tail_score"].tolist() == [0.0, 0.5, 1.0]
    assert out["sigma_dyn_rel"].tolist() == [1.0, 2.0, 3.0]


def test_sigma_model_exponential_percentile_tail():
    scores = pd.DataFrame({"S_dyn_geom": [0.0, 0.5, 1.0]})
    config = OridynConfig(alpha=1.0, sigma_map="exponential", sigma_tail_quantile=0.5)

    out = add_sigma_model(scores, config)

    assert out["sigma_tail_threshold"].tolist() == [0.5, 0.5, 0.5]
    assert out["sigma_tail_score"].tolist() == [0.0, 0.0, 1.0]
    assert out.loc[0, "sigma_dyn_rel"] == 1.0
    assert out.loc[1, "sigma_dyn_rel"] == 1.0
    assert out.loc[2, "sigma_dyn_rel"] == pytest.approx(2.718281828)

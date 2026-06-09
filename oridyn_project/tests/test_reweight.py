from __future__ import annotations

import pandas as pd

from oridyn.reweight import cli_weight_preset, reweight_scores


def test_reweight_scores_adds_preset_columns() -> None:
    table = pd.DataFrame(
        {
            "self_risk_norm": [1.0, 2.0],
            "graph_crowding_norm": [0.5, 0.0],
            "same_laue_zone_crowding_norm": [0.0, 1.0],
            "systematic_row_risk_norm": [0.0, 0.0],
            "frame_axis_risk_norm": [1.0, 1.0],
        }
    )
    presets = {
        "self_only": {"self": 1.0, "graph": 0.0, "zone": 0.0, "row": 0.0, "frame": 0.0, "interaction": 0.0},
        "graph_heavy": {"self": 0.0, "graph": 2.0, "zone": 0.0, "row": 0.0, "frame": 0.0, "interaction": 0.0},
    }
    out = reweight_scores(table, presets, alpha=1.0)
    assert "S_dyn_geom_self_only" in out
    assert "S_dyn_geom_graph_heavy" in out
    assert out.loc[0, "S_dyn_geom_self_only"] == 1.0
    assert out.loc[0, "S_dyn_geom_graph_heavy"] == 0.5
    assert out.loc[0, "S_dyn_geom_weight_denominator_graph_heavy"] == 2.0


def test_reweight_scores_can_use_linear_sigma_without_tail_gate() -> None:
    table = pd.DataFrame(
        {
            "self_risk_norm": [0.0, 1.0],
            "graph_crowding_norm": [0.0, 0.0],
            "same_laue_zone_crowding_norm": [0.0, 0.0],
            "systematic_row_risk_norm": [0.0, 0.0],
            "frame_axis_risk_norm": [0.0, 0.0],
            "sigma": [2.0, 2.0],
        }
    )
    presets = {"self_only": {"self": 1.0, "graph": 0.0, "zone": 0.0, "row": 0.0, "frame": 0.0, "interaction": 0.0}}

    out = reweight_scores(table, presets, alpha=2.0, sigma_map="linear", overwrite_single=True)

    assert out["S_dyn_geom"].tolist() == [0.0, 1.0]
    assert out["sigma_dyn_rel"].tolist() == [1.0, 3.0]
    assert out["sigma_dyn"].tolist() == [2.0, 6.0]


def test_cli_weight_preset_uses_named_values() -> None:
    presets = cli_weight_preset(1, 2, 3, 4, 5, 6, name="test")
    assert presets["test"]["graph"] == 2.0
    assert presets["test"]["interaction"] == 6.0

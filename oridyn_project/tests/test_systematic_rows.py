from pathlib import Path

import pandas as pd

from oridyn.config import OridynConfig
from oridyn.excitation import score_observed_reflections
from oridyn.hkl_generation import generate_candidate_hkls
from oridyn.stream_parser import parse_crystfel_stream
from oridyn.systematic_rows import add_affine_row_terms_from_nodes, add_systematic_row_terms


def test_systematic_row_terms_are_added():
    stream = parse_crystfel_stream(Path("examples") / "minimal_example.stream")
    config = OridynConfig(dmin=2.0, dmax=10.0, sg0=0.05, neighbor_excitation_min=0.0)
    candidates, _ = generate_candidate_hkls(stream.unit_cell, dmin=2.0, dmax=10.0)
    scores = score_observed_reflections(stream, config)
    out = add_systematic_row_terms(scores, stream, candidates, config)

    assert {
        "nearest_row_direction",
        "best_affine_row_direction",
        "row_excited_count",
        "row_excitation_sum",
        "affine_row_crowding_raw",
        "systematic_row_risk_raw",
    } <= set(out.columns)
    assert out["systematic_row_risk_raw"].ge(0).all()


def test_affine_row_terms_detect_offset_row():
    config = OridynConfig(row_direction_limit=1, row_max_steps=2)
    scores = pd.DataFrame({"frame": [0], "h": [16], "k": [-11], "l": [-5]})
    nodes = pd.DataFrame(
        {
            "h": [15, 17],
            "k": [-11, -11],
            "l": [-6, -4],
            "excitation_weight": [0.8, 0.6],
        }
    )

    out = add_affine_row_terms_from_nodes(scores, nodes, config)

    assert out.loc[0, "best_affine_row_direction"] == "(1 0 1)"
    assert out.loc[0, "row_excited_count"] == 2
    assert out.loc[0, "row_excitation_sum"] > 0.0
    assert out.loc[0, "systematic_row_risk_raw"] > 0.0
    assert out.loc[0, "affine_row_crowding_raw"] == out.loc[0, "systematic_row_risk_raw"]


def test_affine_row_terms_still_detect_origin_row_case():
    config = OridynConfig(row_direction_limit=1, row_max_steps=2)
    scores = pd.DataFrame({"frame": [0], "h": [-6], "k": [0], "l": [6]})
    nodes = pd.DataFrame(
        {
            "h": [-7, -5],
            "k": [0, 0],
            "l": [7, 5],
            "excitation_weight": [0.4, 0.7],
        }
    )

    out = add_affine_row_terms_from_nodes(scores, nodes, config)

    assert out.loc[0, "best_affine_row_direction"] == "(1 0 -1)"
    assert out.loc[0, "row_excited_count"] == 2
    assert out.loc[0, "systematic_row_risk_raw"] > 0.0

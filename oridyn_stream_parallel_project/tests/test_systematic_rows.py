from pathlib import Path

from oridyn.config import OridynConfig
from oridyn.excitation import score_observed_reflections
from oridyn.hkl_generation import generate_candidate_hkls
from oridyn.stream_parser import parse_crystfel_stream
from oridyn.systematic_rows import add_systematic_row_terms


def test_systematic_row_terms_are_added():
    stream = parse_crystfel_stream(Path("examples") / "minimal_example.stream")
    config = OridynConfig(dmin=2.0, dmax=10.0, sg0=0.05, neighbor_excitation_min=0.0)
    candidates, _ = generate_candidate_hkls(stream.unit_cell, dmin=2.0, dmax=10.0)
    scores = score_observed_reflections(stream, config)
    out = add_systematic_row_terms(scores, stream, candidates, config)

    assert {"nearest_row_direction", "row_excited_count", "row_excitation_sum", "systematic_row_risk_raw"} <= set(
        out.columns
    )
    assert out["systematic_row_risk_raw"].ge(0).all()

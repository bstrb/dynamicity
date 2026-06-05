from pathlib import Path

from oridyn.config import OridynConfig
from oridyn.excitation import score_observed_reflections
from oridyn.graph_crowding import add_graph_crowding_terms
from oridyn.hkl_generation import generate_candidate_hkls
from oridyn.self_risk import add_self_risk_terms
from oridyn.stream_parser import parse_crystfel_stream


def test_graph_crowding_adds_sparse_neighbor_terms():
    stream = parse_crystfel_stream(Path("examples") / "minimal_example.stream")
    candidates, _ = generate_candidate_hkls(stream.unit_cell, dmin=2.0, dmax=10.0)
    config = OridynConfig(dmin=2.0, dmax=10.0, sg0=0.05, neighbor_excitation_min=0.0)
    scores = add_self_risk_terms(score_observed_reflections(stream, config), config)
    out = add_graph_crowding_terms(scores, stream, candidates, config)

    assert "graph_crowding_raw" in out
    assert "effective_neighbor_count" in out
    assert (out["graph_crowding_raw"] >= 0.0).all()

from pathlib import Path

from oridyn.axis_prediction import predict_problematic_axes
from oridyn.config import OridynConfig
from oridyn.excitation import score_observed_reflections
from oridyn.frame_scoring import compute_frame_scores
from oridyn.hkl_generation import generate_candidate_hkls
from oridyn.laue_zone import add_laue_zone_terms
from oridyn.stream_parser import parse_crystfel_stream


def test_laue_zone_terms_are_added():
    stream = parse_crystfel_stream(Path("examples") / "minimal_example.stream")
    config = OridynConfig(dmin=2.0, dmax=10.0, uvw_max=2, sg0=0.05, neighbor_excitation_min=0.0)
    candidates, _ = generate_candidate_hkls(stream.unit_cell, dmin=2.0, dmax=10.0)
    axes = predict_problematic_axes(candidates, uvw_max=2)
    frames = compute_frame_scores(stream, axes, candidates, config)
    scores = score_observed_reflections(stream, config)
    out = add_laue_zone_terms(scores, frames, stream, candidates, config)

    assert {"assigned_zone_axis", "laue_n", "is_zolz", "same_laue_zone_crowding_raw"} <= set(out.columns)
    assert out["abs_laue_n"].ge(0).all()

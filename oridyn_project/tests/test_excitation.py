from pathlib import Path

from oridyn.config import OridynConfig
from oridyn.excitation import score_observed_reflections
from oridyn.stream_parser import parse_crystfel_stream


def test_observed_excitation_scoring_ignores_intensity_columns():
    stream = parse_crystfel_stream(Path("examples") / "minimal_example.stream")
    scores = score_observed_reflections(stream, OridynConfig(sg0=0.02))

    assert len(scores) == 6
    assert {"sg", "excitation_weight", "q_invA", "d_angstrom"} <= set(scores.columns)
    assert "I" not in scores.columns
    assert "peak" not in scores.columns
    assert "background" not in scores.columns

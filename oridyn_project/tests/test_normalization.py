import pandas as pd

from oridyn.config import OridynConfig
from oridyn.normalization import normalize_frame_terms, normalize_reflection_terms


def test_reflection_and_frame_normalization():
    scores = pd.DataFrame(
        {
            "q_invA": [0.1, 0.2, 0.3, 0.4],
            "self_risk_raw": [0.0, 1.0, 2.0, 3.0],
            "graph_crowding_raw": [0.0, 0.5, 1.0, 1.5],
            "same_laue_zone_crowding_raw": [0.0, 0.0, 1.0, 1.0],
            "systematic_row_risk_raw": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out, metadata = normalize_reflection_terms(scores, OridynConfig(resolution_shells=2, normalization="percentile"))

    assert "resolution_shell" in out
    assert "self_risk_norm" in out
    assert metadata

    frames = pd.DataFrame({"frame": [0, 1, 2], "frame_axis_risk_raw": [0.0, 0.5, 1.0]})
    frame_out, frame_metadata = normalize_frame_terms(frames, OridynConfig(normalization="percentile"))
    assert "frame_axis_risk_norm" in frame_out
    assert frame_metadata


def test_global_minmax_normalization_is_global_and_unclipped():
    scores = pd.DataFrame(
        {
            "q_invA": [0.1, 0.2, 0.3, 0.4],
            "self_risk_raw": [0.0, 1.0, 2.0, 4.0],
            "graph_crowding_raw": [10.0, 20.0, 30.0, 50.0],
            "same_laue_zone_crowding_raw": [0.0, 0.0, 0.0, 0.0],
            "systematic_row_risk_raw": [2.0, 2.0, 2.0, 2.0],
        }
    )

    out, metadata = normalize_reflection_terms(scores, OridynConfig(normalization="global_minmax"))

    assert "resolution_shell" not in out
    assert out["self_risk_norm"].tolist() == [0.0, 0.25, 0.5, 1.0]
    assert out["graph_crowding_norm"].tolist() == [0.0, 0.25, 0.5, 1.0]
    assert out["same_laue_zone_crowding_norm"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert {item["scope"] for item in metadata} == {"global"}


def test_normalization_clip_is_configurable():
    frames = pd.DataFrame({"frame": [0, 1, 2], "frame_axis_risk_raw": [0.0, 1.0, 100.0]})
    frame_out, _ = normalize_frame_terms(frames, OridynConfig(normalization="median_mad", normalization_clip=2.0))

    assert frame_out["frame_axis_risk_norm"].max() == 2.0


def test_frame_rank_normalization_override_spreads_values():
    frames = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "frame_axis_risk_raw": [1e-12, 1e-6, 0.1, 0.8],
        }
    )
    config = OridynConfig(
        normalization="median_mad",
        frame_normalization="rank",
        normalization_clip=6.0,
    )

    frame_out, metadata = normalize_frame_terms(frames, config)

    assert frame_out["frame_axis_risk_norm"].tolist() == [0.0, 2.0, 4.0, 6.0]
    assert metadata[0]["method"] == "rank"
    assert metadata[0]["frame_normalization_override"] == "rank"

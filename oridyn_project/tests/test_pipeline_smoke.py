from pathlib import Path

import pandas as pd

from oridyn.config import OridynConfig
from oridyn.outputs import COMPACT_REFLECTION_COLUMNS
from oridyn.pipeline import run_pipeline


def test_pipeline_smoke_writes_required_outputs(tmp_path):
    output = tmp_path / "oridyn_out"
    config = OridynConfig(
        dmin=2.0,
        dmax=10.0,
        uvw_max=2,
        sg0=0.05,
        neighbor_excitation_min=0.0,
        max_candidates=5000,
        resolution_shells=2,
        workers=2,
    )
    run_pipeline(Path("examples") / "minimal_example.stream", output, config)

    required = [
        "problematic_axes.csv",
        "frame_summary.csv",
        "reflection_scores.csv",
        "score_terms_summary.csv",
        "run_metadata.json",
        "top_self_risk.csv",
        "top_graph_crowding_risk.csv",
        "top_systematic_row_risk.csv",
        "top_laue_zone_risk.csv",
        "summary_close_risky_axis_frames.csv",
        "summary_high_dynamical_frames.csv",
        "summary_frame_metric_correlations.csv",
        "summary_axis_group_metrics.csv",
        "summary_top_reflections_in_high_dynamical_frames.csv",
        "plots/score_distributions.png",
        "plots/frame_risk_trace.png",
        "plots/score_term_correlations.png",
    ]
    for relative in required:
        assert (output / relative).exists(), relative

    scores = pd.read_csv(output / "reflection_scores.csv")
    frames = pd.read_csv(output / "frame_summary.csv")
    assert list(scores.columns) == COMPACT_REFLECTION_COLUMNS
    assert not (output / "reflection_scores_full.csv").exists()
    assert set(scores["source_filename"]) == {"examples/frame-0001.h5", "examples/frame-0002.h5"}
    assert "S_dyn_geom" in scores
    assert "assigned_axis_rank" in frames
    assert "sigma_dyn_rel" in scores
    assert "I" not in scores.columns
    assert "self_risk_raw" not in scores.columns


def test_pipeline_writes_full_reflection_scores_when_requested(tmp_path):
    output = tmp_path / "oridyn_out_full"
    config = OridynConfig(
        dmin=2.0,
        dmax=10.0,
        uvw_max=2,
        sg0=0.05,
        neighbor_excitation_min=0.0,
        max_candidates=5000,
        resolution_shells=2,
        workers=1,
        write_full_reflection_scores=True,
    )
    run_pipeline(Path("examples") / "minimal_example.stream", output, config)

    compact = pd.read_csv(output / "reflection_scores.csv")
    full = pd.read_csv(output / "reflection_scores_full.csv")
    assert list(compact.columns) == COMPACT_REFLECTION_COLUMNS
    assert "source_filename" in full.columns
    assert "self_risk_raw" in full.columns
    assert "top_neighbor_summary" in full.columns
    assert len(full.columns) > len(compact.columns)

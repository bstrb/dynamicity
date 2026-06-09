from oridyn.axis_prediction import mark_active_problematic_axes, predict_problematic_axes
from oridyn.hkl_generation import generate_candidate_hkls
from oridyn.stream_parser import UnitCell


def test_axis_prediction_scores_zolz_density():
    cell = UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    candidates, _ = generate_candidate_hkls(cell, dmin=3.0, dmax=10.0)
    axes = predict_problematic_axes(candidates, uvw_max=2)

    assert {"u", "v", "w", "axis_score", "zolz_weighted_count"} <= set(axes.columns)
    assert axes["axis_score"].max() == 1.0
    assert len(axes) > 0


def test_active_axis_cap_marks_only_top_ranked_axes():
    cell = UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    candidates, _ = generate_candidate_hkls(cell, dmin=3.0, dmax=10.0)
    axes = predict_problematic_axes(candidates, uvw_max=3)
    marked = mark_active_problematic_axes(axes, max_problematic_axes=5)

    assert marked["used_for_scoring"].sum() == 5
    assert marked.loc[marked["used_for_scoring"], "axis_rank"].max() == 5
    assert not marked.loc[marked["axis_rank"] > 5, "used_for_scoring"].any()

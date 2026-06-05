import pandas as pd

from oridyn.summaries import make_information_summaries


def test_information_summaries_pick_high_frames_and_reflections():
    frames = pd.DataFrame(
        {
            "frame": [0, 1],
            "frame_number": [1, 2],
            "event": ["a", "b"],
            "assigned_risky_axis": ["[1 0 0]", "[0 1 0]"],
            "assigned_axis_angle_deg": [0.5, 5.0],
            "nearest_zone_axis": ["[1 0 0]", "[0 1 0]"],
            "nearest_zone_axis_angle_deg": [0.5, 5.0],
            "frame_axis_risk_raw": [0.9, 0.1],
            "frame_axis_risk_norm": [6.0, 0.0],
            "n_excited": [10, 5],
            "sum_excitation_weight": [8.0, 2.0],
            "n_observed_targets": [2, 2],
            "mean_S_dyn_geom": [10.0, 1.0],
            "p95_S_dyn_geom": [12.0, 1.5],
            "mean_sigma_dyn_rel": [20.0, 2.0],
        }
    )
    reflections = pd.DataFrame(
        {
            "reflection_id": [0, 1, 2, 3],
            "frame": [0, 0, 1, 1],
            "frame_number": [1, 1, 2, 2],
            "event": ["a", "a", "b", "b"],
            "h": [1, 2, 3, 4],
            "k": [0, 0, 0, 0],
            "l": [0, 0, 0, 0],
            "q_invA": [0.1, 0.2, 0.3, 0.4],
            "d_angstrom": [10.0, 5.0, 3.3, 2.5],
            "sg": [0.0, 0.01, 0.02, 0.03],
            "excitation_weight": [1.0, 0.9, 0.8, 0.7],
            "self_risk_raw": [1.0, 0.9, 0.1, 0.2],
            "graph_crowding_raw": [1.0, 0.5, 0.2, 0.1],
            "same_laue_zone_crowding_raw": [2.0, 1.0, 0.2, 0.1],
            "systematic_row_risk_raw": [0.5, 0.4, 0.1, 0.1],
            "S_dyn_geom": [30.0, 20.0, 2.0, 1.0],
            "sigma_dyn_rel": [100.0, 50.0, 2.0, 1.5],
            "assigned_zone_axis": ["[1 0 0]"] * 4,
            "laue_n": [1, 2, 3, 4],
            "nearest_row_direction": ["(1 0 0)"] * 4,
            "top_neighbor_summary": [""] * 4,
        }
    )

    summaries = make_information_summaries(frames, reflections, axis_sigma_deg=2.0, top_reflections_per_frame=1)

    assert "summary_high_dynamical_frames.csv" in summaries
    assert summaries["summary_close_risky_axis_frames.csv"].iloc[0]["frame"] == 0
    top = summaries["summary_top_reflections_in_high_dynamical_frames.csv"]
    assert len(top) == 1
    assert top.iloc[0]["reflection_id"] == 0
    assert "mean_S_dyn_geom" in top.columns

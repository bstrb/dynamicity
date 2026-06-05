from __future__ import annotations

from pathlib import Path

import pandas as pd

from oridyn.config import OridynConfig
from oridyn.hkl_trace import build_hkl_target_table, load_hkl_targets, run_hkl_trace
from oridyn.stream_parser import parse_crystfel_stream


def test_load_hkl_targets(tmp_path: Path) -> None:
    path = tmp_path / "hkls.csv"
    path.write_text("h,k,l,label\n1,0,0,100\n1,0,0,duplicate\n0,1,0,010\n")
    targets = load_hkl_targets(path)
    assert list(targets["hkl_label"]) == ["100", "010"]
    assert len(targets) == 2


def test_build_hkl_target_table_crosses_frames_and_hkls() -> None:
    stream = parse_crystfel_stream("examples/minimal_example.stream")
    targets = pd.DataFrame({"target_index": [0, 1], "h": [1, 0], "k": [0, 1], "l": [0, 0], "hkl_label": ["100", "010"]})
    table = build_hkl_target_table(stream, targets)
    assert len(table) == len(stream.crystal_table) * len(targets)
    assert {"frame", "h", "k", "l", "hkl_label", "target_source"}.issubset(table.columns)
    assert set(table["target_source"]) == {"selected_hkl"}


def test_run_hkl_trace_writes_trajectory_table(tmp_path: Path) -> None:
    hkls = tmp_path / "hkls.csv"
    hkls.write_text("h,k,l,label\n1,0,0,100\n0,1,0,010\n")
    out = tmp_path / "trace"
    cfg = OridynConfig(
        dmin=0.6,
        dmax=20.0,
        uvw_max=2,
        max_candidates=2000,
        normalization="rank",
        frame_normalization="rank",
        progress=False,
    )
    run_hkl_trace("examples/minimal_example.stream", hkls, out, cfg)
    trajectories = pd.read_csv(out / "hkl_frame_trajectories.csv")
    assert len(trajectories) == 4
    assert {"S_dyn_geom", "self_risk_norm", "graph_crowding_norm", "hkl_label"}.issubset(trajectories.columns)
    assert (out / "plots" / "hkl_score_heatmap.png").exists()

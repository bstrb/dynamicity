from pathlib import Path

import pandas as pd

from oridyn.stream_rewrite import rewrite_stream_sigmas


def test_rewrite_stream_sigmas_replaces_reflection_sigma(tmp_path):
    scores = pd.DataFrame(
        {
            "frame": [0, 0, 0, 1, 1, 1],
            "h": [1, 0, 1, 1, 0, 1],
            "k": [0, 1, 1, 0, 1, 1],
            "l": [0, 0, 0, 0, 0, 0],
            "sigma": [10.0, 9.0, 8.0, 9.5, 8.5, 7.5],
            "sigma_dyn_rel": [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
            "sigma_dyn": [20.0, 18.0, 16.0, 28.5, 25.5, 22.5],
        }
    )
    scores_path = tmp_path / "scores.csv"
    scores.to_csv(scores_path, index=False)
    output = tmp_path / "rewritten.stream"

    stats = rewrite_stream_sigmas(Path("examples") / "minimal_example.stream", scores_path, output)
    text = output.read_text()

    assert stats.reflection_rows_seen == 6
    assert stats.reflection_rows_rewritten == 6
    assert "          20" in text
    assert "        28.5" in text

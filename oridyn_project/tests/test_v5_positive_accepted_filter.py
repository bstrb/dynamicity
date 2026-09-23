from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "filter_stream_by_v5_nonself_positive_accepted_fraction.py"
SPEC = importlib.util.spec_from_file_location("v5_positive_filter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
v5_filter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v5_filter)


def make_row(index: int, score: float, h: int = 1, k: int = 0, l: int = 0) -> dict[str, object]:
    return {
        "source_filename": "image.h5",
        "event": str(index),
        "h": h,
        "k": k,
        "l": l,
        "nonself_local_excitation_raw": score,
    }


def test_positive_keep_fraction_never_removes_zero_scores_and_respects_min_keep() -> None:
    rows = [make_row(index, 0.0) for index in range(15)]
    rows.extend(make_row(15 + index, float(index + 1)) for index in range(10))
    accepted = pd.DataFrame(rows)

    key_to_mask, sweep, accepted_hkl_summary, removed_by_hkl, selection_table, stats = v5_filter.build_filter_masks(
        accepted,
        "nonself_local_excitation_raw",
        [0.5],
        min_accepted_keep=20,
        positive_threshold=0.0,
    )

    variant = v5_filter.variant_name(0.5)
    removed = [key for key, mask in key_to_mask.items() if mask]
    removed_events = sorted(int(key[1]) for key in removed)

    assert len(key_to_mask) == 25
    assert removed_events == [20, 21, 22, 23, 24]
    assert all(int(key[1]) >= 15 for key in removed)
    assert int(removed_by_hkl[variant]["remove_n"].iloc[0]) == 5
    assert int(selection_table[f"remove_{variant}"].sum()) == 5
    assert int(removed_by_hkl[variant]["kept_accepted_after_filter"].iloc[0]) == 20
    assert int(accepted_hkl_summary["n_zero_or_below_threshold"].iloc[0]) == 15
    assert int(sweep["positive_risk_accepted_observations"].iloc[0]) == 10
    assert stats["zero_or_below_threshold_accepted_observations"] == 15


def test_low_count_and_no_positive_hkls_are_kept_unchanged() -> None:
    low_count_positive = [make_row(index, float(index + 1), h=2) for index in range(19)]
    no_positive = [make_row(100 + index, 0.0, h=3) for index in range(25)]
    accepted = pd.DataFrame(low_count_positive + no_positive)

    key_to_mask, _sweep, _accepted_hkl_summary, removed_by_hkl, _selection_table, stats = v5_filter.build_filter_masks(
        accepted,
        "nonself_local_excitation_raw",
        [0.5],
        min_accepted_keep=20,
        positive_threshold=0.0,
    )

    variant = v5_filter.variant_name(0.5)
    assert all(mask == 0 for mask in key_to_mask.values())
    assert removed_by_hkl[variant].empty
    assert stats["unique_signed_hkls_low_count_kept_unchanged"] == 1
    assert stats["unique_signed_hkls_no_positive_risk_kept_unchanged"] == 1


def test_selection_only_key_csvs_include_removed_and_kept_positive_sets(tmp_path: Path) -> None:
    rows = [make_row(index, 0.0) for index in range(15)]
    rows.extend(make_row(15 + index, float(index + 1)) for index in range(10))
    accepted = pd.DataFrame(rows)

    _key_to_mask, _sweep, _accepted_hkl_summary, _removed_by_hkl, selection_table, _stats = v5_filter.build_filter_masks(
        accepted,
        "nonself_local_excitation_raw",
        [0.5],
        min_accepted_keep=20,
        positive_threshold=0.0,
    )
    paths = v5_filter.output_paths(tmp_path, [0.5])
    key_stats = v5_filter.write_selection_key_csvs(paths, selection_table, [0.5], "nonself_local_excitation_raw")

    variant = v5_filter.variant_name(0.5)
    removed = pd.read_csv(paths[variant]["removed_keys_csv"])
    kept_positive = pd.read_csv(paths[variant]["kept_positive_keys_csv"])

    assert key_stats[variant] == {"removed_key_rows": 5, "kept_positive_key_rows": 5}
    assert set(removed["event"].astype(int)) == {20, 21, 22, 23, 24}
    assert set(kept_positive["event"].astype(int)) == {15, 16, 17, 18, 19}
    assert (removed["nonself_local_excitation_raw"] > 0.0).all()
    assert (kept_positive["nonself_local_excitation_raw"] > 0.0).all()


def test_loads_accepted_keys_and_joins_v5_scores_by_normalized_exact_key(tmp_path: Path) -> None:
    accepted_csv = tmp_path / "accepted.csv"
    v5_csv = tmp_path / "v5.csv"
    pd.DataFrame(
        [
            {"source_filename": " image.h5 ", "event": "//1", "h": 1, "k": 0, "l": 0},
            {"source_filename": "image.h5", "event": "1", "h": 2, "k": 0, "l": 0},
        ]
    ).to_csv(accepted_csv, index=False)
    pd.DataFrame(
        [
            {"source_filename": "image.h5", "event": "1", "h": 1, "k": 0, "l": 0, "nonself_local_excitation_raw": 2.5},
            {"source_filename": "image.h5", "event": "//1", "h": 2, "k": 0, "l": 0, "nonself_local_excitation_raw": 0.0},
            {"source_filename": "image.h5", "event": "1", "h": 9, "k": 9, "l": 9, "nonself_local_excitation_raw": 99.0},
        ]
    ).to_csv(v5_csv, index=False)

    accepted_keys, accepted_stats = v5_filter.load_accepted_keys(accepted_csv, None, chunksize=1)
    accepted_v5, score_stats = v5_filter.load_accepted_v5_scores(
        v5_csv,
        accepted_keys,
        "nonself_local_excitation_raw",
        chunksize=1,
    )

    assert accepted_stats["accepted_unique_keys"] == 2
    assert score_stats["accepted_v5_matched_rows"] == 2
    assert score_stats["accepted_keys_without_v5_score"] == 0
    assert set(accepted_v5["event"]) == {"1"}
    assert set(accepted_v5["h"]) == {1, 2}
    assert 99.0 not in set(accepted_v5["nonself_local_excitation_raw"])


def test_stream_rewrite_removes_only_selected_accepted_positive_rows(tmp_path: Path) -> None:
    stream = tmp_path / "tiny.stream"
    stream.write_text(
        "----- Begin chunk -----\n"
        "Image filename: image.h5\n"
        "Event: //1\n"
        "----- Begin crystal -----\n"
        "Reflections measured after indexing\n"
        " 1 0 0 100.0 1.0\n"
        " 2 0 0 200.0 1.0\n"
        " 3 0 0 300.0 1.0\n"
        "End of reflections\n"
        "----- End crystal -----\n"
        "----- End chunk -----\n"
        "----- Begin chunk -----\n"
        "Image filename: image.h5\n"
        "Event: //2\n"
        "----- Begin crystal -----\n"
        "Reflections measured after indexing\n"
        " 1 0 0 400.0 1.0\n"
        "End of reflections\n"
        "----- End crystal -----\n"
        "----- End chunk -----\n",
        encoding="utf-8",
    )
    accepted = pd.DataFrame(
        [
            {"source_filename": "image.h5", "event": "1", "h": 1, "k": 0, "l": 0, "nonself_local_excitation_raw": 5.0},
            {"source_filename": "image.h5", "event": "2", "h": 1, "k": 0, "l": 0, "nonself_local_excitation_raw": 1.0},
            {"source_filename": "image.h5", "event": "1", "h": 2, "k": 0, "l": 0, "nonself_local_excitation_raw": 0.0},
        ]
    )
    key_to_mask, _sweep, _accepted_hkl_summary, _removed_by_hkl, _selection_table, _stats = v5_filter.build_filter_masks(
        accepted,
        "nonself_local_excitation_raw",
        [0.5],
        min_accepted_keep=1,
        positive_threshold=0.0,
    )
    paths = v5_filter.output_paths(tmp_path, [0.5])
    summary, stream_stats, removed_by_hkl, matched_hkl_counts = v5_filter.write_stream_variants(
        stream,
        paths,
        key_to_mask,
        [0.5],
        progress_every=100,
        max_events=None,
    )

    variant = v5_filter.variant_name(0.5)
    output_text = paths[variant]["stream"].read_text(encoding="utf-8")

    assert " 1 0 0 100.0 1.0\n" not in output_text
    assert " 1 0 0 400.0 1.0\n" in output_text
    assert " 2 0 0 200.0 1.0\n" in output_text
    assert " 3 0 0 300.0 1.0\n" in output_text
    assert int(summary["removed_observations"].iloc[0]) == 1
    assert int(summary["accepted_matched_observations"].iloc[0]) == 3
    assert int(summary["nonaccepted_or_unmatched_observations"].iloc[0]) == 1
    assert stream_stats["stream_observations_seen"] == 4
    assert removed_by_hkl[variant][(1, 0, 0)] == 1
    assert matched_hkl_counts[(2, 0, 0)] == 1
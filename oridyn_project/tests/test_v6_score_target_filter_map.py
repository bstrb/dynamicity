from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_v6_score_target_filter_map.py"
SPEC = importlib.util.spec_from_file_location("v6_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
v6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v6
SPEC.loader.exec_module(v6)


def make_cache(n: int, h: int = 1, k: int = 0, l: int = 0) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        rows.append(
            {
                "source_filename": "img.h5",
                "event": str(idx),
                "h": h,
                "k": k,
                "l": l + idx if h == 99 else l,
                "Eg": 0.50 + idx / max(2 * n, 1),
                "sg": 0.001 * idx,
                "D": float(idx + 1),
                "U": float(idx + 1),
                "M": float(idx + 1),
                "M2": float(idx + 1),
            }
        )
    table = v6.add_exact_key_text(pd.DataFrame(rows))
    table["sg_abs"] = table["sg"].abs()
    table["signed_hkl_id"] = [f"{row.h},{row.k},{row.l}" for row in table.itertuples(index=False)]
    table["source_order"] = range(1, len(table) + 1)
    scored, _stats = v6.compute_scores(table, v6.score_registry(), v6.RunLogger(None))
    return scored


def test_score_registry_has_expected_scores_and_safe_division_zero() -> None:
    scores = v6.score_registry()
    assert [score.score_id for score in scores] == [
        "eg",
        "density_d",
        "legacy_m",
        "eg_m",
        "eg_cmean",
        "eg_c2mean",
        "eg_m2",
        "eg_d1_cmean",
        "eg_d2_cmean",
        "eg_d3_cmean",
        "eg_d2_c2mean",
    ]

    table = v6.add_exact_key_text(
        pd.DataFrame(
            [
                {
                    "source_filename": "img.h5",
                    "event": "1",
                    "h": 1,
                    "k": 0,
                    "l": 0,
                    "Eg": 0.5,
                    "sg": 0.0,
                    "D": 0.0,
                    "U": 0.0,
                    "M": 0.0,
                    "M2": 0.0,
                }
            ]
        )
    )
    table["sg_abs"] = 0.0
    scored, stats = v6.compute_scores(table, scores, v6.RunLogger(None))
    assert float(scored["score_eg_cmean"].iloc[0]) == 0.0
    assert stats["eg_cmean"]["safe_division"]["eg_cmean_expr_mul1_zero_denominator_count"] == 1


def test_experiment_plan_accounts_for_all_88_streams() -> None:
    variants = v6.build_experiment_plan(v6.score_registry())
    assert len(variants) == 88
    assert sum(variant.experiment_type == "diagnostic_low_high" for variant in variants) == 22
    assert sum(variant.filtering_target == "all" for variant in variants) == 18
    assert sum(variant.filtering_target == "higheg" for variant in variants) == 24
    assert sum(variant.filtering_target == "matched" for variant in variants) == 24


def test_diagnostic_even_and_odd_hkl_splits_are_deterministic() -> None:
    even = make_cache(4)
    odd = make_cache(5, h=2)
    cache = pd.concat([even, odd], ignore_index=True)
    variants = v6.build_experiment_plan(v6.score_registry())
    keep_sets, records, hkl_qc = v6.construct_diagnostic_selections(cache, v6.score_registry(), variants, v6.RunLogger(None))

    even_low = keep_sets["diag_eg_low50"] & set(even["exact_key_text"])
    even_high = keep_sets["diag_eg_high50"] & set(even["exact_key_text"])
    assert len(even_low) == 2
    assert len(even_high) == 2
    assert even_low.isdisjoint(even_high)

    odd_low = keep_sets["diag_eg_low50"] & set(odd["exact_key_text"])
    odd_high = keep_sets["diag_eg_high50"] & set(odd["exact_key_text"])
    omitted = set(odd["exact_key_text"]) - odd_low - odd_high
    assert len(odd_low) == 2
    assert len(odd_high) == 2
    assert len(omitted) == 1
    assert any(row["state"] == "diagnostic_odd_middle_omitted" for row in records)
    assert all(row["validation_passed"] for row in hkl_qc)


def test_target_a_removes_highest_scores_with_minimum_remaining() -> None:
    cache = make_cache(10)
    remove_sets, _records, hkl_qc = v6.construct_target_a_selections(cache, ["eg_m2"], v6.RunLogger(None))
    removed = remove_sets["filter_all_eg_m2_drop20"]
    expected = set(cache.sort_values(["score_eg_m2", "exact_key_text"], ascending=[False, True]).head(2)["exact_key_text"])
    assert removed == expected
    drop05_qc = [row for row in hkl_qc if row["variant_id"] == "filter_all_eg_m2_drop05"][0]
    assert drop05_qc["actionable"] is False
    assert drop05_qc["n_removed"] == 0


def test_target_b_high_eg_pool_and_target_c_block_counts() -> None:
    cache = make_cache(34)
    b_remove_sets, _b_records, _b_qc = v6.construct_target_b_selections(cache, ["eg_m2"], v6.RunLogger(None))
    high_pool = v6.high_eg_pool(cache)
    expected_b = set(high_pool.sort_values(["score_eg_m2", "exact_key_text"], ascending=[False, True]).head(2)["exact_key_text"])
    assert b_remove_sets["filter_higheg_eg_m2_drop20"] == expected_b

    blocks, _block_hkl_qc = v6.construct_target_c_blocks(cache, v6.RunLogger(None))
    assert len(blocks) == 10
    c_remove_sets, _c_records, _c_hkl_qc, block_qc = v6.construct_target_c_selections(blocks, ["eg_m2"], v6.RunLogger(None))
    assert len(c_remove_sets["filter_matched_eg_m2_drop20"]) == 2
    assert len(c_remove_sets["filter_matched_eg_m2_drop30"]) == 3
    assert len(c_remove_sets["filter_matched_eg_m2_drop40"]) == 4
    assert len(c_remove_sets["filter_matched_eg_m2_drop50"]) == 5
    assert all(row["n_retained_in_block"] >= 2 for row in block_qc)


def test_final_small_block_is_merged_using_v5_rule() -> None:
    table = make_cache(14)
    blocks = v6.split_excitation_blocks(table)
    assert [len(block) for block in blocks] == [14]


def test_exact_duplicate_reuse_detection(tmp_path: Path) -> None:
    scores = v6.score_registry()
    variants = v6.build_experiment_plan(scores)
    score = {item.score_id: item for item in scores}["eg_d3_cmean"]
    variant_id = "filter_matched_eg_d3_cmean_drop30"
    cache_path = tmp_path / "cached_multi_score_table.csv.gz"
    cache_path.write_text("", encoding="utf-8")
    existing_stream = tmp_path / "p_1p00_he030_bs010_drop030.stream"
    existing_stream.write_text("stream\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "experiment": "p_1p00_he030_bs010_drop030",
                "expression_tree_json": v6.canonical_json(score.expression_tree),
                "high_eg_fraction": 0.30,
                "drop_fraction": 0.30,
                "excitation_block_size": 10,
                "min_final_block_size": 5,
                "min_high_eg_observations": 10,
                "min_remaining_per_block": 2,
                "expected_output_filename": existing_stream.name,
            }
        ]
    ).to_csv(tmp_path / "experiment_plan.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "p_1p00_he030_bs010_drop030",
                "source_filename": "img.h5",
                "event": "1",
                "h": 1,
                "k": 0,
                "l": 0,
            }
        ]
    ).to_csv(tmp_path / "selected_removal_observations.csv", index=False)
    reuse = v6.validate_existing_reuse(
        variants,
        scores,
        {variant_id: {"img.h5|1|1|0|0"}},
        cache_path,
        tmp_path / "source.stream",
        v6.RunLogger(None),
    )
    assert reuse == {variant_id: str(existing_stream)}


def test_small_stream_rewrite_preserves_order_and_removes_requested(tmp_path: Path) -> None:
    stream = tmp_path / "source.stream"
    stream.write_text(
        "\n".join(
            [
                "----- Begin chunk -----",
                "Image filename: img.h5",
                "Event: //1",
                "--- Begin crystal",
                "Reflections measured after indexing",
                "   1    0    0      10.0       1.0",
                "   2    0    0      20.0       1.0",
                "   3    0    0      30.0       1.0",
                "End of reflections",
                "--- End crystal",
                "----- End chunk -----",
                "",
            ]
        ),
        encoding="utf-8",
    )
    all_cache = {"img.h5|1|1|0|0", "img.h5|1|2|0|0", "img.h5|1|3|0|0"}
    spec = v6.RewriteSpec(
        variant_id="diag_eg_low50",
        output_path=tmp_path / "diag_eg_low50.stream",
        mode="diagnostic_keep",
        keys={"img.h5|1|1|0|0", "img.h5|1|3|0|0"},
        requested_removed=1,
    )
    rows = v6.rewrite_stream_batch(stream, [spec], all_cache, v6.RunLogger(None))
    text = spec.output_path.read_text(encoding="utf-8")
    assert rows[0]["removed_observations"] == 1
    assert "   1    0    0" in text
    assert "   2    0    0" not in text
    assert "   3    0    0" in text

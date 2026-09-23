from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "audit_v6_matched_drop30.py"
SPEC = importlib.util.spec_from_file_location("audit_v6", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_stream(path: Path, hkls: list[tuple[int, int, int]]) -> None:
    rows = [
        "----- Begin chunk -----",
        "Image filename: img.h5",
        "Event: //1",
        "--- Begin crystal",
        "Reflections measured after indexing",
    ]
    rows.extend(f"{h:4d} {k:4d} {l:4d} 10.0 1.0" for h, k, l in hkls)
    rows.extend(["End of reflections", "--- End crystal", "----- End chunk -----", ""])
    write(path, "\n".join(rows))


def write_merge(exp_dir: Path, stem: str, stream_path: Path, cc12: float, rsplit: float, redundancy: float) -> Path:
    merge_dir = exp_dir / f"{stem}_partialator_results_20260716T0100"
    write(merge_dir / "cell.cell", "cell\n")
    write(merge_dir / "crystfel.hkl", "")
    write(merge_dir / "crystfel.hkl1", "half1\n")
    write(merge_dir / "crystfel.hkl2", "half2\n")
    write(merge_dir / "partialator_stdout.log", "")
    write(merge_dir / "partialator_stderr.log", "")
    write(
        merge_dir / "parameters.json",
        json.dumps(
            {
                "merge_wrapper": {
                    "stream_file": str(stream_path),
                    "symmetry": "4/mmm",
                    "model": "offset",
                    "iterations": 10,
                    "min_measurements": 1,
                    "lowres": "20.0",
                    "highres": "0.35",
                    "push_res": "inf",
                    "min_res": "inf",
                    "polarisation": "none",
                    "disable_pr": "true",
                    "no_bscale": True,
                    "partialator_command": (
                        f"partialator {stream_path} --model=offset -j 2 -o {merge_dir / 'crystfel.hkl'} "
                        f"-y 4/mmm --min-measurements=1 --push-res=inf --iterations=10 "
                        f"--harvest-file={merge_dir / 'parameters.json'} --log-folder={merge_dir / 'pr-logs'} "
                        "--polarisation=none --max-adu=inf --min-res=inf --no-Bscale --no-pr"
                    ),
                },
                "merging": {
                    "symmetry": "4/mmm",
                    "partiality_model": "offset",
                    "num_iterations": 10,
                    "min_measurements_per_unique_reflection": 1,
                    "Bscale": False,
                    "post_refine": False,
                },
            }
        ),
    )
    write(
        merge_dir / "metadata_and_outputs.txt",
        "\n".join(
            [
                "=== SUMMARY ===",
                "Completeness: 99.00%",
                f"Redundancy:   {redundancy:.6f}x",
                "SNR:         10.00",
                f"CC1/2:        {cc12:.7f}",
                f"Rsplit:       {rsplit:.6f}",
            ]
        ),
    )
    write(
        merge_dir / "qc_stats" / "check_hkl_completeness.log",
        "\n".join(
            [
                "100 measurements in total.",
                "10 reflections in total.",
                "Overall <snr> = 10.123456",
                f"Overall redundancy = {redundancy:.6f} measurements/unique reflection",
                "Overall completeness = 99.000000 %",
            ]
        ),
    )
    write(merge_dir / "qc_stats" / "compare_cc12.log", f"Overall CC = {cc12:.7f}\n")
    write(merge_dir / "qc_stats" / "compare_rsplit.log", f"Overall Rsplit = {rsplit:.6f} %\n")
    write(
        merge_dir / "qc_stats" / "check_shell.tsv",
        "\n".join(
            [
                "Center 1/nm  # refs Possible  Compl       Meas   Red   SNR     Mean I     d(A)    Min 1/nm   Max 1/nm",
                "     5.000       5        5 100.00         50  10.0 11.00     100.0     2.00       0.500     10.000",
                "    15.000       5        5  98.00         50  10.0  9.00      50.0     0.67      10.000     20.000",
            ]
        ),
    )
    write(
        merge_dir / "qc_stats" / "compare_cc12_shell.tsv",
        "\n".join(
            [
                "  1/d centre       CC       nref      d / A   Min 1/nm    Max 1/nm",
                f"     5.000  {cc12:.7f}          5       2.00      0.500      10.000",
                f"    15.000  {cc12 - 0.01:.7f}          5       0.67     10.000      20.000",
            ]
        ),
    )
    write(
        merge_dir / "qc_stats" / "compare_rsplit_shell.tsv",
        "\n".join(
            [
                "  1/d centre Rsplit/%       nref      d / A   Min 1/nm    Max 1/nm",
                f"     5.000       {rsplit:.6f}          5       2.00      0.500      10.000",
                f"    15.000       {rsplit + 1:.6f}          5       0.67     10.000      20.000",
            ]
        ),
    )
    return merge_dir


def test_synthetic_audit_writes_key_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    v6_dir = root / "oridyn_v6_score_target_filter_map_20260716"
    v5_dir = root / "oridyn_v5_p_lambda_screen_20260716"
    source = root / "MFM300-VIII_cut_20-0_3.stream"
    out_dir = root / "audit_out"
    write_stream(source, [(1, 0, 0), (2, 0, 0), (3, 0, 0)])
    for stem in [
        "filter_matched_eg_d2_cmean_drop30",
        "filter_matched_eg_d3_cmean_drop30",
        "filter_all_eg_m2_drop05",
        "filter_all_eg_m2_drop10",
    ]:
        write_stream(v6_dir / f"{stem}.stream", [(1, 0, 0), (3, 0, 0)])
        write_merge(v6_dir, stem, v6_dir / f"{stem}.stream", 0.997 + len(stem) * 1e-8, 5.4, 532.7)
    write_stream(v5_dir / "p_1p00_he030_bs010_drop030.stream", [(1, 0, 0)])
    write_merge(v5_dir, "p_1p00_he030_bs010_drop030", v5_dir / "p_1p00_he030_bs010_drop030.stream", 0.9971192, 5.51, 517.3)
    write_merge(root, source.stem, source, 0.9970285, 5.45, 534.39)

    write(
        v6_dir / "parameters.json",
        json.dumps(
            {
                "source_stream": str(source),
                "cache": str(v5_dir / "cached_multi_score_table.csv.gz"),
                "filtering_parameters": {
                    "high_Eg_fraction": 0.3,
                    "excitation_block_size": 10,
                    "min_final_block_size": 5,
                    "min_high_Eg_observations": 10,
                    "min_remaining": 2,
                },
            }
        ),
    )
    write(
        v6_dir / "validation.json",
        json.dumps(
            {
                "cache_stats": {"cache_rows": 10},
                "source_stream_validation": {
                    "source_reflection_rows": 3,
                    "source_reflection_rows_without_cache_score": 0,
                },
            }
        ),
    )
    pd.DataFrame(
        [
            {
                "variant_id": stem,
                "experiment_type": "targeted_filter",
                "score_id": stem.replace("filter_matched_", "").replace("filter_all_", "").replace("_drop30", "").replace("_drop05", "").replace("_drop10", ""),
                "score_formula": "Eg * M * D^3 / U" if "d3" in stem else "Eg * D^2 * M / U",
                "designation": "drop30",
                "filtering_target": "matched" if "matched" in stem else "all",
                "high_Eg_fraction": 0.3 if "matched" in stem else "",
                "block_size": 10 if "matched" in stem else "",
                "drop_fraction": 0.3 if "drop30" in stem else 0.05,
                "source_stream": str(source),
                "output_stream": str(v6_dir / f"{stem}.stream"),
                "status": "generated",
                "selected_or_removed_count": 1,
                "retained_count": 2,
                "retained_fraction": 2 / 3,
                "global_accepted_observation_fraction_removed": 0.1,
            }
            for stem in [
                "filter_matched_eg_d2_cmean_drop30",
                "filter_matched_eg_d3_cmean_drop30",
                "filter_all_eg_m2_drop05",
                "filter_all_eg_m2_drop10",
            ]
        ]
    ).to_csv(v6_dir / "stream_manifest.csv", index=False)
    pd.read_csv(v6_dir / "stream_manifest.csv").to_csv(v6_dir / "per_variant_selection_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant_id": "filter_matched_eg_d2_cmean_drop30",
                "output_stream": str(v6_dir / "filter_matched_eg_d2_cmean_drop30.stream"),
                "requested_removals": 1,
                "removed_observations": 1,
                "kept_observations": 2,
                "total_reflection_rows_seen": 3,
            },
            {
                "variant_id": "filter_matched_eg_d3_cmean_drop30",
                "output_stream": str(v6_dir / "filter_matched_eg_d3_cmean_drop30.stream"),
                "requested_removals": 1,
                "removed_observations": 1,
                "kept_observations": 2,
                "total_reflection_rows_seen": 3,
            },
        ]
    ).to_csv(v6_dir / "stream_rewrite_qc.csv", index=False)
    with gzip.open(v6_dir / "selected_observations.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = pd.DataFrame(
            [
                {
                    "variant_id": "filter_matched_eg_d2_cmean_drop30",
                    "state": "removed_by_targeted_filter",
                    "exact_key_text": "img.h5|1|1|0|0",
                    "h": 1,
                    "k": 0,
                    "l": 0,
                },
                {
                    "variant_id": "filter_matched_eg_d3_cmean_drop30",
                    "state": "removed_by_targeted_filter",
                    "exact_key_text": "img.h5|1|2|0|0",
                    "h": 2,
                    "k": 0,
                    "l": 0,
                },
            ]
        )
        writer.to_csv(handle, index=False)
    pd.DataFrame(
        [
            {
                "variant_id": "filter_matched_eg_d3_cmean_drop30",
                "n_removed": 1,
                "n_observations": 10,
                "n_high_eg": 3,
                "n_blocks": 1,
                "n_block_observations": 10,
                "actionable": True,
                "validation_passed": True,
            }
        ]
    ).to_csv(v6_dir / "per_variant_per_hkl_qc.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant_id": "filter_matched_eg_d3_cmean_drop30",
                "block_size": 10,
                "n_removed": 1,
                "n_retained_in_block": 9,
                "validation_passed": True,
            }
        ]
    ).to_csv(v6_dir / "per_variant_per_block_qc.csv", index=False)
    write(v6_dir / "experiment_plan.csv", "variant_id,output_filename\n")
    write(v6_dir / "experiment_plan.json", "[]")
    write(v6_dir / "scores.json", "[]")
    write(v6_dir / "run_metadata.json", "{}")
    write(v6_dir / "run.log", "")

    tree = audit.CANONICAL_V6_EG_D3_CMEAN_TREE
    pd.DataFrame(
        [
            {
                "experiment": "p_1p00_he030_bs010_drop030",
                "formula": "Eg * M * D^3 / U",
                "expression_tree_json": json.dumps(tree),
                "high_eg_fraction": 0.3,
                "drop_fraction": 0.3,
                "excitation_block_size": 10,
                "min_final_block_size": 5,
                "min_high_eg_observations": 10,
                "min_remaining_per_block": 2,
                "expected_output_filename": "p_1p00_he030_bs010_drop030.stream",
            }
        ]
    ).to_csv(v5_dir / "experiment_plan.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "p_1p00_he030_bs010_drop030",
                "formula": "Eg * M * D^3 / U",
                "high_eg_fraction": 0.3,
                "drop_fraction": 0.3,
                "excitation_block_size": 10,
                "common_high_Eg_observation_count": 10,
                "common_excitation_block_count": 1,
                "common_actionable_block_count": 1,
                "expected_removal_count": 2,
                "actual_removal_count": 2,
            }
        ]
    ).to_csv(v5_dir / "per_variant_filter_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "p_1p00_he030_bs010_drop030",
                "removed_observations": 2,
                "kept_observations": 1,
                "total_reflection_rows_seen": 3,
            }
        ]
    ).to_csv(v5_dir / "stream_rewrite_qc.csv", index=False)
    pd.DataFrame(
        [
            {"variant": "p_1p00_he030_bs010_drop030", "exact_key_text": "img.h5|1|1|0|0"},
            {"variant": "p_1p00_he030_bs010_drop030", "exact_key_text": "img.h5|1|3|0|0"},
        ]
    ).to_csv(v5_dir / "selected_removal_observations.csv", index=False)
    write(v5_dir / "parameters.json", "{}")
    write(v5_dir / "scores.json", "[]")
    write(v5_dir / "validation.json", "{}")
    write(v5_dir / "run_metadata.json", "{}")

    args = audit.parse_args(
        [
            "--root",
            str(root),
            "--v6-dir",
            str(v6_dir),
            "--v5-dir",
            str(v5_dir),
            "--source-stream",
            str(source),
            "--out-dir",
            str(out_dir),
            "--workers",
            "2",
        ]
    )
    summary = audit.run_audit(args)

    assert (out_dir / "audit_summary.json").is_file()
    removal = pd.read_csv(out_dir / "removal_accounting.csv")
    d3 = removal.loc[removal["label"] == "filter_matched_eg_d3_cmean_drop30"].iloc[0]
    assert int(d3["removed_by_stream_rewrite_qc"]) == 1
    equivalence = pd.read_csv(out_dir / "selection_equivalence.csv")
    d3_vs_v5 = equivalence.loc[equivalence["comparison"] == "v6_d3_vs_v5_equivalent"].iloc[0]
    assert bool(d3_vs_v5["exactly_equal"]) is False
    globals_table = pd.read_csv(out_dir / "merge_global_metrics.csv")
    assert "full_reference" in set(globals_table["label"])
    d2d3 = pd.read_csv(out_dir / "d2_vs_d3_shell_comparison.csv")
    assert len(d2d3) == 2
    assert summary["questions"]["3_v6_d3_equivalent_to_v5_p1"]["v6_d3_vs_v5_removed_set_equal"] is False

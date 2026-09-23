from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "audit_v6_complete_experiment.py"
SPEC = importlib.util.spec_from_file_location("audit_v6_complete", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_stream(path: Path) -> None:
    write(
        path,
        "\n".join(
            [
                "----- Begin chunk -----",
                "Image filename: img.h5",
                "Event: //1",
                "--- Begin crystal",
                "Reflections measured after indexing",
                "   1    0    0 10.0 1.0",
                "   2    0    0 10.0 1.0",
                "End of reflections",
                "--- End crystal",
                "----- End chunk -----",
                "",
            ]
        ),
    )


def write_merge(exp_dir: Path, stem: str, stream_path: Path, cc12: float, rsplit: float, redundancy: float) -> Path:
    merge_dir = exp_dir / f"{stem}_partialator_results_synthetic"
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
                "Overall <snr> = 10.000000",
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


def test_expected_variant_surface_is_complete() -> None:
    specs = audit.expected_variants()
    assert len(specs) == 88
    assert len({spec.variant_id for spec in specs}) == 88
    assert sum(spec.experiment_type == "diagnostic_low_high" for spec in specs) == 22
    assert sum(spec.filtering_target == "all" for spec in specs) == 18
    assert sum(spec.filtering_target == "higheg" for spec in specs) == 24
    assert sum(spec.filtering_target == "matched" for spec in specs) == 24


def test_random_variant_surface_and_replicate_summary() -> None:
    random_specs = audit.expected_random_variants()
    assert len(random_specs) == 39
    globals_rows = [
        {"variant_id": "filter_all_eg_m2_drop05", "cc12": 0.99, "rsplit": 5.0, "snr": 10.0, "completeness": 99.0, "redundancy": 500.0},
        {"variant_id": "random_all_drop05_seed20260717", "cc12": 0.98, "rsplit": 5.5, "snr": 9.0, "completeness": 99.0, "redundancy": 501.0},
        {"variant_id": "random_all_drop05_seed20260718", "cc12": 0.97, "rsplit": 5.2, "snr": 9.5, "completeness": 98.9, "redundancy": 502.0},
        {"variant_id": "random_all_drop05_seed20260719", "cc12": 0.96, "rsplit": 5.3, "snr": 9.2, "completeness": 98.8, "redundancy": 503.0},
    ]
    counts = [
        {
            "variant_id": "filter_all_eg_m2_drop05",
            "removed_or_selected_count": 10,
            "removed_fraction_of_accepted_population": 0.1,
        }
    ]
    rows = audit.generated_random_replicate_results(globals_rows, counts)
    assert len(rows) == 1
    row = rows[0]
    assert row["random_replicate_count"] == 3
    assert row["delta_cc12_oriented_minus_random_mean"] > 0
    assert row["delta_rsplit_oriented_minus_random_mean"] < 0


def test_recalculated_shell_rows_replace_original_variant_rows(tmp_path: Path) -> None:
    merge_dir = tmp_path / "merge"
    record = audit.MergeRecord("variant_a", "variant_a", "v6", None, merge_dir)
    recalc_dir = tmp_path / "recalculated_shells" / "variant_a"
    write_merge(tmp_path, "merge", tmp_path / "stream.stream", 0.997, 5.4, 500.0)
    generated = tmp_path / "merge_partialator_results_synthetic" / "qc_stats"
    recalc_dir.mkdir(parents=True)
    for name in ["check_shell.tsv", "compare_cc12_shell.tsv", "compare_rsplit_shell.tsv"]:
        (recalc_dir / name).write_text((generated / name).read_text(encoding="utf-8"), encoding="utf-8")

    replacements = audit.parse_recalculated_shell_rows(record, recalc_dir)
    combined = audit.replace_shell_rows(
        [
            {"variant_id": "variant_a", "shell_index": 1, "min_invnm": 1.0, "max_invnm": 2.0},
            {"variant_id": "variant_b", "shell_index": 1, "min_invnm": 1.0, "max_invnm": 2.0},
        ],
        replacements,
    )

    assert len(replacements) == 2
    assert all(row["shell_metric_source"] == "recalculated_check_hkl_compare_hkl" for row in replacements)
    assert sum(row["variant_id"] == "variant_a" for row in combined) == 2
    assert sum(row["variant_id"] == "variant_b" for row in combined) == 1


def test_synthetic_complete_audit_accounts_for_all_88_variants(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    v6_dir = root / "oridyn_v6_score_target_filter_map_20260716"
    source = root / "MFM300-VIII_cut_20-0_3.stream"
    out_dir = root / "oridyn_v6_complete_audit_20260717"
    write_stream(source)
    write_merge(root, source.stem, source, 0.9970, 5.45, 534.39)

    plan_rows = []
    manifest_rows = []
    rewrite_rows = []
    selection_rows = []
    hkl_qc_rows = []
    block_qc_rows = []

    specs = audit.expected_variants()
    for idx, spec in enumerate(specs):
        stream_path = v6_dir / spec.output_filename
        write_stream(stream_path)
        cc12 = 0.996 + idx * 1e-6
        rsplit = 5.8 - idx * 0.001
        write_merge(v6_dir, stream_path.stem, stream_path, cc12, rsplit, 500.0 - idx * 0.1)
        selected_state = "diagnostic_low_half" if spec.experiment_type == "diagnostic_low_high" else "removed_by_targeted_filter"
        plan_rows.append({"variant_id": spec.variant_id, "output_filename": spec.output_filename})
        manifest_rows.append(
            {
                "variant_id": spec.variant_id,
                "experiment_type": spec.experiment_type,
                "score_id": spec.score_id,
                "score_formula": audit.SCORE_FORMULAS.get(spec.score_id, ""),
                "designation": spec.designation,
                "filtering_target": spec.filtering_target,
                "drop_fraction": spec.drop_fraction if spec.drop_fraction is not None else "",
                "source_stream": str(source),
                "output_stream": str(stream_path),
                "status": "generated",
                "selected_or_removed_count": 1,
                "retained_count": 1,
                "retained_fraction": 0.5,
                "global_accepted_observation_fraction_removed": 0.01 + idx * 0.0001,
            }
        )
        rewrite_rows.append(
            {
                "variant_id": spec.variant_id,
                "output_stream": str(stream_path),
                "status": "generated",
                "requested_removals": 1,
                "removed_observations": 1,
                "kept_observations": 1,
                "total_reflection_rows_seen": 2,
                "source_order_preserved": True,
                "all_requested_keys_found_exactly_once": True,
                "stream_reflection_row_difference_equals_requested_removals": True,
            }
        )
        selection_rows.append(
            {
                "variant_id": spec.variant_id,
                "state": selected_state,
                "exact_key_text": f"img.h5|1|{idx}|0|0",
                "h": idx,
                "k": 0,
                "l": 0,
            }
        )
        hkl_qc_rows.append(
            {
                "variant_id": spec.variant_id,
                "n_observations": 2,
                "n_selected": 1,
                "n_removed": 1,
                "n_eligible": 2,
                "n_eligible_retained": 1,
                "n_high_eg": 2,
                "n_blocks": 1,
                "n_block_observations": 2,
                "actionable": True,
                "validation_passed": True,
            }
        )
        block_qc_rows.append(
            {
                "variant_id": spec.variant_id,
                "block_size": 2,
                "n_removed": 1,
                "n_retained_in_block": 1,
                "validation_passed": True,
            }
        )

    pd.DataFrame(plan_rows).to_csv(v6_dir / "experiment_plan.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(v6_dir / "stream_manifest.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(v6_dir / "per_variant_selection_summary.csv", index=False)
    pd.DataFrame(rewrite_rows).to_csv(v6_dir / "stream_rewrite_qc.csv", index=False)
    pd.DataFrame(hkl_qc_rows).to_csv(v6_dir / "per_variant_per_hkl_qc.csv", index=False)
    pd.DataFrame(block_qc_rows).to_csv(v6_dir / "per_variant_per_block_qc.csv", index=False)
    with gzip.open(v6_dir / "selected_observations.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame(selection_rows).to_csv(handle, index=False)
    write(v6_dir / "parameters.json", json.dumps({"source_stream": str(source)}))
    write(v6_dir / "validation.json", json.dumps({"source_stream_validation": {"source_reflection_rows": 2}, "cache_stats": {"cache_rows": 2}}))
    write(v6_dir / "scores.json", "[]")
    write(v6_dir / "run_metadata.json", "{}")
    write(v6_dir / "run.log", "")

    args = audit.parse_args(
        [
            "--root",
            str(root),
            "--v6-dir",
            str(v6_dir),
            "--source-stream",
            str(source),
            "--historical-search-root",
            str(root),
            "--out-dir",
            str(out_dir),
            "--workers",
            "4",
            "--skip-stream-scan",
            "--skip-plots",
            "--no-shell-recalculation",
        ]
    )

    assert audit.run_audit(args) == 0
    accounting = pd.read_csv(out_dir / "experiment_accounting.csv")
    assert len(accounting) == 88
    assert set(accounting["status"]) == {"validated"}
    summary = json.loads((out_dir / "complete_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["validated_variants"] == 88
    diagnostics = pd.read_csv(out_dir / "diagnostic_pair_results.csv")
    assert len(diagnostics) == 11
    sweeps = pd.read_csv(out_dir / "filter_sweeps.csv")
    assert len(sweeps) == 66
    for name in audit.REQUIRED_OUTPUT_FILES:
        assert (out_dir / name).exists(), name

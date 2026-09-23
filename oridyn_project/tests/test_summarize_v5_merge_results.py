from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "summarize_v5_merge_results.py"


def write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_stream(exp_dir: Path, stem: str) -> Path:
    path = exp_dir / f"{stem}.stream"
    write(path, f"CrystFEL stream placeholder for {stem}\n")
    return path


def create_merge(exp_dir: Path, stem: str, timestamp: str, stream_path: Path, cc12: float, rsplit: float, shell_shift: float = 0.0, complete: bool = True) -> Path:
    merge_dir = exp_dir / f"{stem}_partialator_results_{timestamp}"
    merge_dir.mkdir(parents=True)
    write(merge_dir / "crystfel.hkl", "")
    write(merge_dir / "crystfel.hkl1", "")
    write(merge_dir / "crystfel.hkl2", "")
    write(
        merge_dir / "parameters.json",
        json.dumps(
            {
                "merge_wrapper": {
                    "stream_file": str(stream_path),
                    "run_id": timestamp,
                    "run_started": "2026-07-16T01:00",
                    "symmetry": "4/mmm",
                    "model": "offset",
                    "iterations": 10,
                    "min_measurements": 1,
                    "threads": 2,
                    "disable_pr": "true",
                    "no_bscale": True,
                    "lowres": "20.0",
                    "highres": "0.35",
                },
                "merging": {"symmetry": "4/mmm", "partiality_model": "offset", "num_iterations": 10, "Bscale": False, "post_refine": False},
            }
        ),
    )
    write(
        merge_dir / "metadata_and_outputs.txt",
        "\n".join(
            [
                "Run started: 2026-07-16T01:00",
                f"Run ID: {timestamp}",
                f"STREAM: {stream_path}",
                "THREADS: 2",
                "SYM: 4/mmm",
                "ITERATIONS: 10",
                "MIN_MEASUREMENTS: 1",
                "MODEL: offset",
                "DISABLE_PR: true",
                "PARTIALATOR_NO_BSCALE: true",
                "Range: 20.0-0.35 A; shells=2",
                "=== SUMMARY ===",
                "Completeness: 99.00%",
                "Redundancy:   20.00x",
                "SNR:         10.00",
                f"CC1/2:        {cc12}",
                f"Rsplit:       {rsplit}",
            ]
        ),
    )
    write(merge_dir / "partialator_stdout.log", "")
    write(merge_dir / "partialator_stderr.log", "")
    if not complete:
        return merge_dir
    min1 = 0.5 + shell_shift
    max1 = 10.0 + shell_shift
    min2 = max1
    max2 = 20.0 + shell_shift
    write(
        merge_dir / "qc_stats" / "check_hkl_completeness.log",
        "\n".join(
            [
                "1/d goes from 0.500000 to 20.000000 nm^-1",
                "Overall values within specified resolution range:",
                "200 measurements in total.",
                "20 reflections in total.",
                "20 reflections possible.",
                "Overall <snr> = 10.123456",
                "Overall redundancy = 10.000000 measurements/unique reflection",
                "Overall completeness = 100.000000 %",
            ]
        ),
    )
    write(
        merge_dir / "qc_stats" / "compare_cc12.log",
        f"Accepted resolution range: 0.500000 to 20.000000 nm^-1 (20.00 to 0.50 Angstroms).\nOverall CC = {cc12}\n",
    )
    write(
        merge_dir / "qc_stats" / "compare_rsplit.log",
        f"Accepted resolution range: 0.500000 to 20.000000 nm^-1 (20.00 to 0.50 Angstroms).\nOverall Rsplit = {rsplit} %\n",
    )
    write(
        merge_dir / "qc_stats" / "check_shell.tsv",
        "\n".join(
            [
                "Center 1/nm  # refs Possible  Compl       Meas   Red   SNR     Mean I     d(A)    Min 1/nm   Max 1/nm",
                f"     5.250       10       10 100.00        100  10.0 11.00     100.0     1.90       {min1:.3f}     {max1:.3f}",
                f"    15.000       10       10 100.00        100  10.0  9.00      50.0     0.67      {min2:.3f}     {max2:.3f}",
            ]
        ),
    )
    write(
        merge_dir / "qc_stats" / "compare_cc12_shell.tsv",
        "\n".join(
            [
                "  1/d centre       CC       nref      d / A   Min 1/nm    Max 1/nm",
                f"     5.250  {cc12:.7f}         10       1.90      {min1:.3f}      {max1:.3f}",
                f"    15.000  {cc12 - 0.01:.7f}         10       0.67      {min2:.3f}      {max2:.3f}",
            ]
        ),
    )
    write(
        merge_dir / "qc_stats" / "compare_rsplit_shell.tsv",
        "\n".join(
            [
                "  1/d centre Rsplit/%       nref      d / A   Min 1/nm    Max 1/nm",
                f"     5.250       {rsplit:.2f}         10       1.90      {min1:.3f}      {max1:.3f}",
                f"    15.000       {rsplit + 1:.2f}         10       0.67      {min2:.3f}      {max2:.3f}",
            ]
        ),
    )
    return merge_dir


def test_duplicate_selection_prefers_complete_run_over_newer_incomplete(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment"
    ref_stream = create_stream(exp_dir, "alpha_ref")
    beta_stream = create_stream(exp_dir, "beta")
    complete_dir = create_merge(exp_dir, "alpha_ref", "20260716T0100", ref_stream, 0.95, 5.0)
    incomplete_dir = create_merge(exp_dir, "alpha_ref", "20260716T0200", ref_stream, 0.97, 4.0, complete=False)
    create_merge(exp_dir, "gamma", "20260716T0130", create_stream(exp_dir, "gamma"), 0.90, 6.0, shell_shift=0.25)
    write(beta_stream, "CrystFEL stream placeholder for beta\n")
    write(
        exp_dir / "experiment_plan.csv",
        "experiment,score_name,filter_name,expected_output_filename,is_reference\nalpha_ref,alpha_ref,filter,alpha_ref.stream,true\ngamma,gamma,filter,gamma.stream,false\nbeta,beta,filter,beta.stream,false\n",
    )
    config = tmp_path / "config.json"
    write(config, json.dumps({"source_directories": [str(exp_dir)]}))
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config), "--out-dir", str(out_dir), "--workers", "2"],
        cwd=SCRIPT.parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    selected = pd.read_csv(out_dir / "selected_merge_runs.csv")
    alpha = selected.loc[selected["stream_stem"] == "alpha_ref"].iloc[0]
    assert alpha["merge_results_dir"] == str(complete_dir)
    assert alpha["selection_reason"] == "selected_latest_complete_parseable_run"
    assert "explicit_metadata_reference_field" in alpha["reference_identification_method"]
    all_runs = pd.read_csv(out_dir / "all_merge_runs.csv")
    newer = all_runs.loc[all_runs["merge_results_dir"] == str(incomplete_dir)].iloc[0]
    assert bool(newer["selected"]) is False
    assert "incomplete" in newer["selection_reason"]
    missing = pd.read_csv(out_dir / "missing_merges.csv")
    assert set(missing["stream_stem"]) == {"beta"}


def test_mismatched_shell_schemes_are_recorded_without_shell_delta_join(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment"
    ref_stream = create_stream(exp_dir, "alpha_ref")
    gamma_stream = create_stream(exp_dir, "gamma")
    create_merge(exp_dir, "alpha_ref", "20260716T0100", ref_stream, 0.95, 5.0)
    create_merge(exp_dir, "gamma", "20260716T0130", gamma_stream, 0.90, 6.0, shell_shift=0.25)
    write(
        exp_dir / "experiment_plan.csv",
        "experiment,score_name,filter_name,expected_output_filename,is_reference\nalpha_ref,alpha_ref,filter,alpha_ref.stream,true\ngamma,gamma,filter,gamma.stream,false\n",
    )
    config = tmp_path / "config.json"
    write(config, json.dumps({"source_directories": [str(exp_dir)]}))
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config), "--out-dir", str(out_dir), "--workers", "2"],
        cwd=SCRIPT.parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["counts"]["shell_scheme_mismatch_experiments"] == 1
    shell_schemes = pd.read_csv(out_dir / "shell_schemes.csv")
    assert set(shell_schemes["scheme_consistency_within_experiment"]) == {"mismatch"}
    deltas = pd.read_csv(out_dir / "shell_deltas_vs_reference.csv")
    assert set(deltas["stream_stem"]) == {"alpha_ref"}
    warnings = pd.read_csv(out_dir / "parse_warnings.csv")
    assert warnings["message"].str.contains("shell_scheme_id differ").any()

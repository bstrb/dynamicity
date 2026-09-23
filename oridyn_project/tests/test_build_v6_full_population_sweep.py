from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_v6_full_population_sweep.py"
SPEC = importlib.util.spec_from_file_location("v6_full_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def write(path: Path, text: str) -> None:
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
                "   1    0    0      10.0       1.0",
                "   2    0    0      20.0       1.0",
                "   3    0    0      30.0       1.0",
                "   4    0    0      40.0       1.0",
                "End of reflections",
                "--- End crystal",
                "----- End chunk -----",
                "",
            ]
        ),
    )


def write_multi_stream(path: Path, hkls: list[tuple[int, int, int]], frame_count: int) -> None:
    rows: list[str] = []
    for frame in range(1, frame_count + 1):
        rows.extend(
            [
                "----- Begin chunk -----",
                f"Image filename: img_{frame}.h5",
                f"Event: //{frame}",
                "--- Begin crystal",
                "Reflections measured after indexing",
            ]
        )
        for h, k, l in hkls:
            rows.append(f"{h:4d} {k:4d} {l:4d}      {10.0 + h:.1f}       1.0")
        rows.extend(["End of reflections", "--- End crystal", "----- End chunk -----"])
    rows.append("")
    write(path, "\n".join(rows))


def write_accepted_and_scores(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source.stream"
    accepted = root / "accepted.csv"
    scores = root / "v5_scores.csv"
    write_stream(source)
    rows = []
    for idx, h in enumerate([1, 2, 3, 4], start=1):
        rows.append(
            {
                "source_filename": "img.h5",
                "event": "1",
                "h": h,
                "k": 0,
                "l": 0,
                "d_angstrom": 1.0 + idx * 0.1,
                "inv_nm": 10.0 / (1.0 + idx * 0.1),
                "sg_target": 0.0001 * idx,
                "target_excitation_Eg": 0.9 + idx * 0.01,
            }
        )
    pd.DataFrame(rows)[["source_filename", "event", "h", "k", "l"]].to_csv(accepted, index=False)
    pd.DataFrame(rows).to_csv(scores, index=False)
    return source, accepted, scores


def write_parallel_accepted_and_scores(root: Path, *, hkl_count: int = 12, frame_count: int = 12) -> tuple[Path, Path, Path, int]:
    source = root / "source_parallel.stream"
    accepted = root / "accepted_parallel.csv"
    scores = root / "v5_scores_parallel.csv"
    hkls = [(h, h % 3, -(h % 2)) for h in range(1, hkl_count + 1)]
    write_multi_stream(source, hkls, frame_count)
    rows = []
    for frame in range(1, frame_count + 1):
        for h, k, l in hkls:
            idx = frame * 100 + h
            rows.append(
                {
                    "source_filename": f"img_{frame}.h5",
                    "event": str(frame),
                    "h": h,
                    "k": k,
                    "l": l,
                    "d_angstrom": 1.0 + (idx % 17) * 0.03,
                    "inv_nm": 10.0 / (1.0 + (idx % 17) * 0.03),
                    "sg_target": 0.00003 * ((idx % 11) + 1),
                    "target_excitation_Eg": 0.75 + 0.015 * ((idx * 7) % 19),
                }
            )
    pd.DataFrame(rows)[["source_filename", "event", "h", "k", "l"]].to_csv(accepted, index=False)
    pd.DataFrame(rows).to_csv(scores, index=False)
    return source, accepted, scores, len(rows)


def parse_args(
    root: Path,
    source: Path,
    out: Path,
    mode: str,
    accepted: Path,
    scores: Path,
    *,
    expected_count: int = 4,
    workers: int = 1,
    plan_batch_hkls: int | None = None,
    max_pending_batches: int | None = None,
) -> object:
    argv = [
            "--root",
            str(root),
            "--source-stream",
            str(source),
            "--out-dir",
            str(out),
            "--mode",
            mode,
            "--accepted-population",
            str(accepted),
            "--v5-scores",
            str(scores),
            "--expected-accepted-count",
            str(expected_count),
            "--expected-source-reflection-rows",
            str(expected_count),
            "--max-cache-rows",
            str(expected_count),
            "--workers",
            str(workers),
            "--skip-halfset-help",
        ]
    if plan_batch_hkls is not None:
        argv.extend(["--plan-batch-hkls", str(plan_batch_hkls)])
    if max_pending_batches is not None:
        argv.extend(["--max-pending-batches", str(max_pending_batches)])
    return builder.parse_args(argv)


def test_variant_surface_includes_88_scores_and_39_random_controls() -> None:
    assert len(builder.score_variants()) == 88
    assert len(builder.random_variants()) == 39
    assert len(builder.all_planned_streams()) == 127
    assert builder.stable_hash_u64(20260717, "img.h5|1|1|0|0") == builder.stable_hash_u64(20260717, "img.h5|1|1|0|0")
    assert builder.stable_hash_u64(20260717, "img.h5|1|1|0|0") != builder.stable_hash_u64(20260718, "img.h5|1|1|0|0")


def test_cache_plan_and_streams_synthetic_smoke(tmp_path: Path) -> None:
    source, accepted, scores = write_accepted_and_scores(tmp_path)
    out = tmp_path / "out"

    assert builder.run_cache(parse_args(tmp_path, source, out, "cache", accepted, scores)) == 0
    cache_validation = json.loads((out / "accepted_population_validation.json").read_text(encoding="utf-8"))
    assert cache_validation["cache_gate"]["score_cache_rows"] == 4

    assert builder.run_plan(parse_args(tmp_path, source, out, "plan", accepted, scores)) == 0
    manifest = pd.read_csv(out / "stream_manifest.csv")
    assert len(manifest) == 127
    assert manifest["variant_id"].str.startswith("random_").sum() == 39
    assert (out / "selection_masks" / "diag_eg_low50.keep.bitset").is_file()
    assert (out / "block_definitions.csv").is_file()
    assert len(pd.read_csv(out / "selection_overlap.csv")) == (127 * 126) // 2
    assert pd.read_csv(out / "score_correlations.csv")["score_id_a"].nunique() == 11

    assert builder.run_streams(parse_args(tmp_path, source, out, "streams", accepted, scores)) == 0
    qc = pd.read_csv(out / "stream_rewrite_qc.csv")
    assert len(qc) == 127
    assert (out / "diag_eg_low50.stream").is_file()
    refreshed_manifest = pd.read_csv(out / "stream_manifest.csv")
    assert set(refreshed_manifest["status"]) == {"generated"}


def test_parallel_plan_uses_two_worker_pids_and_reports_real_eta(tmp_path: Path) -> None:
    source, accepted, scores, expected_count = write_parallel_accepted_and_scores(tmp_path, hkl_count=12, frame_count=12)
    out = tmp_path / "out_parallel"

    assert builder.run_cache(parse_args(tmp_path, source, out, "cache", accepted, scores, expected_count=expected_count)) == 0
    assert (
        builder.run_plan(
            parse_args(
                tmp_path,
                source,
                out,
                "plan",
                accepted,
                scores,
                expected_count=expected_count,
                workers=2,
                plan_batch_hkls=1,
                max_pending_batches=4,
            )
        )
        == 0
    )

    parallel = json.loads((out / "parallelism_validation.json").read_text(encoding="utf-8"))
    assert parallel["passed"] is True
    assert parallel["distinct_worker_pid_count"] >= 2
    assert parallel["total_batches"] == 12

    progress_lines = [
        line
        for line in (out / "run.log").read_text(encoding="utf-8").splitlines()
        if "plan: parallel batch progress:" in line and "(100.0%)" not in line
    ]
    assert progress_lines
    assert any("eta=0.0s" not in line for line in progress_lines)
    assert all(token in progress_lines[0] for token in ["hkls=", "parent_rss=", "worker_pids=", "pending_futures="])


def test_parallel_selections_match_single_worker_reference(tmp_path: Path) -> None:
    source, accepted, scores, expected_count = write_parallel_accepted_and_scores(tmp_path, hkl_count=8, frame_count=12)
    serial_out = tmp_path / "out_single_worker"
    parallel_out = tmp_path / "out_two_workers"

    assert builder.run_cache(parse_args(tmp_path, source, serial_out, "cache", accepted, scores, expected_count=expected_count)) == 0
    assert builder.run_plan(parse_args(tmp_path, source, serial_out, "plan", accepted, scores, expected_count=expected_count, workers=1, plan_batch_hkls=1)) == 0
    assert builder.run_cache(parse_args(tmp_path, source, parallel_out, "cache", accepted, scores, expected_count=expected_count)) == 0
    assert (
        builder.run_plan(
            parse_args(
                tmp_path,
                source,
                parallel_out,
                "plan",
                accepted,
                scores,
                expected_count=expected_count,
                workers=2,
                plan_batch_hkls=1,
                max_pending_batches=4,
            )
        )
        == 0
    )

    serial_counts = pd.read_csv(serial_out / "selection_counts.csv").sort_values("variant_id").reset_index(drop=True)
    parallel_counts = pd.read_csv(parallel_out / "selection_counts.csv").sort_values("variant_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(serial_counts, parallel_counts)

    for serial_mask in sorted((serial_out / "selection_masks").glob("*.bitset")):
        parallel_mask = parallel_out / "selection_masks" / serial_mask.name
        assert parallel_mask.read_bytes() == serial_mask.read_bytes()


def test_target_c_ties_use_abs_sg_before_exact_key(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE score_cache (
            ordinal INTEGER PRIMARY KEY,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL UNIQUE,
            source_order INTEGER NOT NULL,
            sg REAL NOT NULL,
            abs_sg REAL NOT NULL,
            Eg REAL NOT NULL,
            D REAL NOT NULL,
            U REAL NOT NULL,
            M REAL NOT NULL,
            M2 REAL NOT NULL
        )
        """
    )
    for ordinal in range(10):
        source = f"img_{ordinal}.h5"
        event = str(ordinal)
        key = f"{source}|{event}|1|0|0"
        conn.execute(
            """
            INSERT INTO score_cache(ordinal,source_filename,event,h,k,l,exact_key_text,source_order,sg,abs_sg,Eg,D,U,M,M2)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (ordinal, source, event, 1, 0, 0, key, ordinal, 10.0 - ordinal, 10.0 - ordinal, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
    conn.commit()
    conn.close()

    result = builder.plan_batch_worker((1, str(db), [(1, 0, 0)], {(1, 0, 0): {1: np.arange(10, dtype=np.int64)}}))
    removed = set(result["mask_ordinals"]["filter_matched_eg_d3_cmean_drop30"].tolist())
    assert removed == {7, 8, 9}


def test_interrupted_plan_outputs_are_not_accepted_by_streams(tmp_path: Path) -> None:
    source, accepted, scores = write_accepted_and_scores(tmp_path)
    out = tmp_path / "out_interrupted"
    assert builder.run_cache(parse_args(tmp_path, source, out, "cache", accepted, scores)) == 0

    write(out / "validation.json", json.dumps({"passed": True, "stream_gate": {"status": "pending_streams_mode"}}))
    (out / "selection_masks").mkdir(exist_ok=True)
    (out / "selection_masks" / "diag_eg_low50.keep.bitset").write_bytes(b"\x00")

    with pytest.raises(SystemExit, match="Corrected parallel plan gate"):
        builder.run_streams(parse_args(tmp_path, source, out, "streams", accepted, scores))


def test_streams_refuses_incomplete_parallel_plan(tmp_path: Path) -> None:
    source, accepted, scores = write_accepted_and_scores(tmp_path)
    out = tmp_path / "out_incomplete_parallel"
    assert builder.run_cache(parse_args(tmp_path, source, out, "cache", accepted, scores)) == 0

    write(
        out / "validation.json",
        json.dumps(
            {
                "passed": True,
                "plan_gate": {"status": "complete", "builder_version": builder.PLAN_BUILDER_VERSION},
                "stream_gate": {"status": "pending_streams_mode"},
            }
        ),
    )
    write(out / "plan_completion.json", json.dumps({"passed": True, "builder_version": builder.PLAN_BUILDER_VERSION, "required_artifacts": []}))

    with pytest.raises(SystemExit, match="parallelism_validation"):
        builder.run_streams(parse_args(tmp_path, source, out, "streams", accepted, scores))


def test_cache_gate_rejects_wrong_count(tmp_path: Path) -> None:
    source, accepted, scores = write_accepted_and_scores(tmp_path)
    out = tmp_path / "out_bad"
    args = builder.parse_args(
        [
            "--root",
            str(tmp_path),
            "--source-stream",
            str(source),
            "--out-dir",
            str(out),
            "--mode",
            "cache",
            "--accepted-population",
            str(accepted),
            "--v5-scores",
            str(scores),
            "--expected-accepted-count",
            "5",
            "--expected-source-reflection-rows",
            "4",
            "--workers",
            "1",
            "--skip-halfset-help",
        ]
    )
    with pytest.raises(SystemExit):
        builder.run_cache(args)

#!/usr/bin/env python3
"""Run the OriDyn v2 score/filter/split workflow for a prepared stream.

This wrapper intentionally stops before merging, partialator, QC, and SHELXL.
The input stream is assumed to already have the desired resolution cutoff.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_COLUMNS = ["trust_risk_v2_full_norm", "trust_risk_v2_core_norm"]
DEFAULT_KEEP_FRACTIONS = [0.90, 0.80, 0.70]
STREAM_HASH_MAX_BYTES = 2_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path, help="Prepared input CrystFEL stream")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--score-columns", nargs="+", default=DEFAULT_SCORE_COLUMNS)
    parser.add_argument("--keep-fractions", nargs="+", type=float, default=DEFAULT_KEEP_FRACTIONS)
    parser.add_argument("--min-obs-per-hkl", type=int, default=10)
    parser.add_argument("--keep-low-count-hkls", action="store_true")
    parser.add_argument("--make-splits", action="store_true")
    parser.add_argument("--split-fraction", type=float, default=0.50)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--max-events", type=int, default=None, help="Smoke-test limit: score/filter first N crystal blocks")
    parser.add_argument("--max-frames", type=int, default=None, help="Alias for smoke-test crystal-block limit")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running stages")
    parser.add_argument("--force", action="store_true", help="Overwrite known workflow outputs")
    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if args.min_obs_per_hkl < 1:
        raise SystemExit("--min-obs-per-hkl must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    if args.split_fraction <= 0.0 or args.split_fraction > 0.50:
        raise SystemExit("--split-fraction must satisfy 0 < fraction <= 0.50")
    if args.max_events is not None and args.max_events < 1:
        raise SystemExit("--max-events must be >= 1 when provided")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be >= 1 when provided")
    keep_fractions = []
    for value in args.keep_fractions:
        fraction = float(value)
        if not (0.0 < fraction <= 1.0):
            raise SystemExit("--keep-fractions values must satisfy 0 < fraction <= 1")
        keep_fractions.append(fraction)
    args.keep_fractions = sorted(set(keep_fractions), reverse=True)
    args.score_columns = list(dict.fromkeys(str(column) for column in args.score_columns))
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def score_label(score_column: str) -> str:
    label = str(score_column)
    for prefix in ("trust_risk_", "manybeam_coupling_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    for suffix in ("_norm", "_raw"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "score"


def workflow_paths(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root
    score_dirs = {column: root / f"filtered_{score_label(column)}" for column in args.score_columns}
    split_dirs = {column: root / f"splits_{score_label(column)}" for column in args.score_columns}
    return {
        "root": root,
        "base_scores": root / "base_scores",
        "v2_scores": root / "v2_scores",
        "filtered": score_dirs,
        "splits": split_dirs,
        "summary_md": root / "v2_workflow_summary.md",
        "summary_csv": root / "v2_workflow_summary.csv",
        "metadata": root / "run_metadata.json",
        "smoke_stream": root / f"_smoke_input_first_{smoke_limit(args) or 0}_crystals.stream",
    }


def known_output_paths(paths: dict[str, Any], make_splits: bool) -> list[Path]:
    out = [
        paths["base_scores"],
        paths["v2_scores"],
        paths["summary_md"],
        paths["summary_csv"],
        paths["metadata"],
    ]
    out.extend(paths["filtered"].values())
    if make_splits:
        out.extend(paths["splits"].values())
    smoke_stream = paths.get("smoke_stream")
    if isinstance(smoke_stream, Path):
        out.append(smoke_stream)
    return out


def prepare_output_root(args: argparse.Namespace, paths: dict[str, Any]) -> None:
    root = paths["root"]
    known = known_output_paths(paths, args.make_splits)
    existing_known = [path for path in known if path.exists()]
    existing_any = list(root.iterdir()) if root.exists() else []
    if args.dry_run:
        return
    if existing_any and not args.force:
        formatted = "\n".join(f"  {path}" for path in existing_any[:20])
        more = "" if len(existing_any) <= 20 else f"\n  ... and {len(existing_any) - 20} more"
        raise SystemExit(f"Refusing to write into non-empty output root without --force:\n{formatted}{more}")
    if args.force:
        for path in existing_known:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    root.mkdir(parents=True, exist_ok=True)


def smoke_limit(args: argparse.Namespace) -> int | None:
    values = [value for value in [args.max_events, args.max_frames] if value is not None]
    if not values:
        return None
    return min(int(value) for value in values)


def make_smoke_stream(stream: Path, out: Path, max_crystals: int) -> dict[str, int | str]:
    """Write a tiny stream ending at the chunk containing max_crystals crystals."""

    out.parent.mkdir(parents=True, exist_ok=True)
    crystals = 0
    chunks = 0
    lines = 0
    stop_after_chunk = False
    with stream.open("r", encoding="utf-8", errors="replace") as src, out.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            lines += 1
            if "----- Begin chunk -----" in raw_line:
                chunks += 1
            if "Begin crystal" in raw_line:
                crystals += 1
                if crystals >= int(max_crystals):
                    stop_after_chunk = True
            dst.write(raw_line)
            if stop_after_chunk and "----- End chunk -----" in raw_line:
                break
    return {"path": str(out), "chunks": chunks, "crystals": min(crystals, int(max_crystals)), "lines": lines}


def command_to_text(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def run_command(cmd: list[str], dry_run: bool) -> None:
    log(("DRY-RUN " if dry_run else "Running ") + command_to_text(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        n_lines = sum(1 for _ in handle)
    return max(n_lines - 1, 0)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hash_if_feasible(path: Path) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if stat.st_size <= STREAM_HASH_MAX_BYTES:
        payload["sha256"] = sha256_file(path)
    else:
        payload["sha256"] = None
        payload["sha256_skipped_reason"] = f"file larger than {STREAM_HASH_MAX_BYTES} bytes"
    return payload


def git_capture() -> dict[str, str | None]:
    def run_git(args: list[str]) -> str | None:
        proc = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    return {
        "head": run_git(["rev-parse", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
    }


def script_hashes(split_used: bool) -> dict[str, str | None]:
    paths = [
        PROJECT_ROOT / "tools/run_oridyn_v2_score_filter_split_workflow.py",
        PROJECT_ROOT / "oridyn/cli.py",
        PROJECT_ROOT / "oridyn/pipeline.py",
        PROJECT_ROOT / "oridyn/parallel_scoring.py",
        PROJECT_ROOT / "oridyn/graph_crowding.py",
        PROJECT_ROOT / "oridyn/laue_zone.py",
        PROJECT_ROOT / "oridyn/systematic_rows.py",
        PROJECT_ROOT / "oridyn/axis_prediction.py",
        PROJECT_ROOT / "oridyn/geometry.py",
        PROJECT_ROOT / "oridyn/normalization.py",
        PROJECT_ROOT / "oridyn/outputs.py",
        PROJECT_ROOT / "oridyn/coupling_exposure_v2.py",
        PROJECT_ROOT / "tools/compute_geometry_coupling_v2_scores.py",
        PROJECT_ROOT / "tools/filter_stream_by_geometry_coupling_v2_keep_fraction.py",
    ]
    if split_used:
        paths.append(PROJECT_ROOT / "tools/split_stream_by_enh_feed_observation_risk_50.py")
    out: dict[str, str | None] = {}
    for path in paths:
        try:
            out[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path) if path.exists() else None
        except ValueError:
            out[str(path)] = sha256_file(path) if path.exists() else None
    return out


def build_commands(args: argparse.Namespace, paths: dict[str, Any], workflow_stream: Path) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    base_cmd = [
        sys.executable,
        "-m",
        "oridyn.cli",
        "run",
        "--stream",
        str(workflow_stream),
        "--output",
        str(paths["base_scores"]),
        "--workers",
        str(int(args.workers)),
    ]
    commands.append(("base_scoring", base_cmd))

    v2_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools/compute_geometry_coupling_v2_scores.py"),
        "--scores-csv",
        str(paths["base_scores"] / "reflection_scores.csv"),
        "--outdir",
        str(paths["v2_scores"]),
        "--workers",
        str(int(args.workers)),
        "--progress-every-frames",
        str(int(args.progress_every)),
    ]
    if args.force:
        v2_cmd.append("--overwrite")
    commands.append(("v2_scoring", v2_cmd))

    v2_scores_csv = paths["v2_scores"] / "geometry_coupling_v2_scores.csv"
    for score_column in args.score_columns:
        filter_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools/filter_stream_by_geometry_coupling_v2_keep_fraction.py"),
            "--stream",
            str(workflow_stream),
            "--v2-scores-csv",
            str(v2_scores_csv),
            "--output-root",
            str(paths["filtered"][score_column]),
            "--score-column",
            str(score_column),
            "--keep-fractions",
            *[f"{fraction:.12g}" for fraction in args.keep_fractions],
            "--min-obs-per-hkl",
            str(int(args.min_obs_per_hkl)),
            "--progress-every",
            str(int(args.progress_every)),
        ]
        if args.keep_low_count_hkls:
            filter_cmd.append("--keep-low-count-hkls")
        commands.append((f"filter_{score_label(score_column)}", filter_cmd))

    if args.make_splits:
        for score_column in args.score_columns:
            split_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "tools/split_stream_by_enh_feed_observation_risk_50.py"),
                "--stream",
                str(workflow_stream),
                "--joined-observations-csv",
                str(v2_scores_csv),
                "--outdir",
                str(paths["splits"][score_column]),
                "--seed",
                str(int(args.random_seed)),
                "--score-column",
                str(score_column),
                "--min-obs-per-hkl",
                str(int(args.min_obs_per_hkl)),
                "--fraction",
                f"{float(args.split_fraction):.12g}",
            ]
            commands.append((f"split_{score_label(score_column)}", split_cmd))
    return commands


def collect_filter_rows(args: argparse.Namespace, paths: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score_column in args.score_columns:
        summary_path = paths["filtered"][score_column] / "filter_sweep_summary.csv"
        if not summary_path.exists():
            continue
        table = pd.read_csv(summary_path)
        for row in table.to_dict(orient="records"):
            rows.append(
                {
                    "stage": "filter",
                    "score_column": score_column,
                    "variant": row.get("variant", ""),
                    "keep_fraction": row.get("keep_fraction", ""),
                    "matched_observations": row.get("matched_observations", ""),
                    "unmatched_observations": row.get("unmatched_observations", ""),
                    "kept_observations": row.get("kept_observations", ""),
                    "removed_observations": row.get("removed_observations", ""),
                    "removed_fraction": row.get("removed_fraction", ""),
                    "affected_signed_hkls": row.get("number_of_signed_hkls_affected", ""),
                    "low_count_hkls_kept_unchanged": row.get("number_of_low_count_hkls_kept_unchanged", ""),
                    "output_path": row.get("output_stream_path", ""),
                }
            )
    return rows


def collect_split_rows(args: argparse.Namespace, paths: dict[str, Any]) -> list[dict[str, Any]]:
    if not args.make_splits:
        return []
    rows: list[dict[str, Any]] = []
    for score_column in args.score_columns:
        meta = read_json(paths["splits"][score_column] / "run_metadata.json")
        output_paths = meta.get("output_paths", {})
        row_counts = meta.get("row_counts", {})
        for split, output_path in output_paths.items():
            if split not in {"low_enh_50", "high_enh_50", "random_enh_50"} and not split.startswith(("low_enh_", "high_enh_", "random_enh_")):
                continue
            rows.append(
                {
                    "stage": "split",
                    "score_column": score_column,
                    "variant": split,
                    "keep_fraction": float(args.split_fraction),
                    "matched_observations": "",
                    "unmatched_observations": "",
                    "kept_observations": row_counts.get(f"{split}_stream_observations_kept", ""),
                    "removed_observations": "",
                    "removed_fraction": "",
                    "affected_signed_hkls": row_counts.get("signed_hkls_passing_min_observations", ""),
                    "low_count_hkls_kept_unchanged": "",
                    "output_path": output_path,
                }
            )
    return rows


def write_summary_files(
    args: argparse.Namespace,
    paths: dict[str, Any],
    workflow_stream: Path,
    smoke_info: dict[str, Any] | None,
    commands: list[tuple[str, list[str]]],
) -> None:
    summary_rows: list[dict[str, Any]] = []
    base_scores = paths["base_scores"] / "reflection_scores.csv"
    v2_scores = paths["v2_scores"] / "geometry_coupling_v2_scores.csv"
    summary_rows.append(
        {
            "stage": "base_scores",
            "score_column": "",
            "variant": "",
            "keep_fraction": "",
            "matched_observations": "",
            "unmatched_observations": "",
            "kept_observations": count_csv_rows(base_scores),
            "removed_observations": "",
            "removed_fraction": "",
            "affected_signed_hkls": "",
            "low_count_hkls_kept_unchanged": "",
            "output_path": str(base_scores),
        }
    )
    summary_rows.append(
        {
            "stage": "v2_scores",
            "score_column": ",".join(args.score_columns),
            "variant": "",
            "keep_fraction": "",
            "matched_observations": "",
            "unmatched_observations": "",
            "kept_observations": count_csv_rows(v2_scores),
            "removed_observations": "",
            "removed_fraction": "",
            "affected_signed_hkls": "",
            "low_count_hkls_kept_unchanged": "",
            "output_path": str(v2_scores),
        }
    )
    summary_rows.extend(collect_filter_rows(args, paths))
    summary_rows.extend(collect_split_rows(args, paths))
    for row in summary_rows:
        if row.get("stage") == "filter":
            try:
                unmatched = int(float(row.get("unmatched_observations", 0)))
            except (TypeError, ValueError):
                unmatched = 0
            if unmatched:
                log(f"Warning: {row.get('variant', 'filter')} has {unmatched:,} unmatched stream observations")
    write_summary_csv(paths["summary_csv"], summary_rows)
    write_summary_md(paths["summary_md"], args, paths, workflow_stream, smoke_info, summary_rows, commands)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stage",
        "score_column",
        "variant",
        "keep_fraction",
        "matched_observations",
        "unmatched_observations",
        "kept_observations",
        "removed_observations",
        "removed_fraction",
        "affected_signed_hkls",
        "low_count_hkls_kept_unchanged",
        "output_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = [
        "stage",
        "score_column",
        "variant",
        "keep_fraction",
        "matched_observations",
        "unmatched_observations",
        "kept_observations",
        "removed_observations",
        "affected_signed_hkls",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_summary_md(
    path: Path,
    args: argparse.Namespace,
    paths: dict[str, Any],
    workflow_stream: Path,
    smoke_info: dict[str, Any] | None,
    summary_rows: list[dict[str, Any]],
    commands: list[tuple[str, list[str]]],
) -> None:
    stream_stat = args.stream.stat()
    lines = [
        "# OriDyn V2 Score/Filter/Split Workflow Summary",
        "",
        "## Scope",
        "",
        "- This workflow stops after scoring, stream filtering, and optional low/high/random splitting.",
        "- It does not run stream resolution cutting, merging, partialator, QC, or SHELXL.",
        "- Observation matching uses exact `source_filename + event + signed h,k,l`; HKLs are not canonicalized.",
        "",
        "## Inputs",
        "",
        f"- Original input stream: `{args.stream}`",
        f"- Original stream size bytes: `{stream_stat.st_size}`",
        f"- Workflow stream used: `{workflow_stream}`",
        f"- Output root: `{paths['root']}`",
        f"- Score columns: `{', '.join(args.score_columns)}`",
        f"- Keep fractions: `{', '.join(f'{x:.6g}' for x in args.keep_fractions)}`",
        f"- make_splits: `{bool(args.make_splits)}`",
    ]
    if smoke_info:
        lines.extend(["", "## Smoke Mode", "", *[f"- `{k}`: `{v}`" for k, v in smoke_info.items()]])
    lines.extend(
        [
            "",
            "## Summary",
            "",
            markdown_table(summary_rows),
            "",
            "## Commands Run",
            "",
        ]
    )
    for name, cmd in commands:
        lines.extend([f"### {name}", "", "```bash", command_to_text(cmd), "```", ""])
    lines.extend(
        [
            "## Output Files",
            "",
            f"- Base scores: `{paths['base_scores'] / 'reflection_scores.csv'}`",
            f"- V2 scores: `{paths['v2_scores'] / 'geometry_coupling_v2_scores.csv'}`",
            f"- Combined CSV: `{paths['summary_csv']}`",
            f"- Metadata: `{paths['metadata']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    args: argparse.Namespace,
    paths: dict[str, Any],
    workflow_stream: Path,
    smoke_info: dict[str, Any] | None,
    commands: list[tuple[str, list[str]]],
) -> None:
    payload = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stream": file_hash_if_feasible(args.stream),
            "workflow_stream": str(workflow_stream),
        },
        "output_root": str(paths["root"]),
        "score_columns": args.score_columns,
        "keep_fractions": [float(x) for x in args.keep_fractions],
        "min_obs_per_hkl": int(args.min_obs_per_hkl),
        "keep_low_count_hkls": bool(args.keep_low_count_hkls),
        "make_splits": bool(args.make_splits),
        "split_fraction": float(args.split_fraction),
        "random_seed": int(args.random_seed),
        "workers": int(args.workers),
        "progress_every": int(args.progress_every),
        "smoke_info": smoke_info,
        "git": git_capture(),
        "script_hashes": script_hashes(args.make_splits),
        "commands": [{"stage": name, "command": cmd} for name, cmd in commands],
        "warnings": [
            "This workflow does not run stream cutting, merging, partialator, QC, or SHELXL.",
            "Use HIGHRES=0.5 in downstream merge/QC for 0.5 A streams.",
            "Split stage reuses the existing generic observation split script; its output filenames use the historical low_enh/high_enh/random_enh labels.",
        ],
    }
    paths["metadata"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = workflow_paths(args)
    prepare_output_root(args, paths)
    log(f"Output root: {paths['root']}")
    log(f"Base scores: {paths['base_scores']}")
    log(f"V2 scores: {paths['v2_scores']}")
    for score_column in args.score_columns:
        log(f"Filter output for {score_column}: {paths['filtered'][score_column]}")
        if args.make_splits:
            log(f"Split output for {score_column}: {paths['splits'][score_column]}")

    smoke_info = None
    workflow_stream = args.stream
    limit = smoke_limit(args)
    if limit is not None:
        smoke_info = make_smoke_stream(args.stream, paths["smoke_stream"], int(limit)) if not args.dry_run else {
            "path": str(paths["smoke_stream"]),
            "crystals": int(limit),
            "dry_run": True,
        }
        workflow_stream = paths["smoke_stream"]
        log(f"Smoke stream for scoring/filtering: {workflow_stream}")

    commands = build_commands(args, paths, workflow_stream)
    if args.dry_run:
        for name, cmd in commands:
            print(f"[dry-run:{name}] {command_to_text(cmd)}")
        return 0

    for name, cmd in commands:
        log(f"Stage start: {name}")
        started = datetime.now()
        run_command(cmd, dry_run=False)
        elapsed = (datetime.now() - started).total_seconds()
        log(f"Stage end: {name} ({elapsed:.1f} s)")

    write_summary_files(args, paths, workflow_stream, smoke_info, commands)
    write_metadata(args, paths, workflow_stream, smoke_info, commands)
    log("Workflow complete")
    print(f"Wrote: {paths['summary_md']}")
    print(f"Wrote: {paths['summary_csv']}")
    print(f"Wrote: {paths['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run controlled split-data SHELXL refinements from the full-data model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/low_vs_high"
)
FULL_RESULT_DIR = "MFM300-VIII_cut_20-0_3_partialator_results"
SPLIT_DATASETS = (
    ("random", "random_B_frames_50_partialator_results"),
    ("low_risk", "low_risk_frames_50_partialator_results"),
    ("high_risk", "high_risk_frames_50_partialator_results"),
)
DEFAULT_SHELXL_COMMAND = "/opt/ccp4-9/bin/shelxl -a50000 -b3000 -c624 -t18 shelx"


@dataclass(frozen=True)
class RefinementTarget:
    dataset: str
    result_dir: Path
    refine_dir: Path
    hkl_source: Path


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cutoff-label", default="99-0_35")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Prepare inputs and summaries without running SHELXL.")
    parser.add_argument("--summarize-only", action="store_true", help="Only parse existing outputs and regenerate summaries/plots.")
    parser.add_argument("--shelxl-command", default=DEFAULT_SHELXL_COMMAND)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--plot-axis-percentile",
        type=float,
        default=98.0,
        help="Shared Fo/Fc plot axis upper percentile; use 100 for full range.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def first_match(text: str, pattern: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def first_int(text: str, pattern: str, flags: int = 0) -> int | None:
    value = first_match(text, pattern, flags)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def active_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.upper().startswith("END"):
            break
        if stripped and not stripped.upper().startswith("REM"):
            lines.append(stripped)
    return lines


def active_command(text: str, predicate: str) -> str | None:
    pattern = re.compile(predicate, flags=re.IGNORECASE)
    for line in active_lines(text):
        if pattern.match(line):
            return line
    return None


def active_exti(text: str) -> str | None:
    for line in active_lines(text):
        if line.upper().startswith("EXTI"):
            return line
    return None


def parse_ls_cycles(command: str | None) -> int | None:
    if not command:
        return None
    match = re.search(r"(\d+)", command)
    return int(match.group(1)) if match else None


def parse_refinement_stats(lst_text: str) -> dict[str, Any]:
    r_match = re.search(
        r"R1\s*=\s*([0-9.]+)\s+for\s+(\d+)\s+Fo\s*>\s*4sig\(Fo\)\s+and\s+([0-9.]+)\s+for all\s+(\d+)\s+data",
        lst_text,
    )
    wr_match = re.search(
        r"wR2\s*=\s*([0-9.]+),\s*GooF\s*=\s*S\s*=\s*([0-9.]+),\s*Restrained GooF\s*=\s*([0-9.]+)",
        lst_text,
    )
    return {
        "r1_gt_4sigma": float(r_match.group(1)) if r_match else None,
        "data_gt_4sigma": int(r_match.group(2)) if r_match else None,
        "r1_all": float(r_match.group(3)) if r_match else None,
        "data_all": int(r_match.group(4)) if r_match else None,
        "wr2": float(wr_match.group(1)) if wr_match else None,
        "goof": float(wr_match.group(2)) if wr_match else None,
        "restrained_goof": float(wr_match.group(3)) if wr_match else None,
        "parameters": first_int(lst_text, r"Total number of l\.s\. parameters =\s+(\d+)"),
        "restraints": first_int(lst_text, r"(\d+)\s+restraints"),
        "residual_peak": float(first_match(lst_text, r"Highest peak\s+([-0-9.]+)") or "nan"),
        "residual_hole": float(first_match(lst_text, r"Deepest hole\s+([-0-9.]+)") or "nan"),
        "lst_resolution_line": first_match(lst_text, r"(Number of data for d >[^\n]+)"),
        "outside_shel_rejected": first_int(lst_text, r"(\d+)\s+Reflections outside SHEL resolution limits rejected"),
        "shelxl_command_line": first_match(lst_text, r"Command line parameters:\s*(.+)$", re.MULTILINE),
    }


def normalize_stat_values(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("residual_peak", "residual_hole"):
        value = out.get(key)
        if isinstance(value, float) and math.isnan(value):
            out[key] = None
    return out


def parse_active_settings(ins_text: str, res_text: str) -> dict[str, Any]:
    ls_command = active_command(ins_text, r"^(?:L\.S\.|LS|CGLS)\s+")
    return {
        "active_shel": active_command(ins_text, r"^SHEL\b") or active_command(res_text, r"^SHEL\b"),
        "active_wght": active_command(ins_text, r"^WGHT\b") or active_command(res_text, r"^WGHT\b"),
        "active_exti": active_exti(ins_text) or active_exti(res_text),
        "active_hklf": active_command(ins_text, r"^HKLF\b") or active_command(res_text, r"^HKLF\b"),
        "ls_command": ls_command,
        "ls_cycles": parse_ls_cycles(ls_command),
    }


def truncate_at_end(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        out.append(raw.rstrip("\r\n"))
        if raw.strip().upper().startswith("END"):
            return out
    raise SystemExit("Starting model has no END line")


def patch_full_model_as_split_ins(
    full_res_text: str,
    active_shel: str,
    ls_cycles: int,
    target_wght: str,
    target_exti: str,
) -> str:
    lines = truncate_at_end(full_res_text)
    out: list[str] = []
    saw_ls = False
    saw_shel = False
    saw_wght = False
    saw_exti = False
    saw_hklf = False

    for raw in lines:
        stripped = raw.strip()
        upper = stripped.upper()
        if upper.startswith("END"):
            if not saw_ls:
                out.append(f"L.S. {ls_cycles}")
            if not saw_shel:
                out.append(active_shel)
            if not saw_wght:
                out.append(target_wght)
            if not saw_exti:
                out.append(target_exti)
            if not saw_hklf:
                out.append("HKLF 4")
            out.append("END")
            break
        if re.match(r"^(?:L\.S\.|LS|CGLS)\s+", stripped, flags=re.IGNORECASE):
            out.append(f"L.S. {ls_cycles}")
            saw_ls = True
            continue
        if re.match(r"^SHEL\b", stripped, flags=re.IGNORECASE):
            out.append(active_shel)
            saw_shel = True
            continue
        if re.match(r"^WGHT\b", stripped, flags=re.IGNORECASE):
            out.append(target_wght)
            saw_wght = True
            continue
        if upper.startswith("EXTI"):
            out.append(target_exti)
            saw_exti = True
            continue
        if re.match(r"^HKLF\b", stripped, flags=re.IGNORECASE):
            if not saw_shel:
                out.append(active_shel)
                saw_shel = True
            if not saw_wght:
                out.append(target_wght)
                saw_wght = True
            if not saw_exti:
                out.append(target_exti)
                saw_exti = True
            out.append("HKLF 4")
            saw_hklf = True
            continue
        out.append(raw)

    return "\n".join(out).rstrip() + "\n"


def discover_targets(root: Path, cutoff_label: str) -> tuple[Path, list[RefinementTarget]]:
    full_dir = root / FULL_RESULT_DIR / f"shelx_{cutoff_label}"
    targets = [
        RefinementTarget(
            dataset=dataset,
            result_dir=root / result_dir,
            refine_dir=root / result_dir / f"shelx_{cutoff_label}_from_full_model",
            hkl_source=root / result_dir / "shelx" / "shelx.hkl",
        )
        for dataset, result_dir in SPLIT_DATASETS
    ]
    return full_dir, targets


def ensure_inputs(full_dir: Path, targets: list[RefinementTarget]) -> None:
    required = [full_dir / name for name in ("shelx.res", "shelx.ins", "shelx.lst", "shelx.fcf", "shelx.hkl")]
    missing = [path for path in required if not path.exists()]
    missing.extend(target.hkl_source for target in targets if not target.hkl_source.exists())
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(str(path) for path in missing))


def prepare_target_folder(
    target: RefinementTarget,
    full_res_text: str,
    full_model_path: Path,
    active_shel: str,
    ls_cycles: int,
    args: argparse.Namespace,
) -> None:
    if target.refine_dir.exists() and any(target.refine_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Refinement folder already exists and is non-empty: {target.refine_dir}\n"
            "Pass --overwrite only if you really want to replace prepared/refined files."
        )
    target.refine_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target.hkl_source, target.refine_dir / "shelx.hkl")
    ins_text = patch_full_model_as_split_ins(
        full_res_text=full_res_text,
        active_shel=active_shel,
        ls_cycles=ls_cycles,
        target_wght="WGHT 0 0",
        target_exti="EXTI 10000",
    )
    write_text(target.refine_dir / "shelx.ins", ins_text)
    write_protocol(
        target=target,
        full_model_path=full_model_path,
        active_shel=active_shel,
        active_wght="WGHT 0 0",
        active_exti="EXTI 10000",
        active_hklf="HKLF 4",
        ls_cycles=ls_cycles,
        shelxl_command=args.shelxl_command,
        status="prepared",
        returncode=None,
    )


def write_protocol(
    target: RefinementTarget,
    full_model_path: Path,
    active_shel: str,
    active_wght: str,
    active_exti: str,
    active_hklf: str,
    ls_cycles: int,
    shelxl_command: str,
    status: str,
    returncode: int | None,
) -> None:
    lines = [
        "Controlled split refinement from full-data model",
        f"date_time: {datetime.now().isoformat(timespec='seconds')}",
        f"dataset: {target.dataset}",
        f"source_hkl_path: {target.hkl_source}",
        f"starting_model_path: {full_model_path}",
        f"active_shel: {active_shel}",
        f"active_wght: {active_wght}",
        f"active_exti: {active_exti}",
        f"active_hklf: {active_hklf}",
        f"ls_cycles: {ls_cycles}",
        f"shelxl_command: {shelxl_command}",
        f"status: {status}",
        f"returncode: {'' if returncode is None else returncode}",
        "",
    ]
    write_text(target.refine_dir / "refinement_protocol.txt", "\n".join(lines))


def run_shelxl(target: RefinementTarget, args: argparse.Namespace) -> tuple[int | None, str]:
    if args.skip_run:
        return None, "skipped"
    command = shlex.split(args.shelxl_command)
    log(f"Running SHELXL for {target.dataset}: {' '.join(command)}")
    try:
        proc = subprocess.run(
            command,
            cwd=target.refine_dir,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        write_text(target.refine_dir / "shelxl_stdout.txt", "")
        write_text(target.refine_dir / "shelxl_stderr.txt", f"{exc}\n")
        return None, "missing_shelxl_executable"
    except subprocess.TimeoutExpired as exc:
        write_text(target.refine_dir / "shelxl_stdout.txt", exc.stdout or "")
        write_text(target.refine_dir / "shelxl_stderr.txt", exc.stderr or "")
        return None, "timeout"
    write_text(target.refine_dir / "shelxl_stdout.txt", proc.stdout)
    write_text(target.refine_dir / "shelxl_stderr.txt", proc.stderr)
    return proc.returncode, "completed" if proc.returncode == 0 else "failed"


def parse_fcf(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    in_refln_loop = False
    labels: list[str] = []
    for raw in read_text(path).splitlines():
        stripped = raw.strip()
        if stripped == "loop_":
            in_refln_loop = False
            labels = []
            continue
        if stripped.startswith("_"):
            if stripped.startswith("_refln_"):
                labels.append(stripped)
                if "_refln_F_squared_calc" in labels and "_refln_F_squared_meas" in labels:
                    in_refln_loop = True
            elif in_refln_loop:
                break
            continue
        if not in_refln_loop or not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 7:
            continue
        try:
            records.append(
                {
                    "h": int(parts[0]),
                    "k": int(parts[1]),
                    "l": int(parts[2]),
                    "fc2": float(parts[3]),
                    "fo2": float(parts[4]),
                    "sigma_fo2": float(parts[5]),
                    "status": parts[6],
                }
            )
        except ValueError:
            continue
    return records


def summarize_fcf(dataset: str, fcf_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = parse_fcf(fcf_path)
    fc2 = np.array([row["fc2"] for row in records], dtype=float)
    fo2 = np.array([row["fo2"] for row in records], dtype=float)
    observed = sum(1 for row in records if str(row["status"]).lower() == "o")
    finite = np.isfinite(fc2) & np.isfinite(fo2)
    corr = float(np.corrcoef(fc2[finite], fo2[finite])[0, 1]) if finite.sum() > 1 else None
    summary = {
        "dataset": dataset,
        "fcf_path": str(fcf_path),
        "n_reflections_fcf": len(records),
        "n_observed_status_o": observed,
        "fc2_min": float(np.min(fc2)) if len(fc2) else None,
        "fc2_max": float(np.max(fc2)) if len(fc2) else None,
        "fo2_min": float(np.min(fo2)) if len(fo2) else None,
        "fo2_max": float(np.max(fo2)) if len(fo2) else None,
        "fc2_mean": float(np.mean(fc2)) if len(fc2) else None,
        "fo2_mean": float(np.mean(fo2)) if len(fo2) else None,
        "fc2_median": float(np.median(fc2)) if len(fc2) else None,
        "fo2_median": float(np.median(fo2)) if len(fo2) else None,
        "fo2_fc2_pearson": corr,
    }
    return summary, records


def parse_result_row(dataset: str, refine_dir: Path, run_status: str, returncode: int | None) -> dict[str, Any]:
    ins_text = read_text(refine_dir / "shelx.ins")
    res_text = read_text(refine_dir / "shelx.res")
    lst_text = read_text(refine_dir / "shelx.lst")
    settings = parse_active_settings(ins_text, res_text)
    stats = parse_refinement_stats(lst_text)
    unstable = "REFINEMENT UNSTABLE" in lst_text.upper()
    completed = run_status == "reference" or (
        run_status in {"completed", "existing"}
        and returncode in {None, 0}
        and (refine_dir / "shelx.lst").exists()
        and (refine_dir / "shelx.res").exists()
        and (refine_dir / "shelx.fcf").exists()
        and stats.get("r1_all") is not None
        and not unstable
    )
    failure_reason = ""
    if not completed and run_status != "reference":
        if unstable:
            failure_reason = "REFINEMENT UNSTABLE"
        elif returncode not in (None, 0):
            failure_reason = f"SHELXL return code {returncode}"
        elif not (refine_dir / "shelx.fcf").exists():
            failure_reason = "missing shelx.fcf"
        elif stats.get("r1_all") is None:
            failure_reason = "missing final R factors"
        else:
            failure_reason = run_status
    row = {
        "dataset": dataset,
        "refine_dir": str(refine_dir),
        "run_status": run_status,
        "shelxl_returncode": returncode,
        "refinement_completed": completed,
        "unstable": unstable,
        "failure_reason": failure_reason,
        "lst_path": str(refine_dir / "shelx.lst") if (refine_dir / "shelx.lst").exists() else "",
        "res_path": str(refine_dir / "shelx.res") if (refine_dir / "shelx.res").exists() else "",
        "fcf_path": str(refine_dir / "shelx.fcf") if (refine_dir / "shelx.fcf").exists() else "",
    }
    row.update(settings)
    row.update(normalize_stat_values(stats))
    return row


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for _, key in columns:
            value = row.get(key)
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cutoff_label_for_text(cutoff_label: str) -> str:
    return cutoff_label.replace("_", ".")


def write_summary_md(
    rows: list[dict[str, Any]],
    path: Path,
    plot_path: Path,
    cutoff_label: str,
    axis_percentile: float,
) -> None:
    columns = [
        ("dataset", "dataset"),
        ("completed", "refinement_completed"),
        ("failure", "failure_reason"),
        ("SHEL", "active_shel"),
        ("WGHT", "active_wght"),
        ("EXTI", "active_exti"),
        ("LS", "ls_command"),
        ("R1 >4sig", "r1_gt_4sigma"),
        ("R1 all", "r1_all"),
        ("wR2", "wr2"),
        ("GooF", "goof"),
        ("data >4sig", "data_gt_4sigma"),
        ("data all", "data_all"),
        ("params", "parameters"),
        ("restr", "restraints"),
        ("peak", "residual_peak"),
        ("hole", "residual_hole"),
    ]
    lines = [
        f"# Controlled {cutoff_label_for_text(cutoff_label)} A split refinement summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        markdown_table(rows, columns),
        "",
        f"Fo/Fc comparison plot: `{plot_path}`",
        f"Fo/Fc plot axis cap: shared p{axis_percentile:g} of all finite Fo/Fc values.",
        "",
        f"The split refinements use the full-data `shelx_{cutoff_label}/shelx.res` model as the common starting point, with active `WGHT 0 0`, `EXTI 10000`, `HKLF 4`, and the same active `SHEL`/LS cycle count as the full reference.",
        "",
    ]
    write_text(path, "\n".join(lines))


def plot_fo2_fc2(
    ordered_records: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
    cutoff_label: str,
    axis_percentile: float,
) -> None:
    all_values: list[float] = []
    for _, records in ordered_records:
        all_values.extend(math.sqrt(max(float(row["fc2"]), 0.0)) for row in records)
        all_values.extend(math.sqrt(max(float(row["fo2"]), 0.0)) for row in records)
    finite = np.array([value for value in all_values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return
    axis_min = min(0.0, float(np.min(finite)))
    if axis_percentile >= 100:
        axis_max = float(np.max(finite))
    else:
        axis_max = float(np.percentile(finite, axis_percentile))
    pad = 0.04 * (axis_max - axis_min if axis_max > axis_min else 1.0)
    axis_min -= pad
    axis_max += pad

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, (dataset, records) in zip(axes_flat, ordered_records):
        fc = np.sqrt(np.maximum(np.array([row["fc2"] for row in records], dtype=float), 0.0))
        fo = np.sqrt(np.maximum(np.array([row["fo2"] for row in records], dtype=float), 0.0))
        ax.scatter(fc, fo, s=5, alpha=0.28, linewidths=0)
        ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linewidth=1, alpha=0.8)
        ax.set_title(f"{dataset} (n={len(records)})")
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.grid(True, alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fo")
    for ax in axes[-1, :]:
        ax.set_xlabel("Fc")
    fig.suptitle(f"Fo vs Fc, common p{axis_percentile:g} axes, {cutoff_label_for_text(cutoff_label)} A protocol")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    full_dir, targets = discover_targets(args.root, args.cutoff_label)
    ensure_inputs(full_dir, targets)

    full_ins_text = read_text(full_dir / "shelx.ins")
    full_res_text = read_text(full_dir / "shelx.res")
    full_settings = parse_active_settings(full_ins_text, full_res_text)
    active_shel = full_settings.get("active_shel")
    ls_cycles = full_settings.get("ls_cycles")
    if active_shel is None or ls_cycles is None:
        raise SystemExit(f"Could not parse active SHEL/LS cycles from {full_dir}")

    comparison_dir = args.root / f"comparison_{args.cutoff_label}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = [
        parse_result_row("full", full_dir, "reference", None)
    ]

    for target in targets:
        if args.summarize_only:
            rows.append(parse_result_row(target.dataset, target.refine_dir, "existing", None))
            continue
        log(f"Preparing {target.dataset} in {target.refine_dir}")
        prepare_target_folder(
            target=target,
            full_res_text=full_res_text,
            full_model_path=full_dir / "shelx.res",
            active_shel=active_shel,
            ls_cycles=int(ls_cycles),
            args=args,
        )
        returncode, run_status = run_shelxl(target, args)
        row = parse_result_row(target.dataset, target.refine_dir, run_status, returncode)
        protocol_status = "completed" if row["refinement_completed"] else (row.get("failure_reason") or run_status)
        write_protocol(
            target=target,
            full_model_path=full_dir / "shelx.res",
            active_shel=row.get("active_shel") or active_shel,
            active_wght=row.get("active_wght") or "WGHT 0 0",
            active_exti=row.get("active_exti") or "EXTI 10000",
            active_hklf=row.get("active_hklf") or "HKLF 4",
            ls_cycles=int(row.get("ls_cycles") or ls_cycles),
            shelxl_command=args.shelxl_command,
            status=str(protocol_status),
            returncode=returncode,
        )
        rows.append(row)

    fcf_summaries: list[dict[str, Any]] = []
    plot_records: list[tuple[str, list[dict[str, Any]]]] = []
    for row in rows:
        fcf_value = row.get("fcf_path")
        if not fcf_value:
            continue
        fcf_path = Path(str(fcf_value))
        if fcf_path.is_file():
            summary, records = summarize_fcf(str(row["dataset"]), fcf_path)
            fcf_summaries.append(summary)
            plot_records.append((str(row["dataset"]), records))

    plot_path = comparison_dir / f"fo_vs_fc_full_random_low_high_{args.cutoff_label}.png"
    plot_fo2_fc2(
        plot_records,
        plot_path,
        cutoff_label=args.cutoff_label,
        axis_percentile=float(args.plot_axis_percentile),
    )

    summary_csv = comparison_dir / f"refinement_summary_{args.cutoff_label}.csv"
    summary_md = comparison_dir / f"refinement_summary_{args.cutoff_label}.md"
    fcf_csv = comparison_dir / f"fo2_fc2_summary_{args.cutoff_label}.csv"
    write_csv(rows, summary_csv)
    write_csv(fcf_summaries, fcf_csv)
    write_summary_md(rows, summary_md, plot_path, args.cutoff_label, float(args.plot_axis_percentile))

    log(f"Wrote {summary_csv}")
    log(f"Wrote {summary_md}")
    log(f"Wrote {fcf_csv}")
    log(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()

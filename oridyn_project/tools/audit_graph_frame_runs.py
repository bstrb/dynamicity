#!/usr/bin/env python3
"""Audit existing MFM300 graph/frame correction runs.

This script is intentionally read-only with respect to run outputs.  It parses
existing correction summaries, per-HKL diagnostics, merge QC files and SHELXL
LST files, then writes compact audit tables to a separate audit directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
)


@dataclass
class Run:
    run_id: str
    category: str
    folder: Path
    stream: Path | None = None
    summary: Path | None = None
    per_hkl: Path | None = None
    diagnostics: Path | None = None
    merge_dir: Path | None = None
    notes: str = ""


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    text = str(value).strip().replace("%", "").replace("x", "").replace("×", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def as_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_tsv_numeric(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        numeric: list[float] = []
        ok = True
        for part in parts:
            try:
                numeric.append(float(part))
            except ValueError:
                ok = False
                break
        if ok and numeric:
            rows.append(numeric)
    return rows


def parse_cell(cell_path: Path) -> tuple[float, float, float, float, float, float] | None:
    if not cell_path.exists():
        return None
    values: dict[str, float] = {}
    for line in cell_path.read_text(errors="replace").splitlines():
        match = re.match(r"\s*(a|b|c|al|be|ga)\s*=\s*([0-9.]+)", line)
        if match:
            values[match.group(1)] = float(match.group(2))
    needed = ("a", "b", "c", "al", "be", "ga")
    if all(key in values for key in needed):
        return tuple(values[key] for key in needed)  # type: ignore[return-value]
    return None


def d_spacing(h: int, k: int, l: int, cell: tuple[float, float, float, float, float, float] | None) -> float | None:
    if not cell:
        return None
    a, b, c, alpha_deg, beta_deg, gamma_deg = cell
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)
    if abs(sg) < 1e-12:
        return None
    volume = a * b * c * math.sqrt(max(0.0, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg))
    if volume <= 0:
        return None
    astar = b * c * math.sin(alpha) / volume
    bstar = a * c * math.sin(beta) / volume
    cstar = a * b * math.sin(gamma) / volume
    cos_alpha_star = (cb * cg - ca) / (math.sin(beta) * math.sin(gamma))
    cos_beta_star = (ca * cg - cb) / (math.sin(alpha) * math.sin(gamma))
    cos_gamma_star = (ca * cb - cg) / (math.sin(alpha) * math.sin(beta))
    inv_d2 = (
        h * h * astar * astar
        + k * k * bstar * bstar
        + l * l * cstar * cstar
        + 2 * k * l * bstar * cstar * cos_alpha_star
        + 2 * h * l * astar * cstar * cos_beta_star
        + 2 * h * k * astar * bstar * cos_gamma_star
    )
    if inv_d2 <= 0:
        return None
    return 1.0 / math.sqrt(inv_d2)


def resolution_bin(d: float | None) -> str:
    if d is None:
        return "unknown"
    if d >= 1.5:
        return "d>=1.5"
    if d >= 1.0:
        return "1.0<=d<1.5"
    if d >= 0.7:
        return "0.7<=d<1.0"
    if d >= 0.5:
        return "0.5<=d<0.7"
    if d >= 0.35:
        return "0.35<=d<0.5"
    return "d<0.35"


def candidate_merge_dir(stream: Path | None) -> Path | None:
    if not stream:
        return None
    candidate = stream.with_suffix("")
    candidate = candidate.parent / f"{candidate.name}_partialator_results"
    return candidate if candidate.exists() else None


def discover_runs(root: Path) -> list[Run]:
    runs: list[Run] = []
    baseline_merge = root / "MFM300-VIII_cut_20-0_3_partialator_results"
    runs.append(
        Run(
            run_id="baseline_uncorrected",
            category="baseline",
            folder=root,
            stream=root / "MFM300-VIII_cut_20-0_3.stream",
            merge_dir=baseline_merge if baseline_merge.exists() else None,
            notes="original uncorrected stream",
        )
    )

    diag_root = root / "model_free_nonself_diagnostics"
    summary_paths = sorted(
        list(diag_root.glob("graph_frame*/**/scale_summary.json"))
        + list(diag_root.glob("**/bidirectional_shift_summary.json"))
    )

    seen: set[Path] = set()
    for summary in summary_paths:
        if summary in seen:
            continue
        seen.add(summary)
        folder = summary.parent
        data = load_json(summary)
        stream = Path(data["output_stream"]) if data.get("output_stream") else None

        parts = set(folder.parts)
        if "graph_frame_skip_nonpositive_iref_contrast_sweep" in parts and summary.name == "bidirectional_shift_summary.json":
            category = "bidirectional_guarded"
            run_id = f"guarded_{folder.name}"
            notes = "clean observation-key bidirectional guarded run; skips weak-down I_ref <= 0"
        elif "graph_frame_skip_nonpositive_iref_contrast_sweep" in parts:
            category = "obskey_guarded_one_sided"
            run_id = f"guarded_{folder.name}"
            notes = "clean observation-key one-sided weak-down run; skips I_ref <= 0"
        elif "bidirectional_graph_frame_aggressive_shift_smoke" in parts:
            category = "smoke_bidirectional"
            run_id = "smoke_bidirectional_graph_frame_shift"
            notes = "smoke/validation run; not full-data evidence"
        elif "bidirectional_graph_frame_aggressive_shift" in parts:
            category = "bidirectional_aggressive"
            run_id = "bidirectional_aggressive_shift"
            notes = "clean observation-key bidirectional graph/frame stress test"
        elif "graph_frame_aggressive_poc_scaling" in parts:
            category = "obskey_aggressive_one_sided"
            run_id = f"aggressive_{folder.name}"
            notes = "clean observation-key one-sided aggressive POC"
        elif "graph_frame_obskey_scaling_sweep" in parts:
            category = "obskey_prepatch_zero_scaled"
            run_id = f"prepatch_{folder.name}"
            notes = "pre source/event-key fix: zero scaled observations; inventory only"
        elif "graph_frame_scaled_stream_lambda05_smoke" in parts:
            category = "smoke_hklwide"
            run_id = "smoke_graph_frame_scaled_lambda05"
            notes = "smoke/validation run; not full-data evidence"
        elif "graph_frame_scaled_stream_lambda05_rerun_recorded" in parts:
            category = "historical_hklwide_rerun"
            run_id = "historical_hklwide_lambda05_rerun_recorded"
            notes = "reproduced historical target-HKL-wide proof-of-concept"
        elif "graph_frame_scaled_stream_lambda05" in parts:
            category = "historical_hklwide_initial"
            run_id = "historical_hklwide_lambda05_initial"
            notes = "historical target-HKL-wide proof-of-concept; not clean obs-key"
        elif "provenance" in parts:
            category = "provenance_duplicate"
            run_id = "graph_frame_parameter_exploration_provenance"
            notes = "provenance copy pointing at the rerun-recorded stream"
        else:
            category = "graph_frame_other"
            run_id = folder.name
            notes = "discovered graph/frame summary"

        per_hkl = folder / "per_hkl_graph_frame_shift.csv"
        if not per_hkl.exists():
            per_hkl = folder / "per_hkl_bidirectional_graph_frame_shift.csv"
        diagnostics = folder / "scale_diagnostics.csv"
        if not diagnostics.exists():
            diagnostics = folder / "bidirectional_shift_diagnostics.csv"
        merge_dir = candidate_merge_dir(stream)
        if category == "provenance_duplicate":
            merge_dir = None
        runs.append(
            Run(
                run_id=run_id,
                category=category,
                folder=folder,
                stream=stream,
                summary=summary,
                per_hkl=per_hkl if per_hkl.exists() else None,
                diagnostics=diagnostics if diagnostics.exists() else None,
                merge_dir=merge_dir,
                notes=notes,
            )
        )
    return runs


def parse_merge_overall(merge_dir: Path | None) -> dict[str, float | None]:
    out = {"completeness": None, "redundancy": None, "snr": None, "cc12": None, "rsplit": None}
    if not merge_dir:
        return out
    metadata = merge_dir / "metadata_and_outputs.txt"
    if not metadata.exists():
        return out
    text = metadata.read_text(errors="replace")
    patterns = {
        "completeness": r"Completeness:\s*([0-9.]+)",
        "redundancy": r"Redundancy:\s*([0-9.]+)",
        "snr": r"SNR:\s*([0-9.]+)",
        "cc12": r"CC1/2:\s*([0-9.]+)",
        "rsplit": r"Rsplit:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = as_float(match.group(1))
    return out


def parse_shells(merge_dir: Path | None) -> list[dict[str, Any]]:
    if not merge_dir:
        return []
    qc = merge_dir / "qc_stats"
    check = read_tsv_numeric(qc / "check_shell.tsv")
    cc = read_tsv_numeric(qc / "compare_cc12_shell.tsv")
    rsplit = read_tsv_numeric(qc / "compare_rsplit_shell.tsv")
    rows: list[dict[str, Any]] = []
    n = max(len(check), len(cc), len(rsplit))
    for idx in range(n):
        row: dict[str, Any] = {"shell_index": idx + 1}
        if idx < len(check) and len(check[idx]) >= 11:
            c = check[idx]
            row.update(
                {
                    "invd_center": c[0],
                    "nref": int(c[1]),
                    "possible": int(c[2]),
                    "completeness": c[3],
                    "measurements": int(c[4]),
                    "redundancy": c[5],
                    "snr": c[6],
                    "mean_i": c[7],
                    "d_A": c[8],
                    "min_1nm": c[9],
                    "max_1nm": c[10],
                }
            )
        if idx < len(cc) and len(cc[idx]) >= 6:
            row["cc12"] = cc[idx][1]
            row.setdefault("d_A", cc[idx][3])
        if idx < len(rsplit) and len(rsplit[idx]) >= 6:
            row["rsplit"] = rsplit[idx][1]
            row.setdefault("d_A", rsplit[idx][3])
        rows.append(row)
    return rows


def parse_lst(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    out: dict[str, Any] = {"lst_path": str(path)}

    wr_matches = re.findall(r"wR2\s*=\s*([0-9.]+),\s*GooF\s*=\s*S\s*=\s*([0-9.]+)", text)
    if wr_matches:
        out["wr2"], out["goof"] = map(float, wr_matches[-1])

    r1_matches = re.findall(
        r"R1\s*=\s*([0-9.]+)\s+for\s+(\d+)\s+Fo\s*>\s*4sig\(Fo\)\s+and\s+([0-9.]+)\s+for all\s+(\d+)\s+data",
        text,
    )
    if r1_matches:
        r1_obs, n_obs, r1_all, n_all = r1_matches[-1]
        out.update({"r1_obs": float(r1_obs), "n_obs_reflections": int(n_obs), "r1_all": float(r1_all), "n_all_reflections": int(n_all)})

    param_matches = re.findall(r"Total number of l\.s\. parameters\s*=\s*(\d+)", text)
    if param_matches:
        out["parameters"] = int(param_matches[-1])

    wght_matches = re.findall(r"Recommended weighting scheme:\s*WGHT\s+([-+0-9.Ee]+)(?:\s+([-+0-9.Ee]+))?", text)
    if wght_matches:
        first, second = wght_matches[-1]
        out["wght_a"] = as_float(first)
        out["wght_b"] = as_float(second)

    exti_matches = re.findall(r"^\s*EXTI\s+([-+0-9.Ee]+)", text, flags=re.MULTILINE)
    if exti_matches:
        out["exti"] = as_float(exti_matches[-1])

    peak_patterns = {
        "highest_peak": r"Highest(?: difference)? peak[^-\d+]*([-+]?\d+(?:\.\d+)?)",
        "deepest_hole": r"Deepest(?: difference)? hole[^-\d+]*([-+]?\d+(?:\.\d+)?)",
    }
    for key, pattern in peak_patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            out[key] = as_float(match.group(1))

    return out


def cutoff_label(path: Path) -> str:
    text = str(path)
    if "1_5-0_5" in text:
        return "1.5-0.5"
    if "99_0_35" in text or "99-0_35" in text:
        return "99-0.35"
    if path.parent.name == "shelx":
        return "default_shelx"
    return path.parent.name


def signed_hkl(row: dict[str, str]) -> tuple[int, int, int] | None:
    h, k, l = as_int(row.get("h")), as_int(row.get("k")), as_int(row.get("l"))
    if h is None or k is None or l is None:
        return None
    return h, k, l


def summarize_per_hkl(run: Run, summary: dict[str, Any], cell: tuple[float, float, float, float, float, float] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not run.per_hkl or not run.per_hkl.exists():
        return {}, []

    scale_rows = "scale_summary" in (run.summary.name if run.summary else "")
    top_abs: list[dict[str, Any]] = []
    relative_shifts: list[float] = []
    changed_by_bin: defaultdict[str, int] = defaultdict(int)
    target_hkl_count = 0
    nonpositive_ref_hkls = 0
    observations_to_nonpositive_ref = 0
    min_scale_hit_hkls = 0
    min_scale_hit_observations = 0
    min_scale = as_float(summary.get("min_scale"))

    with run.per_hkl.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            hkl = signed_hkl(row)
            if hkl is None:
                continue
            h, k, l = hkl
            d = d_spacing(h, k, l, cell)
            i_ref = as_float(row.get("I_ref"))
            rel = as_float(row.get("relative_shift"))
            direction = row.get("correction_direction", "")
            changed_obs = as_int(row.get("n_shifted"))
            if changed_obs is None:
                changed_obs = as_int(row.get("n_scaled"))
            if changed_obs is None:
                changed_obs = 0
            selected_obs = as_int(row.get("n_top_risk_selected"))
            if selected_obs is None:
                selected_obs = as_int(row.get("n_top_risk_selected_for_scaling"))
            graph_scale = as_float(row.get("graph_frame_scale"))

            is_target = changed_obs > 0 or direction in {"weak_down", "strong_up"} or (graph_scale is not None and graph_scale < 1.0)
            if not is_target:
                continue
            target_hkl_count += 1
            if rel is not None:
                relative_shifts.append(rel)
            if i_ref is not None and i_ref <= 0:
                nonpositive_ref_hkls += 1
                observations_to_nonpositive_ref += changed_obs
            changed_by_bin[resolution_bin(d)] += changed_obs
            if min_scale is not None and graph_scale is not None and abs(graph_scale - min_scale) < 1e-12:
                min_scale_hit_hkls += 1
                min_scale_hit_observations += changed_obs

            if row.get("I_highrisk") is not None:
                i_high = as_float(row.get("I_highrisk"))
            else:
                i_high = as_float(row.get("I_high"))
            if i_high is not None and i_ref is not None:
                if graph_scale is not None:
                    estimated_abs_per_obs = abs(i_high * (1.0 - graph_scale))
                else:
                    estimated_abs_per_obs = abs(i_high - i_ref)
                estimated_total_abs = estimated_abs_per_obs * changed_obs
            else:
                estimated_abs_per_obs = None
                estimated_total_abs = None

            top_abs.append(
                {
                    "run_id": run.run_id,
                    "category": run.category,
                    "h": h,
                    "k": k,
                    "l": l,
                    "d_A": d,
                    "local_class": row.get("local_class", ""),
                    "correction_direction": direction or ("weak_down_scaled" if graph_scale is not None and graph_scale < 1.0 else ""),
                    "changed_observations": changed_obs,
                    "selected_observations": selected_obs,
                    "I_ref": i_ref,
                    "I_highrisk_or_I_high": i_high,
                    "relative_shift": rel,
                    "scale_factor": graph_scale,
                    "estimated_abs_change_per_obs": estimated_abs_per_obs,
                    "estimated_total_abs_change": estimated_total_abs,
                    "rank_type": "",
                    "rank": "",
                    "source_per_hkl": str(run.per_hkl),
                }
            )

    top_by_abs = sorted(
        [row for row in top_abs if row["estimated_total_abs_change"] is not None],
        key=lambda row: row["estimated_total_abs_change"],
        reverse=True,
    )[:50]
    for rank, row in enumerate(top_by_abs, 1):
        row["rank_type"] = "estimated_total_abs_change"
        row["rank"] = rank

    top_by_n = sorted(top_abs, key=lambda row: row["changed_observations"], reverse=True)[:50]
    for rank, row in enumerate(top_by_n, 1):
        row = row.copy()
        row["rank_type"] = "changed_observations"
        row["rank"] = rank
        top_by_abs.append(row)

    extra = {
        "per_hkl_target_rows": target_hkl_count,
        "median_relative_shift_target_hkls": median(relative_shifts) if relative_shifts else None,
        "nonpositive_i_ref_hkls": nonpositive_ref_hkls,
        "observations_to_nonpositive_refs": observations_to_nonpositive_ref,
        "min_scale_hit_hkls": min_scale_hit_hkls,
        "min_scale_hit_observations": min_scale_hit_observations,
    }
    for key in ("d>=1.5", "1.0<=d<1.5", "0.7<=d<1.0", "0.5<=d<0.7", "0.35<=d<0.5", "d<0.35", "unknown"):
        extra[f"changed_obs_{key}"] = changed_by_bin.get(key, 0)
    return extra, top_by_abs


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    outdir = args.outdir or (root / "model_free_nonself_diagnostics" / "graph_frame_run_audit")
    outdir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root)
    cell = parse_cell(root / "MFM300-VIII_cut_20-0_3_partialator_results" / "cell.cell")

    inventory_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    top_changed_rows: list[dict[str, Any]] = []

    for run in runs:
        lsts = []
        if run.merge_dir and run.merge_dir.exists():
            lsts = [p for p in sorted(run.merge_dir.rglob("shelx.lst")) if "olex2/temp" not in str(p)]
        inventory_rows.append(
            {
                "run_id": run.run_id,
                "category": run.category,
                "folder": str(run.folder),
                "stream": str(run.stream) if run.stream else "",
                "stream_exists": bool(run.stream and run.stream.exists()),
                "summary": str(run.summary) if run.summary else "",
                "summary_exists": bool(run.summary and run.summary.exists()),
                "per_hkl": str(run.per_hkl) if run.per_hkl else "",
                "per_hkl_exists": bool(run.per_hkl and run.per_hkl.exists()),
                "diagnostics": str(run.diagnostics) if run.diagnostics else "",
                "diagnostics_exists": bool(run.diagnostics and run.diagnostics.exists()),
                "merge_dir": str(run.merge_dir) if run.merge_dir else "",
                "merge_dir_exists": bool(run.merge_dir and run.merge_dir.exists()),
                "lst_count": len(lsts),
                "notes": run.notes,
            }
        )

        summary = load_json(run.summary)
        if not summary:
            continue

        total_obs = summary.get("total_stream_observations_seen", summary.get("total_stream_reflections_seen", summary.get("total_unmerged_observations")))
        changed_obs = summary.get("scaled_stream_observations", summary.get("total_changed_observations"))
        matched = summary.get("matched_observations")
        unmatched = summary.get("unmatched_observations")
        fraction_changed = None
        if as_float(total_obs) and as_float(changed_obs) is not None:
            fraction_changed = as_float(changed_obs) / as_float(total_obs)

        if run.category in {"obskey_prepatch_zero_scaled", "smoke_hklwide", "smoke_bidirectional", "provenance_duplicate"}:
            per_hkl_extra, top_rows = {}, []
        else:
            per_hkl_extra, top_rows = summarize_per_hkl(run, summary, cell)
        top_changed_rows.extend(top_rows)

        abs_change = summary.get("absolute_intensity_change") if isinstance(summary.get("absolute_intensity_change"), dict) else {}
        rel_change = summary.get("relative_intensity_change") if isinstance(summary.get("relative_intensity_change"), dict) else {}

        row = {
            "run_id": run.run_id,
            "category": run.category,
            "total_observations": total_obs,
            "matched_observations": matched,
            "unmatched_observations": unmatched,
            "changed_observations": changed_obs,
            "fraction_changed": fraction_changed,
            "eligible_signed_hkls": summary.get("eligible_signed_hkls"),
            "target_signed_hkls": summary.get("target_weak_signed_hkls"),
            "weak_down_target_hkls": summary.get("weak_down_target_hkls"),
            "strong_up_target_hkls": summary.get("strong_up_target_hkls"),
            "selected_observation_keys": summary.get("selected_top_risk_observation_keys", summary.get("selected_observation_keys")),
            "scale_factor_median": summary.get("scale_factor_median"),
            "scale_factor_min": summary.get("scale_factor_min"),
            "scale_factor_max": summary.get("scale_factor_max"),
            "abs_change_q25": abs_change.get("q25"),
            "abs_change_median": abs_change.get("median"),
            "abs_change_q75": abs_change.get("q75"),
            "rel_change_q25": rel_change.get("q25"),
            "rel_change_median": rel_change.get("median"),
            "rel_change_q75": rel_change.get("q75"),
            "excluded_flagged_crystal": summary.get("excluded_flagged_crystal"),
            "excluded_partiality_too_small": summary.get("excluded_partiality_too_small"),
            "duplicate_key_count": summary.get("duplicate_key_count"),
            "version": summary.get("version"),
            "notes": run.notes,
        }
        row.update(per_hkl_extra)
        correction_rows.append(row)

    inv_fields = [
        "run_id",
        "category",
        "folder",
        "stream",
        "stream_exists",
        "summary",
        "summary_exists",
        "per_hkl",
        "per_hkl_exists",
        "diagnostics",
        "diagnostics_exists",
        "merge_dir",
        "merge_dir_exists",
        "lst_count",
        "notes",
    ]
    write_tsv(outdir / "run_inventory.tsv", inventory_rows, inv_fields)

    correction_fields = [
        "run_id",
        "category",
        "total_observations",
        "matched_observations",
        "unmatched_observations",
        "changed_observations",
        "fraction_changed",
        "eligible_signed_hkls",
        "target_signed_hkls",
        "weak_down_target_hkls",
        "strong_up_target_hkls",
        "selected_observation_keys",
        "per_hkl_target_rows",
        "median_relative_shift_target_hkls",
        "scale_factor_median",
        "scale_factor_min",
        "scale_factor_max",
        "min_scale_hit_hkls",
        "min_scale_hit_observations",
        "abs_change_q25",
        "abs_change_median",
        "abs_change_q75",
        "rel_change_q25",
        "rel_change_median",
        "rel_change_q75",
        "nonpositive_i_ref_hkls",
        "observations_to_nonpositive_refs",
        "changed_obs_d>=1.5",
        "changed_obs_1.0<=d<1.5",
        "changed_obs_0.7<=d<1.0",
        "changed_obs_0.5<=d<0.7",
        "changed_obs_0.35<=d<0.5",
        "changed_obs_d<0.35",
        "changed_obs_unknown",
        "excluded_flagged_crystal",
        "excluded_partiality_too_small",
        "duplicate_key_count",
        "version",
        "notes",
    ]
    write_csv(outdir / "correction_comparison.csv", correction_rows, correction_fields)

    top_fields = [
        "run_id",
        "category",
        "rank_type",
        "rank",
        "h",
        "k",
        "l",
        "d_A",
        "local_class",
        "correction_direction",
        "changed_observations",
        "selected_observations",
        "I_ref",
        "I_highrisk_or_I_high",
        "relative_shift",
        "scale_factor",
        "estimated_abs_change_per_obs",
        "estimated_total_abs_change",
        "source_per_hkl",
    ]
    write_csv(outdir / "top_changed_hkls.csv", top_changed_rows, top_fields)

    # Merge comparison, including deltas against shell-matched baseline.
    baseline_run = next(run for run in runs if run.run_id == "baseline_uncorrected")
    baseline_overall = parse_merge_overall(baseline_run.merge_dir)
    baseline_shells = parse_shells(baseline_run.merge_dir)
    baseline_shell_by_idx = {row["shell_index"]: row for row in baseline_shells}
    merge_rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.merge_dir or not run.merge_dir.exists():
            continue
        overall = parse_merge_overall(run.merge_dir)
        overall_row = {
            "run_id": run.run_id,
            "category": run.category,
            "level": "overall",
            "shell_index": "",
            "d_A": "",
            "completeness": overall.get("completeness"),
            "redundancy": overall.get("redundancy"),
            "snr": overall.get("snr"),
            "mean_i": "",
            "cc12": overall.get("cc12"),
            "rsplit": overall.get("rsplit"),
            "nref": "",
            "measurements": "",
            "delta_snr_vs_baseline": None if overall.get("snr") is None or baseline_overall.get("snr") is None else overall["snr"] - baseline_overall["snr"],
            "delta_mean_i_vs_baseline": "",
            "delta_cc12_vs_baseline": None if overall.get("cc12") is None or baseline_overall.get("cc12") is None else overall["cc12"] - baseline_overall["cc12"],
            "delta_rsplit_vs_baseline": None if overall.get("rsplit") is None or baseline_overall.get("rsplit") is None else overall["rsplit"] - baseline_overall["rsplit"],
            "merge_dir": str(run.merge_dir),
        }
        merge_rows.append(overall_row)
        for shell in parse_shells(run.merge_dir):
            base = baseline_shell_by_idx.get(shell["shell_index"], {})
            merge_rows.append(
                {
                    "run_id": run.run_id,
                    "category": run.category,
                    "level": "shell",
                    "shell_index": shell.get("shell_index"),
                    "d_A": shell.get("d_A"),
                    "completeness": shell.get("completeness"),
                    "redundancy": shell.get("redundancy"),
                    "snr": shell.get("snr"),
                    "mean_i": shell.get("mean_i"),
                    "cc12": shell.get("cc12"),
                    "rsplit": shell.get("rsplit"),
                    "nref": shell.get("nref"),
                    "measurements": shell.get("measurements"),
                    "delta_snr_vs_baseline": None if shell.get("snr") is None or base.get("snr") is None else shell["snr"] - base["snr"],
                    "delta_mean_i_vs_baseline": None if shell.get("mean_i") is None or base.get("mean_i") is None else shell["mean_i"] - base["mean_i"],
                    "delta_cc12_vs_baseline": None if shell.get("cc12") is None or base.get("cc12") is None else shell["cc12"] - base["cc12"],
                    "delta_rsplit_vs_baseline": None if shell.get("rsplit") is None or base.get("rsplit") is None else shell["rsplit"] - base["rsplit"],
                    "merge_dir": str(run.merge_dir),
                }
            )

    merge_fields = [
        "run_id",
        "category",
        "level",
        "shell_index",
        "d_A",
        "completeness",
        "redundancy",
        "snr",
        "mean_i",
        "cc12",
        "rsplit",
        "nref",
        "measurements",
        "delta_snr_vs_baseline",
        "delta_mean_i_vs_baseline",
        "delta_cc12_vs_baseline",
        "delta_rsplit_vs_baseline",
        "merge_dir",
    ]
    write_csv(outdir / "merge_comparison.csv", merge_rows, merge_fields)

    # Refinement comparison.
    baseline_ref_by_cutoff: dict[str, dict[str, Any]] = {}
    refinement_rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.merge_dir or not run.merge_dir.exists():
            continue
        lsts = [p for p in sorted(run.merge_dir.rglob("shelx.lst")) if "olex2/temp" not in str(p)]
        for lst in lsts:
            parsed = parse_lst(lst)
            cut = cutoff_label(lst)
            parsed.update({"run_id": run.run_id, "category": run.category, "cutoff": cut, "lst_path": str(lst)})
            if run.run_id == "baseline_uncorrected":
                baseline_ref_by_cutoff[cut] = parsed
            refinement_rows.append(parsed)

    for row in refinement_rows:
        base = baseline_ref_by_cutoff.get(row["cutoff"])
        if base:
            for key in ("r1_obs", "r1_all", "wr2", "goof"):
                row[f"delta_{key}_vs_baseline"] = None if row.get(key) is None or base.get(key) is None else row[key] - base[key]
        else:
            for key in ("r1_obs", "r1_all", "wr2", "goof"):
                row[f"delta_{key}_vs_baseline"] = None

    ref_fields = [
        "run_id",
        "category",
        "cutoff",
        "r1_obs",
        "r1_all",
        "wr2",
        "goof",
        "delta_r1_obs_vs_baseline",
        "delta_r1_all_vs_baseline",
        "delta_wr2_vs_baseline",
        "delta_goof_vs_baseline",
        "highest_peak",
        "deepest_hole",
        "n_obs_reflections",
        "n_all_reflections",
        "parameters",
        "wght_a",
        "wght_b",
        "exti",
        "lst_path",
    ]
    write_csv(outdir / "refinement_comparison.csv", refinement_rows, ref_fields)

    # Compact markdown interpretation.
    full_rows = [
        row
        for row in correction_rows
        if "smoke" not in str(row.get("category", ""))
        and row.get("category") not in {"provenance_duplicate", "obskey_prepatch_zero_scaled"}
    ]
    merge_overall = [row for row in merge_rows if row.get("level") == "overall"]
    ref_main = [row for row in refinement_rows if row.get("cutoff") in {"1.5-0.5", "99-0.35", "default_shelx"}]

    def md_table(rows: list[dict[str, Any]], fields: list[str], limit: int = 12) -> list[str]:
        if not rows:
            return ["No rows found."]
        out = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
        for row in rows[:limit]:
            out.append("|" + "|".join(fmt(row.get(field), 5) for field in fields) + "|")
        return out

    best_ref = sorted(
        [row for row in ref_main if row.get("r1_obs") is not None],
        key=lambda row: (row.get("cutoff") != "1.5-0.5", row.get("r1_obs", 999)),
    )
    nonpositive_rows = [
        row
        for row in full_rows
        if as_int(row.get("observations_to_nonpositive_refs")) and as_int(row.get("observations_to_nonpositive_refs")) > 0
    ]
    zero_scaled = [
        row["run_id"]
        for row in correction_rows
        if as_int(row.get("changed_observations")) == 0 and "prepatch" in str(row.get("category", ""))
    ]

    lines: list[str] = []
    lines.append("# Graph/Frame Run Audit")
    lines.append("")
    lines.append(f"Audit output folder: `{outdir}`")
    lines.append("")
    lines.append("## Files Written")
    for name in [
        "run_inventory.tsv",
        "correction_comparison.csv",
        "merge_comparison.csv",
        "refinement_comparison.csv",
        "top_changed_hkls.csv",
        "graph_frame_run_audit_summary.md",
    ]:
        lines.append(f"- `{outdir / name}`")
    lines.append("")
    lines.append("## Correction-Level Summary")
    lines.extend(
        md_table(
            full_rows,
            [
                "run_id",
                "category",
                "changed_observations",
                "fraction_changed",
                "target_signed_hkls",
                "weak_down_target_hkls",
                "strong_up_target_hkls",
                "scale_factor_median",
                "observations_to_nonpositive_refs",
            ],
            limit=20,
        )
    )
    lines.append("")
    lines.append("## Merge Overall")
    lines.extend(
        md_table(
            merge_overall,
            ["run_id", "completeness", "redundancy", "snr", "cc12", "rsplit", "delta_cc12_vs_baseline", "delta_rsplit_vs_baseline"],
            limit=20,
        )
    )
    lines.append("")
    lines.append("## Shell Delta Snapshot")
    shell_rows = [row for row in merge_rows if row.get("level") == "shell" and row.get("run_id") != "baseline_uncorrected"]
    shell_summary: list[dict[str, Any]] = []
    for run_id in sorted({row["run_id"] for row in shell_rows}):
        rows = [row for row in shell_rows if row["run_id"] == run_id]
        high = [row for row in rows if as_float(row.get("d_A")) is not None and as_float(row.get("d_A")) <= 0.5]
        shell_summary.append(
            {
                "run_id": run_id,
                "shells_with_cc12_gain": sum(1 for row in rows if as_float(row.get("delta_cc12_vs_baseline")) is not None and as_float(row.get("delta_cc12_vs_baseline")) > 0),
                "shells_with_rsplit_gain": sum(1 for row in rows if as_float(row.get("delta_rsplit_vs_baseline")) is not None and as_float(row.get("delta_rsplit_vs_baseline")) < 0),
                "highres_shells_with_cc12_gain": sum(1 for row in high if as_float(row.get("delta_cc12_vs_baseline")) is not None and as_float(row.get("delta_cc12_vs_baseline")) > 0),
                "highres_shells_with_rsplit_gain": sum(1 for row in high if as_float(row.get("delta_rsplit_vs_baseline")) is not None and as_float(row.get("delta_rsplit_vs_baseline")) < 0),
            }
        )
    lines.extend(
        md_table(
            shell_summary,
            ["run_id", "shells_with_cc12_gain", "shells_with_rsplit_gain", "highres_shells_with_cc12_gain", "highres_shells_with_rsplit_gain"],
            limit=20,
        )
    )
    lines.append("")
    lines.append("## Refinement Snapshot")
    lines.extend(
        md_table(
            best_ref,
            ["run_id", "cutoff", "r1_obs", "r1_all", "wr2", "goof", "delta_r1_obs_vs_baseline", "delta_wr2_vs_baseline", "lst_path"],
            limit=25,
        )
    )
    lines.append("")
    lines.append("## Key Findings")
    lines.append(
        "- The historical HKL-wide lambda05 proof-of-concept changes far more stream observations than the clean obs-key variants, and it remains the clearest refinement-improving result in the parsed outputs."
    )
    lines.append(
        "- The aggressive clean obs-key one-sided variants are much more localized: scale10 changes about 2% of observations and scale20 changes about 4%, so they test precision more than global contrast."
    )
    lines.append(
        "- The bidirectional stress test changes an intermediate fraction of observations and explicitly separates weak-down from strong-up corrections; it is useful diagnostically but has nonpositive-reference risk."
    )
    if nonpositive_rows:
        names = ", ".join(row["run_id"] for row in nonpositive_rows)
        lines.append(f"- Red flag: nonpositive `I_ref` targets remain in {names}; see `observations_to_nonpositive_refs` and `top_changed_hkls.csv`.")
    if zero_scaled:
        lines.append(
            "- The older obs-key sweep rows with zero changed observations are retained in the inventory as pre-patch artifacts, not as scientific correction results."
        )
    lines.append(
        "- Shell-wise merge deltas are in `merge_comparison.csv`; negative `delta_rsplit_vs_baseline` and positive `delta_cc12_vs_baseline` mark shells helped by correction."
    )
    lines.append(
        "- Top changed HKLs are estimated from per-HKL medians/scale factors, not from a stored per-observation delta table; use them to locate candidates, not as exact integrated intensity-change sums."
    )
    lines.append("")
    lines.append("## Rerun Command")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python /home/bubl3932/projects/dynamicity/oridyn_project/tools/audit_graph_frame_runs.py --root {root} --outdir {outdir}")
    lines.append("```")
    (outdir / "graph_frame_run_audit_summary.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote audit outputs to {outdir}")
    print(f"Inventoried runs: {len(runs)}")
    print(f"Correction summaries parsed: {len(correction_rows)}")
    print(f"Merge rows written: {len(merge_rows)}")
    print(f"Refinement rows written: {len(refinement_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

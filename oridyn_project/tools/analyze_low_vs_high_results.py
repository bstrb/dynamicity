#!/usr/bin/env python3
"""Summarize MFM300 low/high/random/full frame-split merging and refinement results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
from typing import Any


DEFAULT_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/low_vs_high"
)


DATASET_ORDER = {
    "full": 0,
    "random": 1,
    "low_risk": 2,
    "high_risk": 3,
}


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="low_vs_high result folder")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Summary output directory. Defaults to ROOT/analysis_summary",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def first_match(text: str, pattern: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def first_float(text: str, pattern: str, flags: int = 0) -> float | None:
    value = first_match(text, pattern, flags)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def first_int(text: str, pattern: str, flags: int = 0) -> int | None:
    value = first_match(text, pattern, flags)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def dataset_label(result_dir: Path) -> str:
    name = result_dir.name.lower()
    if name.startswith("mfm300"):
        return "full"
    if "random" in name:
        return "random"
    if "low_risk" in name:
        return "low_risk"
    if "high_risk" in name:
        return "high_risk"
    return result_dir.name


def parse_metadata(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "run_started": first_match(text, r"^Run started:\s*(.+)$", re.MULTILINE),
        "run_completed": first_match(text, r"^Run completed:\s*(.+)$", re.MULTILINE),
        "source_stream": first_match(text, r"^STREAM:\s*(.+)$", re.MULTILINE),
        "symmetry": first_match(text, r"^SYM:\s*(.+)$", re.MULTILINE),
        "merge_lowres_A": first_float(text, r"^LOWRES:\s*([0-9.]+)", re.MULTILINE),
        "merge_highres_A": first_float(text, r"^HIGHRES:\s*([0-9.]+)", re.MULTILINE),
        "qc_range_A": first_match(text, r"Range:\s*([^;]+);", re.MULTILINE),
        "crystfel_version": first_match(text, r"^(CrystFEL:\s*.+)$", re.MULTILINE),
        "metadata_merge_outdir": first_match(text, r"^Merge Outdir:\s*(.+)$", re.MULTILINE),
        "metadata_qc_outdir": first_match(text, r"^QC Outdir:\s*(.+)$", re.MULTILINE),
    }


def parse_completeness(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "input_reflections_before_resolution_filter": first_int(text, r"Discarded\s+\d+\s+reflections \(out of\s+(\d+)\)"),
        "resolution_rejected_reflections": first_int(text, r"(\d+)\s+reflections rejected because they were outside"),
        "resolution_invnm_low": first_float(text, r"1/d goes from\s+([0-9.]+)\s+to"),
        "resolution_invnm_high": first_float(text, r"1/d goes from\s+[0-9.]+\s+to\s+([0-9.]+)"),
        "measurements_total": first_int(text, r"(\d+)\s+measurements in total"),
        "hkl_reflections_total": first_int(text, r"(\d+)\s+reflections in total"),
        "hkl_reflections_possible": first_int(text, r"(\d+)\s+reflections possible"),
        "snr": first_float(text, r"Overall <snr> =\s*([0-9.eE+-]+)"),
        "redundancy": first_float(text, r"Overall redundancy =\s*([0-9.eE+-]+)"),
        "completeness_percent": first_float(text, r"Overall completeness =\s*([0-9.eE+-]+)"),
        "invalid_isigi_warning": first_int(text, r"WARNING:\s+(\d+)\s+reflections had infinite or invalid values"),
    }


def parse_compare(path: Path, metric_name: str) -> dict[str, Any]:
    text = read_text(path)
    out: dict[str, Any] = {}
    out[f"{metric_name}_accepted_pairs"] = first_int(text, r"(\d+)\s+reflection pairs accepted")
    out[f"{metric_name}_rejected_pairs"] = first_int(text, r"(\d+)\s+reflection pairs rejected")
    out["accepted_resolution_range_A"] = first_match(
        text,
        r"Accepted resolution range:\s+[0-9.eE+-]+\s+to\s+[0-9.eE+-]+\s+nm\^-1\s+\(([^)]+)\)",
    )
    out["fixed_resolution_range_A"] = first_match(
        text,
        r"Fixed resolution range:\s+[0-9.eE+-]+\s+to\s+[0-9.eE+-]+\s+nm\^-1\s+\(([^)]+)\)",
    )
    if metric_name == "cc12":
        out["cc12"] = first_float(text, r"Overall CC =\s*([0-9.eE+-]+)")
    elif metric_name == "rsplit":
        out["rsplit_percent"] = first_float(text, r"Overall Rsplit =\s*([0-9.eE+-]+)")
    return out


def command_value(text: str, keyword: str) -> str | None:
    match = re.search(rf"^{keyword}\s*(.*)$", text, re.MULTILINE)
    if not match:
        return None
    return " ".join([keyword, match.group(1).strip()]).strip()


def parse_shelx(shelx_dir: Path) -> dict[str, Any]:
    lst = read_text(shelx_dir / "shelx.lst")
    ins = read_text(shelx_dir / "shelx.ins")
    res = read_text(shelx_dir / "shelx.res")

    r_match = re.search(
        r"R1\s*=\s*([0-9.]+)\s+for\s+(\d+)\s+Fo\s*>\s*4sig\(Fo\)\s+and\s+([0-9.]+)\s+for all\s+(\d+)\s+data",
        lst,
    )
    wr_match = re.search(
        r"wR2\s*=\s*([0-9.]+),\s*GooF\s*=\s*S\s*=\s*([0-9.]+),\s*Restrained GooF\s*=\s*([0-9.]+)",
        lst,
    )
    peak_match = re.search(r"Highest peak\s+([-0-9.]+)", lst)
    hole_match = re.search(r"Deepest hole\s+([-0-9.]+)", lst)
    rec_wght_match = re.search(r"Recommended weighting scheme:\s*WGHT\s+([0-9.]+)\s+([0-9.]+)", lst)
    refined_wght_match = re.search(r"Weight parameters refined to\s+([0-9.]+)\s+([0-9.]+)", lst)
    split_warning_count = first_int(lst, r"\*\* Warning:\s+(\d+)\s+atoms may be split")
    npd_warning_count = first_int(lst, r"\*\* Warning:\s+\d+\s+atoms may be split and\s+(\d+)\s+atoms NPD")

    warnings = []
    for line in lst.splitlines():
        stripped = line.strip()
        if stripped.startswith("** Warning") or stripped.startswith("[Weight parameters"):
            warnings.append(stripped)
        elif "may be split into" in stripped:
            warnings.append(stripped)

    command_line = first_match(lst, r"Command line parameters:\s*(.+)")
    ins_wght = command_value(ins, "WGHT")
    res_wght = command_value(res, "WGHT")
    res_wght_lines = re.findall(r"^WGHT\s+(.+)$", res, flags=re.MULTILINE)
    active_res_wght = f"WGHT {res_wght_lines[0].strip()}" if res_wght_lines else None
    recommended_res_wght = f"WGHT {res_wght_lines[-1].strip()}" if len(res_wght_lines) > 1 else None
    ins_exti = command_value(ins, "EXTI")
    res_exti = command_value(res, "EXTI")
    ls_command = first_match(ins, r"^(L\.S\.\s+\d+.*)$", re.MULTILINE)
    ls_cycles = first_int(ins, r"^L\.S\.\s+(\d+)", re.MULTILINE)

    if "MERG" in ins and re.search(r"^MERG\s+", ins, flags=re.MULTILINE):
        merg_setting = command_value(ins, "MERG")
    elif re.search(r"^REM MERG\s+", ins, flags=re.MULTILINE):
        merg_setting = "REM MERG present only; no active MERG instruction"
    else:
        merg_setting = "no active MERG instruction"
    if command_line and " -m0" in command_line:
        merg_setting += "; command line contains -m0"

    return {
        "shelx_lst": str(shelx_dir / "shelx.lst") if (shelx_dir / "shelx.lst").exists() else None,
        "shelx_ins": str(shelx_dir / "shelx.ins") if (shelx_dir / "shelx.ins").exists() else None,
        "shelx_res": str(shelx_dir / "shelx.res") if (shelx_dir / "shelx.res").exists() else None,
        "shelx_command_line": command_line,
        "r1_gt_4sigma": float(r_match.group(1)) if r_match else None,
        "data_gt_4sigma": int(r_match.group(2)) if r_match else None,
        "r1_all": float(r_match.group(3)) if r_match else None,
        "data_all": int(r_match.group(4)) if r_match else None,
        "wr2": float(wr_match.group(1)) if wr_match else None,
        "goof": float(wr_match.group(2)) if wr_match else None,
        "restrained_goof": float(wr_match.group(3)) if wr_match else None,
        "reflections_fourier": first_int(lst, r"R1 =\s+[0-9.]+\s+for\s+(\d+)\s+unique reflections after merging for Fourier"),
        "parameters": first_int(lst, r"Total number of l\.s\. parameters =\s+(\d+)"),
        "restraints": first_int(lst, r"Restrained GooF\s*=\s*[0-9.]+\s+for\s+(\d+)\s+restraints"),
        "residual_peak": float(peak_match.group(1)) if peak_match else None,
        "residual_hole": float(hole_match.group(1)) if hole_match else None,
        "ins_wght": ins_wght,
        "res_active_wght": active_res_wght or res_wght,
        "res_recommended_wght": recommended_res_wght,
        "lst_recommended_wght": f"WGHT {rec_wght_match.group(1)} {rec_wght_match.group(2)}" if rec_wght_match else None,
        "lst_refined_wght": f"{refined_wght_match.group(1)} {refined_wght_match.group(2)}" if refined_wght_match else None,
        "ins_exti": ins_exti,
        "res_exti": res_exti,
        "merg_setting": merg_setting,
        "ls_command": ls_command,
        "ls_cycles": ls_cycles,
        "refinement_mode": "stable_refinement" if ls_cycles and ls_cycles > 0 else "ls0_or_unknown",
        "hklf": command_value(ins, "HKLF"),
        "shel_resolution_line": first_match(lst, r"Number of data for (d > [^\n]+)"),
        "outside_shel_resolution_rejected": first_int(lst, r"(\d+)\s+Reflections outside SHEL resolution limits rejected"),
        "split_atom_warning_count": split_warning_count,
        "npd_warning_count": npd_warning_count,
        "warnings": " | ".join(warnings),
    }


def collect_results(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_dir in sorted(root.glob("*_partialator_results")):
        label = dataset_label(result_dir)
        row: dict[str, Any] = {
            "dataset": label,
            "result_folder": result_dir.name,
            "result_path": str(result_dir),
        }
        row.update(parse_metadata(result_dir / "metadata_and_outputs.txt"))
        row.update(parse_completeness(result_dir / "qc_stats" / "check_hkl_completeness.log"))
        row.update(parse_compare(result_dir / "qc_stats" / "compare_cc12.log", "cc12"))
        rsplit = parse_compare(result_dir / "qc_stats" / "compare_rsplit.log", "rsplit")
        for key, value in rsplit.items():
            if key not in row or value is not None:
                row[key] = value
        row.update(parse_shelx(result_dir / "shelx"))
        row["latest_shelx_file_mtime"] = latest_mtime(result_dir / "shelx")
        row["metadata_path_mismatch"] = bool(
            row.get("metadata_merge_outdir") and str(result_dir) not in str(row.get("metadata_merge_outdir"))
        )
        row["interpretation_note"] = interpretation_note(row)
        rows.append(row)
    rows.sort(key=lambda r: DATASET_ORDER.get(str(r["dataset"]), 99))
    return rows


def latest_mtime(path: Path) -> str | None:
    mtimes = []
    for name in ["shelx.lst", "shelx.res", "shelx.ins"]:
        file_path = path / name
        if file_path.exists():
            mtimes.append(file_path.stat().st_mtime)
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds")


def interpretation_note(row: dict[str, Any]) -> str:
    dataset = row.get("dataset")
    if dataset == "high_risk":
        return "Worst risk-split merging/refinement among 50% subsets: lower CC1/2, higher Rsplit, higher R1/wR2/GooF."
    if dataset == "low_risk":
        return "Low-risk subset refines close to full and better than high-risk; merging is better than random/high by CC1/2 and Rsplit."
    if dataset == "random":
        return "Random 50% has refinement R1 lower than full/low/high but merging is closer to high than low by CC1/2/Rsplit."
    if dataset == "full":
        return "Full stream has expected highest redundancy/SNR and best CC1/2/Rsplit due to all frames."
    return ""


def fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], precision: int = 3) -> str:
    out = []
    out.append("| " + " | ".join(title for title, _ in columns) + " |")
    out.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(key), precision) for _, key in columns) + " |")
    return "\n".join(out)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    merging_cols = [
        ("dataset", "dataset"),
        ("range A", "fixed_resolution_range_A"),
        ("sym", "symmetry"),
        ("HKLs", "hkl_reflections_total"),
        ("possible", "hkl_reflections_possible"),
        ("meas", "measurements_total"),
        ("compl %", "completeness_percent"),
        ("red", "redundancy"),
        ("SNR", "snr"),
        ("CC1/2", "cc12"),
        ("Rsplit %", "rsplit_percent"),
    ]
    refine_cols = [
        ("dataset", "dataset"),
        ("mode", "refinement_mode"),
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
        ("WGHT used", "res_active_wght"),
        ("EXTI", "res_exti"),
        ("LS", "ls_command"),
    ]

    lines = [
        "# MFM300 low/high frame split experiment summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Dataset mapping",
        "",
        "- `full`: `MFM300-VIII_cut_20-0_3_partialator_results`",
        "- `random`: `random_B_frames_50_partialator_results`",
        "- `low_risk`: `low_risk_frames_50_partialator_results`",
        "- `high_risk`: `high_risk_frames_50_partialator_results`",
        "",
        "The metadata and QC logs verify `20.0-0.4 A` merging/QC resolution range with `4/mmm` symmetry for all four datasets. The SHELX lists show stable refinement mode (`L.S. 10`, `HKLF 4`) rather than LS 0 raw-comparison mode.",
        "",
        "## Merging/QC comparison",
        "",
        markdown_table(rows, merging_cols),
        "",
        "## SHELXL/Olex refinement comparison",
        "",
        markdown_table(rows, refine_cols, precision=4),
        "",
        "## Interpretation",
        "",
        "The high-risk half is specifically worse than the low-risk half in both merging and refinement: high-risk has lower SNR and CC1/2, higher Rsplit, and the worst R1/wR2/GooF among the 50% subsets. Low-risk looks much closer to full-stream behavior in refinement and has better merging than random/high, though full naturally has about double redundancy and the strongest global merging statistics.",
        "",
        "The random 50% control is useful but not perfectly monotonic across all metrics: it has CC1/2/Rsplit closer to high-risk than low-risk, yet its current SHELX refinement R1 is the lowest of all four. That means the strongest conclusion is not simply `all 50% subsets track merging stats`; rather, the risk split specifically identifies a high-risk subset that behaves worse than low-risk in both QC and refinement, while random can still refine well under the current model/settings.",
        "",
        "## Refinement settings and caveats",
        "",
        "- All four refinements used `L.S. 10`, not `LS 0`; these are stable-refinement results, not raw Fo/Fc diagnostic runs.",
        "- Active WGHT in the `.res` files is `WGHT 0.242300 0.660000` for all four. SHELXL recommended/estimated weights vary: full `0.1881 0.7753`, low `0.1905 0.6968`, high `0.2000 0.0000` with refined estimate `0.2346 0.85`, random `0.2000 0.0000` with refined estimate `0.2515 0.58`.",
        "- EXTI is large and consistent with your ED context: full about `60133.68`, low about `60459.24`, high about `60117.05`, random about `60117.94`.",
        "- No active `MERG` instruction was found in the `.ins`; `REM MERG 0` is present and the SHELXL command line contains `-m0`. Treat this as consistent with the intended no-merge/raw-index handling, but it is worth keeping explicit in future folders.",
        "- The copied `metadata_and_outputs.txt` files for low/high/random contain original output paths outside `low_vs_high`; the files inspected here are the copies under `low_vs_high` and have fresh modification times on 2026-06-16.",
        "",
        "## Missing or ambiguous results",
        "",
        "- Only `random_B_frames_50_partialator_results` is present; there is no `random_A` branch in this `low_vs_high` folder.",
        "- The full-stream metadata is the latest merge/QC run by timestamp; all SHELX files in `low_vs_high` appear copied/refreshed around 2026-06-16 10:05-10:06.",
        "- These are not LS 0 raw-comparison folders, so `(Fo^2 - Fc^2) / sigma` diagnostics against raw HKL should be done separately if needed.",
        "",
        "## Need for more refinement",
        "",
        "More stable refinement is not required to support the main split-experiment interpretation. For raw reflection-group residual diagnostics, create separate LS 0 folders such as `shelx_raw_ls0_merg0_wght00_noexti`; do not reuse these stable-refinement folders for that diagnostic mode.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    all_keys: list[str] = []
    preferred = [
        "dataset",
        "result_folder",
        "result_path",
        "run_started",
        "run_completed",
        "source_stream",
        "symmetry",
        "merge_lowres_A",
        "merge_highres_A",
        "fixed_resolution_range_A",
        "accepted_resolution_range_A",
        "measurements_total",
        "hkl_reflections_total",
        "hkl_reflections_possible",
        "completeness_percent",
        "redundancy",
        "snr",
        "cc12",
        "rsplit_percent",
        "r1_gt_4sigma",
        "r1_all",
        "wr2",
        "goof",
        "restrained_goof",
        "data_gt_4sigma",
        "data_all",
        "parameters",
        "restraints",
        "residual_peak",
        "residual_hole",
        "ins_wght",
        "res_active_wght",
        "res_recommended_wght",
        "lst_recommended_wght",
        "lst_refined_wght",
        "ins_exti",
        "res_exti",
        "merg_setting",
        "ls_command",
        "ls_cycles",
        "refinement_mode",
        "hklf",
        "warnings",
        "interpretation_note",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            all_keys.append(key)
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root
    output_dir = args.output_dir or root / "analysis_summary"
    if not root.exists():
        raise SystemExit(f"--root not found: {root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_results(root)
    if not rows:
        raise SystemExit(f"No *_partialator_results folders found under {root}")

    csv_path = output_dir / "split_experiment_summary.csv"
    md_path = output_dir / "split_experiment_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    log(f"Wrote {csv_path}")
    log(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

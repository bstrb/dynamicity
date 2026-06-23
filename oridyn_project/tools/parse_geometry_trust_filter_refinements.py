#!/usr/bin/env python3
"""Parse completed SHELXL refinements for the geometry-trust filter sweep.

This script only reads existing refinement and merge/QC outputs. It does not
run SHELXL, partialator, merging, filtering, or any correction step.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


DATASETS = [
    ("full", "full/shelx", "full"),
    ("keep90", "geometry_trust_keep90_partialator_results_20260623T0204/shelx", "geometry_trust_keep90_partialator_results_20260623T0204"),
    ("keep80", "geometry_trust_keep80_partialator_results_20260623T0206/shelx", "geometry_trust_keep80_partialator_results_20260623T0206"),
    ("keep70", "geometry_trust_keep70_partialator_results_20260623T0208/shelx", "geometry_trust_keep70_partialator_results_20260623T0208"),
    ("keep60", "geometry_trust_keep60_partialator_results_20260623T0210/shelx", "geometry_trust_keep60_partialator_results_20260623T0210"),
    ("keep50", "geometry_trust_keep50_partialator_results_20260623T0212/shelx", "geometry_trust_keep50_partialator_results_20260623T0212"),
]

OUTPUT_CSV = "refinement_filter_sweep_summary.csv"
OUTPUT_MD = "refinement_filter_sweep_summary.md"

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"

FINAL_WR2_RE = re.compile(
    rf"wR2\s*=\s*({FLOAT_RE}),\s*GooF\s*=\s*S\s*=\s*({FLOAT_RE}).*?for all data",
    re.IGNORECASE,
)
R1_RE = re.compile(
    rf"R1\s*=\s*({FLOAT_RE})\s*for\s+(\d+)\s+Fo\s*>\s*4sig\(Fo\)\s+and\s+"
    rf"({FLOAT_RE})\s*for all\s+(\d+)\s+data",
    re.IGNORECASE,
)
PARAM_RE = re.compile(r"Total number of l\.s\. parameters\s*=\s*(\d+)", re.IGNORECASE)
PEAK_RE = re.compile(rf"Highest peak\s+({FLOAT_RE})", re.IGNORECASE)
HOLE_RE = re.compile(rf"Deepest hole\s+({FLOAT_RE})", re.IGNORECASE)
EXTI_INSTRUCTION_RE = re.compile(rf"^\s*EXTI\s*({FLOAT_RE})", re.IGNORECASE | re.MULTILINE)
EXTI_PARAMETER_RE = re.compile(rf"^\s*\d+\s+({FLOAT_RE})\s+{FLOAT_RE}\s+{FLOAT_RE}\s+EXTI\b", re.IGNORECASE | re.MULTILINE)
WGHT_RE = re.compile(r"^\s*WGHT\s+(.+)$", re.IGNORECASE | re.MULTILINE)
SHEL_RE = re.compile(r"^\s*SHEL\s+(.+)$", re.IGNORECASE | re.MULTILINE)

SNR_RE = re.compile(rf"Overall <snr>\s*=\s*({FLOAT_RE})", re.IGNORECASE)
REDUNDANCY_RE = re.compile(rf"Overall redundancy\s*=\s*({FLOAT_RE})", re.IGNORECASE)
COMPLETENESS_RE = re.compile(rf"Overall completeness\s*=\s*({FLOAT_RE})\s*%", re.IGNORECASE)
MEASUREMENTS_RE = re.compile(r"(\d+)\s+measurements in total", re.IGNORECASE)
REFLECTIONS_RE = re.compile(r"(\d+)\s+reflections in total", re.IGNORECASE)
CC_RE = re.compile(rf"Overall CC\s*=\s*({FLOAT_RE})", re.IGNORECASE)
RSPLIT_RE = re.compile(rf"Overall Rsplit\s*=\s*({FLOAT_RE})\s*%", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    shelx_dir: Path
    run_dir: Path
    lst_path: Path
    qc_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        required=True,
        type=Path,
        help="geometry_trust_filter_sweep_keep90_80_70_60_50_20260623 root",
    )
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--output-md", default=OUTPUT_MD)
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists() or path.name.endswith(":Zone.Identifier"):
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def last_float_match(pattern: re.Pattern[str], text: str, group: int = 1) -> float | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group(group))


def last_int_match(pattern: re.Pattern[str], text: str, group: int = 1) -> int | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return int(matches[-1].group(group))


def first_text_match(pattern: re.Pattern[str], text: str, group: int = 1) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(group).strip()


def parse_shelxl_lst(lst_path: Path) -> dict[str, Any]:
    text = read_text(lst_path)
    row: dict[str, Any] = {
        "lst_path": str(lst_path),
        "lst_exists": bool(text),
        "R1_obs": None,
        "R1_all": None,
        "wR2": None,
        "GooF": None,
        "observed_reflections": None,
        "all_reflections": None,
        "number_of_parameters": None,
        "residual_peak": None,
        "residual_hole": None,
        "exti": None,
        "wght": None,
        "shel_resolution": None,
    }
    if not text:
        return row

    if match := list(FINAL_WR2_RE.finditer(text)):
        last = match[-1]
        row["wR2"] = float(last.group(1))
        row["GooF"] = float(last.group(2))

    if match := list(R1_RE.finditer(text)):
        last = match[-1]
        row["R1_obs"] = float(last.group(1))
        row["observed_reflections"] = int(last.group(2))
        row["R1_all"] = float(last.group(3))
        row["all_reflections"] = int(last.group(4))

    row["number_of_parameters"] = last_int_match(PARAM_RE, text)
    row["residual_peak"] = last_float_match(PEAK_RE, text)
    row["residual_hole"] = last_float_match(HOLE_RE, text)
    row["exti"] = last_float_match(EXTI_INSTRUCTION_RE, text)
    if row["exti"] is None:
        row["exti"] = last_float_match(EXTI_PARAMETER_RE, text)
    row["wght"] = first_text_match(WGHT_RE, text)
    row["shel_resolution"] = first_text_match(SHEL_RE, text)
    return row


def parse_qc(qc_dir: Path) -> dict[str, Any]:
    completeness_text = read_text(qc_dir / "check_hkl_completeness.log")
    cc_text = read_text(qc_dir / "compare_cc12.log")
    rsplit_text = read_text(qc_dir / "compare_rsplit.log")
    return {
        "qc_completeness_percent": last_float_match(COMPLETENESS_RE, completeness_text),
        "qc_redundancy": last_float_match(REDUNDANCY_RE, completeness_text),
        "qc_snr": last_float_match(SNR_RE, completeness_text),
        "qc_measurements": last_int_match(MEASUREMENTS_RE, completeness_text),
        "qc_reflections": last_int_match(REFLECTIONS_RE, completeness_text),
        "qc_CC1_2": last_float_match(CC_RE, cc_text),
        "qc_Rsplit_percent": last_float_match(RSPLIT_RE, rsplit_text),
    }


def dataset_paths(root: Path) -> list[DatasetPaths]:
    paths = []
    for dataset, shelx_rel, run_rel in DATASETS:
        shelx_dir = root / shelx_rel
        run_dir = root / run_rel
        paths.append(
            DatasetPaths(
                dataset=dataset,
                shelx_dir=shelx_dir,
                run_dir=run_dir,
                lst_path=shelx_dir / "shelx.lst",
                qc_dir=run_dir / "qc_stats",
            )
        )
    return paths


def parse_all(root: Path) -> list[dict[str, Any]]:
    rows = []
    for paths in dataset_paths(root):
        row: dict[str, Any] = {
            "dataset": paths.dataset,
            "shelx_dir": str(paths.shelx_dir),
        }
        row.update(parse_shelxl_lst(paths.lst_path))
        row.update(parse_qc(paths.qc_dir))
        rows.append(row)
    return rows


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_delta_columns(rows: list[dict[str, Any]]) -> None:
    full = next((row for row in rows if row["dataset"] == "full"), None)
    if full is None:
        return
    metrics = ["R1_obs", "R1_all", "wR2", "GooF", "qc_completeness_percent", "qc_redundancy", "qc_snr", "qc_CC1_2", "qc_Rsplit_percent"]
    for row in rows:
        for metric in metrics:
            value = as_float(row.get(metric))
            baseline = as_float(full.get(metric))
            row[f"delta_vs_full_{metric}"] = None if value is None or baseline is None else value - baseline


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "R1_obs",
        "R1_all",
        "wR2",
        "GooF",
        "observed_reflections",
        "all_reflections",
        "number_of_parameters",
        "residual_peak",
        "residual_hole",
        "qc_completeness_percent",
        "qc_redundancy",
        "qc_snr",
        "qc_CC1_2",
        "qc_Rsplit_percent",
        "qc_measurements",
        "qc_reflections",
        "delta_vs_full_R1_obs",
        "delta_vs_full_R1_all",
        "delta_vs_full_wR2",
        "delta_vs_full_GooF",
        "delta_vs_full_qc_completeness_percent",
        "delta_vs_full_qc_redundancy",
        "delta_vs_full_qc_snr",
        "delta_vs_full_qc_CC1_2",
        "delta_vs_full_qc_Rsplit_percent",
        "exti",
        "wght",
        "shel_resolution",
        "lst_path",
        "shelx_dir",
        "lst_exists",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("dataset", "dataset"),
        ("R1_obs", "R1 obs"),
        ("observed_reflections", "N obs refl"),
        ("R1_all", "R1 all"),
        ("all_reflections", "N all refl"),
        ("wR2", "wR2"),
        ("GooF", "GooF"),
        ("qc_completeness_percent", "completeness %"),
        ("qc_redundancy", "redundancy"),
        ("qc_snr", "SNR"),
        ("qc_CC1_2", "CC1/2"),
        ("qc_Rsplit_percent", "Rsplit %"),
        ("delta_vs_full_R1_obs", "delta R1 obs"),
    ]
    lines = ["| " + " | ".join(label for _key, label in columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for key, _label in columns:
            digits = 6 if key in {"qc_CC1_2"} else 4
            values.append(str(row.get(key, "")) if key == "dataset" else fmt(row.get(key), digits=digits))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def interpret(rows: list[dict[str, Any]]) -> list[str]:
    full = next((row for row in rows if row["dataset"] == "full"), None)
    if full is None or as_float(full.get("R1_obs")) is None:
        return ["- Full-data baseline R1 could not be parsed, so trend interpretation is limited."]

    full_r1 = float(full["R1_obs"])
    full_wr2 = as_float(full.get("wR2"))
    improved = []
    worsened = []
    for row in rows:
        if row["dataset"] == "full":
            continue
        r1 = as_float(row.get("R1_obs"))
        if r1 is None:
            continue
        if r1 < full_r1:
            improved.append(row["dataset"])
        elif r1 > full_r1:
            worsened.append(row["dataset"])

    lines = []
    if improved:
        lines.append(f"- R1(obs) improves relative to full for: {', '.join(improved)}.")
    else:
        lines.append("- No filtered dataset improves R1(obs) relative to full.")
    if worsened:
        lines.append(f"- R1(obs) worsens relative to full for: {', '.join(worsened)}.")

    if full_wr2 is not None:
        wr2_improved = [
            row["dataset"]
            for row in rows
            if row["dataset"] != "full" and as_float(row.get("wR2")) is not None and float(row["wR2"]) < full_wr2
        ]
        if wr2_improved:
            lines.append(f"- wR2 improves relative to full for: {', '.join(wr2_improved)}.")
        else:
            lines.append("- No filtered dataset improves wR2 relative to full.")

    qc_worse_but_r1_better = []
    full_cc = as_float(full.get("qc_CC1_2"))
    full_rsplit = as_float(full.get("qc_Rsplit_percent"))
    for row in rows:
        if row["dataset"] == "full" or row["dataset"] not in improved:
            continue
        cc = as_float(row.get("qc_CC1_2"))
        rsplit = as_float(row.get("qc_Rsplit_percent"))
        cc_worse = full_cc is not None and cc is not None and cc < full_cc
        rsplit_worse = full_rsplit is not None and rsplit is not None and rsplit > full_rsplit
        if cc_worse or rsplit_worse:
            qc_worse_but_r1_better.append(row["dataset"])
    if qc_worse_but_r1_better:
        lines.append(
            "- At least one filtered dataset improves R1(obs) despite worse merge/QC indicators: "
            + ", ".join(qc_worse_but_r1_better)
            + "."
        )
    else:
        lines.append("- No filtered dataset clearly improves R1(obs) while showing worse parsed CC1/2 or Rsplit than full.")

    if improved:
        lines.append(
            "- This supports testing practical observation-level geometry-risk filtering at mild cutoffs, "
            "but the trend should be judged alongside redundancy loss and model-bias controls."
        )
    else:
        lines.append(
            "- This does not support hard observation filtering as a practical default; post-partialator "
            "downweighting is likely the safer next experiment."
        )
    return lines


def write_markdown(rows: list[dict[str, Any]], path: Path, root: Path, csv_path: Path) -> None:
    lines = [
        "# Geometry-Trust Filter Sweep Refinement Summary",
        "",
        "## Scope",
        "",
        "- This parses already completed SHELXL refinements only.",
        "- No SHELXL, partialator, merging, filtering, or correction commands were run by this parser.",
        "- This is a cSerialED MFM300-V(III) observation-level geometry-risk filtering comparison.",
        "- Filtered streams removed individual high geometry-coupling-risk observations per signed HKL, not whole frames.",
        "",
        "## Inputs",
        "",
        f"- Experiment root: `{root}`",
        "- Datasets: full, keep90, keep80, keep70, keep60, keep50",
        "",
        "## Summary Table",
        "",
        markdown_table(rows),
        "",
        "## Interpretation",
        "",
        *interpret(rows),
        "",
        "## Notes",
        "",
        "- Lower R1/wR2/GooF is generally better for model compatibility.",
        "- `N obs refl` and `N all refl` are the reflection counts used for the R1(obs) and R1(all) denominators.",
        "- Higher CC1/2, completeness, redundancy, and SNR are generally better merge/QC indicators.",
        "- Lower Rsplit is generally better.",
        "- A filtered dataset improving refinement despite worse merge/QC is the key signal for geometry-risk filtering.",
        "- If mild filtering helps but aggressive filtering worsens, this argues for a narrow high-risk-tail strategy or downweighting rather than hard removal.",
        "",
        "## Output",
        "",
        f"- CSV: `{csv_path}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(rows: list[dict[str, Any]]) -> None:
    print(markdown_table(rows))


def main() -> int:
    args = parse_args()
    root = args.experiment_root
    if not root.exists():
        raise SystemExit(f"Experiment root not found: {root}")

    rows = parse_all(root)
    add_delta_columns(rows)

    csv_path = root / args.output_csv
    md_path = root / args.output_md
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, root, csv_path)
    print_summary(rows)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

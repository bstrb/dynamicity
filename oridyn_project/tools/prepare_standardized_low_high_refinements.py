#!/usr/bin/env python3
"""Audit and prepare standardized SHELX refinements for MFM300 frame-split data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/low_vs_high"
)

DEFAULT_CUTOFFS = ("0.8-0.5", "0.8-0.4", "1.0-0.5", "1.0-0.4", "1.5-0.5", "1.5-0.4")
DATASET_ORDER = {"full": 0, "random": 1, "low_risk": 2, "high_risk": 3}
SHELX_COMMAND = "shelx -a50000 -b3000 -c624 -g0 -m0 -t18 shelx"


@dataclass(frozen=True)
class Dataset:
    label: str
    result_dir: Path
    shelx_dir: Path
    hkl: Path
    ins: Path
    res: Path
    lst: Path
    fcf: Path


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--standardized-root", type=Path, default=None)
    parser.add_argument("--cutoffs", nargs="*", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--stable-wght", nargs=2, type=float, default=(0.27, 0.70))
    parser.add_argument("--stable-exti", type=float, default=60000.0)
    parser.add_argument("--stable-ls-cycles", type=int, default=10)
    parser.add_argument("--common-model-dataset", choices=["full", "random", "low_risk", "high_risk"], default="full")
    parser.add_argument("--prepare-folders", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def dataset_label(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("mfm300"):
        return "full"
    if "random" in name:
        return "random"
    if "low_risk" in name:
        return "low_risk"
    if "high_risk" in name:
        return "high_risk"
    return path.name


def discover_datasets(root: Path) -> list[Dataset]:
    datasets: list[Dataset] = []
    for result_dir in sorted(root.glob("*_partialator_results")):
        label = dataset_label(result_dir)
        shelx_dir = result_dir / "shelx"
        datasets.append(
            Dataset(
                label=label,
                result_dir=result_dir,
                shelx_dir=shelx_dir,
                hkl=shelx_dir / "shelx.hkl",
                ins=shelx_dir / "shelx.ins",
                res=shelx_dir / "shelx.res",
                lst=shelx_dir / "shelx.lst",
                fcf=shelx_dir / "shelx.fcf",
            )
        )
    datasets.sort(key=lambda d: DATASET_ORDER.get(d.label, 99))
    return datasets


def active_command(text: str, keyword_regex: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("REM "):
            continue
        if re.match(keyword_regex, stripped, flags=re.IGNORECASE):
            return stripped
    return None


def is_ls_command(stripped: str) -> bool:
    return bool(re.match(r"^(?:L\.S\.|LS)\s+", stripped, flags=re.IGNORECASE))


def is_exti_command(stripped: str) -> bool:
    # SHELXL .res files may emit EXTI without a separating space, e.g. EXTI60133.6.
    return stripped.upper().startswith("EXTI")


def all_commands(text: str, keyword_regex: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(keyword_regex, stripped, flags=re.IGNORECASE):
            out.append(stripped)
    return out


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
        "restraints": first_int(lst_text, r"Restrained GooF\s*=\s*[0-9.]+\s+for\s+(\d+)\s+restraints"),
        "residual_peak": first_float(lst_text, r"Highest peak\s+([-0-9.]+)"),
        "residual_hole": first_float(lst_text, r"Deepest hole\s+([-0-9.]+)"),
        "lst_data_resolution": first_match(lst_text, r"Number of data for (d > [^\n]+)"),
        "outside_shel_rejected": first_int(lst_text, r"(\d+)\s+Reflections outside SHEL resolution limits rejected"),
    }


def audit_dataset(dataset: Dataset) -> dict[str, Any]:
    ins_text = read_text(dataset.ins)
    res_text = read_text(dataset.res)
    lst_text = read_text(dataset.lst)
    metadata = read_text(dataset.result_dir / "metadata_and_outputs.txt")

    active_shel = active_command(ins_text, r"^SHEL\b")
    active_wght = active_command(res_text, r"^WGHT\b") or active_command(ins_text, r"^WGHT\b")
    active_exti = active_command(res_text, r"^EXTI") or active_command(ins_text, r"^EXTI")
    active_merg = active_command(ins_text, r"^MERG\b")
    ls_command = active_command(ins_text, r"^(?:L\.S\.|LS)\s+") or active_command(ins_text, r"^CGLS\b")
    hklf = active_command(ins_text, r"^HKLF\b")

    row: dict[str, Any] = {
        "dataset": dataset.label,
        "result_folder": dataset.result_dir.name,
        "result_path": str(dataset.result_dir),
        "source_ins": str(dataset.ins),
        "source_res": str(dataset.res),
        "source_lst": str(dataset.lst),
        "source_hkl": str(dataset.hkl),
        "source_fcf": str(dataset.fcf) if dataset.fcf.exists() else "",
        "source_hkl_exists": dataset.hkl.exists(),
        "source_fcf_exists": dataset.fcf.exists(),
        "source_stream": first_match(metadata, r"^STREAM:\s*(.+)$", re.MULTILINE),
        "merge_symmetry": first_match(metadata, r"^SYM:\s*(.+)$", re.MULTILINE),
        "merge_lowres_A": first_float(metadata, r"^LOWRES:\s*([0-9.]+)", re.MULTILINE),
        "merge_highres_A": first_float(metadata, r"^HIGHRES:\s*([0-9.]+)", re.MULTILINE),
        "active_shel": active_shel,
        "active_wght": active_wght,
        "all_wght_lines_res": " | ".join(all_commands(res_text, r"^WGHT\b")),
        "active_exti": active_exti,
        "active_merg": active_merg or "no active MERG instruction",
        "rem_merg": first_match(ins_text, r"^(REM\s+MERG\s+.+)$", re.MULTILINE),
        "shelx_command_line": first_match(lst_text, r"Command line parameters:\s*(.+)$", re.MULTILINE),
        "ls_command": ls_command,
        "ls_cycles": first_int(ls_command or "", r"(\d+)") if ls_command else None,
        "hklf": hklf,
    }
    row.update(parse_refinement_stats(lst_text))
    row["comparable_now"] = "no"
    row["comparability_reason"] = comparability_reason(row)
    return row


def comparability_reason(row: dict[str, Any]) -> str:
    reasons: list[str] = []
    if row.get("active_shel") != "SHEL 1 0.4":
        reasons.append(f"active SHEL differs ({row.get('active_shel')})")
    if row.get("active_merg") == "no active MERG instruction":
        reasons.append("MERG 0 is only REM/commented; command line contains -m0")
    if row.get("ls_cycles") != 10:
        reasons.append(f"L.S. cycles differ ({row.get('ls_command')})")
    if row.get("active_wght") != "WGHT    0.242300    0.660000" and row.get("active_wght") != "WGHT 0.2423 0.66":
        reasons.append(f"WGHT differs ({row.get('active_wght')})")
    if not reasons:
        reasons.append("mostly comparable to current 1.0-0.4 stable mode, but not a strict standardized matrix")
    return "; ".join(reasons)


def cutoff_slug(cutoff: str) -> str:
    return cutoff.replace(".", "p").replace("-", "_")


def parse_cutoff(cutoff: str) -> tuple[str, str]:
    parts = cutoff.split("-")
    if len(parts) != 2:
        raise SystemExit(f"Invalid cutoff {cutoff!r}; expected e.g. 1.0-0.4")
    float(parts[0])
    float(parts[1])
    return parts[0], parts[1]


def patch_shelx_text(
    source_text: str,
    cutoff: str,
    ls_cycles: int,
    wght: tuple[float, float],
    exti: float,
    mode: str,
) -> str:
    lowres, highres = parse_cutoff(cutoff)
    lines = source_text.splitlines()
    if not any(line.strip().upper().startswith("END") for line in lines):
        raise SystemExit("Source SHELX model has no END line")

    out: list[str] = []
    saw_merg = False
    saw_shel = False
    saw_wght = False
    saw_exti = False
    saw_hklf = False

    for raw in lines:
        line = raw.rstrip("\r\n")
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("END"):
            break
        if not stripped:
            out.append(line)
            continue
        if upper.startswith("REM"):
            out.append(line)
            continue
        if re.match(r"^MERG\b", stripped, flags=re.IGNORECASE):
            if not saw_merg:
                out.append("MERG 0")
                saw_merg = True
            continue
        if is_ls_command(stripped) or re.match(r"^CGLS\b", stripped, flags=re.IGNORECASE):
            if not saw_merg:
                out.append("MERG 0")
                saw_merg = True
            out.append(f"L.S. {int(ls_cycles)}")
            continue
        if re.match(r"^SHEL\b", stripped, flags=re.IGNORECASE):
            out.append(f"SHEL {lowres} {highres}")
            saw_shel = True
            continue
        if re.match(r"^WGHT\b", stripped, flags=re.IGNORECASE):
            if mode == "stable":
                out.append(f"WGHT {wght[0]:.4g} {wght[1]:.4g}")
            else:
                out.append("WGHT 0 0")
            saw_wght = True
            continue
        if is_exti_command(stripped):
            if mode == "stable":
                out.append(f"EXTI {exti:.6g}")
            else:
                out.append(f"REM {stripped}")
            saw_exti = True
            continue
        if re.match(r"^FVAR\b", stripped, flags=re.IGNORECASE):
            if not saw_shel:
                out.append(f"SHEL {lowres} {highres}")
                saw_shel = True
            if not saw_wght:
                out.append(f"WGHT {wght[0]:.4g} {wght[1]:.4g}" if mode == "stable" else "WGHT 0 0")
                saw_wght = True
            if mode == "stable" and not saw_exti:
                out.append(f"EXTI {exti:.6g}")
                saw_exti = True
            out.append(line)
            continue
        if re.match(r"^HKLF\b", stripped, flags=re.IGNORECASE):
            if not saw_merg:
                out.append("MERG 0")
                saw_merg = True
            if not saw_shel:
                out.append(f"SHEL {lowres} {highres}")
                saw_shel = True
            if not saw_wght:
                out.append(f"WGHT {wght[0]:.4g} {wght[1]:.4g}" if mode == "stable" else "WGHT 0 0")
                saw_wght = True
            if mode == "stable" and not saw_exti:
                out.append(f"EXTI {exti:.6g}")
                saw_exti = True
            out.append("HKLF 4")
            saw_hklf = True
            continue
        out.append(line)

    if not saw_hklf:
        if not saw_merg:
            out.append("MERG 0")
        if not saw_shel:
            out.append(f"SHEL {lowres} {highres}")
        if not saw_wght:
            out.append(f"WGHT {wght[0]:.4g} {wght[1]:.4g}" if mode == "stable" else "WGHT 0 0")
        if mode == "stable" and not saw_exti:
            out.append(f"EXTI {exti:.6g}")
        out.append("HKLF 4")
    out.append("")
    out.append("END")
    out.append("")
    return "\n".join(out)


def prepare_stable_folder(
    dataset: Dataset,
    common_model_text: str,
    stable_dir: Path,
    cutoff: str,
    args: argparse.Namespace,
) -> str:
    if stable_dir.exists() and any(stable_dir.iterdir()) and not args.overwrite:
        return "exists_skipped"
    stable_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset.hkl, stable_dir / "shelx.hkl")
    shutil.copy2(dataset.hkl, stable_dir / f"source_{dataset.label}.hkl")
    ins_text = patch_shelx_text(
        common_model_text,
        cutoff=cutoff,
        ls_cycles=int(args.stable_ls_cycles),
        wght=(float(args.stable_wght[0]), float(args.stable_wght[1])),
        exti=float(args.stable_exti),
        mode="stable",
    )
    write_text(stable_dir / "shelx.ins", ins_text)
    write_text(
        stable_dir / "README_standardized_stable.txt",
        "\n".join(
            [
                "Standardized stable refinement input.",
                f"Dataset: {dataset.label}",
                f"Source HKL: {dataset.hkl}",
                f"Common starting model dataset: {args.common_model_dataset}",
                f"SHEL cutoff: {cutoff}",
                f"Active WGHT: {args.stable_wght[0]} {args.stable_wght[1]}",
                f"Active EXTI: {args.stable_exti}",
                "Active MERG: MERG 0",
                f"Run command: {SHELX_COMMAND}",
                "",
            ]
        ),
    )
    return "prepared"


def prepare_ls0_folder(stable_dir: Path, cutoff: str, args: argparse.Namespace) -> str:
    ls0_dir = stable_dir / "shelx_raw_ls0_merg0_wght00_noexti"
    stable_res = stable_dir / "shelx.res"
    stable_hkl = stable_dir / "shelx.hkl"
    if ls0_dir.exists() and any(ls0_dir.iterdir()) and not args.overwrite:
        if (ls0_dir / "shelx.ins").exists():
            return "exists_skipped"
        if not (stable_res.exists() and stable_hkl.exists()):
            return "placeholder_waiting_for_stable_res"
    ls0_dir.mkdir(parents=True, exist_ok=True)
    if stable_res.exists() and stable_hkl.exists():
        shutil.copy2(stable_hkl, ls0_dir / "shelx.hkl")
        ls0_text = patch_shelx_text(
            read_text(stable_res),
            cutoff=cutoff,
            ls_cycles=0,
            wght=(0.0, 0.0),
            exti=0.0,
            mode="ls0",
        )
        write_text(ls0_dir / "shelx.ins", ls0_text)
        status = "prepared_from_stable_res"
    else:
        status = "placeholder_waiting_for_stable_res"
    write_text(
        ls0_dir / "README_ls0_diagnostic.txt",
        "\n".join(
            [
                "LS 0 raw-comparison diagnostic folder.",
                "Purpose: generate Fcalc/Fo columns from a converged stable model without refining it.",
                "This folder should be generated from the converged stable shelx.res.",
                "Required active instructions for diagnostic mode:",
                "L.S. 0",
                "MERG 0",
                "WGHT 0 0",
                "No active EXTI (EXTI line should be REM/commented).",
                "HKLF 4",
                f"Run command after shelx.ins and shelx.hkl are present: {SHELX_COMMAND}",
                f"Status at preparation time: {status}",
                "",
            ]
        ),
    )
    return status


def planned_rows(datasets: list[Dataset], standardized_root: Path, cutoffs: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for cutoff in cutoffs:
            slug = cutoff_slug(cutoff)
            stable_dir = standardized_root / dataset.label / f"shelx_refine_{slug}_stable"
            ls0_dir = stable_dir / "shelx_raw_ls0_merg0_wght00_noexti"
            lst_text = read_text(stable_dir / "shelx.lst")
            stats = parse_refinement_stats(lst_text) if lst_text else {}
            rows.append(
                {
                    "dataset": dataset.label,
                    "cutoff_A": cutoff,
                    "stable_dir": str(stable_dir),
                    "ls0_dir": str(ls0_dir),
                    "stable_ins": str(stable_dir / "shelx.ins"),
                    "stable_hkl": str(stable_dir / "shelx.hkl"),
                    "stable_command": f"cd {stable_dir} && {SHELX_COMMAND}",
                    "ls0_command": f"cd {ls0_dir} && {SHELX_COMMAND}",
                    "stable_result_exists": (stable_dir / "shelx.lst").exists(),
                    "ls0_result_exists": (ls0_dir / "shelx.lst").exists(),
                    "prepared_wght": f"{args.stable_wght[0]} {args.stable_wght[1]}",
                    "prepared_exti": args.stable_exti,
                    "prepared_shel": f"SHEL {parse_cutoff(cutoff)[0]} {parse_cutoff(cutoff)[1]}",
                    "prepared_merg": "MERG 0",
                    "prepared_hklf": "HKLF 4",
                    "prepared_ls": f"L.S. {args.stable_ls_cycles}",
                    **stats,
                }
            )
    return rows


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


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for label, _ in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in cols) + " |")
    return "\n".join(lines)


def write_audit(audit_rows: list[dict[str, Any]], path: Path) -> None:
    cols = [
        ("dataset", "dataset"),
        ("active SHEL", "active_shel"),
        ("L.S.", "ls_command"),
        ("WGHT", "active_wght"),
        ("EXTI", "active_exti"),
        ("MERG", "active_merg"),
        ("HKLF", "hklf"),
        ("data", "data_all"),
        ("R1 >4sig", "r1_gt_4sigma"),
        ("R1 all", "r1_all"),
        ("wR2", "wr2"),
        ("GooF", "goof"),
        ("peak", "residual_peak"),
        ("hole", "residual_hole"),
        ("comparable?", "comparable_now"),
    ]
    lines = [
        "# Existing MFM300 refinement audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Existing Active Settings",
        "",
        markdown_table(audit_rows, cols),
        "",
        "## Comparability Notes",
        "",
    ]
    for row in audit_rows:
        lines.append(f"- `{row['dataset']}`: {row['comparability_reason']}")
    lines.extend(
        [
            "",
            "## Key Finding",
            "",
            "The existing folders are not a fully standardized comparison. The most important active-instruction mismatch is `random`, which uses `SHEL 99 0.4`, while full/low/high use `SHEL 1 0.4`. All folders also rely on `REM MERG 0` plus the SHELXL command-line `-m0`, rather than an explicit active `MERG 0` line in the `.ins` file. The standardized folders prepared by this helper make `MERG 0`, `SHEL`, `WGHT`, `EXTI`, `L.S.`, and `HKLF` explicit and consistent.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def write_plan(
    datasets: list[Dataset],
    audit_rows: list[dict[str, Any]],
    planned: list[dict[str, Any]],
    analysis_dir: Path,
    standardized_root: Path,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Standardized MFM300 refinement plan",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Folder Plan",
        "",
        f"Standardized root: `{standardized_root}`",
        "",
        "For each dataset and cutoff, create:",
        "",
        "```text",
        "standardized_refinements/<dataset>/shelx_refine_<cutoff>_stable/",
        "standardized_refinements/<dataset>/shelx_refine_<cutoff>_stable/shelx_raw_ls0_merg0_wght00_noexti/",
        "```",
        "",
        "## Datasets",
        "",
    ]
    for dataset in datasets:
        lines.append(f"- `{dataset.label}`: source HKL `{dataset.hkl}`")
    lines.extend(
        [
            "",
            "## Stable Refinement Preparation",
            "",
            f"- Common starting model: `{args.common_model_dataset}` stable `.res` model.",
            f"- Active `SHEL`: one of `{', '.join(args.cutoffs)}`.",
            f"- Active `L.S.`: `L.S. {args.stable_ls_cycles}`.",
            f"- Active `MERG`: `MERG 0`.",
            f"- Active `HKLF`: `HKLF 4`.",
            f"- Active `WGHT`: `WGHT {args.stable_wght[0]} {args.stable_wght[1]}`.",
            f"- Active `EXTI`: `EXTI {args.stable_exti:g}`.",
            "- Dataset-specific `shelx.hkl` is copied from each corresponding result folder.",
            "- Stable folders are prepared but SHELXL is not run by this script.",
            "",
            "## LS 0 Raw Diagnostic Preparation",
            "",
            "For each stable refinement folder, the diagnostic subfolder is reserved as `shelx_raw_ls0_merg0_wght00_noexti`. After the stable SHELXL run has produced a converged `shelx.res`, rerun the helper with `--prepare-folders`; it will create LS0 `shelx.ins` from that stable `.res` with:",
            "",
            "- `L.S. 0`",
            "- active `MERG 0`",
            "- active `WGHT 0 0`",
            "- no active `EXTI`",
            "- `HKLF 4`",
            "",
            "This LS0 mode is for SRES/Fcalc diagnostics only, not for judging refinement quality.",
            "",
            "## Manual Stable Refinement Commands",
            "",
            f"Command list file: `{analysis_dir / 'run_standardized_stable_refinements.sh'}`",
            "",
            "Run only the cutoffs you want to evaluate. Each command has the form:",
            "",
            "```bash",
            f"cd <stable_folder> && {SHELX_COMMAND}",
            "```",
            "",
            "## Manual LS0 Diagnostic Commands",
            "",
            f"After stable runs finish, rerun folder preparation with:\n\n```bash\npython {Path(__file__).resolve()} --root {args.root} --prepare-folders\n```\n",
            f"Then run commands from: `{analysis_dir / 'run_ls0_diagnostics_after_stable.sh'}`",
            "",
            "## Planned Stable Rows",
            "",
            markdown_table(
                planned,
                [
                    ("dataset", "dataset"),
                    ("cutoff", "cutoff_A"),
                    ("stable result?", "stable_result_exists"),
                    ("LS0 result?", "ls0_result_exists"),
                    ("stable folder", "stable_dir"),
                ],
            ),
            "",
        ]
    )
    write_text(analysis_dir / "standardized_refinement_plan.md", "\n".join(lines))


def write_command_files(planned: list[dict[str, Any]], analysis_dir: Path) -> None:
    stable_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Standardized stable refinement commands. Review and run manually.",
        "",
    ]
    ls0_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# LS0 diagnostic commands. Run only after LS0 folders contain shelx.ins from converged stable shelx.res.",
        "",
    ]
    for row in planned:
        stable_lines.append(f"# {row['dataset']} {row['cutoff_A']}")
        stable_lines.append(row["stable_command"])
        stable_lines.append("")
        ls0_lines.append(f"# {row['dataset']} {row['cutoff_A']}")
        ls0_lines.append(row["ls0_command"])
        ls0_lines.append("")
    stable_path = analysis_dir / "run_standardized_stable_refinements.sh"
    ls0_path = analysis_dir / "run_ls0_diagnostics_after_stable.sh"
    write_text(stable_path, "\n".join(stable_lines))
    write_text(ls0_path, "\n".join(ls0_lines))
    stable_path.chmod(0o755)
    ls0_path.chmod(0o755)


def prepare_folders(
    datasets: list[Dataset],
    standardized_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    common = {d.label: d for d in datasets}.get(args.common_model_dataset)
    if common is None:
        raise SystemExit(f"Common model dataset not found: {args.common_model_dataset}")
    common_model_text = read_text(common.res)
    if not common_model_text:
        raise SystemExit(f"Common model .res not found or empty: {common.res}")

    statuses: list[dict[str, Any]] = []
    for dataset in datasets:
        if not dataset.hkl.exists():
            raise SystemExit(f"Missing source HKL for {dataset.label}: {dataset.hkl}")
        for cutoff in args.cutoffs:
            stable_dir = standardized_root / dataset.label / f"shelx_refine_{cutoff_slug(cutoff)}_stable"
            stable_status = prepare_stable_folder(dataset, common_model_text, stable_dir, cutoff, args)
            ls0_status = prepare_ls0_folder(stable_dir, cutoff, args)
            statuses.append(
                {
                    "dataset": dataset.label,
                    "cutoff_A": cutoff,
                    "stable_dir": str(stable_dir),
                    "stable_status": stable_status,
                    "ls0_status": ls0_status,
                }
            )
    return statuses


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"--root not found: {args.root}")
    args.analysis_dir = args.analysis_dir or args.root / "standardized_analysis"
    args.standardized_root = args.standardized_root or args.root / "standardized_refinements"

    datasets = discover_datasets(args.root)
    if not datasets:
        raise SystemExit(f"No *_partialator_results folders found under {args.root}")

    audit_rows = [audit_dataset(dataset) for dataset in datasets]

    prepare_statuses: list[dict[str, Any]] = []
    if args.prepare_folders:
        prepare_statuses = prepare_folders(datasets, args.standardized_root, args)
        write_csv(prepare_statuses, args.analysis_dir / "folder_preparation_status.csv")

    planned = planned_rows(datasets, args.standardized_root, list(args.cutoffs), args)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_audit(audit_rows, args.analysis_dir / "existing_refinement_audit.md")
    write_plan(datasets, audit_rows, planned, args.analysis_dir, args.standardized_root, args)
    write_csv(planned, args.analysis_dir / "standardized_refinement_summary.csv")
    write_command_files(planned, args.analysis_dir)

    log(f"Wrote {args.analysis_dir / 'existing_refinement_audit.md'}")
    log(f"Wrote {args.analysis_dir / 'standardized_refinement_plan.md'}")
    log(f"Wrote {args.analysis_dir / 'standardized_refinement_summary.csv'}")
    if args.prepare_folders:
        log(f"Prepared folders under {args.standardized_root}")
        log(f"Wrote {args.analysis_dir / 'folder_preparation_status.csv'}")
    else:
        log("No refinement folders prepared because --prepare-folders was not passed")


if __name__ == "__main__":
    main()

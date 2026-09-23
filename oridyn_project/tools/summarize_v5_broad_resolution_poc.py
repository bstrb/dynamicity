#!/usr/bin/env python3
"""Summarize completed broad v5 Dexc actionability outputs by resolution shell."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", message="All-NaN slice encountered")

HKL_COLUMNS = ["h", "k", "l"]
SHELL_COLUMNS = ["resolution_shell_10_index", "resolution_shell_10_label", "resolution_shell_20_index", "resolution_shell_20_label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--actionability-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"--manifest not found: {args.manifest}")
    if not args.actionability_dir.is_dir():
        raise SystemExit(f"--actionability-dir not found: {args.actionability_dir}")
    for filename in ["actionability_decision_table.csv", "candidate_best_model.csv", "candidate_filter_simulation.csv", "candidate_symmetry_orbit_summary.csv"]:
        if not (args.actionability_dir / filename).is_file():
            raise SystemExit(f"Missing input: {args.actionability_dir / filename}")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def hkl_label(h: int, k: int, l: int) -> str:
    return f"({int(h)},{int(k)},{int(l)})"


def local_four_mmm_orbit_id(h: int, k: int, l: int) -> str:
    variants = {(h, k, l), (-k, h, l), (-h, -k, l), (k, -h, l), (k, h, l), (h, -k, l), (-h, k, l), (-k, -h, l)}
    representative = sorted(variants)[0]
    return f"{representative[0]}_{representative[1]}_{representative[2]}"


def read_csv_maybe_empty(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0 or not path.read_text(encoding="utf-8", errors="replace").strip():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def normalize_hkl(table: pd.DataFrame, label: str) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    require_columns(table, HKL_COLUMNS, label)
    out = table.copy()
    for column in HKL_COLUMNS:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any():
            raise SystemExit(f"{label} contains missing/noninteger {column} values")
        out[column] = values.astype(int)
    out["hkl"] = [hkl_label(row.h, row.k, row.l) for row in out.loc[:, HKL_COLUMNS].itertuples(index=False)]
    return out


def finite_series(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(np.nan, index=table.index)
    return pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_fraction(mask: pd.Series) -> float:
    clean = mask.dropna()
    return float(clean.mean()) if len(clean) else np.nan


def safe_quantile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.quantile(q)) if len(clean) else np.nan


def safe_median(values: pd.Series) -> float:
    return safe_quantile(values, 0.50)


def json_counts(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    return json.dumps(dict(Counter(clean)), sort_keys=True)


def model_family(model_name: Any) -> str:
    if pd.isna(model_name):
        return "missing"
    text = str(model_name)
    if "quintile" in text:
        return "deficit_quintile"
    if "threshold" in text:
        return "threshold"
    if "piecewise" in text:
        return "piecewise_linear"
    if "interaction" in text:
        return "interaction"
    if "raw" in text:
        return "linear_raw_deficit"
    if "v5" in text:
        return "linear_v5_score"
    if "norm" in text:
        return "linear_norm_deficit"
    return text


def signed_label(row: pd.Series) -> str:
    delta = row.get("best_median_delta_mae", np.nan)
    stability = row.get("best_slope_sign_stability", np.nan)
    if not np.isfinite(delta):
        return "insufficient_data"
    if delta > 0.0 and np.isfinite(stability) and stability >= 0.8:
        return "broad_consistent_signal"
    if delta <= 0.0:
        return "no_clear_signal"
    return "mixed_signal"


def orbit_label(row: dict[str, Any]) -> str:
    usable = int(row.get("usable_mate_count", 0))
    if usable == 0:
        return "insufficient_data"
    positive = row.get("fraction_positive_heldout_delta_mae", np.nan)
    stable = row.get("stable_sign_fraction", np.nan)
    if np.isfinite(positive) and np.isfinite(stable) and positive >= 0.75 and stable >= 0.75:
        return "broad_consistent_signal"
    if np.isfinite(positive) and positive <= 0.25:
        return "no_clear_signal"
    return "mixed_signal"


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = normalize_hkl(pd.read_csv(path, low_memory=False), str(path))
    require_columns(manifest, [*SHELL_COLUMNS, "resolution_angstrom"], str(path))
    if "four_mmm_orbit_id" not in manifest.columns:
        manifest["four_mmm_orbit_id"] = [local_four_mmm_orbit_id(row.h, row.k, row.l) for row in manifest.loc[:, HKL_COLUMNS].itertuples(index=False)]
    return manifest.drop_duplicates(HKL_COLUMNS, keep="first").reset_index(drop=True)


def summarize_filtering(filtering: pd.DataFrame) -> pd.DataFrame:
    if filtering.empty or not set(HKL_COLUMNS).issubset(filtering.columns):
        return pd.DataFrame(columns=[*HKL_COLUMNS, "best_filter_name", "best_filter_median_delta_mae_vs_random", "best_filter_fraction_beats_random"])
    filtering = normalize_hkl(filtering, "candidate_filter_simulation.csv")
    delta_column = "filter_delta_mae_vs_random_positive_improves"
    beat_column = "beats_matched_random_mae"
    if delta_column not in filtering.columns:
        return pd.DataFrame(columns=[*HKL_COLUMNS, "best_filter_name", "best_filter_median_delta_mae_vs_random", "best_filter_fraction_beats_random"])
    rows: list[dict[str, Any]] = []
    for hkl, group in filtering.groupby(HKL_COLUMNS, sort=False):
        best_rows = []
        for filter_name, filter_group in group.groupby("filter_name", sort=False):
            beats = filter_group[beat_column].astype(bool) if beat_column in filter_group.columns else pd.Series(dtype=bool)
            best_rows.append({"best_filter_name": filter_name, "best_filter_median_delta_mae_vs_random": safe_median(filter_group[delta_column]), "best_filter_fraction_beats_random": float(beats.mean()) if len(beats) else np.nan})
        best = sorted(best_rows, key=lambda item: (-np.inf if not np.isfinite(item["best_filter_median_delta_mae_vs_random"]) else item["best_filter_median_delta_mae_vs_random"]), reverse=True)[0]
        rows.append({"h": int(hkl[0]), "k": int(hkl[1]), "l": int(hkl[2]), **best})
    return pd.DataFrame.from_records(rows)


def build_signed_results(manifest: pd.DataFrame, actionability_dir: Path) -> pd.DataFrame:
    decision = normalize_hkl(read_csv_maybe_empty(actionability_dir / "actionability_decision_table.csv"), "actionability_decision_table.csv")
    best = normalize_hkl(read_csv_maybe_empty(actionability_dir / "candidate_best_model.csv"), "candidate_best_model.csv")
    filtering = summarize_filtering(read_csv_maybe_empty(actionability_dir / "candidate_filter_simulation.csv"))
    decision_keep = [column for column in [*HKL_COLUMNS, "recommended_correction_direction", "symmetry_transfer_supported", "correction_ready_flag", "filtering_ready_flag"] if column in decision.columns]
    best_keep = [column for column in [*HKL_COLUMNS, "best_model_name", "best_median_delta_mae", "best_slope_sign_stability", "best_n_ok_splits"] if column in best.columns]
    out = manifest.copy()
    if decision_keep:
        out = out.merge(decision.loc[:, decision_keep].drop_duplicates(HKL_COLUMNS, keep="first"), on=HKL_COLUMNS, how="left", validate="one_to_one")
    if best_keep:
        out = out.merge(best.loc[:, best_keep].drop_duplicates(HKL_COLUMNS, keep="first"), on=HKL_COLUMNS, how="left", validate="one_to_one")
    if not filtering.empty:
        out = out.merge(filtering, on=HKL_COLUMNS, how="left", validate="one_to_one")
    out["best_model_family"] = out.get("best_model_name", pd.Series(index=out.index, dtype=object)).map(model_family)
    if "recommended_correction_direction" not in out.columns:
        out["recommended_correction_direction"] = np.nan
    if "symmetry_transfer_supported" not in out.columns:
        out["symmetry_transfer_supported"] = False
    out["best_median_delta_mae"] = finite_series(out, "best_median_delta_mae")
    out["best_slope_sign_stability"] = finite_series(out, "best_slope_sign_stability")
    out["best_filter_median_delta_mae_vs_random"] = finite_series(out, "best_filter_median_delta_mae_vs_random")
    out["broad_signal_label"] = out.apply(signed_label, axis=1)
    return out


def summarize_shell(results: pd.DataFrame, shell_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index_column = f"resolution_shell_{shell_count}_index"
    label_column = f"resolution_shell_{shell_count}_label"
    rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    filtering_rows: list[dict[str, Any]] = []
    for (shell_index, shell_label_value), group in results.groupby([index_column, label_column], sort=True, dropna=False):
        delta = finite_series(group, "best_median_delta_mae")
        usable = delta.notna()
        stability = finite_series(group, "best_slope_sign_stability")
        filter_delta = finite_series(group, "best_filter_median_delta_mae_vs_random")
        filter_usable = filter_delta.notna()
        row = {
            "shell_index": shell_index,
            "shell_label": shell_label_value,
            "signed_hkl_count": int(len(group)),
            "orbit_count": int(group["four_mmm_orbit_id"].nunique(dropna=True)),
            "usable_model_count": int(usable.sum()),
            "fraction_positive_heldout_delta_mae": float((delta[usable] > 0.0).mean()) if int(usable.sum()) else np.nan,
            "median_best_median_delta_mae": safe_median(delta),
            "q25_best_median_delta_mae": safe_quantile(delta, 0.25),
            "q75_best_median_delta_mae": safe_quantile(delta, 0.75),
            "fraction_slope_sign_stability_ge_0p8": float((stability.dropna() >= 0.8).mean()) if len(stability.dropna()) else np.nan,
            "fraction_symmetry_transfer_supported": safe_fraction(group["symmetry_transfer_supported"].astype("boolean")) if "symmetry_transfer_supported" in group.columns else np.nan,
            "fraction_best_filter_beats_random": float((filter_delta[filter_usable] > 0.0).mean()) if int(filter_usable.sum()) else np.nan,
            "median_filtering_delta_mae_vs_random": safe_median(filter_delta),
            "selected_model_family_counts": json_counts(group["best_model_family"]),
            "correction_direction_counts": json_counts(group["recommended_correction_direction"]),
        }
        rows.append(row)
        for family, count in group.loc[usable, "best_model_family"].value_counts(dropna=False).items():
            model_rows.append({"shell_index": shell_index, "shell_label": shell_label_value, "model_family": family, "signed_hkl_count": int(count)})
        filtering_rows.append({"shell_index": shell_index, "shell_label": shell_label_value, "filtering_usable_count": int(filter_usable.sum()), "fraction_best_filter_beats_random": row["fraction_best_filter_beats_random"], "median_filtering_delta_mae_vs_random": row["median_filtering_delta_mae_vs_random"], "best_filter_name_counts": json_counts(group.get("best_filter_name", pd.Series(dtype=object)))})
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(model_rows), pd.DataFrame.from_records(filtering_rows)


def modal_assignment(group: pd.DataFrame, column: str) -> Any:
    values = group[column].dropna().astype(str)
    if values.empty:
        return np.nan
    counts = values.value_counts()
    if len(counts) == 1:
        return counts.index[0]
    return "mixed:" + ";".join(counts.index.tolist())


def build_orbit_summary(results: pd.DataFrame, actionability_dir: Path) -> pd.DataFrame:
    orbit_input = read_csv_maybe_empty(actionability_dir / "candidate_symmetry_orbit_summary.csv")
    orbit_support = {}
    if not orbit_input.empty and {"four_mmm_orbit_id", "fraction_transfer_improves_mae"}.issubset(orbit_input.columns):
        orbit_support = dict(zip(orbit_input["four_mmm_orbit_id"].astype(str), pd.to_numeric(orbit_input["fraction_transfer_improves_mae"], errors="coerce"), strict=False))
    rows: list[dict[str, Any]] = []
    for orbit_id, group in results.groupby("four_mmm_orbit_id", sort=False):
        delta = finite_series(group, "best_median_delta_mae")
        usable = delta.notna()
        stability = finite_series(group, "best_slope_sign_stability")
        directions = group.loc[group["recommended_correction_direction"].notna(), "recommended_correction_direction"].astype(str)
        top_direction_fraction = float(directions.value_counts().iloc[0] / len(directions)) if len(directions) else np.nan
        row = {
            "four_mmm_orbit_id": orbit_id,
            "signed_mate_count": int(len(group)),
            "usable_mate_count": int(usable.sum()),
            "fraction_positive_heldout_delta_mae": float((delta[usable] > 0.0).mean()) if int(usable.sum()) else np.nan,
            "median_heldout_delta_mae": safe_median(delta),
            "minimum_heldout_delta_mae": float(delta.dropna().min()) if len(delta.dropna()) else np.nan,
            "stable_sign_fraction": float((stability.dropna() >= 0.8).mean()) if len(stability.dropna()) else np.nan,
            "symmetry_transfer_supported_fraction": float(orbit_support.get(str(orbit_id))) if str(orbit_id) in orbit_support and np.isfinite(orbit_support[str(orbit_id)]) else safe_fraction(group["symmetry_transfer_supported"].astype("boolean")),
            "number_selected_model_families": int(group.loc[usable, "best_model_family"].nunique(dropna=True)),
            "correction_direction_agreement": top_direction_fraction,
            "median_resolution": safe_median(group["resolution_angstrom"]),
            "resolution_shell_10_assignment": modal_assignment(group, "resolution_shell_10_label"),
            "resolution_shell_20_assignment": modal_assignment(group, "resolution_shell_20_label"),
        }
        row["broad_orbit_signal_label"] = orbit_label(row)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def write_readme(out_dir: Path) -> None:
    text = """# V5 Broad Resolution POC Summary

This summary joins completed actionability outputs to a signed-HKL manifest by exact h,k,l. It does not use merged intensity/Fobs as a target, does not refit models, and does not use existing actionability classifications as the main result.

Resolution shells are descriptive only.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    results = build_signed_results(manifest, args.actionability_dir)
    shell10, model10, filtering10 = summarize_shell(results, 10)
    shell20, model20, filtering20 = summarize_shell(results, 20)
    orbit_summary = build_orbit_summary(results, args.actionability_dir)

    results.to_csv(args.out_dir / "broad_signed_hkl_results.csv", index=False)
    shell10.to_csv(args.out_dir / "broad_shell10_summary.csv", index=False)
    shell20.to_csv(args.out_dir / "broad_shell20_summary.csv", index=False)
    orbit_summary.to_csv(args.out_dir / "broad_orbit_summary.csv", index=False)
    model10.to_csv(args.out_dir / "broad_model_family_by_shell10.csv", index=False)
    model20.to_csv(args.out_dir / "broad_model_family_by_shell20.csv", index=False)
    filtering10.to_csv(args.out_dir / "broad_filtering_by_shell10.csv", index=False)
    filtering20.to_csv(args.out_dir / "broad_filtering_by_shell20.csv", index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "actionability_dir": str(args.actionability_dir),
        "scientific_constraints": {
            "joins_by_exact_signed_hkl": True,
            "uses_merged_intensity_or_Fobs_as_target": False,
            "uses_existing_actionability_classification_as_main_result": False,
            "refits_models": False,
            "resolution_shells_affect_results": False,
        },
        "signed_hkl_count": int(len(results)),
        "orbit_count": int(results["four_mmm_orbit_id"].nunique(dropna=True)),
    }
    (args.out_dir / "broad_summary_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    write_readme(args.out_dir)
    log(f"Summarized signed HKLs: {len(results):,}")
    log(f"Summarized 4/mmm orbits: {results['four_mmm_orbit_id'].nunique(dropna=True):,}")
    log(f"Outputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
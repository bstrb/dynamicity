#!/usr/bin/env python3
"""Prepare broad signed-HKL manifests for v5 Dexc resolution POC runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HKL_COLUMNS = ["h", "k", "l"]
REQUIRED_COLUMNS = [
    *HKL_COLUMNS,
    "accepted_count",
    "finite_response_count",
    "nonzero_coupling_count",
    "resolution_angstrom",
    "excitation_deficit_norm_spread_q90_q10",
]
MANIFEST_COLUMNS = [
    *HKL_COLUMNS,
    "hkl",
    "four_mmm_orbit_id",
    "accepted_count",
    "finite_response_count",
    "nonzero_coupling_count",
    "resolution_angstrom",
    "reciprocal_resolution",
    "excitation_deficit_norm_spread_q90_q10",
    "resolution_shell_10_index",
    "resolution_shell_10_label",
    "resolution_shell_20_index",
    "resolution_shell_20_label",
    "extreme_low_resolution_1pct",
    "extreme_high_resolution_5pct",
]
FIXED_MANIFESTS = {
    "broad_permissive_signed_hkls.csv": {"min_observations": 50, "min_finite_response": 50, "min_nonzero_coupling": 20, "min_descriptor_spread": 0.01},
    "broad_moderate_signed_hkls.csv": {"min_observations": 50, "min_finite_response": 50, "min_nonzero_coupling": 30, "min_descriptor_spread": 0.02},
    "broad_conservative_signed_hkls.csv": {"min_observations": 100, "min_finite_response": 100, "min_nonzero_coupling": 50, "min_descriptor_spread": 0.05},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=50)
    parser.add_argument("--min-nonzero-coupling", type=int, default=20)
    parser.add_argument("--min-finite-response", type=int, default=50)
    parser.add_argument("--min-descriptor-spread", type=float, default=0.01)
    args = parser.parse_args()
    screen = args.screen_dir / "all_hkl_screen.csv"
    if not screen.is_file():
        raise SystemExit(f"Missing input: {screen}")
    for name in ["min_observations", "min_nonzero_coupling", "min_finite_response"]:
        if int(getattr(args, name)) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")
    if float(args.min_descriptor_spread) < 0.0:
        raise SystemExit("--min-descriptor-spread must be >= 0")
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
    variants = {
        (h, k, l),
        (-k, h, l),
        (-h, -k, l),
        (k, -h, l),
        (k, h, l),
        (h, -k, l),
        (-h, k, l),
        (-k, -h, l),
    }
    representative = sorted(variants)[0]
    return f"{representative[0]}_{representative[1]}_{representative[2]}"


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def numeric(table: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_screen(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    require_columns(table, REQUIRED_COLUMNS, str(path))
    out = table.copy()
    for column in HKL_COLUMNS:
        values = numeric(out, column)
        if values.isna().any():
            raise SystemExit(f"Input contains missing/noninteger {column} values")
        out[column] = values.astype(int)
    for column in ["accepted_count", "finite_response_count", "nonzero_coupling_count", "resolution_angstrom", "excitation_deficit_norm_spread_q90_q10"]:
        out[column] = numeric(out, column)
    out["hkl"] = [hkl_label(row.h, row.k, row.l) for row in out.loc[:, HKL_COLUMNS].itertuples(index=False)]
    if "four_mmm_orbit_id" not in out.columns:
        out["four_mmm_orbit_id"] = [local_four_mmm_orbit_id(row.h, row.k, row.l) for row in out.loc[:, HKL_COLUMNS].itertuples(index=False)]
    else:
        computed = pd.Series([local_four_mmm_orbit_id(row.h, row.k, row.l) for row in out.loc[:, HKL_COLUMNS].itertuples(index=False)], index=out.index)
        out["four_mmm_orbit_id"] = out["four_mmm_orbit_id"].where(out["four_mmm_orbit_id"].notna(), computed)
    out = out.drop_duplicates(HKL_COLUMNS, keep="first").reset_index(drop=True)
    return out


def eligibility_conditions(table: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, pd.Series]:
    resolution = numeric(table, "resolution_angstrom")
    spread = numeric(table, "excitation_deficit_norm_spread_q90_q10")
    return {
        "accepted_count": numeric(table, "accepted_count") >= float(thresholds["min_observations"]),
        "finite_response_count": numeric(table, "finite_response_count") >= float(thresholds["min_finite_response"]),
        "nonzero_coupling_count": numeric(table, "nonzero_coupling_count") >= float(thresholds["min_nonzero_coupling"]),
        "finite_positive_resolution": np.isfinite(resolution) & (resolution > 0.0),
        "finite_descriptor_spread": np.isfinite(spread),
        "descriptor_spread_threshold": spread >= float(thresholds["min_descriptor_spread"]),
    }


def eligibility_mask(table: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    conditions = eligibility_conditions(table, thresholds)
    mask = pd.Series(True, index=table.index)
    for condition in conditions.values():
        mask &= condition
    return mask


def shell_label(prefix: str, index: int) -> str:
    return f"{prefix}_{int(index):02d}"


def assign_shells(manifest: pd.DataFrame, shell_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = manifest.copy()
    index_column = f"resolution_shell_{shell_count}_index"
    label_column = f"resolution_shell_{shell_count}_label"
    out[index_column] = pd.Series(dtype="Int64")
    out[label_column] = pd.Series(dtype=object)
    if out.empty:
        summary = pd.DataFrame(
            {
                "shell_count": shell_count,
                "shell_index": range(shell_count),
                "shell_label": [shell_label(f"shell{shell_count}", index) for index in range(shell_count)],
                "signed_hkl_count": 0,
                "reciprocal_resolution_min": np.nan,
                "reciprocal_resolution_max": np.nan,
                "resolution_angstrom_min": np.nan,
                "resolution_angstrom_max": np.nan,
            }
        )
        return out, summary
    order = out.sort_values(["reciprocal_resolution", *HKL_COLUMNS], ascending=[True, True, True, True]).index.to_numpy()
    positions = np.arange(len(order), dtype=int)
    shell_indices = np.minimum(np.floor(positions * shell_count / max(1, len(order))).astype(int), shell_count - 1)
    out.loc[order, index_column] = shell_indices
    out[index_column] = out[index_column].astype(int)
    out[label_column] = [shell_label(f"shell{shell_count}", index) for index in out[index_column].to_numpy(dtype=int)]
    rows: list[dict[str, Any]] = []
    for shell_index in range(shell_count):
        group = out.loc[out[index_column] == shell_index]
        rows.append(
            {
                "shell_count": shell_count,
                "shell_index": shell_index,
                "shell_label": shell_label(f"shell{shell_count}", shell_index),
                "signed_hkl_count": int(len(group)),
                "reciprocal_resolution_min": float(group["reciprocal_resolution"].min()) if len(group) else np.nan,
                "reciprocal_resolution_max": float(group["reciprocal_resolution"].max()) if len(group) else np.nan,
                "resolution_angstrom_min": float(group["resolution_angstrom"].min()) if len(group) else np.nan,
                "resolution_angstrom_max": float(group["resolution_angstrom"].max()) if len(group) else np.nan,
            }
        )
    return out, pd.DataFrame.from_records(rows)


def add_extreme_flags(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    out["extreme_low_resolution_1pct"] = False
    out["extreme_high_resolution_5pct"] = False
    if out.empty:
        return out
    ordered = out.sort_values(["reciprocal_resolution", *HKL_COLUMNS], ascending=[True, True, True, True]).index.to_numpy()
    low_n = max(1, int(np.ceil(0.01 * len(ordered))))
    high_n = max(1, int(np.ceil(0.05 * len(ordered))))
    out.loc[ordered[:low_n], "extreme_low_resolution_1pct"] = True
    out.loc[ordered[-high_n:], "extreme_high_resolution_5pct"] = True
    return out


def build_manifest(screen: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = screen.loc[eligibility_mask(screen, thresholds)].copy()
    manifest["reciprocal_resolution"] = 1.0 / numeric(manifest, "resolution_angstrom")
    manifest, shell10 = assign_shells(manifest, 10)
    manifest, shell20 = assign_shells(manifest, 20)
    manifest = add_extreme_flags(manifest)
    return manifest.loc[:, MANIFEST_COLUMNS].sort_values(["h", "k", "l"]).reset_index(drop=True), shell10, shell20


def safe_median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.median()) if len(clean) else np.nan


def sensitivity_row(screen: pd.DataFrame, dimension: str, value: float, thresholds: dict[str, float]) -> dict[str, Any]:
    manifest, _, _ = build_manifest(screen, thresholds)
    row: dict[str, Any] = {
        "dimension": dimension,
        "threshold_value": value,
        "min_observations_used": thresholds["min_observations"],
        "min_finite_response_used": thresholds["min_finite_response"],
        "min_nonzero_coupling_used": thresholds["min_nonzero_coupling"],
        "min_descriptor_spread_used": thresholds["min_descriptor_spread"],
        "signed_hkl_count": int(len(manifest)),
        "four_mmm_orbit_count": int(manifest["four_mmm_orbit_id"].nunique(dropna=True)) if len(manifest) else 0,
        "median_accepted_count": safe_median(manifest["accepted_count"]) if len(manifest) else np.nan,
        "median_nonzero_coupling_count": safe_median(manifest["nonzero_coupling_count"]) if len(manifest) else np.nan,
        "median_resolution": safe_median(manifest["resolution_angstrom"]) if len(manifest) else np.nan,
        "minimum_resolution": float(manifest["resolution_angstrom"].min()) if len(manifest) else np.nan,
        "maximum_resolution": float(manifest["resolution_angstrom"].max()) if len(manifest) else np.nan,
    }
    for shell_index in range(10):
        row[f"shell10_{shell_index:02d}_count"] = int((manifest["resolution_shell_10_index"] == shell_index).sum()) if len(manifest) else 0
    return row


def build_sensitivity(screen: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    base = {
        "min_observations": int(args.min_observations),
        "min_finite_response": int(args.min_finite_response),
        "min_nonzero_coupling": int(args.min_nonzero_coupling),
        "min_descriptor_spread": float(args.min_descriptor_spread),
    }
    rows: list[dict[str, Any]] = []
    for value in [30, 50, 75, 100]:
        thresholds = dict(base, min_observations=value, min_finite_response=value)
        rows.append(sensitivity_row(screen, "observation_and_finite_response", value, thresholds))
    for value in [10, 20, 30, 50, 100]:
        thresholds = dict(base, min_nonzero_coupling=value)
        rows.append(sensitivity_row(screen, "nonzero_coupling", value, thresholds))
    for value in [0.00, 0.01, 0.02, 0.05, 0.10]:
        thresholds = dict(base, min_descriptor_spread=value)
        rows.append(sensitivity_row(screen, "descriptor_spread", value, thresholds))
    return pd.DataFrame.from_records(rows)


def target_location(screen: pd.DataFrame, manifest: pd.DataFrame, hkl: tuple[int, int, int], thresholds: dict[str, float]) -> dict[str, Any]:
    mask = (screen["h"] == hkl[0]) & (screen["k"] == hkl[1]) & (screen["l"] == hkl[2])
    selected = (manifest["h"] == hkl[0]) & (manifest["k"] == hkl[1]) & (manifest["l"] == hkl[2]) if not manifest.empty else pd.Series(False)
    row: dict[str, Any] = {"hkl": hkl_label(*hkl), "present_in_screen": bool(mask.any()), "included_in_main_manifest": bool(selected.any())}
    if mask.any():
        source = screen.loc[mask].iloc[0]
        row.update({"accepted_count": source.get("accepted_count"), "finite_response_count": source.get("finite_response_count"), "nonzero_coupling_count": source.get("nonzero_coupling_count"), "resolution_angstrom": source.get("resolution_angstrom"), "descriptor_spread": source.get("excitation_deficit_norm_spread_q90_q10")})
        failed = [name for name, condition in eligibility_conditions(screen.loc[mask], thresholds).items() if not bool(condition.iloc[0])]
        row["failed_main_conditions"] = ";".join(failed)
    if selected.any():
        hit = manifest.loc[selected].iloc[0]
        row.update({"resolution_shell_10_label": hit["resolution_shell_10_label"], "resolution_shell_20_label": hit["resolution_shell_20_label"], "extreme_low_resolution_1pct": bool(hit["extreme_low_resolution_1pct"]), "extreme_high_resolution_5pct": bool(hit["extreme_high_resolution_5pct"])})
    return row


def write_readme(out_dir: Path) -> None:
    text = """# V5 Broad Resolution POC Manifests

These signed-HKL manifests are selected only by observation availability, finite response availability, nonzero v5 coupling availability, finite positive resolution, and v5 excitation-deficit descriptor spread.

The selection does not use merged intensity/Fobs, intensity slopes, correlations, candidate scores, quintile shapes, symmetry agreement, known-family labels, or actionability classifications. Resolution shells and extreme-resolution flags are descriptive only and do not affect inclusion.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    screen_path = args.screen_dir / "all_hkl_screen.csv"
    screen = load_screen(screen_path)
    thresholds = {"min_observations": int(args.min_observations), "min_finite_response": int(args.min_finite_response), "min_nonzero_coupling": int(args.min_nonzero_coupling), "min_descriptor_spread": float(args.min_descriptor_spread)}

    log(f"Loaded total signed HKLs in screen: {len(screen):,}")
    conditions = eligibility_conditions(screen, thresholds)
    for name, condition in conditions.items():
        log(f"Individual condition pass count - {name}: {int(condition.sum()):,}")

    main_manifest, shell10, shell20 = build_manifest(screen, thresholds)
    main_manifest.to_csv(args.out_dir / "broad_all_signed_hkls.csv", index=False)
    shell10.to_csv(args.out_dir / "broad_resolution_shell_10_summary.csv", index=False)
    shell20.to_csv(args.out_dir / "broad_resolution_shell_20_summary.csv", index=False)

    for filename, fixed_thresholds in FIXED_MANIFESTS.items():
        fixed_manifest, _, _ = build_manifest(screen, fixed_thresholds)
        fixed_manifest.to_csv(args.out_dir / filename, index=False)

    sensitivity = build_sensitivity(screen, args)
    sensitivity.to_csv(args.out_dir / "broad_threshold_sensitivity.csv", index=False)

    locations = [target_location(screen, main_manifest, hkl, thresholds) for hkl in [(0, 0, 4), (0, 0, -4)]]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "screen_dir": str(args.screen_dir),
        "screen_path": str(screen_path),
        "main_thresholds": thresholds,
        "scientific_constraints": {
            "uses_merged_intensity_or_Fobs": False,
            "uses_slopes_or_correlations_or_candidate_scores": False,
            "uses_known_family_or_symmetry_or_actionability": False,
            "signed_hkl_preserved": True,
            "resolution_shells_affect_inclusion": False,
            "extreme_resolution_flags_affect_inclusion": False,
        },
        "named_manifest_thresholds": FIXED_MANIFESTS,
        "target_hkl_locations": locations,
    }
    (args.out_dir / "broad_selection_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    write_readme(args.out_dir)

    log(f"Final main-manifest signed-HKL count: {len(main_manifest):,}")
    log(f"Final main-manifest 4/mmm orbit count: {main_manifest['four_mmm_orbit_id'].nunique(dropna=True):,}")
    if len(main_manifest):
        log(f"Resolution range retained without exclusion: d={main_manifest['resolution_angstrom'].min():.6g}..{main_manifest['resolution_angstrom'].max():.6g} A; 1/d={main_manifest['reciprocal_resolution'].min():.6g}..{main_manifest['reciprocal_resolution'].max():.6g}")
    log("10-shell counts: " + ", ".join(f"{int(row.shell_index)}={int(row.signed_hkl_count)}" for row in shell10.itertuples(index=False)))
    log("20-shell counts: " + ", ".join(f"{int(row.shell_index)}={int(row.signed_hkl_count)}" for row in shell20.itertuples(index=False)))
    log(f"Extreme-low-resolution 1% descriptive flag count: {int(main_manifest['extreme_low_resolution_1pct'].sum()) if len(main_manifest) else 0:,}")
    log(f"Extreme-high-resolution 5% descriptive flag count: {int(main_manifest['extreme_high_resolution_5pct'].sum()) if len(main_manifest) else 0:,}")
    for location in locations:
        log(f"Location of {location['hkl']}: {json.dumps(location, sort_keys=True, default=json_default)}")
    log(f"Outputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
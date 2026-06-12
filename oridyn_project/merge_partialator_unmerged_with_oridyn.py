#!/usr/bin/env python3
"""Manual post-partialator merges from unmerged observations plus OriDyn scores."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)")
UNMERGED_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+(\S+))?\s*$")
UNMERGED_FLAGGED_RE = re.compile(r"^\s*Flagged:\s*(\S+)\s*$", re.IGNORECASE)
SOURCE_COLUMNS = (
    "source_filename",
    "target_source",
    "source",
    "filename",
    "image_filename",
    "image",
    "file",
    "Image filename",
)
SCORE_OPTIONAL_COLUMNS = ("S_dyn_geom", "sigma_dyn_rel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join partialator --unmerged-output rows with OriDyn scores, then write "
            "base and OriDyn-weighted manual merges."
        )
    )
    parser.add_argument("--stream", type=Path, help="Optional original CrystFEL stream for diagnostics only.")
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator --unmerged-output file.")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv.")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--symmetry", choices=["mmm", "1"], default="mmm")
    parser.add_argument(
        "--keep-flagged",
        action="store_true",
        help="Keep rows from flagged crystals and reflections with skip flags.",
    )
    parser.add_argument(
        "--duplicate-unmerged-policy",
        choices=["stop", "mean"],
        default="stop",
        help="How to handle duplicate source+event+h+k+l keys in unmerged observations.",
    )
    parser.add_argument(
        "--missing-score-policy",
        choices=["fill1", "stop"],
        default="fill1",
        help=(
            "How to handle unmerged rows missing score matches or invalid sigma_dyn_rel. "
            "fill1: replace with sigma_dyn_rel=1. stop: fail with examples."
        ),
    )
    parser.add_argument("--min-partiality", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_event(value: object) -> str:
    return str(value).strip()


def normalize_source(value: object) -> str:
    return str(value).strip()


def source_basename(value: object) -> str:
    return Path(str(value).strip()).name


def looks_like_source_filename(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(1000)
    if values.empty:
        return False
    return bool(values.str.contains(r"\.h5\b|/|\\", regex=True).any())


def parse_reflection_row(line: str) -> tuple[int, int, int, float, float] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        intensity = float(parts[3])
        sigma = float(parts[4])
    except ValueError:
        return None
    return h, k, l, intensity, sigma


def parse_stream_observations(stream_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    in_chunk = False
    in_crystal = False
    in_reflections = False
    current_source = ""
    current_event = ""

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("----- Begin chunk -----"):
                in_chunk = True
                in_crystal = False
                in_reflections = False
                current_source = ""
                current_event = ""
                continue
            if line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                continue
            if in_chunk and (match := STREAM_IMAGE_RE.match(line)):
                current_source = normalize_source(match.group(1))
                continue
            if in_chunk and (match := STREAM_EVENT_RE.match(line)):
                current_event = normalize_event(match.group(1))
                continue
            if line.startswith("--- Begin crystal"):
                in_crystal = True
                in_reflections = False
                continue
            if line.startswith("--- End crystal"):
                in_crystal = False
                in_reflections = False
                continue
            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                continue
            if in_reflections and "End of reflections" in line:
                in_reflections = False
                continue
            if in_chunk and in_crystal and in_reflections:
                parsed = parse_reflection_row(line)
                if parsed is None:
                    continue
                h, k, l, intensity, sigma = parsed
                rows.append(
                    {
                        "source": current_source,
                        "event": current_event,
                        "h": h,
                        "k": k,
                        "l": l,
                        "I_original": intensity,
                        "sigma_original": sigma,
                    }
                )

    table = pd.DataFrame.from_records(
        rows,
        columns=["source", "event", "h", "k", "l", "I_original", "sigma_original"],
    )
    return add_match_columns(table)


def parse_unmerged_observations(unmerged_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_source = ""
    current_event = ""
    current_crystal_flagged = False

    with unmerged_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Crystal "):
                current_source = ""
                current_event = ""
                current_crystal_flagged = False
                continue
            if match := UNMERGED_FILENAME_RE.match(line):
                current_source = normalize_source(match.group(1))
                current_event = normalize_event(match.group(2) or "")
                continue
            if match := UNMERGED_FLAGGED_RE.match(line):
                current_crystal_flagged = match.group(1).strip().lower() in {"yes", "y", "true", "1"}
                continue

            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                intensity = float(parts[3])
                w_crystfel = float(parts[4])
            except ValueError:
                continue
            reflection_flag = " ".join(parts[5:]).strip() if len(parts) > 5 else ""
            rows.append(
                {
                    "source": current_source,
                    "event": current_event,
                    "h": h,
                    "k": k,
                    "l": l,
                    "I_scaled": intensity,
                    "w_crystfel": w_crystfel,
                    "partiality": w_crystfel,
                    "reflection_flag": reflection_flag,
                    "flag": reflection_flag,
                    "crystal_flagged": current_crystal_flagged,
                }
            )

    table = pd.DataFrame.from_records(
        rows,
        columns=[
            "source",
            "event",
            "h",
            "k",
            "l",
            "I_scaled",
            "w_crystfel",
            "partiality",
            "reflection_flag",
            "flag",
            "crystal_flagged",
        ],
    )
    return add_match_columns(table)


def load_score_observations(scores_path: Path) -> pd.DataFrame:
    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)
    required = [source_column, "event", "h", "k", "l", "sigma_dyn_rel"]
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"OriDyn scores are missing required column(s): {missing}. Available: {header}")
    usecols = list(dict.fromkeys([*required, "S_dyn_geom", *score_component_columns(header)]))
    table = pd.read_csv(scores_path, usecols=usecols)
    table = table.rename(columns={source_column: "source"})
    table["source"] = table["source"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    table["h"] = pd.to_numeric(table["h"], errors="coerce")
    table["k"] = pd.to_numeric(table["k"], errors="coerce")
    table["l"] = pd.to_numeric(table["l"], errors="coerce")
    table["sigma_dyn_rel"] = pd.to_numeric(table["sigma_dyn_rel"], errors="coerce")
    if "S_dyn_geom" in table:
        table["S_dyn_geom"] = pd.to_numeric(table["S_dyn_geom"], errors="coerce")
    table = table.dropna(subset=["source", "event", "h", "k", "l"])
    table = table.astype({"h": "int64", "k": "int64", "l": "int64"})
    return add_match_columns(table)


def choose_score_source_column(scores_path: Path, header: list[str]) -> str:
    candidates = [column for column in SOURCE_COLUMNS if column in header]
    if not candidates:
        raise SystemExit(
            "OriDyn scores do not contain a source filename column. "
            f"Tried {list(SOURCE_COLUMNS)}. Available columns: {header}"
        )
    sample = pd.read_csv(scores_path, usecols=candidates, nrows=1000)
    for column in candidates:
        if looks_like_source_filename(sample[column]):
            return column
    raise SystemExit(
        "OriDyn scores contain source-like column(s), but none look like HDF5 filenames. "
        f"Candidates: {candidates}. Example values: {sample.head(5).to_dict(orient='records')}"
    )


def score_component_columns(header: list[str]) -> list[str]:
    return [column for column in header if column.startswith("S_") and column not in {"S_dyn_geom"}]


def add_match_columns(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        out = table.copy()
        out["source_norm"] = pd.Series(dtype=str)
        out["source_basename"] = pd.Series(dtype=str)
        out["event_norm"] = pd.Series(dtype=str)
        return out
    out = table.copy()
    out["source_norm"] = out["source"].map(normalize_source)
    out["source_basename"] = out["source"].map(source_basename)
    out["event_norm"] = out["event"].map(normalize_event)
    return out


def key_columns(source_column: str = "source_norm") -> list[str]:
    return [source_column, "event_norm", "h", "k", "l"]


def check_duplicate_keys(table: pd.DataFrame, name: str, keys: list[str] | None = None) -> None:
    if table.empty:
        return
    use_keys = keys or key_columns()
    dup_mask = table.duplicated(use_keys, keep=False)
    if not dup_mask.any():
        return
    examples = table.loc[dup_mask, ["source", "event", "h", "k", "l"]].head(20)
    print(f"ERROR: duplicate source+event+h+k+l keys in {name}: {int(dup_mask.sum()):,} duplicate rows")
    print(examples.to_string(index=False))
    raise SystemExit(f"Duplicate keys found in {name}; refusing to continue.")


def check_output_paths(prefix: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "joined": Path(f"{prefix}_joined_observations.csv"),
        "base_csv": Path(f"{prefix}_manual_base_merge.csv"),
        "oridyn_csv": Path(f"{prefix}_oridyn_sigma_inflated_merge.csv"),
        "base_hkl": Path(f"{prefix}_manual_base_merge.hkl"),
        "oridyn_hkl": Path(f"{prefix}_oridyn_sigma_inflated_merge.hkl"),
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        lines = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Output file(s) already exist; pass --overwrite to replace them:\n{lines}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def _flag_token_mask(series: pd.Series, token: str) -> pd.Series:
    return series.fillna("").astype(str).str.contains(rf"\b{re.escape(token)}\b", case=False, regex=True)


def _join_unique_nonempty_flags(values: pd.Series) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        for token in text.split():
            clean = token.strip()
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)
    return ";".join(ordered)


def filter_unmerged_observations(
    unmerged: pd.DataFrame,
    keep_flagged: bool,
    min_partiality: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {
        "unmerged_rows_parsed": int(len(unmerged)),
        "excluded_flagged_crystals": 0,
        "excluded_partiality_too_small": 0,
        "excluded_nan_esd": 0,
        "excluded_invalid_weight_or_intensity": 0,
        "excluded_min_crystfel_weight": 0,
        "eligible_unmerged_rows": 0,
    }
    if unmerged.empty:
        return add_match_columns(unmerged.copy()), stats

    out = unmerged.copy()
    out["I_scaled"] = pd.to_numeric(out["I_scaled"], errors="coerce")
    out["w_crystfel"] = pd.to_numeric(out["w_crystfel"], errors="coerce")
    out["partiality"] = out["w_crystfel"]
    out["reflection_flag"] = out["reflection_flag"].fillna("").astype(str).str.strip()
    out["flag"] = out["reflection_flag"]
    out["crystal_flagged"] = out["crystal_flagged"].fillna(False).astype(bool)

    flagged_crystal_mask = out["crystal_flagged"]
    partiality_small_mask = _flag_token_mask(out["reflection_flag"], "partiality_too_small")
    nan_esd_mask = _flag_token_mask(out["reflection_flag"], "nan_esd")

    if not keep_flagged:
        stats["excluded_flagged_crystals"] = int(flagged_crystal_mask.sum())
        stats["excluded_partiality_too_small"] = int(partiality_small_mask.sum())
        stats["excluded_nan_esd"] = int(nan_esd_mask.sum())
        keep_mask = ~(flagged_crystal_mask | partiality_small_mask | nan_esd_mask)
        out = out.loc[keep_mask].copy()

    if float(min_partiality) != 0.0:
        print(
            "WARNING: --min-partiality is being applied as min_crystfel_weight on "
            "column-5 weights, not true partiality (not partialator-exact)."
        )
        before = len(out)
        out = out.loc[out["w_crystfel"] >= float(min_partiality)].copy()
        stats["excluded_min_crystfel_weight"] = int(before - len(out))

    valid_mask = np.isfinite(out["I_scaled"]) & np.isfinite(out["w_crystfel"]) & (out["w_crystfel"] > 0.0)
    stats["excluded_invalid_weight_or_intensity"] = int((~valid_mask).sum())
    out = out.loc[valid_mask].copy()

    out["n_unmerged_duplicates"] = 1
    stats["eligible_unmerged_rows"] = int(len(out))
    return add_match_columns(out), stats


def apply_unmerged_duplicate_policy(unmerged: pd.DataFrame, policy: str) -> pd.DataFrame:
    if unmerged.empty:
        out = unmerged.copy()
        out["n_unmerged_duplicates"] = []
        return add_match_columns(out)

    key_cols = ["source", "event", "h", "k", "l"]
    duplicate_counts = (
        unmerged.groupby(key_cols, sort=False, as_index=False)
        .size()
        .rename(columns={"size": "n_unmerged_duplicates"})
        .sort_values("n_unmerged_duplicates", ascending=False)
    )
    duplicate_keys = duplicate_counts.loc[duplicate_counts["n_unmerged_duplicates"] > 1].copy()

    print(f"duplicate unmerged keys before combining: {len(duplicate_keys):,}")
    print(f"number of unmerged rows before combining: {len(unmerged):,}")
    max_duplicate_count = int(duplicate_keys["n_unmerged_duplicates"].max()) if not duplicate_keys.empty else 1
    print(f"max duplicate count: {max_duplicate_count:,}")
    if duplicate_keys.empty:
        print("Top duplicate examples: none")
    else:
        print("Top duplicate examples:")
        print(duplicate_keys.head(20).to_string(index=False))

    if policy == "stop":
        if not duplicate_keys.empty:
            print("number of unmerged rows after combining: not run (policy=stop)")
            raise SystemExit("Duplicate keys found in partialator unmerged; use --duplicate-unmerged-policy mean.")
        out = unmerged.copy()
        out["n_unmerged_duplicates"] = 1
        print(f"number of unmerged rows after combining: {len(out):,}")
        return add_match_columns(out)

    combined = (
        unmerged.groupby(key_cols, sort=False, as_index=False)
        .agg(
            I_scaled=("I_scaled", "mean"),
            w_crystfel=("w_crystfel", "mean"),
            partiality=("partiality", "mean"),
            reflection_flag=("reflection_flag", _join_unique_nonempty_flags),
            crystal_flagged=("crystal_flagged", "max"),
            n_unmerged_duplicates=("h", "size"),
        )
        .copy()
    )
    combined["flag"] = combined["reflection_flag"]
    print(f"number of unmerged rows after combining: {len(combined):,}")
    return add_match_columns(combined)


def join_scores(
    unmerged: pd.DataFrame,
    scores: pd.DataFrame,
    missing_score_policy: str,
) -> tuple[pd.DataFrame, int, int, int]:
    left = unmerged.copy()
    right = scores.copy()
    if "S_dyn_geom" not in right.columns:
        right["S_dyn_geom"] = np.nan

    right_payload = [
        column
        for column in right.columns
        if column
        not in {
            "source",
            "event",
            "h",
            "k",
            "l",
            "source_norm",
            "source_basename",
            "event_norm",
        }
    ]
    merged = left.merge(right[key_columns() + right_payload], on=key_columns(), how="left", indicator=True)
    merged["score_matched"] = merged["_merge"] == "both"
    score_matched_count = int(merged["score_matched"].sum())
    score_missing_count = int((~merged["score_matched"]).sum())
    merged = merged.drop(columns=["_merge"])

    sigma_dyn_rel = pd.to_numeric(merged["sigma_dyn_rel"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sigma_invalid_matched = merged["score_matched"] & sigma_dyn_rel.isna()
    sigma_invalid_matched_count = int(sigma_invalid_matched.sum())

    if missing_score_policy == "stop":
        if score_missing_count > 0:
            print("ERROR: missing score matches for source+event+h+k+l keys (showing up to 20):")
            examples = merged.loc[~merged["score_matched"], ["source", "event", "h", "k", "l"]].head(20)
            print(examples.to_string(index=False))
            raise SystemExit("Missing score matches found; refusing to continue with --missing-score-policy stop.")
        if sigma_invalid_matched_count > 0:
            print("ERROR: matched score rows have invalid sigma_dyn_rel values (showing up to 20):")
            examples = merged.loc[sigma_invalid_matched, ["source", "event", "h", "k", "l", "sigma_dyn_rel"]].head(20)
            print(examples.to_string(index=False))
            raise SystemExit("Invalid sigma_dyn_rel values found; refusing to continue with --missing-score-policy stop.")
        merged["sigma_dyn_rel"] = sigma_dyn_rel.clip(lower=1.0)
    else:
        merged["sigma_dyn_rel"] = sigma_dyn_rel.fillna(1.0).clip(lower=1.0)

    merged["S_dyn_geom"] = pd.to_numeric(merged["S_dyn_geom"], errors="coerce")
    return merged, score_matched_count, score_missing_count, sigma_invalid_matched_count


def add_stream_diagnostics(joined: pd.DataFrame, stream: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    merged = joined.merge(
        stream[key_columns() + ["I_original", "sigma_original"]],
        on=key_columns(),
        how="left",
        indicator=True,
    )
    stream_matched_count = int((merged["_merge"] == "both").sum())
    merged = merged.drop(columns=["_merge"])
    merged["I_original"] = pd.to_numeric(merged["I_original"], errors="coerce")
    merged["sigma_original"] = pd.to_numeric(merged["sigma_original"], errors="coerce")
    denom = merged["I_original"].replace(0.0, np.nan)
    merged["scale_factor"] = merged["I_scaled"] / denom
    merged["sigma_scaled_diag"] = np.abs(merged["scale_factor"]) * merged["sigma_original"]
    return merged, stream_matched_count


def set_merge_hkl(table: pd.DataFrame, symmetry: str) -> pd.DataFrame:
    out = table.copy()
    if symmetry == "mmm":
        out["H"] = out["h"].abs().astype(int)
        out["K"] = out["k"].abs().astype(int)
        out["L"] = out["l"].abs().astype(int)
    else:
        out["H"] = out["h"].astype(int)
        out["K"] = out["k"].astype(int)
        out["L"] = out["l"].astype(int)
    return out


def add_merge_weights(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["w_base"] = pd.to_numeric(out["w_crystfel"], errors="coerce")
    out["sigma_dyn_rel"] = pd.to_numeric(out["sigma_dyn_rel"], errors="coerce")
    out["sigma_dyn_rel"] = out["sigma_dyn_rel"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1.0)
    out["w_oridyn"] = out["w_base"] / np.square(out["sigma_dyn_rel"])
    return out


def weighted_merge(joined: pd.DataFrame, weight_column: str) -> pd.DataFrame:
    table = joined.copy()
    table["_weight"] = pd.to_numeric(table[weight_column], errors="coerce")
    table["I_scaled"] = pd.to_numeric(table["I_scaled"], errors="coerce")
    valid = np.isfinite(table["_weight"]) & (table["_weight"] > 0.0) & np.isfinite(table["I_scaled"])
    table = table.loc[valid].copy()

    output_columns = [
        "H",
        "K",
        "L",
        "I",
        "sigma",
        "n_obs",
        "weight_sum",
        "M2",
        "var",
        "mean_S_dyn_geom",
        "median_S_dyn_geom",
        "mean_sigma_dyn_rel",
        "mean_w_crystfel",
        "mean_weight_used",
    ]
    if table.empty:
        return pd.DataFrame(columns=output_columns)

    table["_weighted_I"] = table["_weight"] * table["I_scaled"]
    grouped = table.groupby(["H", "K", "L"], sort=True)
    merged = grouped.agg(
        weight_sum=("_weight", "sum"),
        weighted_I_sum=("_weighted_I", "sum"),
        n_obs=("I_scaled", "size"),
        mean_S_dyn_geom=("S_dyn_geom", "mean"),
        median_S_dyn_geom=("S_dyn_geom", "median"),
        mean_sigma_dyn_rel=("sigma_dyn_rel", "mean"),
        mean_w_crystfel=("w_crystfel", "mean"),
        mean_weight_used=("_weight", "mean"),
    ).reset_index()
    merged = merged.loc[merged["weight_sum"] > 0.0].copy()
    merged["I"] = merged["weighted_I_sum"] / merged["weight_sum"]

    with_means = table.merge(merged[["H", "K", "L", "I"]], on=["H", "K", "L"], how="inner")
    with_means["_weighted_sq_res"] = with_means["_weight"] * np.square(with_means["I_scaled"] - with_means["I"])
    m2 = with_means.groupby(["H", "K", "L"], sort=True)["_weighted_sq_res"].sum().rename("M2").reset_index()
    merged = merged.merge(m2, on=["H", "K", "L"], how="left")
    merged["M2"] = merged["M2"].fillna(0.0)
    merged["var"] = merged["M2"] / merged["weight_sum"]
    merged["sigma"] = np.sqrt(np.clip(merged["var"], a_min=0.0, a_max=None)) / np.sqrt(merged["n_obs"].astype(float))

    return merged[output_columns]


def write_hkl(table: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in table.itertuples(index=False):
            handle.write(f"{int(row.H):4d}{int(row.K):4d}{int(row.L):4d}{float(row.I):12.2f}{float(row.sigma):12.2f}\n")
        handle.write(f"{0:4d}{0:4d}{0:4d}{0:12.2f}{0:12.2f}\n")


def print_distribution(name: str, values: pd.Series) -> None:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        print(f"{name}: no finite values")
        return
    q = numeric.quantile([0.0, 0.05, 0.5, 0.95, 1.0])
    print(
        f"{name}: n={len(numeric):,} min={q.loc[0.0]:.6g} p05={q.loc[0.05]:.6g} "
        f"median={q.loc[0.5]:.6g} p95={q.loc[0.95]:.6g} max={q.loc[1.0]:.6g}"
    )


def print_merge_diagnostics(joined: pd.DataFrame, base: pd.DataFrame, oridyn: pd.DataFrame) -> None:
    print("Global joined-observation diagnostics:")
    print(f"  observation count: {len(joined):,}")
    print_distribution("I_scaled", joined["I_scaled"])
    print_distribution("w_crystfel", joined["w_crystfel"])
    print_distribution("sigma_dyn_rel", joined["sigma_dyn_rel"])
    print_distribution("w_base", joined["w_base"])
    print_distribution("w_oridyn", joined["w_oridyn"])
    print()

    comparison = base.merge(oridyn, on=["H", "K", "L"], suffixes=("_base", "_oridyn"))
    print("Merged-HKL diagnostics:")
    print(f"  merged HKLs in base: {len(base):,}")
    print(f"  merged HKLs in OriDyn: {len(oridyn):,}")
    print(f"  common HKLs: {len(comparison):,}")
    if comparison.empty:
        print("  No common HKLs between base and OriDyn merges.")
        return

    comparison["delta"] = comparison["I_oridyn"] - comparison["I_base"]
    print(f"  mean delta I_oridyn-I_base: {comparison['delta'].mean():.6g}")
    print(f"  median delta I_oridyn-I_base: {comparison['delta'].median():.6g}")
    print()
    top = comparison.reindex(comparison["delta"].abs().sort_values(ascending=False).index).head(20)
    columns = [
        "H",
        "K",
        "L",
        "I_base",
        "I_oridyn",
        "delta",
        "n_obs_base",
        "n_obs_oridyn",
        "mean_sigma_dyn_rel_base",
        "mean_sigma_dyn_rel_oridyn",
    ]
    print("Top 20 HKLs where OriDyn merge changes I most relative to base:")
    print(top[columns].to_string(index=False))


def main() -> None:
    args = parse_args()
    output_paths = check_output_paths(args.output_prefix, args.overwrite)

    unmerged = parse_unmerged_observations(args.unmerged)
    eligible_unmerged, unmerged_stats = filter_unmerged_observations(unmerged, args.keep_flagged, args.min_partiality)
    eligible_unmerged = apply_unmerged_duplicate_policy(eligible_unmerged, args.duplicate_unmerged_policy)
    scores = load_score_observations(args.scores)

    print("Unmerged filtering diagnostics:")
    print(f"  unmerged rows parsed: {unmerged_stats['unmerged_rows_parsed']:,}")
    print(f"  rows excluded due to flagged crystals: {unmerged_stats['excluded_flagged_crystals']:,}")
    print(f"  rows excluded due to partiality_too_small: {unmerged_stats['excluded_partiality_too_small']:,}")
    print(f"  rows excluded due to nan_esd: {unmerged_stats['excluded_nan_esd']:,}")
    if unmerged_stats["excluded_min_crystfel_weight"] > 0:
        print(
            "  rows excluded by min_crystfel_weight "
            f"(--min-partiality compatibility mode): {unmerged_stats['excluded_min_crystfel_weight']:,}"
        )
    if unmerged_stats["excluded_invalid_weight_or_intensity"] > 0:
        print(
            "  rows excluded due to invalid I_scaled/w_crystfel: "
            f"{unmerged_stats['excluded_invalid_weight_or_intensity']:,}"
        )
    print(f"  eligible unmerged rows: {len(eligible_unmerged):,}")
    print()
    print("Input row counts:")
    print(f"  unmerged rows after filtering/policy: {len(eligible_unmerged):,}")
    print(f"  score rows: {len(scores):,}")

    check_duplicate_keys(scores, "OriDyn scores")

    joined, score_matched_count, score_missing_count, sigma_invalid_matched_count = join_scores(
        eligible_unmerged,
        scores,
        args.missing_score_policy,
    )
    print(f"score matched count: {score_matched_count:,}")
    if args.missing_score_policy == "fill1":
        print(f"score missing count filled with sigma_dyn_rel = 1: {score_missing_count:,}")
        if sigma_invalid_matched_count > 0:
            print(f"matched rows with invalid sigma_dyn_rel filled with 1: {sigma_invalid_matched_count:,}")
    else:
        print("score missing policy: stop")
        print("score missing count: 0")
        print("matched rows with invalid sigma_dyn_rel: 0")

    if args.stream is not None:
        stream = parse_stream_observations(args.stream)
        print(f"optional stream rows parsed: {len(stream):,}")
        check_duplicate_keys(stream, "original stream")
        joined, stream_matched_count = add_stream_diagnostics(joined, stream)
        print(f"stream diagnostic matched count: {stream_matched_count:,}")

    joined = set_merge_hkl(joined, args.symmetry)
    joined = add_merge_weights(joined)
    if joined.empty:
        raise SystemExit("No observations survived filtering/joining; no merge files written.")

    joined_columns = [
        "source",
        "event",
        "h",
        "k",
        "l",
        "H",
        "K",
        "L",
        "crystal_flagged",
        "I_scaled",
        "w_crystfel",
        "partiality",
        "reflection_flag",
        "flag",
        "n_unmerged_duplicates",
        "score_matched",
        "S_dyn_geom",
        "sigma_dyn_rel",
        "w_base",
        "w_oridyn",
    ]
    if args.stream is not None:
        joined_columns += ["I_original", "sigma_original", "scale_factor", "sigma_scaled_diag"]
    extra_score_columns = [column for column in joined.columns if column.startswith("S_") and column not in joined_columns]
    joined[joined_columns + extra_score_columns].to_csv(output_paths["joined"], index=False)

    base = weighted_merge(joined, "w_base")
    oridyn = weighted_merge(joined, "w_oridyn")
    base.to_csv(output_paths["base_csv"], index=False)
    oridyn.to_csv(output_paths["oridyn_csv"], index=False)
    write_hkl(base, output_paths["base_hkl"])
    write_hkl(oridyn, output_paths["oridyn_hkl"])

    print("Outputs:")
    for path in output_paths.values():
        print(f"  {path}")
    print()
    print_merge_diagnostics(joined, base, oridyn)


if __name__ == "__main__":
    main()

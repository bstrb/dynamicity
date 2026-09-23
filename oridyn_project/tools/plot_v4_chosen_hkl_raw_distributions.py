#!/usr/bin/env python3
"""Plot raw v4 local-crowding distributions for selected signed HKLs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HKL_COLUMNS = ["h", "k", "l"]
KEY_COLUMNS = ["source_filename", "event", *HKL_COLUMNS]
DEFAULT_SCORE_COLUMN = "local_crowding_target_gated_raw"
DEFAULT_CHUNKSIZE = 500_000
DEFAULT_N_HKLS = 24
DEFAULT_MIN_OBS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="V4 local-crowding raw score CSV")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory for the selected-HKL plots")
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--n-hkls", type=int, default=DEFAULT_N_HKLS)
    parser.add_argument("--min-obs", type=int, default=DEFAULT_MIN_OBS)
    parser.add_argument("--select", choices=["top-spread", "top-obs", "random"], default="top-spread")
    parser.add_argument(
        "--hkls",
        default=None,
        help='Optional explicit signed HKLs, e.g. "0,4,0;0,27,5;6,6,2". Overrides --select.',
    )
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--survivor-mask", type=Path, default=None, help="Exact-key partialator survivor mask CSV")
    parser.add_argument("--survivors-only", action="store_true", help="Require --survivor-mask and plot survivor rows only")
    parser.add_argument("--min-partiality", type=float, default=None, help="Minimum partialator partiality from --survivor-mask")
    x_group = parser.add_mutually_exclusive_group()
    x_group.add_argument("--shared-x", dest="shared_x", action="store_true", default=True)
    x_group.add_argument("--individual-x", dest="shared_x", action="store_false")
    args = parser.parse_args()

    if not args.scores.is_file():
        raise SystemExit(f"--scores must be a CSV file, not a directory or missing path: {args.scores}")
    if int(args.n_hkls) < 1:
        raise SystemExit("--n-hkls must be >= 1")
    if int(args.min_obs) < 1:
        raise SystemExit("--min-obs must be >= 1")
    if int(args.bins) < 1:
        raise SystemExit("--bins must be >= 1")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if args.survivor_mask is not None and not args.survivor_mask.is_file():
        raise SystemExit(f"--survivor-mask must be a CSV file, not a directory or missing path: {args.survivor_mask}")
    if args.survivors_only and args.survivor_mask is None:
        raise SystemExit("--survivors-only requires --survivor-mask")
    if args.min_partiality is not None:
        if args.survivor_mask is None:
            raise SystemExit("--min-partiality requires --survivor-mask")
        if not math.isfinite(float(args.min_partiality)):
            raise SystemExit("--min-partiality must be finite")
    return args


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def require_columns(header: list[str], columns: list[str]) -> None:
    missing = [column for column in columns if column not in header]
    if missing:
        raise SystemExit(f"Input score CSV is missing required column(s): {missing}")


def normalize_source(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_event(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    return text


def parse_hkls(text: str | None) -> list[tuple[int, int, int]] | None:
    if text is None or not str(text).strip():
        return None
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for token in str(text).split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [part.strip() for part in token.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"Invalid --hkls entry: {token!r}")
        try:
            hkl = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError as exc:
            raise SystemExit(f"Invalid --hkls entry with non-integer value: {token!r}") from exc
        if hkl not in seen:
            out.append(hkl)
            seen.add(hkl)
    if not out:
        raise SystemExit("--hkls was supplied but no valid HKLs were parsed")
    return out


def selection_label(args: argparse.Namespace, explicit_hkls: list[tuple[int, int, int]] | None) -> str:
    return "manual_hkls" if explicit_hkls is not None else str(args.select)


def output_paths(args: argparse.Namespace, explicit_hkls: list[tuple[int, int, int]] | None) -> dict[str, Path]:
    survivor_suffix = ""
    if args.survivor_mask is not None:
        survivor_suffix = "_partialator_survivors"
        if args.min_partiality is not None:
            threshold = f"{float(args.min_partiality):g}".replace("-", "m").replace(".", "p")
            survivor_suffix += f"_minpartiality{threshold}"
    if explicit_hkls is not None:
        stem = f"v4_raw_manual_hkls{survivor_suffix}"
        summary_stem = f"v4_raw_manual_hkls{survivor_suffix}"
    else:
        stem = f"v4_raw_{args.select}_n{int(args.n_hkls)}_minobs{int(args.min_obs)}{survivor_suffix}"
        summary_stem = f"v4_raw_{args.select}{survivor_suffix}"
    return {
        "png": args.outdir / f"{stem}_hist_grid.png",
        "pdf": args.outdir / f"{stem}_hist_grid.pdf",
        "csv": args.outdir / f"{stem}_summary.csv",
        "md": args.outdir / f"{summary_stem}_summary.md",
    }


def ensure_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    blocked = [path for path in paths.values() if path.exists()]
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing output file(s):\n{formatted}\nUse --overwrite if intended.")
    paths["png"].parent.mkdir(parents=True, exist_ok=True)


def usecols_from_header(header: list[str], score_column: str, include_keys: bool) -> list[str]:
    cols = [*(KEY_COLUMNS if include_keys else HKL_COLUMNS), score_column]
    for optional in ["d_angstrom", "inv_nm"]:
        if optional in header:
            cols.append(optional)
    return list(dict.fromkeys(cols))


def numeric_chunk(chunk: pd.DataFrame, score_column: str) -> pd.DataFrame:
    out = chunk.copy()
    if "source_filename" in out:
        out["source_filename"] = out["source_filename"].map(normalize_source)
    if "event" in out:
        out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[score_column] = pd.to_numeric(out[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for optional in ["d_angstrom", "inv_nm"]:
        if optional in out:
            out[optional] = pd.to_numeric(out[optional], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.loc[~out[HKL_COLUMNS].isna().any(axis=1)].copy()
    if out.empty:
        return out
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def load_survivor_mask(path: Path, min_partiality: float | None, chunksize: int) -> pd.DataFrame:
    header = read_header(path)
    require_columns(header, KEY_COLUMNS)
    usecols = [*KEY_COLUMNS]
    if "partiality" in header:
        usecols.append("partiality")
    elif min_partiality is not None:
        raise SystemExit("--min-partiality was supplied but --survivor-mask has no partiality column")

    chunks = []
    rows_read = 0
    rows_kept = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)), start=1):
        rows_read += int(len(chunk))
        work = numeric_chunk(chunk, "partiality") if "partiality" in chunk.columns else numeric_chunk(chunk, HKL_COLUMNS[0])
        if "partiality" in work.columns and min_partiality is not None:
            work = work.loc[work["partiality"] >= float(min_partiality)].copy()
        if not work.empty:
            chunks.append(work.loc[:, KEY_COLUMNS].copy())
            rows_kept += int(len(work))
        if chunk_index == 1 or chunk_index % 5 == 0:
            print(f"Survivor mask pass: chunks={chunk_index:,}, rows_read={rows_read:,}, rows_kept={rows_kept:,}", flush=True)
    if not chunks:
        raise SystemExit("No survivor-mask keys remained after applying --min-partiality")
    keys = pd.concat(chunks, ignore_index=True).drop_duplicates(KEY_COLUMNS, keep="first")
    print(f"Loaded {len(keys):,} unique survivor-mask keys", flush=True)
    return keys


def apply_survivor_mask(work: pd.DataFrame, survivor_keys: pd.DataFrame | None) -> pd.DataFrame:
    if survivor_keys is None or work.empty:
        return work
    return work.merge(survivor_keys, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")


def score_summary_columns(work: pd.DataFrame, score_column: str) -> pd.DataFrame:
    columns = [*HKL_COLUMNS, score_column]
    for optional in ["d_angstrom", "inv_nm"]:
        if optional in work:
            columns.append(optional)
    return work.loc[:, columns].copy()


def compute_hkl_summary(args: argparse.Namespace, usecols: list[str], survivor_keys: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    rows_read = 0
    for chunk_index, chunk in enumerate(pd.read_csv(args.scores, usecols=usecols, chunksize=int(args.chunksize)), start=1):
        rows_read += int(len(chunk))
        work = numeric_chunk(chunk, str(args.score_column))
        work = work.dropna(subset=[args.score_column])
        work = apply_survivor_mask(work, survivor_keys)
        if not work.empty:
            rows.append(score_summary_columns(work, str(args.score_column)))
        if chunk_index == 1 or chunk_index % 5 == 0:
            print(f"Summary pass: chunks={chunk_index:,}, rows_read={rows_read:,}, buffered_chunks={len(rows):,}", flush=True)

    if not rows:
        raise SystemExit("No finite score values were found")
    table = pd.concat(rows, ignore_index=True)
    grouped = table.groupby(HKL_COLUMNS, sort=False)[str(args.score_column)]
    q = grouped.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    summary = grouped.agg(n_obs="size", min="min", max="max").reset_index()
    q = q.rename(columns={0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"}).reset_index()
    summary = summary.merge(q, on=HKL_COLUMNS, how="left")
    if "d_angstrom" in table:
        summary = summary.merge(table.groupby(HKL_COLUMNS, sort=False)["d_angstrom"].median().reset_index(), on=HKL_COLUMNS, how="left")
    if "inv_nm" in table:
        summary = summary.merge(table.groupby(HKL_COLUMNS, sort=False)["inv_nm"].median().reset_index(), on=HKL_COLUMNS, how="left")
    summary["spread_q90_q10"] = summary["q90"] - summary["q10"]
    summary["spread_q75_q25"] = summary["q75"] - summary["q25"]
    return summary


def choose_hkls(
    summary: pd.DataFrame,
    args: argparse.Namespace,
    explicit_hkls: list[tuple[int, int, int]] | None,
) -> list[tuple[int, int, int]]:
    if explicit_hkls is not None:
        return explicit_hkls
    eligible = summary.loc[summary["n_obs"] >= int(args.min_obs)].copy()
    if eligible.empty:
        raise SystemExit(f"No HKLs have n_obs >= {int(args.min_obs)}")
    if args.select == "top-spread":
        eligible = eligible.sort_values(["spread_q90_q10", "n_obs", "h", "k", "l"], ascending=[False, False, True, True, True])
        selected = eligible.head(int(args.n_hkls))
    elif args.select == "top-obs":
        eligible = eligible.sort_values(["n_obs", "spread_q90_q10", "h", "k", "l"], ascending=[False, False, True, True, True])
        selected = eligible.head(int(args.n_hkls))
    else:
        selected = eligible.sample(n=min(int(args.n_hkls), len(eligible)), random_state=int(args.seed))
        selected = selected.sort_values(["h", "k", "l"])
    return [(int(row.h), int(row.k), int(row.l)) for row in selected.itertuples(index=False)]


def collect_selected_rows(
    args: argparse.Namespace,
    usecols: list[str],
    selected_hkls: list[tuple[int, int, int]],
    survivor_keys: pd.DataFrame | None,
) -> pd.DataFrame:
    selected = set(selected_hkls)
    chunks = []
    rows_read = 0
    rows_kept = 0
    for chunk_index, chunk in enumerate(pd.read_csv(args.scores, usecols=usecols, chunksize=int(args.chunksize)), start=1):
        rows_read += int(len(chunk))
        work = numeric_chunk(chunk, str(args.score_column))
        work = work.dropna(subset=[args.score_column])
        work = apply_survivor_mask(work, survivor_keys)
        if not work.empty:
            mask = [(int(h), int(k), int(l)) in selected for h, k, l in work.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
            sub = score_summary_columns(work.loc[mask].copy(), str(args.score_column))
            if not sub.empty:
                chunks.append(sub)
                rows_kept += int(len(sub))
        if chunk_index == 1 or chunk_index % 5 == 0:
            print(f"Collect pass: chunks={chunk_index:,}, rows_read={rows_read:,}, selected_rows={rows_kept:,}", flush=True)
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame(columns=usecols)


def exact_summary(selected_rows: pd.DataFrame, score_column: str, selected_hkls: list[tuple[int, int, int]]) -> pd.DataFrame:
    rows = []
    for order, (h, k, l) in enumerate(selected_hkls, start=1):
        values = selected_rows.loc[(selected_rows["h"] == h) & (selected_rows["k"] == k) & (selected_rows["l"] == l), score_column].dropna()
        record: dict[str, Any] = {"selection_rank": order, "h": h, "k": k, "l": l, "n_obs": int(len(values))}
        if len(values):
            q10, q25, q50, q75, q90 = np.quantile(values.to_numpy(dtype=float), [0.10, 0.25, 0.50, 0.75, 0.90])
            record.update(
                {
                    "min": float(values.min()),
                    "q10": float(q10),
                    "q25": float(q25),
                    "q50": float(q50),
                    "q75": float(q75),
                    "q90": float(q90),
                    "max": float(values.max()),
                    "spread_q90_q10": float(q90 - q10),
                    "spread_q75_q25": float(q75 - q25),
                }
            )
        else:
            record.update({key: np.nan for key in ["min", "q10", "q25", "q50", "q75", "q90", "max", "spread_q90_q10", "spread_q75_q25"]})
        for optional in ["d_angstrom", "inv_nm"]:
            if optional in selected_rows:
                opt = selected_rows.loc[(selected_rows["h"] == h) & (selected_rows["k"] == k) & (selected_rows["l"] == l), optional].dropna()
                record[optional] = float(opt.median()) if len(opt) else np.nan
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def shared_xlim(selected_rows: pd.DataFrame, score_column: str, enabled: bool) -> tuple[float, float] | None:
    if not enabled:
        return None
    values = selected_rows[score_column].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return None
    q01, q99 = np.quantile(values, [0.01, 0.99])
    if not np.isfinite(q01) or not np.isfinite(q99) or q99 <= q01:
        return float(np.nanmin(values)), float(np.nanmax(values))
    lo = max(float(np.nanmin(values)), float(q01 - 0.10 * (q99 - q01)))
    hi = min(float(np.nanmax(values)), float(q99 + 0.10 * (q99 - q01)))
    return lo, hi


def plot_grid(
    selected_rows: pd.DataFrame,
    selected_summary: pd.DataFrame,
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    n = len(selected_summary)
    ncols = min(4, max(1, math.ceil(math.sqrt(n))))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)
    xlim = shared_xlim(selected_rows, str(args.score_column), bool(args.shared_x))
    for ax, row in zip(axes.ravel(), selected_summary.itertuples(index=False), strict=False):
        h, k, l = int(row.h), int(row.k), int(row.l)
        values = selected_rows.loc[
            (selected_rows["h"] == h) & (selected_rows["k"] == k) & (selected_rows["l"] == l), str(args.score_column)
        ].dropna()
        ax.hist(values.to_numpy(dtype=float), bins=int(args.bins), color="#3268a8", alpha=0.85)
        for quantile_name, color in [("q10", "#777777"), ("q50", "#d1495b"), ("q90", "#777777")]:
            value = getattr(row, quantile_name)
            if pd.notna(value):
                ax.axvline(float(value), color=color, linestyle="--" if quantile_name != "q50" else "-", linewidth=1.2)
        ax.set_title(f"HKL {h} {k} {l}, n={int(row.n_obs)}")
        ax.set_xlabel(str(args.score_column))
        ax.set_ylabel("observations")
        if xlim is not None and xlim[1] > xlim[0]:
            ax.set_xlim(*xlim)
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Raw v4 local-crowding distributions by signed HKL", y=0.995)
    fig.tight_layout()
    fig.savefig(paths["png"], dpi=180)
    fig.savefig(paths["pdf"])
    plt.close(fig)


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    explicit_hkls: list[tuple[int, int, int]] | None,
    selected_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    mode = "manual" if explicit_hkls is not None else str(args.select)
    lines = [
        "# V4 Raw Chosen-HKL Distribution Summary",
        "",
        f"- input CSV: `{args.scores}`",
        f"- score column: `{args.score_column}`",
        f"- selection mode: `{mode}`",
        f"- HKLs plotted: {len(selected_summary)}",
        f"- min obs threshold: {int(args.min_obs)}",
        f"- survivor mask: `{args.survivor_mask}`" if args.survivor_mask is not None else "- survivor mask: none",
        f"- min partiality: {float(args.min_partiality):.6g}" if args.min_partiality is not None else "- min partiality: none",
        "- note: no normalization, log transform, filtering, or stream writing was used",
        "",
        "## Outputs",
        f"- PNG: `{paths['png']}`",
        f"- PDF: `{paths['pdf']}`",
        f"- summary CSV: `{paths['csv']}`",
        "",
        "## Selected HKLs",
        "| rank | h | k | l | n_obs | spread_q90_q10 | q10 | q50 | q90 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in selected_summary.head(50).itertuples(index=False):
        lines.append(
            f"| {int(row.selection_rank)} | {int(row.h)} | {int(row.k)} | {int(row.l)} | {int(row.n_obs)} | "
            f"{float(row.spread_q90_q10):.6g} | {float(row.q10):.6g} | {float(row.q50):.6g} | {float(row.q90):.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    header = read_header(args.scores)
    required = [*(KEY_COLUMNS if args.survivor_mask is not None else HKL_COLUMNS), str(args.score_column)]
    require_columns(header, required)
    explicit_hkls = parse_hkls(args.hkls)
    paths = output_paths(args, explicit_hkls)
    ensure_outputs(paths, bool(args.overwrite))
    survivor_keys = None
    if args.survivor_mask is not None:
        print("Loading exact-key partialator survivor mask", flush=True)
        survivor_keys = load_survivor_mask(args.survivor_mask, args.min_partiality, int(args.chunksize))
    usecols = usecols_from_header(header, str(args.score_column), survivor_keys is not None)

    if explicit_hkls is None:
        print("Pass 1/2: computing per-signed-HKL raw score summaries", flush=True)
        summary = compute_hkl_summary(args, usecols, survivor_keys)
        selected_hkls = choose_hkls(summary, args, explicit_hkls)
    else:
        selected_hkls = explicit_hkls
    print(f"Selected {len(selected_hkls)} signed HKLs", flush=True)

    print("Pass 2/2: collecting selected HKL observations", flush=True)
    selected_rows = collect_selected_rows(args, usecols, selected_hkls, survivor_keys)
    selected_summary = exact_summary(selected_rows, str(args.score_column), selected_hkls)
    selected_summary.to_csv(paths["csv"], index=False)
    plot_grid(selected_rows, selected_summary, args, paths)
    write_markdown(paths["md"], args, explicit_hkls, selected_summary, paths)

    print("V4 raw chosen-HKL distribution plots written")
    print(f"png: {paths['png']}")
    print(f"pdf: {paths['pdf']}")
    print(f"summary_csv: {paths['csv']}")
    print(f"summary_md: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
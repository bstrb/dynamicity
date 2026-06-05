"""Information summaries that turn OriDyn score tables into actionable lists."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FRAME_CONTEXT_COLUMNS = [
    "frame",
    "frame_number",
    "event",
    "assigned_risky_axis",
    "assigned_axis_rank",
    "assigned_axis_angle_deg",
    "nearest_zone_axis",
    "nearest_zone_axis_angle_deg",
    "axis_match",
    "axis_angle_delta_deg",
    "frame_axis_risk_raw",
    "frame_axis_risk_norm",
    "n_excited",
    "sum_excitation_weight",
    "n_observed_targets",
    "mean_S_dyn_geom",
    "p95_S_dyn_geom",
    "mean_sigma_dyn_rel",
]


REFLECTION_CONTEXT_COLUMNS = [
    "reflection_id",
    "frame",
    "frame_number",
    "event",
    "h",
    "k",
    "l",
    "q_invA",
    "d_angstrom",
    "sg",
    "excitation_weight",
    "self_risk_raw",
    "graph_crowding_raw",
    "same_laue_zone_crowding_raw",
    "systematic_row_risk_raw",
    "S_dyn_geom",
    "sigma_dyn_rel",
    "assigned_zone_axis",
    "laue_n",
    "nearest_row_direction",
    "top_neighbor_summary",
]


def make_information_summaries(
    frame_summary: pd.DataFrame,
    reflection_scores: pd.DataFrame,
    axis_sigma_deg: float = 2.0,
    high_frame_quantile: float = 0.90,
    top_reflections_per_frame: int = 20,
    top_frames: int = 50,
) -> dict[str, pd.DataFrame]:
    """Build compact CSV-ready summaries for frame/reflection triage."""

    frames = _decorate_frame_summary(frame_summary, axis_sigma_deg, high_frame_quantile)
    summaries = {
        "summary_close_risky_axis_frames.csv": _close_risky_axis_frames(frames, top_frames),
        "summary_high_dynamical_frames.csv": _high_dynamical_frames(frames, top_frames),
        "summary_frame_metric_correlations.csv": _frame_metric_correlations(frames),
        "summary_axis_group_metrics.csv": _axis_group_metrics(frames),
        "summary_top_reflections_in_high_dynamical_frames.csv": _top_reflections_in_high_frames(
            frames, reflection_scores, top_reflections_per_frame
        ),
    }
    return summaries


def write_information_summaries(output_dir: str | Path, summaries: dict[str, pd.DataFrame]) -> None:
    """Write information summary tables."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, table in summaries.items():
        table.to_csv(out / filename, index=False)


def _decorate_frame_summary(
    frame_summary: pd.DataFrame,
    axis_sigma_deg: float,
    high_frame_quantile: float,
) -> pd.DataFrame:
    frames = frame_summary.copy()
    if frames.empty:
        return frames
    if "axis_match" not in frames and {"assigned_risky_axis", "nearest_zone_axis"} <= set(frames.columns):
        frames["axis_match"] = frames["assigned_risky_axis"].astype(str) == frames["nearest_zone_axis"].astype(str)
    if "axis_angle_delta_deg" not in frames and {"assigned_axis_angle_deg", "nearest_zone_axis_angle_deg"} <= set(
        frames.columns
    ):
        frames["axis_angle_delta_deg"] = (
            pd.to_numeric(frames["assigned_axis_angle_deg"], errors="coerce")
            - pd.to_numeric(frames["nearest_zone_axis_angle_deg"], errors="coerce")
        )
    if "assigned_axis_angle_deg" in frames:
        frames["close_to_risky_axis"] = (
            pd.to_numeric(frames["assigned_axis_angle_deg"], errors="coerce") <= float(axis_sigma_deg)
        )
    else:
        frames["close_to_risky_axis"] = False
    for metric in ("mean_S_dyn_geom", "p95_S_dyn_geom", "frame_axis_risk_norm", "frame_axis_risk_raw"):
        if metric not in frames:
            frames[metric] = np.nan
    q = min(max(float(high_frame_quantile), 0.0), 1.0)
    mean_threshold = _quantile(frames["mean_S_dyn_geom"], q)
    p95_threshold = _quantile(frames["p95_S_dyn_geom"], q)
    frames["high_mean_S_dyn_geom"] = pd.to_numeric(frames["mean_S_dyn_geom"], errors="coerce") >= mean_threshold
    frames["high_p95_S_dyn_geom"] = pd.to_numeric(frames["p95_S_dyn_geom"], errors="coerce") >= p95_threshold
    frames["high_dynamical_frame"] = frames["high_mean_S_dyn_geom"] | frames["high_p95_S_dyn_geom"]
    frames["mean_S_dyn_geom_quantile_threshold"] = mean_threshold
    frames["p95_S_dyn_geom_quantile_threshold"] = p95_threshold
    frames["axis_sigma_deg_threshold"] = float(axis_sigma_deg)
    return frames


def _close_risky_axis_frames(frames: pd.DataFrame, top_frames: int) -> pd.DataFrame:
    if frames.empty:
        return pd.DataFrame()
    table = frames.loc[frames["close_to_risky_axis"]].copy()
    if table.empty:
        table = frames.copy()
    table = table.sort_values(["assigned_axis_angle_deg", "frame_axis_risk_raw"], ascending=[True, False])
    return _select_columns(table.head(top_frames), FRAME_CONTEXT_COLUMNS + ["close_to_risky_axis"])


def _high_dynamical_frames(frames: pd.DataFrame, top_frames: int) -> pd.DataFrame:
    if frames.empty:
        return pd.DataFrame()
    table = frames.sort_values(["mean_S_dyn_geom", "p95_S_dyn_geom"], ascending=[False, False]).head(top_frames)
    return _select_columns(
        table,
        FRAME_CONTEXT_COLUMNS
        + [
            "close_to_risky_axis",
            "high_mean_S_dyn_geom",
            "high_p95_S_dyn_geom",
            "high_dynamical_frame",
        ],
    )


def _frame_metric_correlations(frames: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "assigned_axis_angle_deg",
        "nearest_zone_axis_angle_deg",
        "frame_axis_risk_raw",
        "frame_axis_risk_norm",
        "n_excited",
        "sum_excitation_weight",
        "mean_S_dyn_geom",
        "p95_S_dyn_geom",
        "mean_sigma_dyn_rel",
    ]
    columns = [col for col in columns if col in frames]
    if len(columns) < 2:
        return pd.DataFrame(columns=["metric_x", "metric_y", "pearson_r", "spearman_r", "n"])
    numeric = frames[columns].apply(pd.to_numeric, errors="coerce")
    rows: list[dict[str, float | int | str]] = []
    for i, col_x in enumerate(columns):
        for col_y in columns[i + 1 :]:
            pair = numeric[[col_x, col_y]].dropna()
            if len(pair) < 3:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(pair[col_x].corr(pair[col_y], method="pearson"))
                spearman = float(pair[col_x].corr(pair[col_y], method="spearman"))
            rows.append({"metric_x": col_x, "metric_y": col_y, "pearson_r": pearson, "spearman_r": spearman, "n": len(pair)})
    return pd.DataFrame.from_records(rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)


def _axis_group_metrics(frames: pd.DataFrame) -> pd.DataFrame:
    if frames.empty or "assigned_risky_axis" not in frames:
        return pd.DataFrame()
    grouped = frames.groupby("assigned_risky_axis", dropna=False).agg(
        n_frames=("frame", "size"),
        n_close_to_risky_axis=("close_to_risky_axis", "sum"),
        mean_assigned_axis_angle_deg=("assigned_axis_angle_deg", "mean"),
        median_assigned_axis_angle_deg=("assigned_axis_angle_deg", "median"),
        mean_frame_axis_risk_norm=("frame_axis_risk_norm", "mean"),
        mean_S_dyn_geom=("mean_S_dyn_geom", "mean"),
        p95_S_dyn_geom=("p95_S_dyn_geom", "mean"),
        mean_n_excited=("n_excited", "mean"),
        mean_sum_excitation_weight=("sum_excitation_weight", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["fraction_close_to_risky_axis"] = grouped["n_close_to_risky_axis"] / grouped["n_frames"].clip(lower=1)
    return grouped.sort_values(["mean_S_dyn_geom", "mean_frame_axis_risk_norm"], ascending=[False, False])


def _top_reflections_in_high_frames(
    frames: pd.DataFrame,
    reflection_scores: pd.DataFrame,
    top_reflections_per_frame: int,
) -> pd.DataFrame:
    if frames.empty or reflection_scores.empty:
        return pd.DataFrame()
    high_frames = frames.loc[frames["high_mean_S_dyn_geom"], _frame_context_existing(frames)].copy()
    if high_frames.empty:
        high_frames = frames.sort_values("mean_S_dyn_geom", ascending=False).head(10)
    high_frame_ids = set(pd.to_numeric(high_frames["frame"], errors="coerce").dropna().astype(int))
    reflections = reflection_scores.loc[reflection_scores["frame"].isin(high_frame_ids)].copy()
    if reflections.empty:
        return pd.DataFrame()
    top_n = max(int(top_reflections_per_frame), 1)
    reflections = reflections.sort_values(["frame", "S_dyn_geom"], ascending=[True, False])
    reflections = reflections.groupby("frame", group_keys=False).head(top_n).copy()
    reflections["within_frame_S_dyn_rank"] = reflections.groupby("frame")["S_dyn_geom"].rank(
        method="first", ascending=False
    ).astype(int)
    context_columns = [
        "frame",
        "close_to_risky_axis",
        "assigned_axis_angle_deg",
        "frame_axis_risk_norm",
        "mean_S_dyn_geom",
        "p95_S_dyn_geom",
    ]
    context = high_frames[[col for col in context_columns if col in high_frames]].drop_duplicates("frame")
    reflections = reflections.merge(context, on="frame", how="left", suffixes=("", "_frame"))
    ordered = _select_columns(
        reflections.sort_values(["mean_S_dyn_geom", "frame", "within_frame_S_dyn_rank"], ascending=[False, True, True]),
        REFLECTION_CONTEXT_COLUMNS
        + [
            "within_frame_S_dyn_rank",
            "close_to_risky_axis",
            "assigned_axis_angle_deg_frame",
            "frame_axis_risk_norm_frame",
            "mean_S_dyn_geom",
            "p95_S_dyn_geom",
        ],
    )
    return ordered


def _frame_context_existing(frames: pd.DataFrame) -> list[str]:
    return [col for col in FRAME_CONTEXT_COLUMNS + ["close_to_risky_axis", "high_mean_S_dyn_geom"] if col in frames]


def _select_columns(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return table[[col for col in columns if col in table.columns]].copy()


def _quantile(values: pd.Series, q: float) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return np.nan
    return float(finite.quantile(q))

"""Output table and metadata writers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


TERM_SUMMARY_COLUMNS = [
    "self_risk_raw",
    "graph_crowding_raw",
    "same_laue_zone_crowding_raw",
    "systematic_row_risk_raw",
    "frame_axis_risk_norm",
    "self_risk_norm",
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "S_dyn_geom_weighted_sum",
    "S_dyn_geom",
    "sigma_tail_score",
    "sigma_dyn_rel",
]


def summarize_score_terms(scores: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary table for score terms."""

    rows: list[dict[str, float | int | str]] = []
    for column in TERM_SUMMARY_COLUMNS:
        if column not in scores:
            continue
        values = pd.to_numeric(scores[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "term": column,
                "n": int(values.size),
                "min": float(values.min()),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_outputs(
    output_dir: str | Path,
    problematic_axes: pd.DataFrame,
    frame_summary: pd.DataFrame,
    reflection_scores: pd.DataFrame,
    score_terms_summary: pd.DataFrame,
    metadata: dict[str, object],
    candidate_scores: pd.DataFrame | None = None,
    information_summaries: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Write required and optional outputs."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    problematic_axes.to_csv(out / "problematic_axes.csv", index=False)
    frame_summary.to_csv(out / "frame_summary.csv", index=False)
    reflection_scores.to_csv(out / "reflection_scores.csv", index=False)
    score_terms_summary.to_csv(out / "score_terms_summary.csv", index=False)
    _write_top(reflection_scores, out / "top_self_risk.csv", "self_risk_raw")
    _write_top(reflection_scores, out / "top_graph_crowding_risk.csv", "graph_crowding_raw")
    _write_top(reflection_scores, out / "top_systematic_row_risk.csv", "systematic_row_risk_raw")
    _write_top(reflection_scores, out / "top_laue_zone_risk.csv", "laue_zone_risk_raw")
    if information_summaries:
        for filename, table in information_summaries.items():
            table.to_csv(out / filename, index=False)
    if candidate_scores is not None:
        candidate_scores.to_csv(out / "candidate_reflection_scores.csv", index=False)
    payload = dict(metadata)
    payload.setdefault("generated_utc", datetime.now(timezone.utc).isoformat())
    (out / "run_metadata.json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


def _write_top(scores: pd.DataFrame, path: Path, column: str, n: int = 100) -> None:
    if scores.empty or column not in scores:
        pd.DataFrame().to_csv(path, index=False)
        return
    scores.sort_values(column, ascending=False).head(n).to_csv(path, index=False)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

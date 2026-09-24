"""Disagreement statistics and risk/disagreement summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .xds_parser import assign_resolution_shell


def add_correct_status(observations: pd.DataFrame, matches: dict[tuple[int, int, int, float], dict[str, Any]]) -> pd.DataFrame:
    """Attach conservative CORRECT/XDS_ASCII status diagnostics."""

    out = observations.copy()
    statuses: list[str] = []
    corrected_iobs: list[float] = []
    corrected_sigma: list[float] = []
    for row in out.itertuples(index=False):
        key = (int(row.h), int(row.k), int(row.l), round(float(row.ZCAL), 1))
        match = matches.get(key)
        if match:
            statuses.append(str(match["correct_status"]))
            corrected_iobs.append(float(match["corrected_IOBS"]) if match.get("corrected_IOBS") is not None else np.nan)
            corrected_sigma.append(float(match["corrected_SIGMA"]) if match.get("corrected_SIGMA") is not None else np.nan)
        else:
            statuses.append("not_matched_to_XDS_ASCII")
            corrected_iobs.append(np.nan)
            corrected_sigma.append(np.nan)
    out["correct_status"] = statuses
    out["corrected_IOBS"] = corrected_iobs
    out["corrected_SIGMA"] = corrected_sigma
    return out


def add_leave_one_out_disagreement(observations: pd.DataFrame, min_multiplicity: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add leave-one-out reference intensity and disagreement columns."""

    out = observations.copy()
    out["reference_intensity"] = np.nan
    out["reference_I_over_SIGMA"] = np.nan
    out["signed_difference"] = np.nan
    out["relative_disagreement"] = np.nan
    out["disagreement_valid"] = False
    out["exclusion_reason"] = ""
    out["risk_rank_within_class"] = np.nan
    out["disagreement_rank_within_class"] = np.nan

    for _symmetry_id, group in out.groupby("symmetry_id", sort=True):
        indices = group.index.to_numpy()
        intensities = group["IOBS"].to_numpy(dtype=float)
        isig = group["I_over_SIGMA"].to_numpy(dtype=float)
        multiplicity = len(group)
        for local_idx, index in enumerate(indices):
            others = np.delete(intensities, local_idx)
            others_isig = np.delete(isig, local_idx)
            finite_ref = others[np.isfinite(others)]
            finite_isig = others_isig[np.isfinite(others_isig)]
            reasons: list[str] = []
            if multiplicity < min_multiplicity:
                reasons.append("symmetry_multiplicity_lt_3")
            if finite_ref.size == 0:
                reasons.append("no_finite_leave_one_out_reference")
                reference = np.nan
            else:
                reference = float(np.median(finite_ref))
            reference_isig = float(np.median(finite_isig)) if finite_isig.size else np.nan
            out.at[index, "reference_intensity"] = reference
            out.at[index, "reference_I_over_SIGMA"] = reference_isig
            if not np.isfinite(reference):
                reasons.append("nonfinite_reference_intensity")
            elif reference <= 0.0:
                reasons.append("nonpositive_reference_intensity")
            if not np.isfinite(out.at[index, "S_risk"]):
                reasons.append("nonfinite_S_risk")
            if reasons:
                out.at[index, "exclusion_reason"] = ";".join(dict.fromkeys(reasons))
                continue
            signed = float(out.at[index, "IOBS"]) - reference
            out.at[index, "signed_difference"] = signed
            out.at[index, "relative_disagreement"] = abs(signed) / abs(reference)
            out.at[index, "disagreement_valid"] = True

    out = add_within_class_ranks(out)
    excluded = out.loc[out["exclusion_reason"] != "", ["observation_id", "h", "k", "l", "symmetry_id", "exclusion_reason"]].copy()
    return out, excluded


def add_within_class_ranks(observations: pd.DataFrame) -> pd.DataFrame:
    """Add pooled fractional within-class ranks for risk and disagreement."""

    out = observations.copy()
    valid = out["disagreement_valid"].astype(bool) & np.isfinite(out["S_risk"]) & np.isfinite(out["relative_disagreement"])
    for _symmetry_id, group in out.loc[valid].groupby("symmetry_id", sort=True):
        if len(group) < 3:
            continue
        risk_rank = group["S_risk"].rank(method="average")
        disagreement_rank = group["relative_disagreement"].rank(method="average")
        denom = max(len(group) - 1, 1)
        out.loc[group.index, "risk_rank_within_class"] = (risk_rank - 1.0) / denom
        out.loc[group.index, "disagreement_rank_within_class"] = (disagreement_rank - 1.0) / denom
    return out


def risk_distribution(observations: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for risk components."""

    records: list[dict[str, Any]] = []
    for column in ("E_target", "R_env", "S_risk"):
        values = pd.to_numeric(observations[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            records.append({"quantity": column, "count": 0})
            continue
        records.append(
            {
                "quantity": column,
                "count": int(values.size),
                "min": float(np.min(values)),
                "q01": float(np.quantile(values, 0.01)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "q99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame.from_records(records)


def correlation_tables(observations: pd.DataFrame) -> pd.DataFrame:
    """Return global and within-class risk/disagreement correlations."""

    rows: list[dict[str, Any]] = []
    valid = observations["disagreement_valid"].astype(bool)
    for predictor, role in (
        ("S_risk", "primary"),
        ("E_target", "diagnostic"),
        ("R_env", "diagnostic"),
    ):
        result = spearman(observations.loc[valid, predictor], observations.loc[valid, "relative_disagreement"])
        rows.append(
            {
                "analysis": "global",
                "predictor": predictor,
                "role": role,
                **result,
                "rank_transform": "ordinary global average ranks",
            }
        )
    within_valid = (
        valid
        & np.isfinite(observations["risk_rank_within_class"])
        & np.isfinite(observations["disagreement_rank_within_class"])
    )
    within = spearman(
        observations.loc[within_valid, "risk_rank_within_class"],
        observations.loc[within_valid, "disagreement_rank_within_class"],
    )
    rows.append(
        {
            "analysis": "within_symmetry_class",
            "predictor": "S_risk",
            "role": "primary",
            **within,
            "rank_transform": "average ties within class, transformed as (rank - 1) / (n_class - 1), then pooled",
        }
    )
    return pd.DataFrame.from_records(rows)


def spearman(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Calculate Spearman rho with average-rank ties and deterministic filtering."""

    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return {"n": int(len(frame)), "rho": np.nan}
    xr = frame["x"].rank(method="average")
    yr = frame["y"].rank(method="average")
    if float(xr.std(ddof=0)) == 0.0 or float(yr.std(ddof=0)) == 0.0:
        return {"n": int(len(frame)), "rho": np.nan}
    return {"n": int(len(frame)), "rho": float(np.corrcoef(xr, yr)[0, 1])}


def add_resolution_shells(observations: pd.DataFrame, shells: pd.DataFrame) -> pd.DataFrame:
    """Attach XDS resolution shell identifiers."""

    out = observations.copy()
    if shells.empty:
        out["resolution_shell"] = np.nan
    else:
        out["resolution_shell"] = [assign_resolution_shell(float(value), shells) for value in out["resolution"]]
    return out


def resolution_summary(observations: pd.DataFrame, shells: pd.DataFrame) -> pd.DataFrame:
    """Summarize relationship by XDS resolution shell."""

    if shells.empty or "resolution_shell" not in observations:
        return pd.DataFrame(
            [
                {
                    "shell": np.nan,
                    "note": "No reliable XDS resolution-shell boundaries were parsed; no invented shells were created.",
                }
            ]
        )
    rows: list[dict[str, Any]] = []
    for shell, group in observations.groupby("resolution_shell", dropna=True, sort=True):
        valid = group["disagreement_valid"].astype(bool)
        global_corr = spearman(group.loc[valid, "S_risk"], group.loc[valid, "relative_disagreement"])
        within_valid = valid & np.isfinite(group["risk_rank_within_class"]) & np.isfinite(group["disagreement_rank_within_class"])
        within_corr = spearman(group.loc[within_valid, "risk_rank_within_class"], group.loc[within_valid, "disagreement_rank_within_class"])
        rows.append(
            {
                "shell": int(shell),
                "observation_count": int(len(group)),
                "symmetry_class_count": int(group["symmetry_id"].nunique()),
                "valid_disagreement_count": int(valid.sum()),
                "median_disagreement": float(group.loc[valid, "relative_disagreement"].median()) if valid.any() else np.nan,
                "global_spearman_n": global_corr["n"],
                "global_spearman_rho": global_corr["rho"],
                "within_class_rank_n": within_corr["n"],
                "within_class_rank_rho": within_corr["rho"],
            }
        )
    summary = pd.DataFrame.from_records(rows)
    return shells.merge(summary, on="shell", how="left")


def finalize_group_table(observations: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Add primary-analysis counts to the group table."""

    valid_counts = observations.groupby("symmetry_id", sort=True)["disagreement_valid"].sum().rename("valid_disagreement_count")
    out = groups.merge(valid_counts, on="symmetry_id", how="left")
    out["valid_disagreement_count"] = out["valid_disagreement_count"].fillna(0).astype(int)
    return out

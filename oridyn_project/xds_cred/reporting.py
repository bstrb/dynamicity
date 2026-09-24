"""Reporting, progress logging, and plotting for the XDS/cRED workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_OUTPUTS = (
    "observations.csv.gz",
    "groups.csv.gz",
    "excluded.csv.gz",
    "geometry.csv.gz",
    "geometry_summary.txt",
    "orientation_audit.csv",
    "neighbor_audit.csv.gz",
    "risk_distribution.csv",
    "correlations.csv",
    "resolution.csv",
    "scales_integrate.csv",
    "scales_correct.csv",
    "summary.txt",
    "parameters.json",
    "input_files.json",
    "software_versions.json",
    "run.log",
)


def prepare_output_dir(output_dir: Path, allow_overwrite: bool) -> None:
    """Create output directory and stop before overwriting expected results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in EXPECTED_OUTPUTS if (output_dir / name).exists()]
    if existing and not allow_overwrite:
        raise FileExistsError(
            f"Output directory already contains expected result files: {', '.join(existing)}. "
            "Choose a new output directory or move existing results before running production."
        )
    (output_dir / "figures").mkdir(exist_ok=True)


def setup_logger(output_dir: Path) -> logging.Logger:
    """Set up timestamped logging to stdout and run.log."""

    logger = logging.getLogger("xds_cred")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_and_flush(logger: logging.Logger, message: str) -> None:
    """Log and flush all handlers immediately."""

    logger.info(message)
    for handler in logger.handlers:
        handler.flush()


class Progress:
    """Timestamped progress reporter with rate and ETA."""

    def __init__(self, logger: logging.Logger, label: str, total: int | None):
        self.logger = logger
        self.label = label
        self.total = total
        self.completed = 0
        self.started = time.monotonic()
        self.last_report = 0.0

    def update(self, count: int, force: bool = False) -> None:
        self.completed += int(count)
        now = time.monotonic()
        if not force and now - self.last_report < 5.0 and (self.total is None or self.completed < self.total):
            return
        elapsed = max(now - self.started, 1e-9)
        rate = self.completed / elapsed
        if self.total:
            pct = 100.0 * min(self.completed, self.total) / self.total
            remaining = max(self.total - self.completed, 0)
            eta = remaining / rate if rate > 0 else float("inf")
            detail = f"{self.completed}/{self.total} ({pct:.1f}%), elapsed {elapsed:.1f}s, {rate:.1f}/s, ETA {eta:.1f}s"
        else:
            detail = f"{self.completed} completed, elapsed {elapsed:.1f}s, {rate:.1f}/s"
        stamp = datetime.now(timezone.utc).isoformat()
        log_and_flush(self.logger, f"{self.label}: {detail}, timestamp {stamp}")
        self.last_report = now


def write_geometry_summary(path: Path, summaries: list[dict[str, Any]], selected_source: str, reason: str) -> None:
    """Write a human-readable geometry audit summary."""

    lines = [
        "XDS geometry audit",
        "==================",
        "",
        reason,
        "",
        "Primary criterion: agreement with ZCAL because risk is evaluated at the continuous orientation.",
        "XCAL and YCAL residuals are additional detector-projection checks.",
        "",
    ]
    for summary in summaries:
        marker = "SELECTED" if summary["geometry_source"] == selected_source else "not selected"
        lines.extend(
            [
                f"{summary['geometry_source']} ({marker})",
                f"  audited observations: {summary['n_audited']}",
                f"  angle offset: {summary.get('angle_offset_frames', 0.0):.6g} frames",
                f"  INTEGRATE batch count: {summary.get('batch_count', 0)}",
                f"  median |s_g(ZCAL)|: {summary['median_abs_s_target_invA']:.6g} A^-1",
                f"  p95 |s_g(ZCAL)|: {summary['p95_abs_s_target_invA']:.6g} A^-1",
                f"  median Zpred-ZCAL: {summary['median_z_residual_frames']:.6g} frames",
                f"  median |Zpred-ZCAL|: {summary['median_abs_z_residual_frames']:.6g} frames",
                f"  median |Zpred-ZCAL| angle: {summary.get('median_abs_z_residual_degrees', float('nan')):.6g} degrees",
                f"  p95 |Zpred-ZCAL|: {summary['p95_abs_z_residual_frames']:.6g} frames",
                f"  median |XCAL residual|: {summary['median_abs_x_residual_px']:.6g} px",
                f"  median |YCAL residual|: {summary['median_abs_y_residual_px']:.6g} px",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    dataset_name: str,
    observations: pd.DataFrame,
    groups: pd.DataFrame,
    correlations: pd.DataFrame,
    risk_dist: pd.DataFrame,
    resolution: pd.DataFrame,
    geometry_reason: str,
    scale_integrate: pd.DataFrame,
    scale_correct: pd.DataFrame,
    parameters: dict[str, Any],
) -> None:
    """Write clear scientific run summary."""

    valid = observations["disagreement_valid"].astype(bool)
    global_primary = correlations[(correlations["analysis"] == "global") & (correlations["predictor"] == "S_risk")]
    within = correlations[correlations["analysis"] == "within_symmetry_class"]
    z_residual = observations["z_residual_frames"].replace([np.inf, -np.inf], np.nan).dropna()
    z_abs = z_residual.abs()
    z_abs_deg = observations["z_residual_degrees"].replace([np.inf, -np.inf], np.nan).dropna().abs()
    lines = [
        f"OriDyn cRED/XDS validation summary: {dataset_name}",
        "=" * (38 + len(dataset_name)),
        "",
        "This run assigns one geometry-only OriDyn risk score to each finite INTEGRATE.HKL observation at its exact fractional ZCAL.",
        "Measured intensity is used only after risk calculation, for symmetry-class disagreement diagnostics.",
        "",
        "Geometry",
        "--------",
        geometry_reason,
        f"Selected geometry source: {parameters.get('geometry_source', 'unknown')}",
        f"Angle equation: {parameters.get('xds_orientation_convention', {}).get('angle_equation', 'not recorded')}",
        f"ZCAL angle offset: {parameters.get('zcal_angle_offset_frames', 'not recorded')} frames",
        f"Median target |s_g(ZCAL)|: {observations['s_target'].abs().median():.6g} A^-1",
        f"Median E(g): {observations['E_target'].median():.6g}",
        f"Median |Zpred-ZCAL|: {z_abs.median():.6g} frames ({z_abs_deg.median():.6g} degrees)",
        f"P95 |Zpred-ZCAL|: {z_abs.quantile(0.95):.6g} frames",
        "",
        "Eligibility",
        "-----------",
        f"Finite INTEGRATE.HKL observations retained: {len(observations)}",
        f"Primary disagreement-valid observations: {int(valid.sum())}",
        f"Symmetry classes: {observations['symmetry_id'].nunique()}",
        f"Contributing classes with valid observations: {observations.loc[valid, 'symmetry_id'].nunique()}",
        "No I/SIGMA, PEAK, CORR, resolution, or risk cutoffs were applied.",
        "",
        "Risk Distribution",
        "-----------------",
    ]
    for row in risk_dist.itertuples(index=False):
        lines.append(
            f"{row.quantity}: n={row.count}, median={getattr(row, 'median', np.nan):.6g}, "
            f"q05={getattr(row, 'q05', np.nan):.6g}, q95={getattr(row, 'q95', np.nan):.6g}"
        )
    lines.extend(["", "Risk vs Disagreement", "--------------------"])
    if not global_primary.empty:
        row = global_primary.iloc[0]
        lines.append(f"Global descriptive Spearman S_risk vs D_i: rho={row['rho']:.6g}, n={int(row['n'])}")
    if not within.empty:
        row = within.iloc[0]
        lines.append(
            "Primary within-class rank relationship: "
            f"rho={row['rho']:.6g}, n={int(row['n'])}; "
            "ties use average ranks and ranks are transformed as (rank - 1)/(n_class - 1)."
        )
    lines.extend(
        [
            "",
            "Resolution",
            "----------",
            "Resolution shells come from XDS CORRECT.LP. No new shells were invented.",
            f"Resolution rows written: {len(resolution)}",
            "",
            "Scale Diagnostics",
            "-----------------",
            f"INTEGRATE image-scale rows: {len(scale_integrate)}",
            f"CORRECT correction/scale diagnostic rows: {len(scale_correct)}",
            "INTEGRATE image scaling and CORRECT corrections are saved separately and are not interpreted as the same quantity.",
            "",
            "Observed Evidence vs Interpretation",
            "-----------------------------------",
            "The reported correlations are descriptive validation outputs. This first version does not choose risk thresholds, remove observations, run a permutation test, or rerun XDS.",
            "",
            "Key Parameters",
            "--------------",
            f"s0={parameters['score']['s0']} A^-1, sigma_c={parameters['score']['sigma_c']} A^-1, r_cut={parameters['score']['r_cut']} A^-1",
            f"worker_count={parameters['worker_count_resolved']}, chunk_size={parameters['chunk_size']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(output_dir: Path, observations: pd.DataFrame, geometry_audit: pd.DataFrame) -> None:
    """Write standard diagnostic figures."""

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    _plot_risk_distribution(fig_dir / "risk_distribution.png", observations)
    _plot_risk_disagreement(fig_dir / "risk_disagreement.png", observations)
    _plot_risk_intensity(fig_dir / "risk_intensity.png", observations)
    _plot_disagreement_intensity(fig_dir / "disagreement_intensity.png", observations)
    _plot_geometry_zcal(fig_dir / "geometry_zcal.png", observations, geometry_audit)
    _plot_geometry_zcal_vs_zcal(fig_dir / "geometry_zcal_vs_zcal.png", observations)
    _plot_geometry_batch(fig_dir / "geometry_zcal_by_batch.png", observations)
    _plot_geometry_xy(fig_dir / "geometry_xy.png", geometry_audit)


def _plot_risk_distribution(path: Path, observations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, column in zip(axes, ("E_target", "R_env", "S_risk"), strict=True):
        values = observations[column].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(values, bins=40, color="#206a5d", alpha=0.85)
        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel("observations")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_risk_disagreement(path: Path, observations: pd.DataFrame) -> None:
    valid = observations["disagreement_valid"].astype(bool)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(observations.loc[valid, "S_risk"], observations.loc[valid, "relative_disagreement"], s=8, alpha=0.5)
    ax.set_xlabel("S_risk")
    ax.set_ylabel("relative disagreement D_i")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_risk_intensity(path: Path, observations: pd.DataFrame) -> None:
    valid = observations["disagreement_valid"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(observations.loc[valid, "reference_intensity"], observations.loc[valid, "S_risk"], s=8, alpha=0.5)
    axes[0].set_xlabel("leave-one-out reference intensity")
    axes[0].set_ylabel("S_risk")
    axes[1].scatter(observations.loc[valid, "reference_I_over_SIGMA"], observations.loc[valid, "S_risk"], s=8, alpha=0.5)
    axes[1].set_xlabel("leave-one-out reference I/sigma")
    axes[1].set_ylabel("S_risk")
    for ax in axes:
        positive_x = ax.collections[0].get_offsets()[:, 0]
        if np.all(positive_x[np.isfinite(positive_x)] > 0):
            ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_disagreement_intensity(path: Path, observations: pd.DataFrame) -> None:
    valid = observations["disagreement_valid"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(observations.loc[valid, "reference_intensity"], observations.loc[valid, "relative_disagreement"], s=8, alpha=0.5)
    axes[0].set_xlabel("leave-one-out reference intensity")
    axes[0].set_ylabel("relative disagreement D_i")
    axes[1].scatter(observations.loc[valid, "reference_I_over_SIGMA"], observations.loc[valid, "relative_disagreement"], s=8, alpha=0.5)
    axes[1].set_xlabel("leave-one-out reference I/sigma")
    axes[1].set_ylabel("relative disagreement D_i")
    for ax in axes:
        positive_x = ax.collections[0].get_offsets()[:, 0]
        if np.all(positive_x[np.isfinite(positive_x)] > 0):
            ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_geometry_zcal(path: Path, observations: pd.DataFrame, audit: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if "z_residual_frames" in observations:
        values = observations["z_residual_frames"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(values, bins=40, alpha=0.75, label="selected geometry")
    else:
        for source, group in audit.groupby("geometry_source", sort=True):
            values = group["z_residual"].replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(values, bins=40, alpha=0.5, label=source)
    ax.set_xlabel("Zpred - ZCAL (frames)")
    ax.set_ylabel("observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_geometry_zcal_vs_zcal(path: Path, observations: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if {"ZCAL", "z_residual_frames"} <= set(observations.columns):
        ax.scatter(observations["ZCAL"], observations["z_residual_frames"], s=6, alpha=0.45)
    ax.set_xlabel("ZCAL")
    ax.set_ylabel("Zpred - ZCAL (frames)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_geometry_batch(path: Path, observations: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if {"geometry_batch", "z_residual_frames"} <= set(observations.columns):
        labels = sorted(str(value) for value in observations["geometry_batch"].dropna().unique())
        data = [
            observations.loc[observations["geometry_batch"].astype(str) == label, "z_residual_frames"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
            for label in labels
        ]
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel("INTEGRATE batch")
    ax.set_ylabel("Zpred - ZCAL (frames)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_geometry_xy(path: Path, audit: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for source, group in audit.groupby("geometry_source", sort=True):
        axes[0].scatter(group["x_residual"], group["y_residual"], s=8, alpha=0.5, label=source)
        axes[1].scatter(group["x_residual"], group["z_residual"], s=8, alpha=0.5, label=source)
    axes[0].set_xlabel("XCAL residual (px)")
    axes[0].set_ylabel("YCAL residual (px)")
    axes[1].set_xlabel("XCAL residual (px)")
    axes[1].set_ylabel("Z residual (frames)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

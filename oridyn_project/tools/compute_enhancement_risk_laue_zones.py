"""Synthetic enhancement-only feed-in risk maps for oriented Laue zones.

This standalone prototype computes a multibeam-inspired two-step feed-in score:

    S_enh(g) = sum_q E(q) * source_strength(q) * coupling_strength(g - q)

It is not a full multibeam dynamical diffraction solver. It ignores phases,
thickness oscillations, depletion, and back-coupling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oridyn.geometry import normalize_vector, reciprocal_matrix_from_cell
from oridyn.hkl_generation import allowed_by_centering
from oridyn.stream_parser import UnitCell


MAX_SEARCH_GRID_POINTS = 2_500_000
TOP_PATH_TARGETS = 50
TOP_PATHS_PER_TARGET = 20
SOURCE_WEIGHT_EPS = 1e-90
COUPLING_WEIGHT_EPS = 1e-12
SIMPLE_PATH_SUMMARY_TARGETS = (
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (2, 0, 0),
    (0, 2, 0),
    (2, 1, 0),
    (1, 2, 0),
    (2, 2, 0),
)


def cell_to_real_basis(cell: UnitCell) -> np.ndarray:
    """Return direct-space basis vectors as columns in angstrom."""

    alpha = np.deg2rad(cell.alpha)
    beta = np.deg2rad(cell.beta)
    gamma = np.deg2rad(cell.gamma)
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) < 1e-12:
        raise ValueError("Unit-cell gamma is too close to 0 or 180 degrees.")

    a_vec = np.asarray([cell.a, 0.0, 0.0], dtype=float)
    b_vec = np.asarray([cell.b * np.cos(gamma), cell.b * sin_gamma, 0.0], dtype=float)
    c_x = cell.c * np.cos(beta)
    c_y = cell.c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    c_z = np.sqrt(max(cell.c * cell.c - c_x * c_x - c_y * c_y, 0.0))
    c_vec = np.asarray([c_x, c_y, c_z], dtype=float)
    return np.column_stack([a_vec, b_vec, c_vec])


def reciprocal_basis(real_basis: np.ndarray) -> np.ndarray:
    """Return crystallographic reciprocal basis columns without 2*pi."""

    basis = np.asarray(real_basis, dtype=float)
    if basis.shape != (3, 3):
        raise ValueError("real_basis must have shape (3, 3).")
    return np.linalg.inv(basis).T


def hkl_to_G(hkl: tuple[int, int, int] | np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    """Map Miller indices to a reciprocal vector."""

    return np.asarray(reciprocal, dtype=float) @ np.asarray(hkl, dtype=float)


def laue_zone_index(hkl: tuple[int, int, int], uvw: tuple[int, int, int]) -> int:
    """Return the Laue-zone label h*u + k*v + l*w."""

    return int(np.dot(np.asarray(hkl, dtype=int), np.asarray(uvw, dtype=int)))


def excitation_error(G: np.ndarray, wavelength_angstrom: float, beam_direction: np.ndarray) -> np.ndarray:
    """Return the requested excitation-error-like value in inverse angstrom."""

    vectors = np.asarray(G, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, 3)
    k_norm = 1.0 / float(wavelength_angstrom)
    k_in = normalize_vector(beam_direction) * k_norm
    return (np.sum((vectors + k_in[None, :]) ** 2, axis=1) - k_norm**2) / (2.0 * k_norm)


def excitation_weight(s: np.ndarray, s_max: float) -> np.ndarray:
    """Use s_max as both excitation width and plotting inclusion cutoff."""

    scale = max(float(s_max), 1e-12)
    return np.exp(-0.5 * (np.asarray(s, dtype=float) / scale) ** 2)


def proxy_strength(G_norm: np.ndarray, scale: float) -> np.ndarray:
    """Smooth low-order proxy for source or coupling strength."""

    s = max(float(scale), 1e-12)
    return np.exp(-((np.asarray(G_norm, dtype=float) / s) ** 2))


def automatic_hkl_bounds(cell: UnitCell, d_min: float) -> tuple[int, int, int]:
    """Derive conservative HKL bounds from |h| <= |G|max * |a|."""

    qmax = 1.0 / float(d_min)
    direct_norms = np.linalg.norm(cell_to_real_basis(cell), axis=0)
    bounds = np.ceil(qmax * direct_norms).astype(int) + 1
    return tuple(int(x) for x in bounds)


def generate_reflection_table(cell: UnitCell, d_min: float, d_max: float) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    """Generate all candidate HKLs satisfying d_min <= d <= d_max."""

    if d_min <= 0.0 or d_max <= 0.0:
        raise ValueError("d-min and d-max must be positive.")
    if d_min > d_max:
        raise ValueError("d-min must be smaller than or equal to d-max.")

    reciprocal = reciprocal_matrix_from_cell(cell)
    hmax, kmax, lmax = automatic_hkl_bounds(cell, d_min)
    n_grid = (2 * hmax + 1) * (2 * kmax + 1) * (2 * lmax + 1)
    if n_grid > MAX_SEARCH_GRID_POINTS:
        raise ValueError(
            f"Automatic HKL search grid has {n_grid:,} points, above the internal safety limit "
            f"of {MAX_SEARCH_GRID_POINTS:,}. Increase d-min or narrow the cell search."
        )

    qmin = 1.0 / float(d_max)
    qmax = 1.0 / float(d_min)
    records: list[dict[str, float | int]] = []
    for h in range(-hmax, hmax + 1):
        for k in range(-kmax, kmax + 1):
            for l in range(-lmax, lmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if not allowed_by_centering(h, k, l, cell.centering):
                    continue
                G = hkl_to_G((h, k, l), reciprocal)
                G_norm = float(np.linalg.norm(G))
                if qmin <= G_norm <= qmax:
                    records.append(
                        {
                            "h": h,
                            "k": k,
                            "l": l,
                            "d_spacing": 1.0 / G_norm,
                            "Gx": float(G[0]),
                            "Gy": float(G[1]),
                            "Gz": float(G[2]),
                            "G_norm": G_norm,
                        }
                    )

    if not records:
        raise ValueError("No reflections matched the requested d-spacing range.")
    table = pd.DataFrame.from_records(records).sort_values(["G_norm", "h", "k", "l"]).reset_index(drop=True)
    return table, (hmax, kmax, lmax)


def detector_plane_axes(zone_axis_real_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal detector-plane axes perpendicular to the incident beam."""

    beam = normalize_vector(zone_axis_real_vector)
    x_ref = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(x_ref, beam))) > 0.95:
        x_ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
    x_axis = normalize_vector(x_ref - float(np.dot(x_ref, beam)) * beam)
    y_axis = normalize_vector(np.cross(beam, x_axis))
    return x_axis, y_axis


def project_to_detector_plane(G: np.ndarray, zone_axis_real_vector: np.ndarray) -> tuple[float, float]:
    """Project one reciprocal vector onto the detector plane."""

    x_axis, y_axis = detector_plane_axes(zone_axis_real_vector)
    return float(np.dot(G, x_axis)), float(np.dot(G, y_axis))


def compute_enhancement_scores(
    table: pd.DataFrame,
    cell: UnitCell,
    reciprocal: np.ndarray,
    uvw: tuple[int, int, int],
    wavelength_angstrom: float,
    s_max: float,
) -> tuple[pd.DataFrame, dict[tuple[int, int, int], int], np.ndarray, np.ndarray, np.ndarray]:
    """Compute enhancement-only feed-in scores for generated reflections."""

    hkls = table[["h", "k", "l"]].to_numpy(dtype=int)
    G = table[["Gx", "Gy", "Gz"]].to_numpy(dtype=float)
    G_norm = table["G_norm"].to_numpy(dtype=float)
    direct = cell_to_real_basis(cell)
    zone_axis_real = direct @ np.asarray(uvw, dtype=float)
    s = excitation_error(G, wavelength_angstrom, zone_axis_real)
    E = excitation_weight(s, s_max)

    min_nonzero_G = float(np.min(G_norm[G_norm > 0.0]))
    G0 = 3.0 * min_nonzero_G
    W = proxy_strength(G_norm, G0)
    source_weight = E * W
    coupling_weight = W

    hkl_min = np.min(hkls, axis=0)
    hkl_max = np.max(hkls, axis=0)
    grid_shape = tuple(int(hkl_max[axis] - hkl_min[axis] + 1) for axis in range(3))
    target_index_grid = np.full(grid_shape, -1, dtype=np.int32)
    target_coords = tuple((hkls[:, axis] - hkl_min[axis]).astype(int) for axis in range(3))
    target_index_grid[target_coords] = np.arange(len(hkls), dtype=np.int32)
    lookup: dict[tuple[int, int, int], int] = {}
    for idx, hkl in enumerate(hkls):
        lookup[tuple(int(x) for x in hkl)] = idx

    raw_scores = np.zeros(len(table), dtype=float)
    n_paths = np.zeros(len(table), dtype=int)
    squared_contrib_sums = np.zeros(len(table), dtype=float)
    effective_paths = np.zeros(len(table), dtype=float)

    active_source_indices = np.flatnonzero(source_weight > SOURCE_WEIGHT_EPS)
    active_coupling_indices = np.flatnonzero(coupling_weight > COUPLING_WEIGHT_EPS)
    coupling_hkls = hkls[active_coupling_indices]
    coupling_G = G[active_coupling_indices]
    coupling_norm = G_norm[active_coupling_indices]
    active_coupling_weight = coupling_weight[active_coupling_indices]

    for q_idx in active_source_indices:
        q_hkl = hkls[q_idx]
        q_G = G[q_idx]
        q_norm = G_norm[q_idx]
        candidate_targets = coupling_hkls + q_hkl[None, :]
        in_bounds = np.all((candidate_targets >= hkl_min[None, :]) & (candidate_targets <= hkl_max[None, :]), axis=1)
        if not np.any(in_bounds):
            continue

        candidate_targets = candidate_targets[in_bounds]
        r_G = coupling_G[in_bounds]
        r_norm = coupling_norm[in_bounds]
        r_weight = active_coupling_weight[in_bounds]
        coords = tuple((candidate_targets[:, axis] - hkl_min[axis]).astype(int) for axis in range(3))
        target_indices = target_index_grid[coords]
        exists = target_indices >= 0
        if not np.any(exists):
            continue

        target_indices = target_indices[exists]
        r_G = r_G[exists]
        r_norm = r_norm[exists]
        r_weight = r_weight[exists]
        target_G = G[target_indices]
        target_norm = G_norm[target_indices]
        dot_q = target_G @ q_G
        dot_r = np.einsum("ij,ij->i", r_G, target_G)
        valid = (q_norm < target_norm) & (r_norm < target_norm) & (dot_q > 0.0) & (dot_r > 0.0)
        if not np.any(valid):
            continue

        target_indices = target_indices[valid]
        contributions = source_weight[q_idx] * r_weight[valid]
        np.add.at(raw_scores, target_indices, contributions)
        np.add.at(squared_contrib_sums, target_indices, contributions * contributions)
        np.add.at(n_paths, target_indices, 1)

    nonzero_path_mask = squared_contrib_sums > 0.0
    effective_paths[nonzero_path_mask] = (raw_scores[nonzero_path_mask] ** 2) / squared_contrib_sums[nonzero_path_mask]

    positive_global = raw_scores[raw_scores > 0.0]
    p95_global = float(np.quantile(positive_global, 0.95)) if positive_global.size else 1.0
    norm_global = raw_scores / max(p95_global, 1e-12)

    x_axis, y_axis = detector_plane_axes(zone_axis_real)
    out = table.copy()
    out["laue_zone"] = [laue_zone_index(tuple(int(x) for x in hkl), uvw) for hkl in hkls]
    norm_zone = np.zeros(len(out), dtype=float)
    for _zone, zone_group in out.groupby("laue_zone", sort=False):
        zone_idx = zone_group.index.to_numpy(dtype=int)
        positive_zone = raw_scores[zone_idx][raw_scores[zone_idx] > 0.0]
        p95_zone = float(np.quantile(positive_zone, 0.95)) if positive_zone.size else 1.0
        norm_zone[zone_idx] = raw_scores[zone_idx] / max(p95_zone, 1e-12)
    out["x_proj"] = G @ x_axis
    out["y_proj"] = G @ y_axis
    out["excitation_error_s"] = s
    out["abs_excitation_error_s"] = np.abs(s)
    out["target_excitation_Eg"] = E
    out["S_enh_raw"] = raw_scores
    out["S_enh_norm_global"] = norm_global
    out["S_enh_norm_zone"] = norm_zone
    out["S_enh_norm"] = norm_zone
    out["n_paths"] = n_paths
    out["effective_n_paths"] = effective_paths
    out["included_in_diffraction_plot"] = out["abs_excitation_error_s"] <= float(s_max)
    # Display-only contrast normalization for plotted spots; keep raw and full-zone
    # normalizations for quantitative comparisons across Laue zones.
    out["S_enh_plot_norm_zone"] = np.nan
    plotted_mask = out["included_in_diffraction_plot"].to_numpy(dtype=bool)
    for _zone, zone_group in out.loc[plotted_mask].groupby("laue_zone", sort=False):
        zone_idx = zone_group.index.to_numpy(dtype=int)
        positive_zone = raw_scores[zone_idx][raw_scores[zone_idx] > 0.0]
        p95_zone = float(np.quantile(positive_zone, 0.95)) if positive_zone.size else 1.0
        denominator = p95_zone if p95_zone > 0.0 else 1.0
        out.loc[zone_idx, "S_enh_plot_norm_zone"] = raw_scores[zone_idx] / denominator
    ordered = [
        "h",
        "k",
        "l",
        "laue_zone",
        "d_spacing",
        "Gx",
        "Gy",
        "Gz",
        "G_norm",
        "x_proj",
        "y_proj",
        "excitation_error_s",
        "abs_excitation_error_s",
        "target_excitation_Eg",
        "S_enh_raw",
        "S_enh_norm_global",
        "S_enh_norm_zone",
        "S_enh_norm",
        "S_enh_plot_norm_zone",
        "n_paths",
        "effective_n_paths",
        "included_in_diffraction_plot",
    ]
    out = out[ordered]
    return out, lookup, source_weight, coupling_weight, E


def collect_top_paths(
    scored: pd.DataFrame,
    lookup: dict[tuple[int, int, int], int],
    source_weight: np.ndarray,
    coupling_weight: np.ndarray,
    E: np.ndarray,
) -> pd.DataFrame:
    """Collect top valid build-up paths for strong and simple low-index targets."""

    hkls = scored[["h", "k", "l"]].to_numpy(dtype=int)
    G = scored[["Gx", "Gy", "Gz"]].to_numpy(dtype=float)
    G_norm = scored["G_norm"].to_numpy(dtype=float)
    active_source_indices = np.flatnonzero(source_weight > SOURCE_WEIGHT_EPS)
    target_indices = list(scored.sort_values("S_enh_raw", ascending=False).head(TOP_PATH_TARGETS).index.to_numpy(dtype=int))
    for target in SIMPLE_PATH_SUMMARY_TARGETS:
        target_idx = lookup.get(target)
        if target_idx is not None:
            target_indices.append(int(target_idx))
    target_indices = list(dict.fromkeys(target_indices))
    rows: list[dict[str, float | int]] = []
    for target_idx in target_indices:
        g = tuple(int(x) for x in hkls[target_idx])
        g_G = G[target_idx]
        g_norm = G_norm[target_idx]
        candidates: list[tuple[float, tuple[int, int, int], tuple[int, int, int]]] = []
        for q_idx in active_source_indices:
            q = tuple(int(x) for x in hkls[q_idx])
            r = (g[0] - q[0], g[1] - q[1], g[2] - q[2])
            r_idx = lookup.get(r)
            if r_idx is None:
                continue
            if coupling_weight[r_idx] <= COUPLING_WEIGHT_EPS:
                continue
            if G_norm[q_idx] >= g_norm or G_norm[r_idx] >= g_norm:
                continue
            if float(np.dot(G[q_idx], g_G)) <= 0.0 or float(np.dot(G[r_idx], g_G)) <= 0.0:
                continue
            contribution = float(source_weight[q_idx] * coupling_weight[r_idx])
            if contribution > 0.0:
                candidates.append((contribution, q, r))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for rank, (contribution, q, r) in enumerate(candidates[:TOP_PATHS_PER_TARGET], start=1):
            q_idx = lookup[q]
            r_idx = lookup[r]
            rows.append(
                {
                    "target_h": g[0],
                    "target_k": g[1],
                    "target_l": g[2],
                    "q_h": q[0],
                    "q_k": q[1],
                    "q_l": q[2],
                    "r_h": r[0],
                    "r_k": r[1],
                    "r_l": r[2],
                    "E_q": float(E[q_idx]),
                    "source_weight_q": float(source_weight[q_idx]),
                    "coupling_strength_r": float(coupling_weight[r_idx]),
                    "contribution": contribution,
                    "rank_within_target": rank,
                }
            )
    return pd.DataFrame.from_records(rows)


def _marker_sizes(E: pd.Series) -> np.ndarray:
    values = E.to_numpy(dtype=float)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    scaled = np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)
    return 18.0 + 95.0 * scaled


def _plot_limits(plotted: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    if plotted.empty:
        return (-1.0, 1.0), (-1.0, 1.0)
    x = plotted["x_proj"].to_numpy(dtype=float)
    y = plotted["y_proj"].to_numpy(dtype=float)
    x_pad = max((float(np.max(x)) - float(np.min(x))) * 0.06, 0.02)
    y_pad = max((float(np.max(y)) - float(np.min(y))) * 0.06, 0.02)
    return (float(np.min(x)) - x_pad, float(np.max(x)) + x_pad), (float(np.min(y)) - y_pad, float(np.max(y)) + y_pad)


def _plot_excitation(plotted: pd.DataFrame, outdir: Path, xlim: tuple[float, float], ylim: tuple[float, float]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    if plotted.empty:
        ax.text(0.5, 0.5, "No reflections satisfy |s| <= s_max", ha="center", va="center", transform=ax.transAxes)
    else:
        zones = sorted(int(x) for x in plotted["laue_zone"].unique())
        zone_to_color = {zone: plt.get_cmap("tab20")(idx % 20) for idx, zone in enumerate(zones)}
        colors = [zone_to_color[int(zone)] for zone in plotted["laue_zone"]]
        ax.scatter(
            plotted["x_proj"],
            plotted["y_proj"],
            s=_marker_sizes(plotted["target_excitation_Eg"]),
            c=colors,
            edgecolors="none",
            linewidths=0.0,
            alpha=0.86,
        )
        if len(zones) <= 12:
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="", color=zone_to_color[zone], label=str(zone))
                for zone in zones
            ]
            ax.legend(handles=handles, title="Laue zone", fontsize=8, title_fontsize=8, loc="best")
    _finish_pattern_axes(ax, xlim, ylim, "Excited reflections (size ~ Eg)")
    return _save_plot(fig, outdir / "diffraction_excitation")


def _plot_enhancement(plotted: pd.DataFrame, outdir: Path, xlim: tuple[float, float], ylim: tuple[float, float]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    if plotted.empty:
        ax.text(0.5, 0.5, "No reflections satisfy |s| <= s_max", ha="center", va="center", transform=ax.transAxes)
    else:
        vmax = _color_vmax(plotted["S_enh_plot_norm_zone"])
        sc = ax.scatter(
            plotted["x_proj"],
            plotted["y_proj"],
            s=44,
            c=plotted["S_enh_plot_norm_zone"],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            edgecolors="none",
            alpha=0.9,
        )
        fig.colorbar(sc, ax=ax, label="Plotted-zone-normalized enhancement score")
    _finish_pattern_axes(ax, xlim, ylim, "Enhancement risk (display-normalized by plotted Laue zone)")
    return _save_plot(fig, outdir / "diffraction_enhancement")


def _plot_combined(plotted: pd.DataFrame, outdir: Path, xlim: tuple[float, float], ylim: tuple[float, float]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    if plotted.empty:
        ax.text(0.5, 0.5, "No reflections satisfy |s| <= s_max", ha="center", va="center", transform=ax.transAxes)
    else:
        vmax = _color_vmax(plotted["S_enh_plot_norm_zone"])
        sc = ax.scatter(
            plotted["x_proj"],
            plotted["y_proj"],
            s=_marker_sizes(plotted["target_excitation_Eg"]),
            c=plotted["S_enh_plot_norm_zone"],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            edgecolors="none",
            linewidths=0.0,
            alpha=0.9,
        )
        fig.colorbar(sc, ax=ax, label="Plotted-zone-normalized enhancement score")
    _finish_pattern_axes(ax, xlim, ylim, "Excitedness + enhancement risk (size ~ Eg, display-normalized color)")
    return _save_plot(fig, outdir / "diffraction_combined")


def _color_vmax(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return 1.0
    return max(float(finite.quantile(0.99)), 1e-12)


def _finish_pattern_axes(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float], title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x_proj")
    ax.set_ylabel("y_proj")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="#9a9a9a", linewidth=0.5, alpha=0.55)
    ax.axvline(0.0, color="#9a9a9a", linewidth=0.5, alpha=0.55)


def _save_plot(fig: plt.Figure, stem: Path) -> list[Path]:
    fig.tight_layout()
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_diffraction_patterns(scored: pd.DataFrame, outdir: Path) -> list[Path]:
    """Write the three matched diffraction-pattern visualizations."""

    plotted = scored.loc[scored["included_in_diffraction_plot"]].copy()
    xlim, ylim = _plot_limits(plotted)
    paths: list[Path] = []
    paths.extend(_plot_excitation(plotted, outdir, xlim, ylim))
    paths.extend(_plot_enhancement(plotted, outdir, xlim, ylim))
    paths.extend(_plot_combined(plotted, outdir, xlim, ylim))
    return paths


def write_outputs(scored: pd.DataFrame, outdir: Path, plot_paths: list[Path], top_paths: pd.DataFrame | None) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    score_path = outdir / "enhancement_risk_scores.csv"
    scored.to_csv(score_path, index=False)
    output_paths = [score_path, *plot_paths]
    if top_paths is not None:
        top_path = outdir / "enhancement_top_paths.csv"
        top_paths.to_csv(top_path, index=False)
        output_paths.append(top_path)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", nargs=6, type=float, required=True, metavar=("a", "b", "c", "alpha", "beta", "gamma"))
    parser.add_argument("--uvw", nargs=3, type=int, required=True, metavar=("u", "v", "w"))
    parser.add_argument("--lambda-ang", type=float, required=True, help="Incident wavelength in angstrom.")
    parser.add_argument("--d-min", type=float, required=True, help="High-resolution d-spacing limit in angstrom.")
    parser.add_argument("--d-max", type=float, required=True, help="Low-resolution d-spacing limit in angstrom.")
    parser.add_argument("--s-max", type=float, required=True, help="Excitation-error cutoff and Gaussian width.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-top-paths", action="store_true", help="Also write top paths for the strongest targets.")
    return parser.parse_args()


def _format_stats(values: pd.Series) -> str:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return "n/a"
    return f"{float(finite.min()):.6g} / {float(finite.median()):.6g} / {float(finite.max()):.6g}"


def _matching_target_rows(table: pd.DataFrame, target: tuple[int, int, int]) -> pd.DataFrame:
    h, k, l = target
    return table.loc[(table["h"] == h) & (table["k"] == k) & (table["l"] == l)]


def _print_simple_top_paths(scored: pd.DataFrame, top_paths: pd.DataFrame | None) -> None:
    if top_paths is None:
        return

    print("Top paths for simple low-index targets:")
    for target in SIMPLE_PATH_SUMMARY_TARGETS:
        score_rows = _matching_target_rows(scored, target)
        if score_rows.empty:
            continue
        score_row = score_rows.iloc[0]
        path_rows = _matching_target_rows(
            top_paths.rename(columns={"target_h": "h", "target_k": "k", "target_l": "l"}), target
        ).head(3)
        target_label = f"({target[0]}, {target[1]}, {target[2]})"
        score_text = (
            f"S_enh_raw={float(score_row['S_enh_raw']):.6g}, "
            f"S_enh_norm_zone={float(score_row['S_enh_norm_zone']):.6g}"
        )
        if path_rows.empty:
            print(f"  {target_label}: no valid paths ({score_text})")
            continue
        parts = []
        for row in path_rows.itertuples(index=False):
            parts.append(
                f"q=({int(row.q_h)},{int(row.q_k)},{int(row.q_l)}) + "
                f"r=({int(row.r_h)},{int(row.r_k)},{int(row.r_l)}) "
                f"contrib={float(row.contribution):.3g}"
            )
        print(f"  {target_label}: {score_text}; " + "; ".join(parts))


def _print_plotted_zone_stats(plotted: pd.DataFrame) -> None:
    print("Per plotted Laue-zone enhancement stats:")
    if plotted.empty:
        print("  none")
        return
    for zone, group in plotted.groupby("laue_zone", sort=True):
        print(
            f"  zone {int(zone)}: count={len(group)}, "
            f"S_enh_raw={_format_stats(group['S_enh_raw'])}, "
            f"S_enh_norm_zone={_format_stats(group['S_enh_norm_zone'])}, "
            f"S_enh_plot_norm_zone={_format_stats(group['S_enh_plot_norm_zone'])}"
        )


def print_summary(
    scored: pd.DataFrame,
    generated_count: int,
    output_paths: list[Path],
    top_paths: pd.DataFrame | None,
) -> None:
    plotted = scored.loc[scored["included_in_diffraction_plot"]].copy()
    zone_counts = plotted["laue_zone"].value_counts().sort_index()
    zone_summary = ", ".join(f"{int(zone)}:{int(count)}" for zone, count in zone_counts.items()) or "none"

    print(f"Generated reflections: {generated_count}")
    print(f"Scored reflections: {len(scored)}")
    print(f"Included in diffraction plots: {len(plotted)}")
    print(f"Laue-zone counts for plotted reflections: {zone_summary}")
    print(f"Plotted target_excitation_Eg min/median/max: {_format_stats(plotted['target_excitation_Eg'])}")
    print(f"Plotted S_enh_norm_zone min/median/max: {_format_stats(plotted['S_enh_norm_zone'])}")
    _print_plotted_zone_stats(plotted)
    _print_simple_top_paths(scored, top_paths)
    print("Output paths written:")
    for path in output_paths:
        print(f"  {path}")


def main() -> int:
    args = parse_args()
    outdir: Path = args.outdir
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{outdir} exists and is not empty; pass --overwrite to reuse it.")
    outdir.mkdir(parents=True, exist_ok=True)

    cell = UnitCell(*args.cell)
    if tuple(args.uvw) == (0, 0, 0):
        raise SystemExit("--uvw must be a non-zero zone axis.")

    reciprocal = reciprocal_matrix_from_cell(cell)
    generated, _bounds = generate_reflection_table(cell, args.d_min, args.d_max)
    scored, lookup, source_weight, coupling_weight, E = compute_enhancement_scores(
        generated,
        cell,
        reciprocal,
        tuple(int(x) for x in args.uvw),
        args.lambda_ang,
        args.s_max,
    )

    plot_paths = plot_diffraction_patterns(scored, outdir)
    top_paths = collect_top_paths(scored, lookup, source_weight, coupling_weight, E) if args.write_top_paths else None
    output_paths = write_outputs(scored, outdir, plot_paths, top_paths)
    print_summary(scored, len(generated), output_paths, top_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

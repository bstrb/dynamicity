"""Interactive HTML visualization for orientation-dependent dynamical peak predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .pipeline import AnalysisResult


def _downsample_frame_table(
    frame_table: pd.DataFrame,
    score_column: str,
    max_points_per_frame: int,
) -> pd.DataFrame:
    if frame_table.empty or frame_table.shape[0] <= max_points_per_frame:
        return frame_table
    return frame_table.sort_values(score_column, ascending=False).head(max_points_per_frame).copy()


def _nearest_neighbor_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size <= 1:
        return np.full_like(x, 1e9, dtype=float)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1)


def _compute_visual_probability(
    frame_table: pd.DataFrame,
    score_column: str,
    gx_center_x: float,
    gx_center_y: float,
    detector_nx: int,
    detector_ny: int,
    radial_penalty_strength: float = 0.55,
    sg_penalty_strength: float = 0.35,
) -> np.ndarray:
    """Convert raw per-reflection score to a robust display probability in [0, 1].

    This leaves the underlying scientific score untouched but stabilizes visual
    behavior by reducing outlier dominance and edge-driven artifacts.
    """
    if frame_table.empty:
        return np.zeros(0, dtype=float)

    raw = frame_table[score_column].to_numpy(dtype=float)
    q10, q90 = np.percentile(raw, [10, 90])
    if np.isclose(q10, q90):
        base = np.full_like(raw, 0.5)
    else:
        base = np.clip((raw - q10) / (q90 - q10), 0.0, 1.0)

    x = frame_table["x_px"].to_numpy(dtype=float)
    y = frame_table["y_px"].to_numpy(dtype=float)
    radius = np.sqrt((x - gx_center_x) ** 2 + (y - gx_center_y) ** 2)
    r_scale = 0.55 * np.sqrt(float(detector_nx) ** 2 + float(detector_ny) ** 2)
    radial_weight = np.exp(-radial_penalty_strength * (radius / max(r_scale, 1e-9)) ** 2)

    if "sg_invA" in frame_table.columns:
        sg = np.abs(frame_table["sg_invA"].to_numpy(dtype=float))
        sg_q80 = np.percentile(sg, 80)
        sg_scale = max(sg_q80, 1e-9)
        sg_weight = np.exp(-sg_penalty_strength * (sg / sg_scale) ** 2)
    else:
        sg_weight = np.ones_like(base)

    prob = base * radial_weight * sg_weight
    if np.allclose(prob.max(), prob.min()):
        return np.full_like(prob, 0.5)

    return np.clip((prob - prob.min()) / (prob.max() - prob.min()), 0.0, 1.0)


def _gaussian_surface_from_frame(
    frame_table: pd.DataFrame,
    probabilities: np.ndarray,
    detector_nx: int,
    detector_ny: int,
    grid_size: int,
    min_sigma_px: float,
    max_sigma_px: float,
    overlap_mode: str = "max",
    color_locality_gamma: float = 1.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, float(detector_nx), grid_size)
    y = np.linspace(0.0, float(detector_ny), grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)
    color_num = np.zeros_like(X, dtype=float)
    color_den = np.zeros_like(X, dtype=float)

    if frame_table.empty:
        return X, Y, Z, color_num

    probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)

    x0 = frame_table["x_px"].to_numpy(dtype=float)
    y0 = frame_table["y_px"].to_numpy(dtype=float)

    # High dynamical probability -> wider and taller Gaussian peaks.
    sigma = min_sigma_px + probs * (max_sigma_px - min_sigma_px)
    amplitude = 0.25 + 0.95 * probs

    # Prevent strongly overlapping broad peaks from merging into a single ridge.
    nn_dist = _nearest_neighbor_distance(x0, y0)
    sigma_cap = np.clip(0.40 * nn_dist, min_sigma_px, max_sigma_px)
    sigma = np.minimum(sigma, sigma_cap)

    if overlap_mode not in {"sum", "max"}:
        raise ValueError("overlap_mode must be 'sum' or 'max'.")

    for xi, yi, si, ai, pi in zip(x0, y0, sigma, amplitude, probs):
        Gi = ai * np.exp(-(((X - xi) ** 2 + (Y - yi) ** 2) / (2.0 * si**2)))
        # Localize color near each apex: color fades more quickly than height.
        local_factor = np.clip((Gi / max(ai, 1e-12)) ** color_locality_gamma, 0.0, 1.0)

        if overlap_mode == "sum":
            Z += Gi
            color_num += Gi * pi * local_factor
            color_den += Gi
        else:
            mask = Gi > Z
            Z = np.where(mask, Gi, Z)
            color_num = np.where(mask, pi * local_factor, color_num)
            color_den = np.where(mask, 1.0, color_den)

    prob_field = np.divide(color_num, np.maximum(color_den, 1e-12))
    return X, Y, Z, prob_field


def _mask_surface_tails(
    Z: np.ndarray,
    P: np.ndarray,
    tail_clip_fraction: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide low-amplitude tails so color stays local around peak tops."""
    z_max = float(np.max(Z)) if Z.size else 0.0
    if z_max <= 0.0:
        return Z, P

    mask = Z >= (tail_clip_fraction * z_max)
    Z_masked = np.where(mask, Z, np.nan)
    P_masked = np.where(mask, P, np.nan)
    return Z_masked, P_masked


def export_orientation_peaks_html(
    result: AnalysisResult,
    output_html: str | Path,
    score_column: str = "S_comb",
    frame_metric: str = "S_MB",
    max_points_per_frame: int = 700,
    surface_grid_size: int = 130,
    min_sigma_px: float = 8.0,
    max_sigma_px: float = 40.0,
    overlap_mode: str = "max",
    surface_opacity: float = 0.78,
    color_locality_gamma: float = 1.8,
    tail_clip_fraction: float = 0.03,
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    """Export an interactive HTML report of predicted dynamical peaks across orientations.

    The output contains:
    - frame-wise metric trend (left panel)
    - detector scatter for each frame with animation slider (right panel)
    - animated 3D Gaussian peak landscape for each frame (bottom panel)
    """

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "plotly is required for HTML export. Install with: pip install plotly"
        ) from exc

    frame_summary = result.frame_summary.sort_values("frame").reset_index(drop=True)
    reflections = result.reflections_long.copy()

    if frame_summary.empty:
        raise ValueError("frame_summary is empty; nothing to visualize.")
    if reflections.empty:
        raise ValueError("reflections_long is empty; no predicted peaks to render.")
    if score_column not in reflections.columns:
        raise ValueError(f"score column '{score_column}' not found in reflections_long.")
    if frame_metric not in frame_summary.columns:
        raise ValueError(f"frame metric '{frame_metric}' not found in frame_summary.")

    frame_ids = frame_summary["frame"].to_numpy(dtype=int)
    first_frame = int(frame_ids[0])

    cmin = float(reflections[score_column].min())
    cmax = float(reflections[score_column].max())
    if np.isclose(cmin, cmax):
        cmax = cmin + 1e-9

    size_norm = (reflections[score_column] - cmin) / (cmax - cmin)
    reflections["marker_size"] = 7.0 + 18.0 * np.clip(size_norm, 0.0, 1.0)

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.42, 0.58],
        row_heights=[0.46, 0.54],
        horizontal_spacing=0.08,
        vertical_spacing=0.11,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "scene", "colspan": 2}, None]],
        subplot_titles=(
            f"Frame trend: {frame_metric}",
            "Predicted dynamical peaks on detector",
            "3D Gaussian peaks (width/color by dynamical probability)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=frame_summary["frame_number"],
            y=frame_summary[frame_metric],
            mode="lines+markers",
            name=frame_metric,
            line=dict(color="#1f5aa6", width=2.5),
            marker=dict(size=5),
            hovertemplate="Frame %{x}<br>" + frame_metric + ": %{y:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    first_summary = frame_summary.loc[frame_summary["frame"] == first_frame].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[int(first_summary["frame_number"])],
            y=[float(first_summary[frame_metric])],
            mode="markers",
            name="Current frame",
            marker=dict(size=14, color="#e53935", symbol="diamond"),
            hovertemplate="Current frame %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    def frame_scatter(frame_idx: int) -> pd.DataFrame:
        ft = reflections[reflections["frame"] == frame_idx].copy()
        ft = _downsample_frame_table(ft, score_column=score_column, max_points_per_frame=max_points_per_frame)
        if not ft.empty:
            ft["dyn_prob_vis"] = _compute_visual_probability(
                ft,
                score_column=score_column,
                gx_center_x=float(result.gxparm.orgx_px),
                gx_center_y=float(result.gxparm.orgy_px),
                detector_nx=int(result.gxparm.detector_nx),
                detector_ny=int(result.gxparm.detector_ny),
            )
        return ft

    ft0 = frame_scatter(first_frame)
    fig.add_trace(
        go.Scattergl(
            x=ft0["x_px"],
            y=ft0["y_px"],
            mode="markers",
            name=score_column,
            marker=dict(
                size=ft0["marker_size"],
                color=ft0["dyn_prob_vis"],
                colorscale="RdYlGn_r",
                cmin=0.0,
                cmax=1.0,
                opacity=0.84,
                line=dict(width=0),
                colorbar=dict(title="dyn_prob_vis", x=1.0),
            ),
            customdata=np.stack(
                [
                    ft0["h"].to_numpy(dtype=int),
                    ft0["k"].to_numpy(dtype=int),
                    ft0["l"].to_numpy(dtype=int),
                    ft0["sg_invA"].to_numpy(dtype=float),
                    ft0[score_column].to_numpy(dtype=float),
                ],
                axis=1,
            ) if not ft0.empty else None,
            hovertemplate=(
                "x=%{x:.1f}px, y=%{y:.1f}px"
                "<br>(h,k,l)=(%{customdata[0]},%{customdata[1]},%{customdata[2]})"
                f"<br>{score_column}=%{{customdata[4]:.4g}}"
                "<br>dyn_prob_vis=%{marker.color:.3f}"
                "<br>sg=%{customdata[3]:.3e} 1/A"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    X0, Y0, Z0, P0 = _gaussian_surface_from_frame(
        ft0,
        probabilities=ft0["dyn_prob_vis"].to_numpy(dtype=float),
        detector_nx=result.gxparm.detector_nx,
        detector_ny=result.gxparm.detector_ny,
        grid_size=surface_grid_size,
        min_sigma_px=min_sigma_px,
        max_sigma_px=max_sigma_px,
        overlap_mode=overlap_mode,
        color_locality_gamma=color_locality_gamma,
    )
    Z0, P0 = _mask_surface_tails(Z0, P0, tail_clip_fraction=tail_clip_fraction)

    fig.add_trace(
        go.Surface(
            x=X0,
            y=Y0,
            z=Z0,
            surfacecolor=P0,
            colorscale=[
                [0.0, "#1b9e3e"],
                [0.5, "#f5c242"],
                [1.0, "#cb2a2a"],
            ],
            cmin=0.0,
            cmax=1.0,
            opacity=surface_opacity,
            showscale=False,
            hovertemplate=(
                "x=%{x:.1f}px, y=%{y:.1f}px"
                "<br>gaussian intensity=%{z:.3f}"
                "<br>dyn. probability=%{surfacecolor:.3f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    if result.xds_input is not None and result.xds_input.untrusted_rectangles:
        for x1, x2, y1, y2 in result.xds_input.untrusted_rectangles:
            fig.add_shape(
                type="rect",
                xref="x2",
                yref="y2",
                x0=float(x1),
                x1=float(x2),
                y0=float(y1),
                y1=float(y2),
                line=dict(color="rgba(255,255,255,0.5)", width=1),
                fillcolor="rgba(120,120,120,0.24)",
            )

    frames = []
    slider_steps = []

    for frame_idx in frame_ids:
        summary_row = frame_summary.loc[frame_summary["frame"] == frame_idx].iloc[0]
        ft = frame_scatter(int(frame_idx))

        scatter_kwargs: dict[str, object] = dict(
            x=ft["x_px"],
            y=ft["y_px"],
            marker=dict(size=ft["marker_size"], color=ft[score_column]),
        )

        if ft.empty:
            scatter_kwargs["customdata"] = None
        else:
            scatter_kwargs["customdata"] = np.stack(
                [
                    ft["h"].to_numpy(dtype=int),
                    ft["k"].to_numpy(dtype=int),
                    ft["l"].to_numpy(dtype=int),
                    ft["sg_invA"].to_numpy(dtype=float),
                    ft[score_column].to_numpy(dtype=float),
                ],
                axis=1,
            )
            scatter_kwargs["marker"]["color"] = ft["dyn_prob_vis"]

        Xf, Yf, Zf, Pf = _gaussian_surface_from_frame(
            ft,
            probabilities=ft["dyn_prob_vis"].to_numpy(dtype=float),
            detector_nx=result.gxparm.detector_nx,
            detector_ny=result.gxparm.detector_ny,
            grid_size=surface_grid_size,
            min_sigma_px=min_sigma_px,
            max_sigma_px=max_sigma_px,
            overlap_mode=overlap_mode,
            color_locality_gamma=color_locality_gamma,
        )
        Zf, Pf = _mask_surface_tails(Zf, Pf, tail_clip_fraction=tail_clip_fraction)

        frame_data = [
            go.Scatter(x=[int(summary_row["frame_number"])], y=[float(summary_row[frame_metric])]),
            go.Scattergl(**scatter_kwargs),
            go.Surface(x=Xf, y=Yf, z=Zf, surfacecolor=Pf),
        ]

        frame_name = str(int(frame_idx))
        frames.append(go.Frame(data=frame_data, name=frame_name, traces=[1, 2, 3]))
        slider_steps.append(
            {
                "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                "label": str(int(summary_row["frame_number"])),
                "method": "animate",
            }
        )

    fig.frames = frames

    fig.update_xaxes(title_text="Frame number", row=1, col=1)
    fig.update_yaxes(title_text=frame_metric, row=1, col=1)

    fig.update_xaxes(title_text="Detector x / px", row=1, col=2)
    fig.update_yaxes(
        title_text="Detector y / px",
        row=1,
        col=2,
        autorange="reversed",
        scaleanchor="x2",
        scaleratio=1,
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            "Bloch-wave analyser: predicted dynamical peak distribution by orientation"
        ),
        scene=dict(
            xaxis=dict(title="Detector x / px", range=[0, float(result.gxparm.detector_nx)]),
            yaxis=dict(title="Detector y / px", range=[float(result.gxparm.detector_ny), 0]),
            zaxis=dict(title="Gaussian intensity"),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.32),
            camera=dict(eye=dict(x=1.6, y=1.35, z=0.75)),
        ),
        height=1020,
        width=1400,
        legend=dict(orientation="h", y=1.03, x=0.0),
        margin=dict(l=60, r=30, t=80, b=40),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.60,
                "y": 1.11,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 120, "redraw": False},
                                "transition": {"duration": 40},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.04,
                "y": -0.02,
                "len": 0.92,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 15, "b": 10},
                "currentvalue": {"prefix": "Frame: ", "font": {"size": 14}},
                "steps": slider_steps,
            }
        ],
    )

    output_path = Path(output_html).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=include_plotlyjs, full_html=True)
    return output_path


def export_single_frame_gaussian_html(
    result: AnalysisResult,
    frame_number: int,
    output_html: str | Path,
    score_column: str = "S_comb",
    surface_grid_size: int = 170,
    min_sigma_px: float = 8.0,
    max_sigma_px: float = 42.0,
    overlap_mode: str = "max",
    surface_opacity: float = 0.78,
    color_locality_gamma: float = 1.8,
    tail_clip_fraction: float = 0.03,
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    """Export one frame as a 3D Gaussian diffraction landscape.

    Parameters
    ----------
    frame_number:
        One-based frame number (matching frame_number in frame_summary).
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "plotly is required for HTML export. Install with: pip install plotly"
        ) from exc

    reflections = result.reflections_long.copy()
    if reflections.empty:
        raise ValueError("reflections_long is empty; no peaks to render.")
    if score_column not in reflections.columns:
        raise ValueError(f"score column '{score_column}' not found in reflections_long.")

    frame_table = reflections[reflections["frame_number"] == int(frame_number)].copy()
    if frame_table.empty:
        available_min = int(reflections["frame_number"].min())
        available_max = int(reflections["frame_number"].max())
        raise ValueError(
            f"Frame {frame_number} has no peaks. Available frame numbers: {available_min}..{available_max}."
        )

    dyn_prob = _compute_visual_probability(
        frame_table,
        score_column=score_column,
        gx_center_x=float(result.gxparm.orgx_px),
        gx_center_y=float(result.gxparm.orgy_px),
        detector_nx=int(result.gxparm.detector_nx),
        detector_ny=int(result.gxparm.detector_ny),
    )

    X, Y, Z, P = _gaussian_surface_from_frame(
        frame_table,
        probabilities=dyn_prob,
        detector_nx=result.gxparm.detector_nx,
        detector_ny=result.gxparm.detector_ny,
        grid_size=surface_grid_size,
        min_sigma_px=min_sigma_px,
        max_sigma_px=max_sigma_px,
        overlap_mode=overlap_mode,
        color_locality_gamma=color_locality_gamma,
    )
    Z, P = _mask_surface_tails(Z, P, tail_clip_fraction=tail_clip_fraction)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=P,
            colorscale=[
                [0.0, "#1b9e3e"],
                [0.5, "#f5c242"],
                [1.0, "#cb2a2a"],
            ],
            cmin=0.0,
            cmax=1.0,
            colorbar=dict(title="Dynamical probability"),
            opacity=surface_opacity,
            hovertemplate=(
                "x=%{x:.1f}px, y=%{y:.1f}px"
                "<br>gaussian intensity=%{z:.3f}"
                "<br>dyn. probability=%{surfacecolor:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            f"Frame {int(frame_number)}: 3D Gaussian diffraction pattern "
            f"(sigma/color from robust dynamical probability using {score_column})"
        ),
        width=1240,
        height=860,
        margin=dict(l=40, r=30, t=70, b=30),
        scene=dict(
            xaxis=dict(title="Detector x / px", range=[0, float(result.gxparm.detector_nx)]),
            yaxis=dict(title="Detector y / px", range=[float(result.gxparm.detector_ny), 0]),
            zaxis=dict(title="Gaussian intensity"),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.35),
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.80)),
        ),
    )

    output_path = Path(output_html).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=include_plotlyjs, full_html=True)
    return output_path

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_thickness_long(base: Path) -> pd.DataFrame:
    files = [
        base / 'lta_frame0050_pred_t1_100nm.csv',
        base / 'lta_frame0050_pred_t2_200nm.csv',
        base / 'lta_frame0050_pred_t3_350nm.csv',
        base / 'lta_frame0050_pred_t4_600nm.csv',
    ]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError('Missing prediction tables: ' + ', '.join(missing))
    parts = [pd.read_csv(f) for f in files]
    long = pd.concat(parts, ignore_index=True)
    long['h'] = long['h'].astype(int)
    long['k'] = long['k'].astype(int)
    long['l'] = long['l'].astype(int)
    return long


def _load_selected(selected_csv: Path) -> pd.DataFrame:
    sel = pd.read_csv(selected_csv)
    sel['h'] = sel['h'].astype(int)
    sel['k'] = sel['k'].astype(int)
    sel['l'] = sel['l'].astype(int)
    return sel


def _load_detector_frame(frame_number: int = 50) -> tuple[pd.DataFrame, object]:
    # Reuse Bloch analyser pipeline to obtain detector positions for this frame.
    bloch_root = Path('/Users/xiaodong/Desktop/dynamicity/bloch_wave_analyser_project')
    if str(bloch_root) not in sys.path:
        sys.path.insert(0, str(bloch_root))

    from src.parsers import parse_composition, parse_gxparm, parse_integrate_hkl, load_optional_xds_inp
    from src.pipeline import AnalysisConfig, run_analysis

    gxparm = parse_gxparm(bloch_root / 'data' / 'LTA_t1' / 'GXPARM.XDS')
    integrate = parse_integrate_hkl(bloch_root / 'data' / 'LTA_t1' / 'INTEGRATE.HKL')
    xds_inp = load_optional_xds_inp(bloch_root / 'data' / 'LTA_t1' / 'XDS.INP')
    composition = parse_composition('24 Si, 48 O')

    cfg = AnalysisConfig(mode='proxy', dmin_angstrom=0.6, dmax_angstrom=50.0, excitation_tolerance_invA=1.5e-3)
    result = run_analysis(gxparm=gxparm, integrate=integrate, composition=composition, xds_input=xds_inp, config=cfg)

    frame_idx = int(frame_number) - 1
    ft = result.frame_table(frame_idx).copy()
    ft['h'] = ft['h'].astype(int)
    ft['k'] = ft['k'].astype(int)
    ft['l'] = ft['l'].astype(int)
    return ft, gxparm


def _plot_curves(ax, long_df: pd.DataFrame, subset: pd.DataFrame, title: str, color: str) -> None:
    for r in subset.itertuples(index=False):
        g = long_df[(long_df['h'] == int(r.h)) & (long_df['k'] == int(r.k)) & (long_df['l'] == int(r.l))].copy()
        g = g.sort_values('thickness_nm')
        y = g['intensity'].to_numpy(dtype=float)
        if y.size == 0:
            continue
        ref = y[0] if abs(y[0]) > 1e-12 else 1e-12
        y_rel = 100.0 * (y / ref - 1.0)
        label = f"({int(r.h)} {int(r.k)} {int(r.l)})"
        ax.plot(g['thickness_nm'], y_rel, marker='o', linewidth=2.0, label=label, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel('Thickness (nm)')
    ax.set_ylabel('Intensity change (%)')
    ax.grid(alpha=0.25)
    ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8, ncol=2)


def main() -> None:
    out_dir = Path('/Users/xiaodong/Desktop/dynamicity/abtem_single_orientation/results/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_base = Path('/Users/xiaodong/Desktop/dynamicity/bloch_wave_analyser_project/analysis_output')
    selected_csv = out_dir / 'thickness_variation_story_selected.csv'

    long_df = _load_thickness_long(pred_base)
    selected = _load_selected(selected_csv)
    strong = selected[selected['group'] == 'high_variation'].head(6).copy()
    stable = selected[selected['group'] == 'stable'].head(6).copy()

    frame_table, gxparm = _load_detector_frame(frame_number=50)

    # Join selected groups with detector positions
    strong_pos = strong.merge(frame_table[['h', 'k', 'l', 'x_px', 'y_px']], on=['h', 'k', 'l'], how='left')
    stable_pos = stable.merge(frame_table[['h', 'k', 'l', 'x_px', 'y_px']], on=['h', 'k', 'l'], how='left')

    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.30)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Detector-like map
    ax0.set_facecolor('black')
    ax0.scatter(frame_table['x_px'], frame_table['y_px'], s=8, c='white', alpha=0.12, linewidths=0)

    s_ok = strong_pos.dropna(subset=['x_px', 'y_px'])
    t_ok = stable_pos.dropna(subset=['x_px', 'y_px'])

    ax0.scatter(s_ok['x_px'], s_ok['y_px'], s=42, c='#ff5252', label='Strongly varying', edgecolors='none')
    ax0.scatter(t_ok['x_px'], t_ok['y_px'], s=42, c='#4fc3f7', label='Relatively stable', edgecolors='none')

    for r in s_ok.itertuples(index=False):
        ax0.text(r.x_px + 8, r.y_px + 8, f"({int(r.h)} {int(r.k)} {int(r.l)})", color='#ff8a80', fontsize=7)
    for r in t_ok.itertuples(index=False):
        ax0.text(r.x_px + 8, r.y_px - 10, f"({int(r.h)} {int(r.k)} {int(r.l)})", color='#81d4fa', fontsize=7)

    ax0.set_xlim(0, gxparm.detector_nx)
    ax0.set_ylim(gxparm.detector_ny, 0)
    ax0.set_aspect('equal')
    ax0.set_title('Frame 50: selected reflections on detector map')
    ax0.set_xlabel('Detector x / px')
    ax0.set_ylabel('Detector y / px')
    ax0.legend(loc='lower right', fontsize=8)

    _plot_curves(ax1, long_df, strong, 'Strongly varying reflections', color='#ff5252')
    _plot_curves(ax2, long_df, stable, 'Relatively stable reflections', color='#4fc3f7')

    fig.suptitle('Thickness effect at one orientation: where and how reflections differ', y=1.02)
    fig.tight_layout()

    out_png = out_dir / 'thickness_variation_with_detector_map.png'
    fig.savefig(out_png, dpi=240, bbox_inches='tight')
    plt.close(fig)

    print(f'figure={out_png}')


if __name__ == '__main__':
    main()

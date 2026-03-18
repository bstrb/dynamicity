from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_shell_balanced(df: pd.DataFrame, n_total: int, ascending: bool) -> pd.DataFrame:
    ranked = df.sort_values('max_abs_rel_change_pct', ascending=ascending).copy()
    per_shell = max(1, n_total // 4)
    picks = []
    for shell in ['inner', 'mid_low', 'mid_high', 'outer']:
        s = ranked[ranked['shell'] == shell].head(per_shell)
        if not s.empty:
            picks.append(s)
    out = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=df.columns)
    if len(out) < n_total:
        used = set(zip(out['h'], out['k'], out['l'])) if len(out) else set()
        extra = ranked[[ (int(h), int(k), int(l)) not in used for h, k, l in zip(ranked['h'], ranked['k'], ranked['l']) ]]
        out = pd.concat([out, extra.head(n_total - len(out))], ignore_index=True)
    return out.head(n_total)


def _exclude_hkls(df: pd.DataFrame, hkls: set[tuple[int, int, int]]) -> pd.DataFrame:
    if not hkls:
        return df.copy()
    keep = [
        (int(h), int(k), int(l)) not in hkls
        for h, k, l in zip(df['h'].to_numpy(), df['k'].to_numpy(), df['l'].to_numpy())
    ]
    return df.loc[keep].copy()


def _load_bloch_frame_and_thickness(frame_number: int = 50) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    bloch_root = Path('/Users/xiaodong/Desktop/dynamicity/bloch_wave_analyser_project')
    if str(bloch_root) not in sys.path:
        sys.path.insert(0, str(bloch_root))

    from src.parsers import parse_composition, parse_gxparm, parse_integrate_hkl, load_optional_xds_inp
    from src.pipeline import AnalysisConfig, run_analysis

    data_dir = bloch_root / 'data' / 'LTA_t1'
    gxparm = parse_gxparm(data_dir / 'GXPARM.XDS')
    integrate = parse_integrate_hkl(data_dir / 'INTEGRATE.HKL')
    xds_inp = load_optional_xds_inp(data_dir / 'XDS.INP')
    composition = parse_composition('24 Si, 48 O')

    thickness_nm = np.arange(10.0, 501.0, 10.0, dtype=float)
    cfg = AnalysisConfig(
        mode='thickness',
        thickness_nm=thickness_nm,
        dmin_angstrom=0.6,
        dmax_angstrom=50.0,
        excitation_tolerance_invA=1.5e-3,
    )
    result = run_analysis(gxparm=gxparm, integrate=integrate, composition=composition, xds_input=xds_inp, config=cfg)

    frame_idx = int(frame_number) - 1
    frame_table = result.frame_table(frame_idx).copy()
    frame_table['h'] = frame_table['h'].astype(int)
    frame_table['k'] = frame_table['k'].astype(int)
    frame_table['l'] = frame_table['l'].astype(int)

    if result.thickness_long is None or result.thickness_long.empty:
        raise RuntimeError('No thickness_long table returned by Bloch analysis.')

    thick = result.thickness_long.copy()
    thick = thick[thick['frame_number'] == int(frame_number)].copy()
    thick['h'] = thick['h'].astype(int)
    thick['k'] = thick['k'].astype(int)
    thick['l'] = thick['l'].astype(int)

    return frame_table, thick, gxparm


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot HKL thickness-variation story for 10-500 nm range.')
    parser.add_argument('--n-per-group', type=int, default=10, help='Number of HKLs per group (high variation / stable).')
    args = parser.parse_args()

    n_per_group = max(4, int(args.n_per_group))

    out_dir = Path('/Users/xiaodong/Desktop/dynamicity/abtem_single_orientation/results_10to500_step10/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_table, thick, gxparm = _load_bloch_frame_and_thickness(frame_number=50)

    # Join detector coordinates and distance from beam center for shell balancing.
    geo = frame_table[['h', 'k', 'l', 'x_px', 'y_px']].drop_duplicates().copy()
    geo['radius_px'] = np.sqrt((geo['x_px'] - gxparm.orgx_px) ** 2 + (geo['y_px'] - gxparm.orgy_px) ** 2)

    rows = []
    for (h, k, l), g in thick.groupby(['h', 'k', 'l']):
        g = g.sort_values('thickness_nm')
        y = g['intensity'].to_numpy(dtype=float)
        if y.size < 3:
            continue
        ref = y[0] if abs(y[0]) > 1e-12 else 1e-12
        y_rel = 100.0 * (y / ref - 1.0)
        rows.append(
            {
                'h': int(h),
                'k': int(k),
                'l': int(l),
                'i_mean': float(np.mean(np.abs(y))),
                'i_min': float(np.min(y)),
                'i_max': float(np.max(y)),
                'range': float(np.max(y) - np.min(y)),
                'max_abs_rel_change_pct': float(np.max(np.abs(y_rel))),
            }
        )

    stats = pd.DataFrame(rows).merge(geo, on=['h', 'k', 'l'], how='inner')
    if stats.empty:
        raise RuntimeError('No HKL reflections available after joining geometry and thickness tables.')

    # Use an intensity floor + beam-center exclusion to avoid unstable tiny-denominator effects.
    stats = stats[stats['radius_px'] >= 25.0].copy()
    i_floor = float(stats['i_mean'].quantile(0.35))
    eligible = stats[stats['i_mean'] >= i_floor].copy()
    if len(eligible) < 16:
        eligible = stats.copy()

    eligible['shell'] = pd.qcut(
        eligible['radius_px'],
        q=4,
        labels=['inner', 'mid_low', 'mid_high', 'outer'],
        duplicates='drop',
    )

    fallback_pool = stats.copy()
    fallback_pool['shell'] = pd.qcut(
        fallback_pool['radius_px'],
        q=4,
        labels=['inner', 'mid_low', 'mid_high', 'outer'],
        duplicates='drop',
    )

    top_var = _pick_shell_balanced(eligible, n_total=n_per_group, ascending=False)
    var_hkls = {(int(r.h), int(r.k), int(r.l)) for r in top_var.itertuples(index=False)}

    stable_pool = _exclude_hkls(eligible, var_hkls)
    if len(stable_pool) < n_per_group:
        stable_pool = _exclude_hkls(fallback_pool, var_hkls)
    top_stable = _pick_shell_balanced(stable_pool, n_total=n_per_group, ascending=True)

    selected_all = pd.concat([top_var, top_stable], ignore_index=True)
    y_all = []
    for r in selected_all.itertuples(index=False):
        g = thick[(thick['h'] == int(r.h)) & (thick['k'] == int(r.k)) & (thick['l'] == int(r.l))].sort_values('thickness_nm')
        y = g['intensity'].to_numpy(dtype=float)
        ref = y[0] if abs(y[0]) > 1e-12 else 1e-12
        y_all.extend((100.0 * (y / ref - 1.0)).tolist())

    lo = float(np.percentile(y_all, 2)) if y_all else -10.0
    hi = float(np.percentile(y_all, 98)) if y_all else 10.0
    pad = 0.08 * max(hi - lo, 1.0)
    y_lim = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(18.5, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # HKL-based detector map at the chosen orientation.
    ax0.set_facecolor('black')
    ax0.scatter(frame_table['x_px'], frame_table['y_px'], s=8, c='white', alpha=0.12, linewidths=0)
    ax0.scatter(top_var['x_px'], top_var['y_px'], s=46, c='#ff5252', label='Strongly varying HKLs', edgecolors='none')
    ax0.scatter(top_stable['x_px'], top_stable['y_px'], s=46, c='#4fc3f7', label='Relatively stable HKLs', edgecolors='none')

    for i, r in enumerate(top_var.itertuples(index=False), start=1):
        ax0.text(r.x_px + 8, r.y_px + 8, f'V{i}', color='#ff8a80', fontsize=8)
    for i, r in enumerate(top_stable.itertuples(index=False), start=1):
        ax0.text(r.x_px + 8, r.y_px - 10, f'S{i}', color='#81d4fa', fontsize=8)

    ax0.set_xlim(0, gxparm.detector_nx)
    ax0.set_ylim(gxparm.detector_ny, 0)
    ax0.set_aspect('equal')
    ax0.set_title('Frame 50 detector map (HKL-indexed)')
    ax0.set_xlabel('Detector x / px')
    ax0.set_ylabel('Detector y / px')
    ax0.legend(loc='lower right', fontsize=8)

    for i, r in enumerate(top_var.itertuples(index=False), start=1):
        g = thick[(thick['h'] == int(r.h)) & (thick['k'] == int(r.k)) & (thick['l'] == int(r.l))].sort_values('thickness_nm')
        y = g['intensity'].to_numpy(dtype=float)
        ref = y[0] if abs(y[0]) > 1e-12 else 1e-12
        y_rel = 100.0 * (y / ref - 1.0)
        ax1.plot(g['thickness_nm'], y_rel, linewidth=2.0, marker='o', markersize=3, label=f"V{i} ({int(r.h)} {int(r.k)} {int(r.l)})")
    ax1.set_title('Strongly varying HKL reflections')
    ax1.set_xlabel('Thickness (nm)')
    ax1.set_ylabel('Intensity change from 10 nm (%)')
    ax1.set_ylim(*y_lim)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7, ncol=2)

    for i, r in enumerate(top_stable.itertuples(index=False), start=1):
        g = thick[(thick['h'] == int(r.h)) & (thick['k'] == int(r.k)) & (thick['l'] == int(r.l))].sort_values('thickness_nm')
        y = g['intensity'].to_numpy(dtype=float)
        ref = y[0] if abs(y[0]) > 1e-12 else 1e-12
        y_rel = 100.0 * (y / ref - 1.0)
        ax2.plot(g['thickness_nm'], y_rel, linewidth=2.0, marker='o', markersize=3, label=f"S{i} ({int(r.h)} {int(r.k)} {int(r.l)})")
    ax2.set_title('Relatively stable HKL reflections')
    ax2.set_xlabel('Thickness (nm)')
    ax2.set_ylabel('Intensity change from 10 nm (%)')
    ax2.set_ylim(*y_lim)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f'HKL-based thickness sensitivity at one orientation (frame 50), 10-500 nm (n={n_per_group} per group)',
        y=1.02,
    )
    fig.tight_layout()

    out_png = out_dir / 'thickness_variation_story_hkl_10to500.png'
    fig.savefig(out_png, dpi=240, bbox_inches='tight')
    plt.close(fig)

    selected = pd.concat(
        [top_var.assign(group='high_variation'), top_stable.assign(group='stable')],
        ignore_index=True,
    )
    out_csv = out_dir / 'thickness_variation_story_hkl_10to500_selected.csv'
    selected.to_csv(out_csv, index=False)

    print(f'figure={out_png}')
    print(f'selected_csv={out_csv}')


if __name__ == '__main__':
    main()

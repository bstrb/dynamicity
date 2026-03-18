from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_shell_balanced(group: pd.DataFrame, n_total: int, ascending: bool) -> pd.DataFrame:
    picks = []
    per_shell = max(1, n_total // 4)
    ranked = group.sort_values('max_abs_rel_change_pct', ascending=ascending)
    for shell in ['inner_lowres', 'mid_lowres', 'mid_highres', 'outer_highres']:
        s = ranked[ranked['shell'] == shell].head(per_shell)
        if not s.empty:
            picks.append(s)
    combined = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=group.columns)
    if len(combined) < n_total:
        taken = set(combined['hkl']) if 'hkl' in combined.columns else set()
        extra = ranked[~ranked['hkl'].isin(taken)].head(n_total - len(combined))
        combined = pd.concat([combined, extra], ignore_index=True)
    return combined.head(n_total)


def main() -> None:
    root = Path('/Users/xiaodong/Desktop/dynamicity/bloch_wave_analyser_project/analysis_output')
    files = [
        root / 'lta_frame0050_pred_t1_100nm.csv',
        root / 'lta_frame0050_pred_t2_200nm.csv',
        root / 'lta_frame0050_pred_t3_350nm.csv',
        root / 'lta_frame0050_pred_t4_600nm.csv',
    ]

    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError('Missing input tables: ' + ', '.join(missing))

    rows = []
    for f in files:
        df = pd.read_csv(f)
        rows.append(df[['h', 'k', 'l', 'thickness_nm', 'intensity']].copy())
    long = pd.concat(rows, ignore_index=True)
    long['hkl'] = list(zip(long['h'].astype(int), long['k'].astype(int), long['l'].astype(int)))

    stats = (
        long.groupby('hkl', as_index=False)
        .agg(
            h=('h', 'first'),
            k=('k', 'first'),
            l=('l', 'first'),
            n_t=('thickness_nm', 'nunique'),
            i_min=('intensity', 'min'),
            i_max=('intensity', 'max'),
            i_mean=('intensity', 'mean'),
            i_std=('intensity', 'std'),
        )
    )
    stats = stats[stats['n_t'] >= 4].copy()
    stats['range'] = stats['i_max'] - stats['i_min']
    stats['cv'] = stats['i_std'].fillna(0.0) / np.maximum(stats['i_mean'], 1e-12)
    stats = stats.sort_values('range', ascending=False).reset_index(drop=True)

    # Compute the same variation metric that will be shown in the plot:
    # max absolute relative change (%) from the first thickness point.
    rel_rows = []
    for r in stats.itertuples(index=False):
        g = long[(long['h'] == int(r.h)) & (long['k'] == int(r.k)) & (long['l'] == int(r.l))].copy()
        g = g.sort_values('thickness_nm')
        y = g['intensity'].to_numpy(dtype=float)
        if y.size == 0:
            max_abs_rel = np.nan
        else:
            ref = float(y[0])
            ref = ref if abs(ref) > 1e-12 else 1e-12
            y_rel = 100.0 * (y / ref - 1.0)
            max_abs_rel = float(np.max(np.abs(y_rel)))
        rel_rows.append(max_abs_rel)
    stats['max_abs_rel_change_pct'] = rel_rows

    stats['hkl_norm'] = np.sqrt(stats['h'] ** 2 + stats['k'] ** 2 + stats['l'] ** 2)
    stats['mean_abs_int'] = stats['i_mean'].abs()

    # Relative-change metrics can over-rank weak, high-index peaks.
    # Keep only reflections above a modest intensity floor before selecting exemplars.
    intensity_floor = float(stats['mean_abs_int'].quantile(0.35))
    eligible = stats[stats['mean_abs_int'] >= intensity_floor].copy()
    if len(eligible) < 12:
        eligible = stats.copy()

    eligible['shell'] = pd.qcut(
        eligible['hkl_norm'],
        q=4,
        labels=['inner_lowres', 'mid_lowres', 'mid_highres', 'outer_highres'],
        duplicates='drop',
    )

    # Use the same metric for picking "varying" and "stable" groups.
    # This keeps panel assignment consistent with what is actually displayed.
    stats_rel = eligible.sort_values('max_abs_rel_change_pct', ascending=False).reset_index(drop=True)

    top_var = _pick_shell_balanced(stats_rel, n_total=6, ascending=False)
    top_stable = _pick_shell_balanced(stats_rel, n_total=6, ascending=True)

    def rel_change_percent(y: np.ndarray) -> np.ndarray:
        ref = float(y[0]) if len(y) else 1.0
        ref = ref if abs(ref) > 1e-12 else 1e-12
        return 100.0 * (y / ref - 1.0)

    # Build global y-range from all selected curves so both panels share the same scale.
    y_all = []
    selected_all = pd.concat([top_var, top_stable], ignore_index=True)
    for r in selected_all.itertuples(index=False):
        g = long[(long['h'] == int(r.h)) & (long['k'] == int(r.k)) & (long['l'] == int(r.l))].copy()
        g = g.sort_values('thickness_nm')
        y = g['intensity'].to_numpy(dtype=float)
        y_all.extend(rel_change_percent(y).tolist())

    if y_all:
        lo = float(np.percentile(y_all, 2))
        hi = float(np.percentile(y_all, 98))
        pad = 0.08 * max(hi - lo, 1.0)
        y_limits = (lo - pad, hi + pad)
    else:
        y_limits = (-10.0, 10.0)

    def plot_group(ax, group: pd.DataFrame, title: str) -> None:
        for r in group.itertuples(index=False):
            g = long[(long['h'] == int(r.h)) & (long['k'] == int(r.k)) & (long['l'] == int(r.l))].copy()
            g = g.sort_values('thickness_nm')
            y = g['intensity'].to_numpy(dtype=float)
            y_rel = rel_change_percent(y)
            label = f"({int(r.h)} {int(r.k)} {int(r.l)})"
            ax.plot(g['thickness_nm'], y_rel, marker='o', linewidth=2.0, label=label)

        ax.set_title(title)
        ax.set_xlabel('Thickness (nm)')
        ax.set_ylabel('Intensity change from first thickness (%)')
        ax.set_ylim(*y_limits)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    plot_group(axes[0], top_var, 'Reflections that vary strongly with thickness')
    plot_group(axes[1], top_stable, 'Reflections that stay relatively stable')
    fig.suptitle('Single orientation (frame 50): thickness-sensitive vs stable reflections (intensity-screened, shell-balanced)', y=1.02)
    fig.tight_layout()

    out_dir = Path('/Users/xiaodong/Desktop/dynamicity/abtem_single_orientation/results/figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / 'thickness_variation_story.png'
    fig.savefig(out_png, dpi=240, bbox_inches='tight')
    plt.close(fig)

    out_csv = out_dir / 'thickness_variation_story_selected.csv'
    selected = pd.concat([
        top_var.assign(group='high_variation'),
        top_stable.assign(group='stable'),
    ], ignore_index=True)
    selected.to_csv(out_csv, index=False)

    print(f'figure={out_png}')
    print(f'selected_csv={out_csv}')


if __name__ == '__main__':
    main()

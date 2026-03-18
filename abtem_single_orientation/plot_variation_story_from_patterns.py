from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _display_image(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=float)
    floor = max(float(np.percentile(vals, 1.0)), 1e-20)
    log_img = np.log10(np.maximum(arr, floor))
    vmin = float(np.percentile(log_img[np.isfinite(log_img)], 1.0))
    vmax = float(np.percentile(log_img[np.isfinite(log_img)], 99.9))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return np.zeros_like(arr, dtype=float)
    return np.clip((log_img - vmin) / (vmax - vmin), 0.0, 1.0)


def _local_maxima_mask(arr: np.ndarray, threshold: float) -> np.ndarray:
    c = arr
    neighbors = [
        np.roll(np.roll(c, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    ]
    mask = c > threshold
    for n in neighbors:
        mask &= c >= n

    # Remove wrapped borders introduced by roll.
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return mask


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
        used = set(zip(out['y_px'], out['x_px'])) if len(out) else set()
        extra = ranked[[ (int(y), int(x)) not in used for y, x in zip(ranked['y_px'], ranked['x_px']) ]]
        out = pd.concat([out, extra.head(n_total - len(out))], ignore_index=True)
    return out.head(n_total)


def main() -> None:
    base = Path('/Users/xiaodong/Desktop/dynamicity/abtem_single_orientation/results_10to500_step10')
    index_csv = base / 'pattern_index.csv'
    if not index_csv.exists():
        raise FileNotFoundError(f'Missing index CSV: {index_csv}')

    df = pd.read_csv(index_csv).sort_values('thickness_nm').reset_index(drop=True)
    thickness = df['thickness_nm'].to_numpy(float)
    stack = []
    for p in df['npy_path']:
        stack.append(np.load(p).astype(float))
    stack = np.stack(stack, axis=0)

    # Use a mid-thickness frame as reference for spot finding.
    ref_idx = int(np.argmin(np.abs(thickness - 250.0)))
    ref = stack[ref_idx]
    h, w = ref.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    # Candidate spots: local maxima above robust high percentile, excluding central beam disk.
    thr = float(np.percentile(ref, 99.5))
    cand_mask = _local_maxima_mask(ref, threshold=thr)

    yy, xx = np.indices(ref.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cand_mask &= rr >= 14.0

    ys, xs = np.where(cand_mask)
    if ys.size < 20:
        thr = float(np.percentile(ref, 99.0))
        cand_mask = _local_maxima_mask(ref, threshold=thr)
        cand_mask &= rr >= 14.0
        ys, xs = np.where(cand_mask)

    if ys.size == 0:
        raise RuntimeError('No candidate diffraction spots found in pattern stack.')

    records = []
    for y, x in zip(ys, xs):
        series = stack[:, y, x]
        ref_i = float(series[0]) if abs(float(series[0])) > 1e-14 else 1e-14
        rel = 100.0 * (series / ref_i - 1.0)
        mean_i = float(np.mean(np.abs(series)))
        records.append(
            {
                'y_px': int(y),
                'x_px': int(x),
                'radius_px': float(rr[y, x]),
                'i_mean': mean_i,
                'i_min': float(np.min(series)),
                'i_max': float(np.max(series)),
                'range': float(np.max(series) - np.min(series)),
                'max_abs_rel_change_pct': float(np.max(np.abs(rel))),
            }
        )

    spots = pd.DataFrame(records)

    # Filter weak spots to avoid huge relative swings from tiny denominators.
    i_floor = float(spots['i_mean'].quantile(0.35))
    eligible = spots[spots['i_mean'] >= i_floor].copy()
    if len(eligible) < 16:
        eligible = spots.copy()

    eligible['shell'] = pd.qcut(
        eligible['radius_px'],
        q=4,
        labels=['inner', 'mid_low', 'mid_high', 'outer'],
        duplicates='drop',
    )

    top_var = _pick_shell_balanced(eligible, n_total=8, ascending=False)
    top_stable = _pick_shell_balanced(eligible, n_total=8, ascending=True)

    def rel_curve(y: int, x: int) -> np.ndarray:
        s = stack[:, y, x]
        ref_s = float(s[0]) if abs(float(s[0])) > 1e-14 else 1e-14
        return 100.0 * (s / ref_s - 1.0)

    y_all = []
    for rec in pd.concat([top_var, top_stable], ignore_index=True).itertuples(index=False):
        y_all.extend(rel_curve(int(rec.y_px), int(rec.x_px)).tolist())
    lo = float(np.percentile(y_all, 2)) if y_all else -10.0
    hi = float(np.percentile(y_all, 98)) if y_all else 10.0
    pad = 0.08 * max(hi - lo, 1.0)
    y_lim = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(16.5, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    img = _display_image(ref)
    ax0.imshow(img, cmap='magma', origin='lower', vmin=0.0, vmax=1.0, aspect='equal')
    ax0.scatter(eligible['x_px'], eligible['y_px'], s=10, c='white', alpha=0.14, linewidths=0)

    ax0.scatter(top_var['x_px'], top_var['y_px'], s=44, c='#ff5252', label='Strongly varying', edgecolors='none')
    ax0.scatter(top_stable['x_px'], top_stable['y_px'], s=44, c='#4fc3f7', label='Relatively stable', edgecolors='none')

    for i, r in enumerate(top_var.itertuples(index=False), start=1):
        ax0.text(r.x_px + 5, r.y_px + 5, f'V{i}', color='#ff8a80', fontsize=8)
    for i, r in enumerate(top_stable.itertuples(index=False), start=1):
        ax0.text(r.x_px + 5, r.y_px - 8, f'S{i}', color='#81d4fa', fontsize=8)

    ax0.set_title('Detector pattern at 250 nm (selected spots)')
    ax0.set_xlabel('x (px)')
    ax0.set_ylabel('y (px)')
    ax0.legend(loc='lower right', fontsize=8)

    for i, r in enumerate(top_var.itertuples(index=False), start=1):
        y = rel_curve(int(r.y_px), int(r.x_px))
        ax1.plot(thickness, y, linewidth=2.0, marker='o', markersize=3, label=f'V{i} ({int(r.x_px)},{int(r.y_px)})')
    ax1.set_title('Strongly varying spots')
    ax1.set_xlabel('Thickness (nm)')
    ax1.set_ylabel('Intensity change from 10 nm (%)')
    ax1.set_ylim(*y_lim)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7, ncol=1)

    for i, r in enumerate(top_stable.itertuples(index=False), start=1):
        y = rel_curve(int(r.y_px), int(r.x_px))
        ax2.plot(thickness, y, linewidth=2.0, marker='o', markersize=3, label=f'S{i} ({int(r.x_px)},{int(r.y_px)})')
    ax2.set_title('Relatively stable spots')
    ax2.set_xlabel('Thickness (nm)')
    ax2.set_ylabel('Intensity change from 10 nm (%)')
    ax2.set_ylim(*y_lim)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7, ncol=1)

    fig.suptitle('10-500 nm thickness series at one orientation: detector spots with high vs low thickness sensitivity', y=1.02)
    fig.tight_layout()

    out_dir = base / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / 'thickness_variation_story_from_patterns_10to500.png'
    fig.savefig(out_png, dpi=240, bbox_inches='tight')
    plt.close(fig)

    selected = pd.concat(
        [top_var.assign(group='high_variation'), top_stable.assign(group='stable')],
        ignore_index=True,
    )
    out_csv = out_dir / 'thickness_variation_story_from_patterns_10to500_selected.csv'
    selected.to_csv(out_csv, index=False)

    print(f'figure={out_png}')
    print(f'selected_csv={out_csv}')


if __name__ == '__main__':
    main()

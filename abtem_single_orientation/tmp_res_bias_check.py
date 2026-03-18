import numpy as np
import pandas as pd
from pathlib import Path

base = Path('/Users/xiaodong/Desktop/dynamicity/bloch_wave_analyser_project/analysis_output')
files = [
    base / 'lta_frame0050_pred_t1_100nm.csv',
    base / 'lta_frame0050_pred_t2_200nm.csv',
    base / 'lta_frame0050_pred_t3_350nm.csv',
    base / 'lta_frame0050_pred_t4_600nm.csv',
]

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
for c in ['h', 'k', 'l']:
    df[c] = df[c].astype(int)

rows = []
for (h, k, l), x in df.groupby(['h', 'k', 'l']):
    x = x.sort_values('thickness_nm')
    y = x['intensity'].to_numpy(float)
    if y.size < 2:
        continue
    ref = y[0] if abs(y[0]) > 1e-12 else 1e-12
    rel = 100 * (y / ref - 1)
    rows.append({
        'h': h,
        'k': k,
        'l': l,
        'hkl_norm': float(np.sqrt(h * h + k * k + l * l)),
        'mean_abs_int': float(abs(y).mean()),
        'max_abs_rel_pct': float(abs(rel).max()),
        'abs_range': float(y.max() - y.min()),
    })

G = pd.DataFrame(rows)
print('n_reflections', len(G))
print('corr(max_abs_rel_pct,hkl_norm)=', round(float(G[['max_abs_rel_pct', 'hkl_norm']].corr().iloc[0, 1]), 4))
print('corr(abs_range,hkl_norm)=', round(float(G[['abs_range', 'hkl_norm']].corr().iloc[0, 1]), 4))
print('corr(max_abs_rel_pct,mean_abs_int)=', round(float(G[['max_abs_rel_pct', 'mean_abs_int']].corr().iloc[0, 1]), 4))
G['shell'] = pd.qcut(G['hkl_norm'], 4, labels=['inner_lowres', 'mid_lowres', 'mid_highres', 'outer_highres'])
print('shell_median_max_abs_rel_pct')
print(G.groupby('shell')['max_abs_rel_pct'].median().round(2).to_string())
print('shell_median_abs_range')
print(G.groupby('shell')['abs_range'].median().round(2).to_string())

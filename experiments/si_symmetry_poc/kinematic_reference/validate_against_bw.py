from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import abtem

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/si_symmetry_poc/kinematic_reference/validation'
ZARR = ROOT / 'experiments/si_symmetry_poc/results/20260924-121554_Si_CollCode51688_paper_cell_1x1x1000_451/bw.zarr'
KIN = ROOT / 'experiments/si_symmetry_poc/kinematic_reference/finite_thickness_kinematic.npz'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    k = np.load(KIN)
    kin = np.asarray(k['i_kin_alpha_z_h'], dtype=float)
    alpha = np.asarray(k['alpha'], dtype=float)
    thickness = np.asarray(k['thickness'], dtype=float)
    hkls = np.asarray(k['hkls'], dtype=int)
    data = abtem.from_zarr(str(ZARR))
    dyn = np.asarray(data.intensities.compute(), dtype=float)
    assert dyn.shape == kin.shape
    nonzero = ~np.all(hkls == 0, axis=1)
    static_strength = np.asarray(k['i_kin0'], dtype=float)
    eligible = np.flatnonzero(nonzero & np.isfinite(static_strength))
    # Ordinary, nonzero reflections: choose separated strong kinematic reflections.
    order = eligible[np.argsort(static_strength[eligible])[::-1]]
    selected = []
    for index in order:
        if all(np.linalg.norm(hkls[index] - hkls[j]) >= 2 for j in selected):
            selected.append(int(index))
        if len(selected) == 6:
            break
    selected = np.asarray(selected, dtype=int)
    pd.DataFrame({'reflection_index': selected, 'h': hkls[selected,0], 'k': hkls[selected,1], 'l': hkls[selected,2], 'I_kin0': static_strength[selected]}).to_csv(OUT/'selected_reflections.csv', index=False)

    rows=[]
    for z_index, z in enumerate(thickness):
        if z > 50: break
        x=kin[:,z_index,selected]; y=dyn[:,z_index,selected]
        mask=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>=0)
        xv=x[mask]; yv=y[mask]
        slope=float(np.dot(xv,yv)/np.dot(xv,xv)) if len(xv) and np.dot(xv,xv)>0 else np.nan
        corr=float(np.corrcoef(xv,yv)[0,1]) if len(xv)>2 and np.std(xv)>0 and np.std(yv)>0 else np.nan
        rows.append({'thickness':z,'n_points':len(xv),'scale_dyn_on_kin':slope,'pearson_shape_corr':corr,'kin_sum':float(xv.sum()),'dyn_sum':float(yv.sum())})
    pd.DataFrame(rows).to_csv(OUT/'thin_limit_scale_shape_summary.csv',index=False)

    for z in (5.0,10.0,25.0,50.0):
        zi=int(np.flatnonzero(thickness==z)[0])
        fig, axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
        for ax,index in zip(axes.ravel(),selected):
            ax.plot(np.rad2deg(alpha),kin[:,zi,index],label='kinematic',lw=1.5)
            ax.plot(np.rad2deg(alpha),dyn[:,zi,index],label='Bloch wave',lw=1.0,alpha=.8)
            h,k_,l=hkls[index]; ax.set_title(f'({h} {k_} {l})'); ax.set_xlabel('alpha (deg)'); ax.set_ylabel('intensity')
        axes[0,0].legend(); fig.suptitle(f'Kinematic vs BW rocking curves, thickness {z:.0f} A'); fig.tight_layout(); fig.savefig(OUT/f'rocking_curves_{int(z):04d}A.png',dpi=180); plt.close(fig)

        x=kin[:,:,selected][:,zi,:].ravel(); y=dyn[:,zi,selected].ravel(); mask=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>=0); x=x[mask]; y=y[mask]
        slope=float(np.dot(x,y)/np.dot(x,x)) if np.dot(x,x)>0 else np.nan
        fig,ax=plt.subplots(figsize=(5,5)); ax.scatter(x,y,s=8,alpha=.45); lim=max(x.max(),y.max()); ax.plot([0,lim],[0,lim],'k--',lw=.8); ax.plot([0,lim],[0,slope*lim],label=f'fit slope={slope:.3g}'); ax.set(xlabel='I_kin',ylabel='I_dyn',title=f'thickness {z:.0f} A'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/f'scatter_dyn_vs_kin_{int(z):04d}A.png',dpi=180); plt.close(fig)

    print('selected HKLs',hkls[selected].tolist())
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()

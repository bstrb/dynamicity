#!/usr/bin/env python3
"""
Compute SRES for reflections by merging XDS-derived reflections with SHELX .fcf

This script prefers using cctbx to map reflections to the asymmetric unit (ASU).
It will try to extract unit-cell and space-group info from the .fcf (CIF) header.

Usage:
  python scripts/compute_sres.py \
    --reflections /path/to/reflections_long.csv \
    --fcf /path/to/shelx/shelx.fcf \
    --output sres_out.csv

If cctbx is available and cell+spacegroup are found in the .fcf, indices
will be mapped to the ASU using cctbx.miller.array.map_to_asu(). Otherwise the
script falls back to a raw h,k,l join.

Requirements: pandas, numpy. Optional: cctbx (iotbx) for ASU mapping.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd


def _to_float(tok):
    if tok in ("?", "."):
        return np.nan
    try:
        return float(tok)
    except Exception:
        return np.nan


def parse_fcf_cif_style(fcf_path: Path):
    """Parse a CIF-style SHELX .fcf reflection loop and header.

    Returns: (hkl_array, Fo2, sigFo2, Fc2, header_dict)
    header_dict may contain keys like 'cell' (6 floats), 'space_group', 'symops'
    """
    text = fcf_path.read_text(encoding='utf-8', errors='ignore').splitlines()

    # First, scan header for cell/spacegroup tokens
    header = {}
    for ln in text:
        s = ln.strip()
        if s.lower().startswith("_cell_length_a"):
            try:
                header['cell_a'] = float(s.split()[1])
            except Exception:
                pass
        if s.lower().startswith("_cell_length_b"):
            try:
                header['cell_b'] = float(s.split()[1])
            except Exception:
                pass
        if s.lower().startswith("_cell_length_c"):
            try:
                header['cell_c'] = float(s.split()[1])
            except Exception:
                pass
        if s.lower().startswith("_cell_angle_alpha"):
            try:
                header['alpha'] = float(s.split()[1])
            except Exception:
                pass
        if s.lower().startswith("_cell_angle_beta"):
            try:
                header['beta'] = float(s.split()[1])
            except Exception:
                pass
        if s.lower().startswith("_cell_angle_gamma"):
            try:
                header['gamma'] = float(s.split()[1])
            except Exception:
                pass
        if s.lower().startswith("_symmetry_space_group_name_h-m") or s.lower().startswith("_space_group_name_h-m"):
            parts = s.split(None, 1)
            if len(parts) > 1:
                header['space_group'] = parts[1].strip().strip('"')
        if s.lower().startswith("_space_group_crystal_system"):
            pass

    # Now parse loops like in merge/plot_fofc.py
    hkl, Fo2, sigFo2, Fc2 = [], [], [], []
    i = 0
    n = len(text)
    while i < n:
        line = text[i].strip()
        if line.lower() == 'loop_':
            i += 1
            headers = []
            while i < n and text[i].lstrip().startswith('_'):
                headers.append(text[i].strip())
                i += 1
            lower = [h.lower() for h in headers]
            colmap = {lower[j]: j for j in range(len(lower))}

            # Also check for symop loop to capture symops if present
            if "_space_group_symop_operation_xyz" in colmap or "_symmetry_equiv_pos_as_xyz" in colmap:
                op_key = "_space_group_symop_operation_xyz" if "_space_group_symop_operation_xyz" in colmap else "_symmetry_equiv_pos_as_xyz"
                op_idx = colmap[op_key]
                symops = []
                while i < n:
                    s = text[i].strip()
                    if not s or s.startswith('#'):
                        i += 1
                        continue
                    if s.lower() == 'loop_' or s.startswith('_') or s.lower().startswith('data_'):
                        break
                    parts = s.split()
                    if len(parts) > op_idx:
                        symops.append(parts[op_idx].strip().strip("'\"") )
                    i += 1
                if symops:
                    header['symops'] = symops
                continue

            required_base = ["_refln_index_h", "_refln_index_k", "_refln_index_l", "_refln_f_squared_meas"]
            has_required_base = all(req in colmap for req in required_base)
            has_fc2 = "_refln_f_squared_calc" in colmap
            has_fc_amp = "_refln_f_calc" in colmap
            if has_required_base and (has_fc2 or has_fc_amp):
                idx_sigma = colmap.get("_refln_f_squared_sigma")
                while i < n:
                    s = text[i].strip()
                    if not s or s.startswith('#'):
                        i += 1
                        continue
                    if s.lower() == 'loop_' or s.startswith('_') or s.lower().startswith('data_'):
                        break
                    parts = s.split()
                    try:
                        h = int(parts[colmap["_refln_index_h"]])
                        k = int(parts[colmap["_refln_index_k"]])
                        l = int(parts[colmap["_refln_index_l"]])
                        fo2 = _to_float(parts[colmap["_refln_f_squared_meas"]])
                        if has_fc2:
                            fc2 = _to_float(parts[colmap["_refln_f_squared_calc"]])
                        else:
                            fc_amp = _to_float(parts[colmap["_refln_f_calc"]])
                            fc2 = fc_amp*fc_amp if np.isfinite(fc_amp) else np.nan
                        sfo2 = (_to_float(parts[idx_sigma]) if idx_sigma is not None else np.nan)
                    except Exception:
                        i += 1
                        continue
                    if np.isfinite(fo2) and np.isfinite(fc2):
                        hkl.append((h, k, l))
                        Fo2.append(float(fo2))
                        sigFo2.append(float(sfo2) if (sfo2 is not None) else np.nan)
                        Fc2.append(float(fc2))
                    i += 1
                # continue scanning for other loops
                continue
        i += 1

    if not Fo2:
        raise RuntimeError(f"No reflections parsed from .fcf: {fcf_path}")

    # consolidate header into useful fields
    header_out = {}
    if all(k in header for k in ('cell_a','cell_b','cell_c','alpha','beta','gamma')):
        header_out['cell'] = (header['cell_a'], header['cell_b'], header['cell_c'], header['alpha'], header['beta'], header['gamma'])
    if 'space_group' in header:
        header_out['space_group'] = header['space_group']
    if 'symops' in header:
        header_out['symops'] = header['symops']

    return (np.array(hkl, dtype=int), np.array(Fo2, dtype=float), np.array(sigFo2, dtype=float), np.array(Fc2, dtype=float), header_out)


def mad(x: np.ndarray):
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))


def try_map_to_asu(fcf_df, refl_df, header):
    """Attempt mapping indices to ASU using cctbx when possible.

    Returns merged dataframe (on ASU indices) or None on failure.
    """
    try:
        from cctbx import crystal
        from cctbx import miller
        from cctbx.array_family import flex
        from cctbx import sgtbx
    except Exception:
        return None

    if 'cell' not in header:
        return None

    cell = header['cell']
    spg = header.get('space_group')

    sg_obj = None
    if spg:
        try:
            sg_obj = sgtbx.space_group_info(spg).group()
        except Exception:
            sg_obj = None
    if sg_obj is None and 'symops' in header:
        try:
            sg_obj = sgtbx.space_group()
            for op in header['symops']:
                sg_obj.expand_smx(sgtbx.rt_mx(op))
        except Exception:
            sg_obj = None
    if sg_obj is None:
        try:
            sg_obj = sgtbx.space_group_info('P1').group()
        except Exception:
            return None

    try:
        cs = crystal.symmetry(unit_cell=cell, space_group=sg_obj)
    except Exception:
        return None

    try:
        # build miller sets for fcf and reflections, map both to ASU, then join on ASU indices
        indices_f = flex.miller_index(list(map(tuple, fcf_df[['h','k','l']].values.tolist())))
        mset_f = miller.build_set(unit_cell=cs.unit_cell(), space_group=cs.space_group(), indices=indices_f, anomalous_flag=False)
        marr_fc2 = miller.array(mset_f, data=flex.double(fcf_df['Fc2_fcf'].astype(float).tolist()))
        marr_fc2_asu = marr_fc2.map_to_asu()
        mapped_f = np.array(list(map(tuple, marr_fc2_asu.indices())))
        fcf_df = fcf_df.copy()
        fcf_df['h_asu'] = mapped_f[:,0]; fcf_df['k_asu'] = mapped_f[:,1]; fcf_df['l_asu'] = mapped_f[:,2]

        indices_r = flex.miller_index(list(map(tuple, refl_df[['h','k','l']].values.tolist())))
        mset_r = miller.build_set(unit_cell=cs.unit_cell(), space_group=cs.space_group(), indices=indices_r, anomalous_flag=False)
        marr_r = miller.array(mset_r, data=flex.double([1.0]*len(refl_df)))
        marr_r_asu = marr_r.map_to_asu()
        mapped_r = np.array(list(map(tuple, marr_r_asu.indices())))
        refl_df = refl_df.copy()
        refl_df['h_asu'] = mapped_r[:,0]; refl_df['k_asu'] = mapped_r[:,1]; refl_df['l_asu'] = mapped_r[:,2]

        merged = refl_df.merge(fcf_df, how='left', left_on=['h_asu','k_asu','l_asu'], right_on=['h_asu','k_asu','l_asu'])
        return merged
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reflections', required=True)
    ap.add_argument('--fcf', required=True)
    ap.add_argument('--output', default='sres_out.csv')
    ap.add_argument('--keep-unmatched', action='store_true', help='Keep rows with no matching Fo2/Fc2 from .fcf')
    args = ap.parse_args()

    refl_path = Path(args.reflections)
    fcf_path = Path(args.fcf)
    out_path = Path(args.output)

    if not refl_path.exists():
        print('Reflections file not found:', refl_path); sys.exit(1)
    if not fcf_path.exists():
        print('.fcf file not found:', fcf_path); sys.exit(1)

    df = pd.read_csv(refl_path)

    hkl_arr, Fo2_arr, sigFo2_arr, Fc2_arr, header = parse_fcf_cif_style(fcf_path)
    fcf_df = pd.DataFrame({'h': hkl_arr[:,0], 'k': hkl_arr[:,1], 'l': hkl_arr[:,2], 'Fo2_fcf': Fo2_arr, 'sigFo2_fcf': sigFo2_arr, 'Fc2_fcf': Fc2_arr})

    merged = None
    if header and ('cell' in header and 'space_group' in header):
        merged = try_map_to_asu(fcf_df, df, header)
        if merged is None:
            print('cctbx ASU mapping attempted but failed; falling back to raw h,k,l join')

    if merged is None:
        merged = df.merge(fcf_df, how='left', on=['h','k','l'])

    if 'Fo2_fcf' not in merged.columns:
        print('No Fo2 from .fcf merged; aborting'); sys.exit(1)

    if not args.keep_unmatched:
        merged = merged[merged['Fo2_fcf'].notna()].copy()

    fc2_vals = merged['Fc2_fcf'].values
    fo2_vals = merged['Fo2_fcf'].values
    fc2_mad = mad(fc2_vals[np.isfinite(fc2_vals)])
    fo2_mad = mad(fo2_vals[np.isfinite(fo2_vals)])
    if fo2_mad == 0 or not np.isfinite(fo2_mad):
        scale = 1.0
    else:
        scale = (fc2_mad/fo2_mad) if np.isfinite(fc2_mad) and fc2_mad>0 else 1.0

    merged['Fo2_scaled'] = merged['Fo2_fcf'] * scale
    merged['sigFo2_scaled'] = merged['sigFo2_fcf'] * scale

    def compute_sres(r):
        fo2 = r.get('Fo2_scaled')
        fc2 = r.get('Fc2_fcf')
        sig = r.get('sigFo2_scaled')
        if not np.isfinite(fo2) or not np.isfinite(fc2) or not np.isfinite(sig) or sig==0:
            return np.nan
        return (fo2 - fc2) / sig

    merged['SRES'] = merged.apply(compute_sres, axis=1)

    # Build output with requested columns where available
    out = pd.DataFrame()
    out[['h','k','l']] = merged[['h','k','l']]
    out['fo2obs'] = merged['Fo2_fcf']
    out['fo2sigma'] = merged['sigFo2_fcf']
    out['fcalc'] = merged['Fc2_fcf']
    if 'frame' in merged.columns:
        out['frame'] = merged['frame']
    if 'S_orient' in merged.columns:
        out['S_orient'] = merged['S_orient']
    if 'sigma_orient_scale' in merged.columns:
        out['sigma_orient_scale'] = merged['sigma_orient_scale']
    out['SRES'] = merged['SRES']

    out.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(out)} rows). Scale factor: {scale:.6g}')


if __name__ == '__main__':
    main()

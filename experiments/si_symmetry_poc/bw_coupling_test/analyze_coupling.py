"""Reconstruct abTEM's static Bloch off-diagonal coupling without rerunning BW."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read, cif
from scipy.stats import spearmanr

import abtem
from abtem.bloch import StructureFactor
from abtem.bloch.dynamical import calculate_M_matrix
from abtem.core.constants import kappa
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.bloch.utils import excitation_errors


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/si_symmetry_poc/bw_coupling_test"
ZARR = ROOT / "experiments/si_symmetry_poc/results/20260924-121554_Si_CollCode51688_paper_cell_1x1x1000_451/bw.zarr"
CIF = ROOT / "experiments/si_symmetry_poc/Si_CollCode51688_paper_cell.cif"
ENERGY = 200000.0
G_MAX = 8.0
SG_MAX = 0.5
S0_GRID = (0.01, 0.03, 0.10)
REP_Z = 505.0
FAMILY52 = 52
CONTROL = 85


def load_previous():
    import importlib.util
    path = ROOT / "experiments/si_symmetry_poc/oridyn_test/analyze_scores.py"
    spec = importlib.util.spec_from_file_location("previous", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def laue_ids(hkls):
    structure = cif.read_cif(str(CIF)); rotations, _ = structure.info["spacegroup"].get_op()
    inversion = -np.eye(3)
    if not np.any(np.all(rotations == inversion, axis=(-2, -1))):
        rotations = np.vstack((rotations, np.matmul(rotations, inversion).astype(int)))
    operations = np.transpose(np.linalg.inv(rotations), (0, 2, 1)).astype(int)
    _, unique = np.unique(operations, axis=0, return_index=True); operations = operations[unique.sort()][0]
    sym = (operations @ hkls.T).transpose(2, 0, 1)
    a = np.ascontiguousarray(sym[:, 0]).view([("x", int), ("y", int), ("z", int)])
    b = np.ascontiguousarray(sym.reshape(-1, 3)).view([("x", int), ("y", int), ("z", int)])
    _, ia, ib = np.intersect1d(a, b, return_indices=True)
    ids = np.arange(len(hkls)) + len(hkls); np.minimum.at(ids, ia, ib // sym.shape[1])
    return ids.astype(int), operations


def beam_pool(basis0):
    reciprocal_length = float(np.min(np.linalg.svd(basis0)[1])); lim = int(np.ceil(G_MAX / reciprocal_length)) + 1
    grid = np.mgrid[-lim:lim + 1, -lim:lim + 1, -lim:lim + 1].reshape(3, -1).T.astype(int)
    g = grid @ basis0.T; h, k, l = grid.T
    return grid[(np.linalg.norm(g, axis=1) <= G_MAX + 1e-10) & ((h % 2) == (k % 2)) & ((h % 2) == (l % 2))]


def crossings(alpha, sg):
    result = {}
    for index in range(sg.shape[1]):
        changes = np.flatnonzero(sg[:-1, index] * sg[1:, index] <= 0)
        if len(changes):
            i = int(changes[0]); f = 0.0 if sg[i, index] == 0 else -sg[i, index] / (sg[i + 1, index] - sg[i, index])
            result[index] = (alpha[i] + f * (alpha[i + 1] - alpha[i]), i, f)
    return result


def build_static_coupling(atoms, pool, targets):
    sf = StructureFactor(atoms, g_max=G_MAX, parametrization="lobato", thermal_sigma={"Si": 0.1}, occupancy=1.0, centering="F", device="cpu").build(lazy=False)
    source = {tuple(h): value for h, value in zip(sf.hkl.astype(int), np.asarray(sf.array))}
    prefactor = energy2sigma(ENERGY) / (kappa * energy2wavelength(ENERGY) * np.pi)
    lookup = np.zeros((len(targets), len(pool)), dtype=np.complex64)
    for target_index, target in enumerate(targets):
        deltas = pool - target
        values = np.fromiter((source.get(tuple(delta), 0.0) for delta in deltas), dtype=np.complex64, count=len(pool))
        lookup[target_index] = values
    return lookup, sf, prefactor


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = load_previous()
    integrated, sg, alpha, thickness, hkls, basis = previous.load_data()
    ids, operations = laue_ids(hkls)
    complete = previous.complete_mask(sg, thickness)
    rep_index = int(np.flatnonzero(thickness == REP_Z)[0])

    # Longest fixed-member interval with at least two complete members per family.
    intervals = []
    response_rows = []
    target_indices = set()
    for family_id in np.unique(ids):
        sets = [tuple(np.flatnonzero((ids == family_id) & complete[i])) for i in range(len(thickness))]
        runs = []
        start = 0
        while start < len(sets):
            end = start + 1
            while end < len(sets) and sets[end] == sets[start]: end += 1
            if len(sets[start]) >= 2: runs.append((end - start, start, end, sets[start]))
            start = end
        if not runs: continue
        _, begin, end, members = max(runs)
        intervals.append({"family_id": family_id, "valid_start_A": thickness[begin], "valid_end_A": thickness[end - 1], "n_samples": end - begin, "n_members": len(members), "members": ";".join(str(tuple(hkls[i])) for i in members)})
        target_indices.update(members)
        values = integrated["integrated_intensities"][begin:end][:, members]
        log_ratio = np.log(np.maximum(values, np.finfo(float).tiny) / np.exp(np.mean(np.log(np.maximum(values, np.finfo(float).tiny)), axis=1))[:, None])
        for col, member in enumerate(members):
            response_rows.append({"family_id": family_id, "reflection_index": member, "h": hkls[member,0], "k": hkls[member,1], "l": hkls[member,2], "valid_start_A": thickness[begin], "valid_end_A": thickness[end-1], "n_samples": end - begin, "D_RMS": np.sqrt(np.mean(log_ratio[:, col] ** 2)), "D_median_abs": np.median(np.abs(log_ratio[:, col])), "D_p90_abs": np.quantile(np.abs(log_ratio[:, col]), .9)})
    pd.DataFrame(intervals).to_csv(OUT / "fixed_family_intervals.csv", index=False)
    responses = pd.DataFrame(response_rows); responses.to_csv(OUT / "thickness_aggregated_response.csv", index=False)

    targets = np.array(sorted(target_indices), dtype=int); pool = beam_pool(basis[0]); atoms = read(str(CIF))
    cache_path = OUT / "coupling_cache.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        cached_targets = cache["target_indices"]
        if np.array_equal(cached_targets, targets):
            static_f = cache["coupling_structure_factor"]
            pool = cache["pool_hkls"]
            sf_hkl = cache["structure_factor_hkl"]
            sf_array = cache["structure_factor"]
        else:
            static_f, sf_obj, prefactor = build_static_coupling(atoms, pool, hkls[targets])
            sf_hkl, sf_array = sf_obj.hkl, sf_obj.array
            np.savez_compressed(cache_path, target_indices=targets, target_hkls=hkls[targets], pool_hkls=pool, coupling_structure_factor=static_f, structure_factor_hkl=sf_hkl, structure_factor=sf_array)
    else:
        static_f, sf_obj, prefactor = build_static_coupling(atoms, pool, hkls[targets])
        sf_hkl, sf_array = sf_obj.hkl, sf_obj.array
        np.savez_compressed(cache_path, target_indices=targets, target_hkls=hkls[targets], pool_hkls=pool, coupling_structure_factor=static_f, structure_factor_hkl=sf_hkl, structure_factor=sf_array)
    sf = type("StaticStructure", (), {"hkl": sf_hkl, "array": sf_array})()
    prefactor = energy2sigma(ENERGY) / (kappa * energy2wavelength(ENERGY) * np.pi)
    # Verify the static Hermitian relation F(-g)=conj(F(g)) on the source grid.
    source = {tuple(h): value for h, value in zip(sf.hkl.astype(int), np.asarray(sf.array))}
    hermitian_error = max(abs(source.get(tuple(-np.asarray(h)), 0) - np.conjugate(v)) for h, v in source.items())
    pd.DataFrame([{"n_targets":len(targets), "n_pool":len(pool), "prefactor":prefactor, "structure_factor_dtype":str(sf.array.dtype), "max_Friedel_hermitian_error":hermitian_error, "formula":"A[h,q] = F(q-h) * prefactor * M_h(alpha) * M_q(alpha); diagonal replaced by s_g term"}]).to_csv(OUT / "coupling_verification.csv", index=False)

    crossing_map = crossings(alpha, sg)
    target_position = {int(index): position for position, index in enumerate(targets)}
    valid_targets = np.array([index for index in targets if index in crossing_map], dtype=int)
    valid_positions = np.array([target_position[int(index)] for index in valid_targets], dtype=int)
    f_abs = np.abs(static_f[valid_positions]).astype(np.float32)
    f_abs2 = np.square(f_abs)
    for row, target_index in enumerate(valid_targets):
        f_abs[row, np.all(pool == hkls[target_index], axis=1)] = 0.0
        f_abs2[row, np.all(pool == hkls[target_index], axis=1)] = 0.0
    k0 = 1.0 / energy2wavelength(ENERGY)
    exact_scores = {s0: (np.zeros(len(valid_targets)), np.zeros(len(valid_targets))) for s0 in S0_GRID}
    path_scores = {s0: (np.zeros(len(valid_targets)), np.zeros(len(valid_targets))) for s0 in S0_GRID}
    for row, target_index in enumerate(valid_targets):
        alpha_b, bracket, fraction = crossing_map[int(target_index)]
        bmat = basis[bracket] + fraction * (basis[bracket + 1] - basis[bracket])
        gq = pool @ bmat.T; gh = bmat @ hkls[target_index]
        mq = 1.0 / np.sqrt(1.0 + gq[:, 2] / k0); mh = 1.0 / np.sqrt(1.0 + gh[2] / k0)
        sq = excitation_errors(gq, ENERGY, use_wave_eq=True)
        for s0 in S0_GRID:
            w = np.exp(-np.square(sq / s0)); exact_scores[s0][0][row] = np.sum(w * f_abs2[row] * prefactor**2 * mh**2 * mq**2); exact_scores[s0][1][row] = np.sum(w * f_abs[row] * prefactor * abs(mh) * abs(mq))
    for s0 in S0_GRID:
        ru = np.zeros((len(valid_targets), len(alpha))); ra = np.zeros_like(ru)
        for start in range(0, len(alpha), 16):
            stop = min(start + 16, len(alpha)); bchunk = basis[start:stop]
            gq = np.einsum("aij,nj->ani", bchunk, pool)
            sq = excitation_errors(gq.reshape(-1, 3), ENERGY, use_wave_eq=True).reshape(stop - start, len(pool))
            mq2 = 1.0 / (1.0 + np.square(gq[:, :, 2] / k0))
            eq = np.exp(-np.square(sq / s0)) * mq2
            ru[:, start:stop] = (f_abs2 @ eq.T) * prefactor**2
            ra[:, start:stop] = (f_abs @ eq.T) * prefactor
        for row, target_index in enumerate(valid_targets):
            target_sg = sg[:, target_index]; wt = np.exp(-np.square(target_sg / s0)); denom = np.trapz(wt, alpha)
            target_mh2 = 1.0 / (1.0 + np.square(np.einsum("aij,j->ai", basis, hkls[target_index])[:, 2] / k0))
            path_scores[s0][0][row] = np.trapz(wt * target_mh2 * ru[row], alpha) / denom
            path_scores[s0][1][row] = np.trapz(wt * np.sqrt(target_mh2) * ra[row], alpha) / denom
    score_rows = []
    for row, target_index in enumerate(valid_targets):
        alpha_b = crossing_map[int(target_index)][0]; target = hkls[target_index]
        for s0 in S0_GRID:
            score_rows.append({"reflection_index":target_index,"family_id":ids[target_index],"h":target[0],"k":target[1],"l":target[2],"s0":s0,"alpha_B_deg":np.rad2deg(alpha_b),"R_U_B":exact_scores[s0][0][row],"R_Uabs_B":exact_scores[s0][1][row],"R_U_path":path_scores[s0][0][row],"R_Uabs_path":path_scores[s0][1][row]})
    scores = pd.DataFrame(score_rows); scores.to_csv(OUT / "coupling_observation_scores.csv", index=False)
    baseline = pd.read_csv(ROOT / "experiments/si_symmetry_poc/oridyn_test/observation_geometry_scores.csv")
    baseline = baseline[(baseline.s0 == .03) & (baseline.sigma_C == .3) & (baseline.r_cut == 1.0)].drop(columns=["s0","sigma_C","r_cut"])
    scores_rep = scores[scores.s0 == .03].merge(responses, on=["family_id","reflection_index","h","k","l"], how="inner").merge(baseline, on=["family_id","reflection_index","h","k","l"], how="left")
    scores_rep.to_csv(OUT / "coupling_vs_previous_scores.csv", index=False)
    correlation_rows=[]
    for score in ["R_B","R_path","N_excited_B","N_excited_path","R_U_B","R_U_path"]:
        rhos=[]
        for family_id, group in scores_rep.groupby("family_id"):
            if len(group)>=3 and group.D_RMS.nunique()>1 and group[score].nunique()>1: rhos.append(spearmanr(group.D_RMS,group[score]).statistic)
        correlation_rows.append({"score":score,"n_families":len(rhos),"fraction_positive":np.mean(np.asarray(rhos)>0) if rhos else np.nan,"median_within_family_spearman":np.median(rhos) if rhos else np.nan})
    pd.DataFrame(correlation_rows).to_csv(OUT / "coupling_whole_dataset_correlations.csv", index=False)
    for family_id in [FAMILY52, CONTROL]:
        x=scores_rep[scores_rep.family_id==family_id]; fig, ax=plt.subplots(1,2,figsize=(10,4)); ax[0].scatter(x.R_U_B,x.D_RMS,label="R_U_B"); ax[0].scatter(x.R_U_path,x.D_RMS,marker='x',label="R_U_path"); ax[1].scatter(x.R_B,x.D_RMS,label="R_B"); ax[1].scatter(x.N_excited_B,x.D_RMS,marker='x',label="N_excited_B"); ax[0].legend(); ax[1].legend(); fig.suptitle(f"Family {family_id}: D_RMS and coupling/count scores"); fig.tight_layout(); fig.savefig(OUT/f"family{family_id}_coupling_comparison.png",dpi=180); plt.close(fig)


if __name__ == "__main__": main()
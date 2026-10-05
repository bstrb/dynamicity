from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/si_symmetry_poc/kinematic_reference"
INDIVIDUAL = ROOT / "experiments/si_symmetry_poc/individual_complete_observations.csv"
SCORES = ROOT / "experiments/si_symmetry_poc/oridyn_test/observation_geometry_scores.csv"
KIN = np.load(OUT / "finite_thickness_kinematic.npz")

hkls = KIN["hkls"].astype(int)
thickness = KIN["thickness"].astype(float)
j_kin = KIN["j_kin"].astype(float)
dyn = pd.read_csv(INDIVIDUAL)
lookup = {(float(z), int(i)): float(j_kin[zi, i]) for zi, z in enumerate(thickness) for i in range(len(hkls))}
dyn["J_kin"] = [lookup.get((float(z), int(i)), np.nan) for z, i in zip(dyn.thickness, dyn.reflection_index)]
dyn["is_transmitted_000"] = dyn.reflection_index == 0
dyn["log_Jdyn_over_Jkin"] = np.log(dyn.integrated_intensity_sg / dyn.J_kin)
dyn["abs_log_Jdyn_over_Jkin"] = np.abs(dyn.log_Jdyn_over_Jkin)
dyn.to_csv(OUT / "dynamical_kinematic_comparison.csv", index=False)

rep = dyn[(dyn.thickness == 505.0) & (~dyn.is_transmitted_000)].copy()
scores = pd.read_csv(SCORES)
scores = scores[(scores.s0 == 0.03) & (scores.sigma_C == 0.3) & (scores.r_cut == 1.0)]
rep = rep.merge(scores.drop(columns=["s0", "sigma_C", "r_cut"]), on=["family_id", "reflection_index", "h", "k", "l"], how="left")
rep.to_csv(OUT / "representative_dynamical_kinematic_oridyn.csv", index=False)

rows = []
for score in ["R_B", "R_path", "N_excited_B", "N_excited_path"]:
    rhos = []
    for _, group in rep.groupby("family_id"):
        group = group.dropna(subset=[score, "abs_log_Jdyn_over_Jkin"])
        if len(group) >= 3 and group[score].nunique() > 1 and group.abs_log_Jdyn_over_Jkin.nunique() > 1:
            rhos.append(spearmanr(group.abs_log_Jdyn_over_Jkin, group[score]).statistic)
    rows.append({"response": "abs_log_Jdyn_over_Jkin", "score": score, "n_families": len(rhos), "fraction_positive": float(np.mean(np.asarray(rhos) > 0)) if rhos else np.nan, "median_within_family_spearman": float(np.median(rhos)) if rhos else np.nan})
pd.DataFrame(rows).to_csv(OUT / "oridyn_vs_dynamical_kinematic_correlations.csv", index=False)

print("rows", len(dyn), "finite comparisons", dyn.abs_log_Jdyn_over_Jkin.notna().sum())
for family_id in (52, 85):
    x = rep[rep.family_id == family_id]
    print("family", family_id)
    print(x[["h", "k", "l", "J_dyn" if "J_dyn" in x else "integrated_intensity_sg", "J_kin", "log_Jdyn_over_Jkin", "R_B", "R_path"]].to_string(index=False))

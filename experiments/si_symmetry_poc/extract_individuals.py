"""Extract individual Laue-equivalent observations before the authors' R_int sum."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import cif


ROOT = Path(__file__).resolve().parents[2]
PY3DED = ROOT.parent / "py3DED"
ZARR_CANDIDATES = sorted(
    (ROOT / "experiments/si_symmetry_poc/results").glob("*/bw.zarr"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
ZARR = next((path for path in ZARR_CANDIDATES if (path / ".zattrs").exists()), None)
CIF = ROOT / "experiments/si_symmetry_poc/Si_CollCode51688_paper_cell.cif"
OUT = ROOT / "experiments/si_symmetry_poc"


def load_authors_analysis():
    path = PY3DED / "scripts/run_py3DED_hkl+Rint.py"
    spec = importlib.util.spec_from_file_location("authors_rint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def laue_group_ids(miller_indices: np.ndarray) -> np.ndarray:
    """Use the grouping algorithm copied from calculate_rint()."""
    structure = cif.read_cif(str(CIF))
    rotations, _ = structure.info["spacegroup"].get_op()
    inversion = np.eye(3) * -1
    if not np.any(np.all(rotations == inversion, axis=(-2, -1))):
        rotations = np.vstack((rotations, np.matmul(rotations, inversion).astype(int)))
    reciprocal = np.transpose(np.linalg.inv(rotations), (0, 2, 1)).astype(int)
    _, unique_ids = np.unique(reciprocal, axis=0, return_index=True)
    reciprocal = reciprocal[unique_ids.sort()][0]

    hkl_sym = (reciprocal @ miller_indices.T).transpose(2, 0, 1)
    a = np.ascontiguousarray(hkl_sym[:, 0])
    a_view = a.view([("x", a.dtype), ("y", a.dtype), ("z", a.dtype)])
    b = np.ascontiguousarray(hkl_sym.reshape(-1, 3))
    b_view = b.view([("x", b.dtype), ("y", b.dtype), ("z", b.dtype)])
    _, idx_a, idx_b = np.intersect1d(a_view, b_view, return_indices=True)
    origin = idx_b // hkl_sym.shape[1]
    group_ids = np.arange(len(a)) + len(a)
    np.minimum.at(group_ids, idx_a, origin)
    return group_ids.astype(int)


def crossing_angles(alpha_rad: np.ndarray, sg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the first actual s_g zero crossing and local ds_g/dalpha."""
    alpha_b = np.full(sg.shape[1], np.nan, dtype=float)
    slope = np.full(sg.shape[1], np.nan, dtype=float)
    for hkl_index in range(sg.shape[1]):
        values = sg[:, hkl_index]
        exact = np.flatnonzero(values == 0.0)
        if exact.size:
            index = int(exact[0])
            alpha_b[hkl_index] = alpha_rad[index]
            if 0 < index < len(values) - 1:
                slope[hkl_index] = (values[index + 1] - values[index - 1]) / (
                    alpha_rad[index + 1] - alpha_rad[index - 1]
                )
            continue
        changes = np.flatnonzero(values[:-1] * values[1:] < 0.0)
        if not changes.size:
            continue
        index = int(changes[0])
        fraction = -values[index] / (values[index + 1] - values[index])
        alpha_b[hkl_index] = alpha_rad[index] + fraction * (
            alpha_rad[index + 1] - alpha_rad[index]
        )
        slope[hkl_index] = (values[index + 1] - values[index]) / (
            alpha_rad[index + 1] - alpha_rad[index]
        )
    return alpha_b, slope


def main() -> None:
    if ZARR is None:
        raise FileNotFoundError("No completed bw.zarr found under the experiment results.")
    authors = load_authors_analysis()
    integrated = authors.calculate_integrated_intensities(ZARR, g_max=2.0)
    intensities = integrated["integrated_intensities"]
    sg = integrated["excitation_errors"]
    if sg.ndim == 3 and sg.shape[1] == 1:
        sg = sg[:, 0, :]
    hkls = integrated["miller_indices"].astype(int)
    thicknesses = integrated["thicknesses"]

    # This is the authors' criterion before the source's erroneous mask[0] line.
    sg_min = np.min(sg, axis=0)
    sg_max = np.max(sg, axis=0)
    rc_width_min = 2.0 / thicknesses
    complete = (sg_min[None, :] < -rc_width_min[:, None] / 2.0) & (
        sg_max[None, :] > rc_width_min[:, None] / 2.0
    )
    family_ids = laue_group_ids(hkls)
    alpha_rad = np.linspace(0.0, np.deg2rad(45.0), sg.shape[0])
    alpha_b, slope = crossing_angles(alpha_rad, sg)

    rows = []
    for thickness_index, thickness in enumerate(thicknesses):
        for reflection_index in np.flatnonzero(complete[thickness_index]):
            h, k, l = hkls[reflection_index]
            rows.append(
                {
                    "family_id": int(family_ids[reflection_index]),
                    "reflection_index": int(reflection_index),
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                    "thickness": float(thickness),
                    "alpha_B": float(np.rad2deg(alpha_b[reflection_index])),
                    "ds_g_dalpha": float(slope[reflection_index]),
                    "integrated_intensity_sg": float(
                        intensities[thickness_index, reflection_index]
                    ),
                }
            )
    individual = pd.DataFrame.from_records(rows)
    individual.to_csv(OUT / "individual_complete_observations.csv", index=False)

    representative = float(thicknesses[len(thicknesses) // 2])
    at_rep = individual[individual["thickness"] == representative]
    summary_rows = []
    for family_id, group in at_rep.groupby("family_id", sort=False):
        summary_rows.append(
            {
                "family_id": int(family_id),
                "n_complete": len(group),
                "hkls": ";".join(
                    f"({int(row.h)},{int(row.k)},{int(row.l)})"
                    for row in group.itertuples()
                ),
                "alpha_B_values": ";".join(f"{v:.6f}" for v in group["alpha_B"]),
                "min_alpha_B": group["alpha_B"].min(),
                "max_alpha_B": group["alpha_B"].max(),
                "mean_integrated_intensity": group["integrated_intensity_sg"].mean(),
                "median_integrated_intensity": group["integrated_intensity_sg"].median(),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)
    summary["delta_alpha_B"] = summary["max_alpha_B"] - summary["min_alpha_B"]
    summary = summary[summary["n_complete"] >= 2].sort_values(
        ["n_complete", "delta_alpha_B"], ascending=False
    )
    summary.insert(0, "representative_thickness", representative)
    summary.to_csv(OUT / "candidate_family_summary.csv", index=False)

    # Diagnostic plots only: display families with the most complete members,
    # using span as a display tie-breaker rather than a scientific ranking.
    selected = summary.head(3)["family_id"].tolist()
    da = authors.abtem.from_zarr(str(ZARR)).to_data_array()
    for family_id in selected:
        members = at_rep[at_rep["family_id"] == family_id]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for _, row in members.iterrows():
            curve = da.isel(hkl=int(row["reflection_index"]), z=len(thicknesses) // 2).compute()
            axes[0].plot(np.rad2deg(alpha_rad), curve, label=f"({int(row['h'])} {int(row['k'])} {int(row['l'])})")
            axes[0].axvline(row["alpha_B"], color="k", alpha=0.25)
        axes[0].set(xlabel="alpha (deg)", ylabel="I", title=f"family {family_id}, z={representative:.0f} A")
        for (h, k, l), member in individual[individual["family_id"] == family_id].groupby(["h", "k", "l"], sort=False):
            axes[1].plot(member["thickness"], member["integrated_intensity_sg"], label=f"({int(h)} {int(k)} {int(l)})")
        axes[1].set(xlabel="thickness (A)", ylabel="J = |integral I ds_g|", title="individual integrated intensities")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"family_{family_id}_diagnostic.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
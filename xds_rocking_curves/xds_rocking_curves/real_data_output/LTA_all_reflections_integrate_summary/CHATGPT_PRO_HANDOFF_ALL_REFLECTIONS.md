# LTA All-Reflections, All-Datasets Handoff

## Scope
This package summarizes **all indexed reflections** from `INTEGRATE.HKL` for:
- `LTA_t1` (100 nm)
- `LTA_t2` (200 nm)
- `LTA_t3` (350 nm)
- `LTA_t4` (600 nm)

Total rows: **60,813** (15,223 + 15,201 + 15,235 + 15,154).

## Important note
This is a full-reflection summary from `INTEGRATE.HKL` metadata and intensities.
It is not a full 60k local rocking-curve Gaussian-fit run (which would be a much larger compute job).

## Key outputs
- `all_reflections_all_datasets_long.csv`
  - Long table with one row per indexed reflection observation.
  - Columns include: `dataset`, `thickness_nm`, `h,k,l`, `I`, `sigma`, `isig`, `x_cal`, `y_cal`, `z_cal`, `frame_est`, `peak`, `corr`, `psi`, `iseg`, `d_spacing_angstrom`, `resolution_invA`, and percentile ranks.
- `dataset_overview_stats.csv`
  - Per-dataset aggregate stats (counts, I/sigma thresholds, quantiles, d-spacing stats).
- `dataset_quantiles_isig_I.csv`
  - Robust quantiles (`q10..q95`) for `I` and `I/sigma` per dataset.
- `hkl_dataset_overlap_counts.csv`
  - For each HKL triplet, number of datasets where it is present.
- `hkl_overlap_distribution.csv`
  - Distribution of overlap counts (`n_datasets_present` -> `n_hkls`).
- `top_1000_by_isig_all_datasets.csv`
  - Concatenated top-1000 (by `I/sigma`) for each dataset.
- Per-dataset folders:
  - `LTA_t1/all_reflections_dataset.csv`, `LTA_t1/top_1000_by_isig.csv`
  - `LTA_t2/all_reflections_dataset.csv`, `LTA_t2/top_1000_by_isig.csv`
  - `LTA_t3/all_reflections_dataset.csv`, `LTA_t3/top_1000_by_isig.csv`
  - `LTA_t4/all_reflections_dataset.csv`, `LTA_t4/top_1000_by_isig.csv`

## High-level findings
From `dataset_overview_stats.csv`:
- `LTA_t1`: median `I/sigma` ~ 16.39, p90 ~ 41.96
- `LTA_t2`: median `I/sigma` ~ 16.49, p90 ~ 41.97
- `LTA_t3`: median `I/sigma` ~ 15.69, p90 ~ 40.82
- `LTA_t4`: median `I/sigma` ~ 10.07, p90 ~ 33.55

Interpretation: signal quality drops noticeably in `LTA_t4` relative to `t1..t3`.

From `hkl_overlap_distribution.csv`:
- HKLs present in all 4 datasets: **15,018**
- Present in exactly 3 datasets: 134
- Present in exactly 2 datasets: 123
- Present in exactly 1 dataset: 93

Interpretation: overlap is very high, enabling robust cross-thickness comparative analyses.

## Suggested next analyses (for ChatGPT Pro)
1. Build matched-HKL comparative models using the `n_datasets_present=4` subset.
2. Test thickness trend of `I/sigma` with robust methods (median differences, quantile regression).
3. Stratify by resolution (`d_spacing_angstrom`) and test whether thickness effects are resolution-dependent.
4. Detect outlier HKLs with large thickness-dependent deviations (e.g., standardized residual across datasets).
5. Define a high-confidence core set, e.g. HKLs with `I/sigma >= 8` in all four datasets.

## Repro command
```bash
cd /home/bubl3932/projects/dynamicity/xds_rocking_curves/xds_rocking_curves
/home/bubl3932/anaconda3/envs/pyxem-env/bin/python examples/summarize_all_reflections_integrate.py \
  --output-root real_data_output/LTA_all_reflections_integrate_summary \
  --top-n 1000
```

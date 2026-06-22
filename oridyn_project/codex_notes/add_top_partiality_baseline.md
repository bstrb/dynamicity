**What Changed**
- Added `--baseline-top-partiality-fraction` with default `0.10`.
- Added per-HKL baseline computation from the highest-partiality observations: `I_baseline_hkl = median(I_unmerged * partiality)` over the top fraction, with `baseline_n_obs`.
- Local reciprocal weak/strong classification now compares `I_baseline_hkl` against neighboring baseline medians; the risk-shift test still uses all eligible observations and the existing top/bottom risk tails.
- Added `I_baseline_hkl`, `baseline_top_partiality_fraction`, and `baseline_n_obs` to `per_hkl_metric_shifts.csv`, and added a `baseline` metadata block with the baseline definition, fraction, and `baseline_n_obs` summary.

**Validation Status**
- `py_compile`: passed.
- `--help`: passed and shows `--baseline-top-partiality-fraction`.
- Tiny smoke test with `--max-rows 50000`: passed, writing to `/tmp/dynamicity_top_partiality_baseline_smoke`. With the default `--min-obs-per-hkl 50`, that capped slice had 0 eligible HKL groups.

**Exact Full Command**
```bash
python /home/bubl3932/projects/dynamicity/oridyn_project/tools/analyze_global_median_orientation_shift.py --scores /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/oridyn_large_ABC_min1_mildweights/oridyn_scores_mild/reflection_scores.csv --unmerged /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300-VIII_cut_20-0_3_partialator_results/unmerged.hkl --stream /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300-VIII_cut_20-0_3.stream --output-root /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/local_reciprocal_top_partiality_baseline_signed_hkl --classification-mode local_reciprocal --grouping signed_hkl --pointgroup 1 --tail-fraction 0.10 --baseline-top-partiality-fraction 0.10 --min-obs-per-hkl 50 --local-neighbor-count 100 --local-min-neighbors 30 --exclude-partiality-too-small --progress-every 100000 --scores-chunksize 1000000
```

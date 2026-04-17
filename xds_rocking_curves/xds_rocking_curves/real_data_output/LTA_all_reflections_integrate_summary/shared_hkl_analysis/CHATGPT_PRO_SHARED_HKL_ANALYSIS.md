# Shared HKL Thickness-Trend Analysis

Input root: `real_data_output/LTA_all_reflections_integrate_summary`
Output dir: `real_data_output/LTA_all_reflections_integrate_summary/shared_hkl_analysis`

Shared HKLs across all 4 datasets: **15018**
Rows in long table: **60072**

## Median I/sigma by dataset (shared HKLs)
- LTA_t1: 16.4246
- LTA_t2: 16.5114
- LTA_t3: 15.7121
- LTA_t4: 10.0424

## Exported tables
- shared_hkls_long.csv
- shared_hkl_trend_metrics.csv
- top_500_most_stable_hkls.csv
- top_500_most_variable_hkls.csv
- top_500_steepest_negative_isig_trend.csv
- top_500_steepest_positive_isig_trend.csv
- shared_hkl_dataset_summary.csv

## Exported plots
- hist_isig_shared_hkls.png
- box_isig_shared_hkls.png
- hist_slope_isig_per_nm.png
- scatter_mean_isig_vs_cv_isig.png

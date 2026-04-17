# Reflection Variation Ranking

Input root: `real_data_output/LTA_reflections_adaptive_top1000_all4`
Reflections analyzed: **340**

Metrics:
- intensity variation: `cv_peak_intensity`, `delta_peak_intensity`
- shape variation: `mean_shape_rmse` on smoothed normalized aligned curves
- quality context: failed-fit fraction, endpoint flag, median fitted frames

Outputs:
- reflection_variation_metrics.csv
- top_200_most_varying_reflections.csv
- top_200_least_varying_reflections.csv

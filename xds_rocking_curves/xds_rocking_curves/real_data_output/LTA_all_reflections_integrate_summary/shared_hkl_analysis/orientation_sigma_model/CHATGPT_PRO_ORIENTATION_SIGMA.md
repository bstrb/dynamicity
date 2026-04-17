# Orientation-Dependent Sigma Model

Residual model per HKL:
- Fit linear trend vs thickness for `isig` and `logI`.
- Residuals are pooled by orientation bins (`phi`).

Global robust residual sigma (isig): 1.713483
Global robust residual sigma (logI): 0.222605

Recommended additive error model (by orientation bin):
- `sigma_add_isig(phi) = max(0, sigma_resid_isig(phi) - sigma_global_isig)`
- `sigma_add_logI(phi) = max(0, sigma_resid_logI(phi) - sigma_global_logI)`

HKLs modeled: 15018
Observations modeled: 60072

Outputs:
- sigma_by_orientation_bin.csv
- hkl_excess_variability_ranked.csv
- top_500_hkls_excess_variability.csv
- per_hkl_linear_residual_metrics.csv
- per_observation_with_orientation_residuals.csv

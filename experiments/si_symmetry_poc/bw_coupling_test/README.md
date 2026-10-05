# Actual Bloch-coupling test

This directory is post-processing of the completed Si BW result. No BW or
multislice simulation was rerun.

## abTEM implementation

The pinned abTEM implementation is in
`abtem/bloch/dynamical.py`, `calculate_structure_matrix()`.
For matrix row `h` and column `q`, it constructs `q - h` in Miller indices,
retrieves the corresponding structure factor, and applies:

```text
A[h,q] = F(q-h) * prefactor * M_h * M_q
prefactor = energy2sigma(E) / (kappa * wavelength(E) * pi)
M_g = 1 / sqrt(1 + g_z / k0)
```

The diagonal is then replaced by the excitation-error term. The structure
factor itself is calculated in `calculate_structure_factors()` from the same
ASE atoms, Lobato parametrization, thermal sigma, occupancy, and cell volume.
The reconstructed static source grid contains 85,851 F-centered HKLs with
`|g| <= 8 A^-1`. The coupling cache is
`coupling_cache.npz`; it stores target HKLs, pool HKLs, complex `F(q-h)`
values, and the full source structure-factor grid. The exact matrix coupling
used in the scores additionally includes the orientation-dependent `M_h M_q`
factors. Friedel/Hermitian verification error was zero for the source grid.

## Response and intervals

For every family, the response uses the longest contiguous thickness interval
where the exact complete-member set is fixed and contains at least two
members. It exports `D_RMS`, median absolute log deviation, and 90th-percentile
absolute log deviation in `thickness_aggregated_response.csv`, with interval
metadata in `fixed_family_intervals.csv`.

The primary score is the actual squared coupling:

```text
R_U_B    = sum E_q(alpha_B) * |A[h,q](alpha_B)|^2
R_U_path = integral E_h(alpha) * R_U(alpha) d alpha / integral E_h(alpha) d alpha
```

The amplitude variant is also exported as `R_Uabs_B` and `R_Uabs_path`.
The predefined excitation widths remain `s0 = 0.01, 0.03, 0.10 A^-1`.
No intensity is used in any score.

## Main outputs

- `coupling_observation_scores.csv`
- `coupling_vs_previous_scores.csv`
- `coupling_whole_dataset_correlations.csv`
- `thickness_aggregated_response.csv`
- `fixed_family_intervals.csv`
- `coupling_verification.csv`
- `family52_coupling_comparison.png`
- `family85_coupling_comparison.png`

The whole-dataset test contains 49 eligible family comparisons with at least
three members. Summary values are:

```text
score          fraction positive    median within-family rho
R_B                 0.286                   -0.400
R_path              0.367                   -0.200
N_excited_B         0.396                   -0.158
N_excited_path      0.286                   -0.400
R_U_B               0.367                   -0.200
R_U_path            0.306                   -0.200
```

## Interpretation

The actual abTEM coupling does not substantially improve prediction of
dynamical-deviation magnitude over the previous descriptive baselines. The
exact-Bragg and trajectory variants are both non-positive in their median
within-family association. Family 52 remains illustrative, but the result is
also seen in the whole-dataset test and in the independently selected family
85 control.

This is outcome C: a positive scalar local-environment score, even when based
on the actual Bloch off-diagonal coupling rather than reciprocal distance,
does not capture the magnitude robustly. Coherent phase and pathway
interference are likely essential. No increasingly flexible scalar score was
introduced after this result.
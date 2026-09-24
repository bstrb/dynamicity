# OriDyn cRED/XDS Validation

This directory contains the first cRED/XDS validation workflow for OriDyn. It is separate from the existing cSerialED/CrystFEL path and scores each finite `INTEGRATE.HKL` observation at its exact fractional `ZCAL`.

The production command is:

```bash
python /Users/xiaodong/Desktop/dynamicity/oridyn_project/xds_cred/run_analysis.py \
  --config /Users/xiaodong/Desktop/dynamicity/oridyn_project/xds_cred/config_lta1_fixed.json
```

The production run writes only to:

```text
/Users/xiaodong/Desktop/LTA1_0.03deg/oridyn_xds_results/LTA_t1_fixed
```

If expected result files already exist there, the workflow stops before overwriting them.

## What It Does

For each finite `IOBS` row in `INTEGRATE.HKL`, the workflow:

1. parses signed `h,k,l` and exact fractional `ZCAL`;
2. audits `XPARM.XDS`, `GXPARM.XDS`, and the batch-refined orientation matrices printed by `INTEGRATE.LP`;
3. reconstructs the selected reciprocal basis at `ZCAL` using `phi = STARTING_ANGLE + OSCILLATION_RANGE * (ZCAL - STARTING_FRAME + 1.0)`;
4. computes `E_target`, `R_env`, and `S_risk`;
5. groups signed observations by true space-group rotations using `gemmi` when available, otherwise `cctbx`;
6. computes leave-one-out median intensity references within symmetry classes;
7. reports global and within-class rank relationships between risk and relative disagreement.

The risk calculation uses:

```text
R_env(g) = sum_{h != g, d_gh <= r_cut} E(h) C(g,h)
S_risk(g) = E(g) R_env(g)
E(q) = exp[-(s_q / s0)^2]
C(g,h) = exp[-(d_gh / sigma_C)^2]
s_q = ||k0 + q|| - ||k0||
```

Reciprocal-space units are `A^-1` without a `2*pi` factor.

## Outputs

Primary files:

```text
observations.csv.gz
groups.csv.gz
excluded.csv.gz
geometry.csv.gz
geometry_summary.txt
orientation_audit.csv
neighbor_audit.csv.gz
risk_distribution.csv
correlations.csv
resolution.csv
scales_integrate.csv
scales_correct.csv
summary.txt
parameters.json
input_files.json
software_versions.json
run.log
```

Figures are written under `figures/`.

`scales_integrate.csv` contains image-scale rows parsed from `INTEGRATE.LP`. `scales_correct.csv` contains CORRECT shutter-position diagnostics and correction-grid metadata; binary CBF correction grids are documented but not decoded in this first version.

## Smoke Test

A deterministic smoke configuration is provided:

```bash
python /Users/xiaodong/Desktop/dynamicity/oridyn_project/xds_cred/run_analysis.py \
  --config /Users/xiaodong/Desktop/dynamicity/oridyn_project/xds_cred/config_smoke_lta1_fixed.json
```

It processes only the first 300 finite observations and writes to `xds_cred/smoke_output_fixed`.

## Dependencies

The workflow uses the existing OriDyn geometry helpers plus `numpy`, `pandas`, and `matplotlib`. Symmetry grouping requires a crystallographic backend: `gemmi` or `cctbx`.

# Bloch-wave dynamicality analyser

A research-oriented Python project for reconstructing XDS rotation-series orientations, generating candidate reflections, and computing reflection-wise dynamicality metrics inspired by a Bloch-wave coupling model.

The codebase recreates the logic of the supplied browser-based analyser while extending it with an optional thickness-aware Bloch propagation mode.

## Purpose

This project is aimed at method development in electron crystallography and SerialED workflows where you want to:

- parse `GXPARM.XDS`, `INTEGRATE.HKL`, and optionally `XDS.INP`
- parse CrystFEL `.stream` output directly for snapshot SerialED
- parse PETS2 project outputs (`.pts2` + `.ptsopt` + `.rprofall`)
- reconstruct frame orientations from `phi0`, `dphi`, and the XDS rotation axis
- generate candidate reciprocal-lattice vectors from the GXPARM reference basis
- compute excitation errors and reflection-wise dynamicality proxy scores
- compute orientation-only per-reflection uncertainty/risk terms (`orientation_sigma_sg_invA`, `orientation_p_excited`, `S_dyn`, `S_orient`)
- optionally run a pure orientation-only mode that skips Wilson/coupling terms
- optionally propagate a Bloch-wave state through crystal thickness and study thickness sensitivity
- export frame summaries and long-format reflection tables for downstream analysis

## Scientific scope

The default orientation model follows the browser prototype:

- frame orientations are reconstructed from `GXPARM.XDS`
- `INTEGRATE.HKL` is used for observed intensities and frame `z` coordinates, not for independent per-frame orientation matrices
- the original score is a coupling / dynamicality proxy rather than a full physically calibrated Bloch intensity simulation

This repository preserves that behavior in **proxy mode** and adds **thickness-aware mode** as an explicit extension.

For snapshot-style SerialED indexing outputs, you can supply a custom orientation
model built from per-frame UB matrices (`ReciprocalMatrixOrientationModel`).

## Project layout

```text
bloch_wave_analyser_project/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── pyproject.toml
├── examples/
│   └── run_analysis.py
├── notebooks/
│   └── bloch_analyser.ipynb
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── parsers.py
│   ├── geometry.py
│   ├── wilson.py
│   ├── bloch.py
│   ├── metrics.py
│   ├── visualization.py
│   └── pipeline.py
└── tests/
    ├── test_parsers.py
    ├── test_geometry.py
    └── test_metrics.py
```

## Installation

### Option 1: pip + virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: conda / mamba

```bash
mamba env create -f environment.yml
mamba activate bloch-wave-analyser
```

## Running the notebook

From the project root:

```bash
jupyter lab
```

Open `notebooks/bloch_analyser.ipynb` and step through:

1. input file paths
2. composition entry
3. parameter setup
4. parsing
5. Wilson calibration
6. pipeline execution
7. summary plots
8. detector plot for a selected frame
9. thickness-sensitivity plots
10. CSV export

## Running the CLI

Single-thickness proxy-style analysis:

```bash
python examples/run_analysis.py \
  --gxparm path/to/GXPARM.XDS \
  --integrate path/to/INTEGRATE.HKL \
  --xdsinp path/to/XDS.INP \
  --composition "24 Si, 48 O, 12 Na, 27 O" \
  --dmin 0.6 \
  --dmax 50.0 \
  --mode proxy
```

Using CrystFEL `.stream` only (no GXPARM / INTEGRATE required):

```bash
python examples/run_analysis.py \
  --stream path/to/indexing.stream \
  --composition "24 Si, 48 O" \
  --mode proxy
```

`--stream` is mutually exclusive with `--gxparm`, `--integrate`, and `--xdsinp`.
In stream mode, each indexed crystal block is treated as one snapshot frame (ordered by appearance in the stream).

Using PETS2 project files:

```bash
python examples/run_analysis.py \
  --pets-project /path/to/LTA1_PETS/LTA1_petsdata \
  --composition "24 Si, 48 O" \
  --dmin 0.6 \
  --orientation-only \
  --output-dir analysis_output_pets
```

Optional explicit PETS files:

```bash
python examples/run_analysis.py \
  --pets-project /path/to/LTA1_PETS \
  --pets-pts2 /path/to/LTA1_PETS/LTA1.pts2 \
  --pets-ptsopt /path/to/LTA1_PETS/LTA1_petsdata/LTA1.ptsopt \
  --pets-rprofall /path/to/LTA1_PETS/LTA1_petsdata/LTA1.rprofall \
  --composition "24 Si, 48 O" \
  --output-dir analysis_output_pets
```

Pure orientation-only scoring (no Wilson scaling, no Bloch coupling terms):

```bash
python examples/run_analysis.py \
  --gxparm path/to/GXPARM.XDS \
  --integrate path/to/INTEGRATE.HKL \
  --composition "24 Si, 48 O" \
  --mode proxy \
  --orientation-only
```

Thickness-aware analysis at one thickness:

```bash
python examples/run_analysis.py \
  --gxparm path/to/GXPARM.XDS \
  --integrate path/to/INTEGRATE.HKL \
  --composition "24 Si, 48 O, 12 Na, 27 O" \
  --mode thickness \
  --thickness 100
```

Thickness scan:

```bash
python examples/run_analysis.py \
  --gxparm path/to/GXPARM.XDS \
  --integrate path/to/INTEGRATE.HKL \
  --composition "24 Si, 48 O, 12 Na, 27 O" \
  --mode thickness \
  --thickness-start 20 \
  --thickness-stop 300 \
  --thickness-step 10
```

Results are written to `analysis_output/` by default.

## Observation-Level Dynamical Uncertainty (New)

A new intensity-independent pipeline is available in:

- `src/dynamical_uncertainty.py`
- `examples/run_dynamical_uncertainty.py`

This pipeline estimates **per-observation** dynamical uncertainty terms from:

- local orientation sensitivity around the indexed orientation
- thickness sensitivity over a user-selected thickness interval

It does **not** use observed intensities (`I`) or measured `sigma(I)` as model
inputs for risk calculation.

### Run (CrystFEL stream, orientation-driven)

```bash
python examples/run_dynamical_uncertainty.py \
  --stream /path/to/MFM300.stream \
  --dataset-id MFM300 \
  --orientation-axes xyz \
  --orientation-step-deg 0.05 \
  --orientation-n-steps 1 \
  --thickness-min-nm 20 \
  --thickness-max-nm 300 \
  --n-thickness-steps 15 \
  --dyn-sigma-form linear \
  --dyn-sigma-alpha 1.0 \
  --output-dir analysis_output_uncertainty_mfm300
```

### Run (XDS-style)

```bash
python examples/run_dynamical_uncertainty.py \
  --gxparm /path/to/GXPARM.XDS \
  --integrate /path/to/INTEGRATE.HKL \
  --dataset-id sample_xds \
  --output-dir analysis_output_uncertainty_xds
```

### Canonical Observation Table

The new pipeline converts all supported inputs into one canonical observation table (`canonical_observations.csv`) with at least:

- `obs_id`
- `source`, `dataset_id`
- `frame_input`, `frame_index`, `frame_number`, `event_id`
- `h`, `k`, `l`
- orientation matrix columns `UB11..UB33`
- geometry metadata (`wavelength_angstrom`, unit-cell params, detector geometry)

Optional column:

- `sigma_exp` (carried through for optional sigma inflation output only)

### Output Columns

`observation_dynamical_uncertainty.csv` includes:

- orientation metrics:
  - `ori_mean`, `ori_std`, `ori_cv`, `ori_range`
  - `ori_grad_rms`, `ori_curvature`, `ori_multipeak_score`
  - `zone_axis_proximity`, `zone_axis_layer_density`, `zone_axis_score`
- thickness metrics:
  - `thick_mean`, `thick_std`, `thick_cv`, `thick_range`
  - `thick_derivative_rms`, `thick_max_min_ratio`
- combined risk:
  - `risk_orientation`, `risk_thickness`, `risk_total`, `risk_total_norm`
- uncertainty term:
  - `dyn_sigma_rel` (alias `dyn_uncertainty_rel`)
- optional (`--include-sigma-dyn`):
  - `sigma_dyn = sigma_exp * dyn_sigma_rel`

The CLI also writes compact summary tables in `compact_summaries/`:

- `frame_risk_summary.csv`
- `hkl_risk_summary.csv`
- `topN_observations_by_risk.csv`
- `frame_zone_axis_focus_summary.csv` (top-10%-zone-axis observations per frame)

### Formulas (Transparent Defaults)

Local response proxy for one observation uses:

- excitation error `s_g` from indexed geometry
- reciprocal length `q = |g|`
- proxy structure factor `F_g,proxy = fg_scale / (1 + (q / q0)^2)`
- extinction-like scale `xi = pi * V / (lambda * F_g,proxy)`
- two-beam damping `d2 = 1 / (1 + (s_g * xi)^2)`
- oscillation term over thickness `sin^2(pi * t / xi_eff)`, with `xi_eff = xi * sqrt(1 + (s_g*xi)^2)`

Orientation screening perturbs the indexed orientation in configurable symmetric steps around chosen axes (`x/y/z`), then computes orientation metrics from local response variation.

To capture strong zone-axis behavior (including ZOLZ-heavy patterns), the orientation risk also includes a zone-axis/layer-crowding term:

- `zone_axis_proximity = exp(-( |g_z| / (w * |g|) )^2)`
- `zone_axis_layer_density` from Gaussian density in `g_z` over observations in the same frame
- `zone_axis_score = zone_axis_proximity * zone_axis_layer_density`

This contributes to `risk_orientation` with configurable weight (`--zone-axis-boost-weight`).

Thickness screening evaluates the same observation at indexed orientation over a configurable thickness grid, then computes thickness metrics from that response curve.

Combined risk:

`risk_total = sqrt((w_ori * risk_orientation)^2 + (w_thick * risk_thickness)^2)`

Default uncertainty mapping:

- linear: `dyn_sigma_rel = 1 + alpha * risk_total_norm`
- exponential: `dyn_sigma_rel = exp(alpha * risk_total_norm)`

`risk_total_norm` is normalized by a configurable quantile (`risk_normalization_quantile`) for numerical robustness.

## Interactive HTML visualization

To generate an interactive HTML report showing predicted dynamical peaks on the detector across orientations:

```bash
python examples/export_orientation_peaks_html.py \
  --gxparm data/GXPARM.XDS \
  --integrate data/INTEGRATE.HKL \
  --xdsinp data/XDS.INP \
  --composition "24 Si, 48 O" \
  --mode proxy \
  --output-html analysis_output/orientation_dynamical_peaks.html
```

The report contains:

- frame-wise trend panel (e.g., `S_MB` vs frame)
- animated detector panel with predicted reflection positions per frame
- reflection coloring and sizing by your selected score column (default `S_comb`)

Single-frame 3D Gaussian view (for example frame 45):

```bash
python examples/export_single_frame_gaussian_html.py \
  --gxparm data/GXPARM.XDS \
  --integrate data/INTEGRATE.HKL \
  --xdsinp data/XDS.INP \
  --composition "24 Si, 48 O" \
  --frame-number 45 \
  --score-column S_comb \
  --output-html analysis_output/frame45_gaussian_3d.html
```

In this view, each predicted reflection contributes a Gaussian peak at its detector position.
Peak sigma and color are both mapped from the selected score column, so more dynamically affected reflections appear broader and redder.

## Verifying predicted thickness trends against real reflection data

If you have multiple thickness datasets (for example `LTA_t1..LTA_t4` or `STW_t1_360..STW_t4_360`) under `data/`, you can compare predicted intensity-vs-thickness trends against observed `INTEGRATE.HKL` intensities:

```bash
python examples/verify_thickness_against_real.py \
  --sample LTA \
  --composition "24 Si, 48 O" \
  --frame-number 45 \
  --phi-tolerance-deg 0.8 \
  --output-dir analysis_output
```

Outputs include:

- `*_pred_vs_obs_long.csv`: merged predicted/observed long table
- `*_reflection_correlations.csv`: per-reflection Pearson/Spearman agreement over thickness
- `*_summary.txt`: compact run summary
- diagnostic plots (`*_pred_vs_obs_scatter.png`, `*_top_trend_matches.png`)

## Proxy mode vs thickness-aware mode

### Proxy mode

This reproduces the original browser logic as closely as practical:

- candidate reflections are generated from the GXPARM reference reciprocal basis
- each frame orientation is reconstructed from the XDS rotation series
- reflections are marked as excited if they cross the Ewald sphere between frame endpoints or if their mid-frame excitation error is below a tolerance
- the transmitted + diffracted beam coupling matrix is assembled and diagonalized
- per-reflection scores are derived from:
  - extinction distance `xi_g`
  - excitation error `s_g`
  - two-beam score `d_2beam = 1 / (1 + (s_g * xi_g)^2)`
  - effective coupling multiplicity `N_eff`
  - combined proxy score `S_comb = d_2beam * N_eff`
  - orientation-only uncertainty columns:
    - `orientation_sigma_sg_invA` (sigma of `s_g` from orientation uncertainty only)
    - `orientation_p_excited` (probability of being near excitation under orientation uncertainty)
    - `S_dyn` (geometry-only dynamical-risk score from the coupling environment)
    - `S_orient` (alias of `S_dyn` in `--orientation-only` runs, retained for existing plots)
    - `sigma_orient_scale = 1 + alpha * S_orient`

In `--orientation-only` runs, target `s_g` is treated mainly as a visibility/relevance gate.
The dynamical-risk score is instead driven by low-index strength, ZOLZ proximity to low-index
zone axes, excited-neighbor density, reciprocal-row coupling, two-step pathways `0 -> h -> g`,
and local reciprocal-space crowding. Component columns such as `strength_proxy`,
`ZOLZ_zone_axis_risk`, `neighbor_density`, `row_coupling`, `pathway_risk`, and
`local_crowding` are written to `reflections_long.csv`.

### Thickness-aware mode

This keeps the same structure matrix but adds propagation through thickness:

- the real symmetric coupling matrix is diagonalized
- the incident state is taken as a unit-amplitude transmitted beam at `t = 0`
- amplitudes are propagated as `psi(t) = V exp(2 pi i Lambda t) V^T psi(0)`
- thickness is supplied in nanometers and internally converted to angstroms
- diffracted-beam intensities `|psi_g(t)|^2` are recorded for one thickness or a thickness grid
- sensitivity metrics are computed per reflection and summarized per frame

Because the original HTML is a proxy model, absolute scaling should be treated cautiously. The implementation here is designed to be internally consistent with the coupling matrix used by the prototype, not to claim a rigorously normalized dynamical diffraction solver.

## Limitations and assumptions

- `GXPARM.XDS` parsing follows the line layout used by the supplied analyser and typical XDS exports.
- In `--orientation-only` mode, `S_orient` is the geometry-only `S_dyn` score and coupling-derived columns (`S_MB`, `S_comb`, `N_eff`) are set to zero.
- Equivalent-reflection merging uses the deliberately simple absolute-value sorting key inherited from the HTML prototype.
- The reciprocal basis is taken as the inverse of the GXPARM real-space reference matrix, matching the original script.
- Untrusted detector rectangles are parsed and visualized; they are not removed from the dynamical calculation unless you extend the code.
- The thickness-aware mode is a useful research extension, but it is still built on approximate amplitudes calibrated from Wilson-like scaling.
- Rotation-series and snapshot SerialED are both supported; for stream-only snapshot inputs, frame order is the crystal-block order in the stream.

## Future extensions

The code is intentionally structured so that it can be extended to support:

- reflection filtering near problematic orientations
- exporting per-reflection uncertainty terms for scaling studies
- exclusion of ZOLZ reflections
- orientation-aware merging experiments
- alternate structure-factor models and more realistic scattering constants

Frame-specific orientation matrices from an external table are now supported via
`ReciprocalMatrixOrientationModel` and `reciprocal_lookup_from_table` in `src.geometry`.

## Testing

Run:

```bash
pytest
```

The tests cover lightweight parsing, rotation, composition parsing, and metric sanity checks.

## Orientation sigma controls (CLI)

`examples/run_analysis.py` now accepts orientation-uncertainty knobs:

- `--orientation-sigma-deg` isotropic orientation uncertainty (degrees)
- `--orientation-sigma-axis-deg SX SY SZ` anisotropic uncertainty around lab axes
- `--orientation-score-formulation {log_n_eff,linear_n_eff}`
- `--orientation-sigma-alpha` for `sigma_orient_scale`
- `--orientation-only` to skip Wilson/coupling terms and score purely from orientation uncertainty

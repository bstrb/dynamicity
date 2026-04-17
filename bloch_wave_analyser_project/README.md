# Bloch-wave dynamicality analyser

A research-oriented Python project for reconstructing XDS rotation-series orientations, generating candidate reflections, and computing reflection-wise dynamicality metrics inspired by a Bloch-wave coupling model.

The codebase recreates the logic of the supplied browser-based analyser while extending it with an optional thickness-aware Bloch propagation mode.

## Purpose

This project is aimed at method development in electron crystallography and SerialED workflows where you want to:

- parse `GXPARM.XDS`, `INTEGRATE.HKL`, and optionally `XDS.INP`
- parse CrystFEL `.stream` output directly for snapshot SerialED
- parse PETS project outputs (`.pts2/.pts2.backup` + `.rprofall`) directly
- reconstruct frame orientations from `phi0`, `dphi`, and the XDS rotation axis
- generate candidate reciprocal-lattice vectors from the GXPARM reference basis
- compute excitation errors and reflection-wise dynamicality proxy scores
- compute orientation-only per-reflection uncertainty terms (`orientation_sigma_sg_invA`, `orientation_p_excited`, `S_orient`)
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

Using PETS2 `.rprofall` instead of `INTEGRATE.HKL`:

```bash
python examples/run_analysis.py \
  --gxparm path/to/GXPARM.XDS \
  --rprofall path/to/LTA_t1.rprofall \
  --composition "24 Si, 48 O" \
  --mode proxy
```

`--integrate` and `--rprofall` are mutually exclusive.

Using PETS project files only (no GXPARM / INTEGRATE needed):

```bash
python examples/run_analysis.py \
  --pets-project path/to/pets_project_folder \
  --composition "64 C, 8 H, 40 O, 8 V" \
  --mode proxy
```

Optional explicit `.rprofall` override in PETS mode:

```bash
python examples/run_analysis.py \
  --pets-project path/to/lta1_new.pts2.backup \
  --pets-rprofall path/to/lta1_new.rprofall \
  --composition "64 C, 8 H, 40 O, 8 V" \
  --mode proxy
```

Using CrystFEL `.stream` only (no GXPARM / INTEGRATE required):

```bash
python examples/run_analysis.py \
  --stream path/to/indexing.stream \
  --composition "24 Si, 48 O" \
  --mode proxy
```

`--stream` is mutually exclusive with `--pets-project`, `--pets-rprofall`, `--gxparm`, `--integrate`, `--rprofall`, and `--xdsinp`.
In stream mode, each indexed crystal block is treated as one snapshot frame (ordered by appearance in the stream).

Pure orientation-only scoring (no Wilson scaling, no Bloch coupling terms):

```bash
python examples/run_analysis.py \
  --pets-project path/to/pets_project_folder \
  --composition "64 C, 8 H, 40 O, 8 V" \
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
    - `S_orient` (orientation-only dynamical-risk proxy)
    - `sigma_orient_scale = 1 + alpha * S_orient`

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
- PETS-only mode uses `.rprofall` for observations and reads geometry/orientation essentials from `.pts2/.pts2.backup` (`lambda`, `aperpixel`, unit cell, `ubmatrix`, and `imagelist` angles/centers).
- PETS frame rotations are reconstructed from `alpha`, `beta`, and `domega` in the order `Rz(domega) * Ry(beta) * Rx(alpha)`; this is an explicit approximation for interoperability.
- In `--orientation-only` mode, `S_orient` is set directly to `orientation_p_excited` and coupling-derived columns (`S_MB`, `S_comb`, `N_eff`) are set to zero.
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

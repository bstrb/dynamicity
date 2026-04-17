# xds_rocking_curves

`xds_rocking_curves` is a focused Python research project for reconstructing **local rocking curves** directly from raw TIFF diffraction images using **XDS-derived orientations and geometry**.

The first working target is deliberately narrow:

- one dataset
- one manually chosen reflection `(h,k,l)`
- predicted positions from `GXPARM.XDS` / `XPARM.XDS`
- Gaussian fitting on only the **relevant frames**
- per-frame output table and rocking-curve plots

This is designed as a proof-of-concept base for later cross-thickness comparison by **orientation matching**.

## What it does

For a chosen reflection `(h,k,l)` the pipeline:

1. parses `GXPARM.XDS` / `XPARM.XDS`
2. optionally parses `XDS.INP`, `SPOT.XDS`, and `INTEGRATE.HKL`
3. reconstructs frame-wise orientations from the XDS rotation model
4. predicts detector coordinates `(x,y)` for the reflection on each frame
5. selects only **relevant frames** near the excitation condition
6. reads the TIFF images for those frames
7. extracts a small patch around the predicted position
8. fits a **2D elliptical Gaussian + constant background**
9. saves a per-frame rocking curve table and plots

## Main design choices

- **Predicted positions are the backbone**
- `SPOT.XDS` is used only for validation / sanity checking
- fitting uses a **2D elliptical Gaussian**, not a box sum
- only **relevant frames** are analyzed
- output intensity is the **background-subtracted integrated Gaussian intensity**

## Project layout

```text
xds_rocking_curves/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── parsers.py
│   ├── geometry.py
│   ├── prediction.py
│   ├── image_io.py
│   ├── gaussian_fit.py
│   ├── rocking_curve.py
│   ├── plotting.py
│   ├── pipeline.py
│   └── synthetic.py
├── examples/
│   ├── run_single_reflection.py
│   └── synthetic_demo.py
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Example: analyze one reflection in one real dataset

```bash
python examples/run_single_reflection.py \
  --gxparm /path/to/GXPARM.XDS \
  --xds-inp /path/to/XDS.INP \
  --spot-xds /path/to/SPOT.XDS \
  --integrate-hkl /path/to/INTEGRATE.HKL \
  --image-glob "/path/to/images/*.tif" \
  --dataset-name LTA_t3 \
  --thickness-nm 350 \
  --hkl 1 2 0 \
  --relevance-mode sg \
  --sg-threshold 0.02 \
  --patch-half-size 7 \
  --output outputs/LTA_t3_hkl_1_2_0
```

If you prefer a frame window around the predicted crossing instead of an `|sg|` cutoff:

```bash
python examples/run_single_reflection.py \
  --gxparm /path/to/GXPARM.XDS \
  --image-glob "/path/to/images/*.tif" \
  --hkl 1 2 0 \
  --relevance-mode window \
  --window-half-width 4 \
  --output outputs/LTA_t3_hkl_1_2_0
```

## Output files

The pipeline writes:

- `rocking_curve.csv`
- `prediction_table.csv`
- `rocking_curve.png`
- `rocking_curve_normalized.png`
- `detector_track.png`
- `analysis_metadata.json`

The main rocking-curve CSV contains columns such as:

```text
dataset, thickness_nm, frame, phi_deg, h, k, l,
x_pred, y_pred,
x_fit, y_fit,
I_fit, bg, sigma_fit,
fit_success, fit_message,
rmse, r_squared,
sg,
spot_x, spot_y, spot_distance_px, spot_indexed_match,
image_path
```

## Synthetic demo

A small synthetic dataset generator is included so the project can be tested even without real data.

```bash
python examples/synthetic_demo.py
```

This creates a synthetic TIFF stack, matching XDS metadata files, runs the single-reflection pipeline, and writes demo outputs under:

```text
synthetic_demo_output/
```

## Notes on geometry

This code reconstructs frame-wise reciprocal-basis matrices from the XDS rotation model and predicts detector positions by intersecting the diffracted-ray direction with the detector plane.

`INTEGRATE.HKL` is **not** used as the source of rocking-curve intensities. It is only used here as an optional validation source for choosing the rotation sign and checking prediction consistency.

## Limitations

- This is intentionally a **single-reflection PoC** project.
- It does **not** yet do cross-thickness matching.
- It does **not** fit all reflections automatically.
- It assumes a single detector segment in the first working path.
- The detector projection is modular on purpose so you can adjust it if your local convention needs tuning.

## Next step after this project

Once you verify that the predicted spot track and fitted local rocking curve look right for one reflection, the natural next extension is:

- batch extraction for many reflections
- orientation matching across thickness datasets
- curve-shape comparison or thickness-sensitivity scoring

## Running tests

```bash
pytest
```

# serialed_orientation_sigma

`serialed_orientation_sigma` is a Python research project for **orientation-aware dynamical weighting** in SerialED and related electron diffraction workflows.

The code is designed for datasets where each frame has its own indexed orientation matrix. Instead of trying to perform full dynamical diffraction simulation during routine processing, the pipeline computes **fast, orientation-conditioned risk descriptors** and converts them into **sigma inflation factors**, **weights**, or **filtering masks** that can be tested during downstream scaling and merging.

This is especially useful when you want to ask questions such as:

- Which frames are close to strong zone axes?
- Which reflections are close to multi-beam excitation conditions?
- Which reflections are most likely to be unstable under thickness variation?
- Does down-weighting orientation-sensitive reflections improve merged statistics?

## Scientific idea

In SerialED, each frame is usually treated as a weak, approximately kinematical snapshot. In practice, that assumption breaks down selectively:

- frames close to low-index zone axes can excite many reflections simultaneously,
- some reflections sit close to the Ewald sphere and behave like two-beam or near-multi-beam cases,
- thickness variations can amplify orientation-dependent intensity instability.

This project implements a **statistical dynamical weighting** strategy:

1. reconstruct or load per-frame orientations,
2. compute orientation metrics such as nearest zone axis and excitation density,
3. compute reflection-wise excitation error `sg` and a two-beam sensitivity proxy,
4. combine the metrics into a dynamical score `S`,
5. convert `S` into `sigma_new`, `weight_new`, and an optional `keep` mask.

The result is a flat reflection table that can be used for scaling experiments inspired by tools such as CrystFEL or Careless, without claiming to replace full multislice or full Bloch-wave refinement.

## Features

- Two input modes:
  - **XDS rotation series** via `GXPARM.XDS` + `INTEGRATE.HKL` + optional `XDS.INP`
  - **SerialED snapshot mode** via per-frame `UB` matrices + flat reflection table
- Orientation metrics:
  - nearest low-index zone axis
  - zone-axis angular distance
  - multi-beam excitation count `N_excited`
  - excitation density in a configurable resolution shell
- Reflection metrics:
  - exact Ewald-style excitation error proxy `sg`
  - approximate extinction-distance proxy `xi_g`
  - two-beam sensitivity `d_2beam = 1 / (1 + (sg * xi_g)^2)`
  - combined dynamical score `S`
- Weighting schemes:
  - multiplicative sigma inflation
  - weight down-scaling
  - reflection filtering mask
- Validation module:
  - simplified Bloch-like thickness scan for selected reflections
- Utilities:
  - plotting helpers
  - notebooks
  - CSV export for scaling experiments
  - optional multiprocessing over frame chunks

## Project layout

```text
serialed_orientation_sigma/
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
├── src/
├── notebooks/
├── examples/
└── tests/
```

The import namespace is intentionally `src` to match the file layout you requested.

## Installation

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### conda

```bash
conda env create -f environment.yml
conda activate serialed-orientation-sigma
pip install -e .
```

## Input formats

### 1. SerialED snapshot mode

Orientation table example:

```text
frame,UB11,UB12,UB13,UB21,UB22,UB23,UB31,UB32,UB33
1,0.100,0.000,0.000,0.000,0.100,0.000,0.000,0.000,0.100
2,0.099,0.000,0.012,0.000,0.100,0.000,-0.012,0.000,0.099
```

Reflection table example:

```text
frame,h,k,l,I,sigma,x,y
1,1,0,0,1520.0,38.0,1024.2,1018.9
1,0,1,0,1498.0,35.0,1031.6,990.4
2,1,0,0,1304.0,42.0,1002.8,1010.1
```

Cell file example (`cell.json`):

```json
{
  "a": 18.5,
  "b": 18.5,
  "c": 27.2,
  "alpha": 90.0,
  "beta": 90.0,
  "gamma": 120.0,
  "voltage_kv": 200.0,
  "composition": {
    "Zn": 4,
    "C": 24,
    "N": 8,
    "O": 4,
    "H": 24
  }
}
```

### 2. XDS rotation-series mode

Required files:

* `GXPARM.XDS`
* `INTEGRATE.HKL`
* optional `XDS.INP`

The parser extracts:

* reference orientation matrix from the direct cell axes stored in `GXPARM.XDS`,
* `rotation_axis`, `phi0`, and `dphi`,
* frame IDs from `INTEGRATE.HKL` using the `ZD` coordinate when needed.

## Running the pipeline

### Snapshot mode

```bash
python examples/run_pipeline.py \
  --orientations orientations.csv \
  --reflections reflections.csv \
  --cell cell.json \
  --dmin 0.6 \
  --dmax 50 \
  --output results/
```

### XDS mode

```bash
python examples/run_pipeline.py \
  --gxparm GXPARM.XDS \
  --integrate INTEGRATE.HKL \
  --xds-inp XDS.INP \
  --output results/
```

### Recommended high-throughput knobs

For large datasets, start with conservative proxy settings:

```bash
python examples/run_pipeline.py \
  --orientations orientations.csv \
  --reflections reflections.csv \
  --cell cell.json \
  --dmin 0.8 \
  --dmax 20 \
  --hkl-limit 20 \
  --max-candidate-reflections 50000 \
  --chunk-size-frames 5000 \
  --n-workers 4 \
  --output results/
```

`hkl_limit` and `max_candidate_reflections` are important when you work with very large cells. They deliberately turn the excitation-density calculation into a **controlled proxy** instead of an exhaustive enumeration.

## Main outputs

### `frame_summary.csv`

Columns include:

* `frame`
* `phi`
* `nearest_zone_axis`
* `zone_axis_angle`
* `N_excited`
* `excitation_density`
* `mean_dynamical_score`
* `n_reflections`

### `reflection_scores.csv`

Columns include:

* `frame`
* `h`, `k`, `l`
* `I`, `sigma`
* `sg`
* `xi_g`
* `d_2beam`
* `zone_axis_proximity`
* `multi_beam_density`
* `S`
* `sigma_new`
* `weight_new`
* `keep`

### `merge_weights.csv`

A compact export for scaling experiments containing the columns most useful for downstream merging.

### `run_provenance.json`

Machine-readable run log for reproducibility, including:

* CLI arguments,
* UTC timestamp,
* current working directory,
* Python executable,
* git commit/branch/dirty status,
* entrypoint script identifier.

## Notebooks

### `notebooks/orientation_analysis.ipynb`

Shows how to:

* load or synthesize orientations,
* compute frame-wise orientation metrics,
* plot orientation trajectory and zone-axis distance.

### `notebooks/reflection_sensitivity.ipynb`

Shows how to:

* compute reflection-wise `sg`, `d_2beam`, and `S`,
* inspect score distributions,
* view detector-space score maps.

### `notebooks/thickness_validation.ipynb`

Shows how to:

* run simplified thickness scans,
* compare reflections with different `sg` and `xi_g`,
* inspect which reflections are thickness-sensitive.

## Relation to dynamical scattering

This repository is intentionally **not** a full dynamical refinement engine.

Instead, it uses three layers of approximation:

1. **zone-axis proximity** as a risk marker for stronger multi-beam coupling,
2. **excitation density** as a per-orientation crowding proxy,
3. **two-beam sensitivity and thickness scans** as reflection-wise instability proxies.

That makes it suitable for rapid weighting experiments on very large SerialED datasets.

## Limitations

* `sg` is computed with a compact Ewald-based proxy, not a full diffracted-beam solution.
* `xi_g` is estimated from a structure-factor proxy, not from full electron scattering factors.
* `nearest_zone_axis` only searches integer zone axes up to `zone_axis_limit`.
* Thickness validation uses a simplified Bloch-like oscillation model for hypothesis testing, not for production refinement.
* Very large unit cells require careful tuning of `hkl_limit` and `max_candidate_reflections`.

## Suggested workflow

1. Run the pipeline with conservative defaults.
2. Inspect `frame_summary.csv` for suspicious zone-axis neighborhoods.
3. Inspect `reflection_scores.csv` for the distribution of `S`.
4. Test merging with:

   * original `sigma`,
   * `sigma_new`,
   * `weight_new`,
   * `keep == True` filtering.
5. Use `thickness_validation.py` on a few representative reflections to see whether high-`S` reflections also show stronger thickness sensitivity.

## Running tests

```bash
pytest
```

## Future extensions

* symmetry-aware aggregation before export,
* optional calibration of `alpha` against merged-map or R-split behavior,
* empirical fitting of score-to-sigma transforms,
* tighter coupling to CrystFEL stream or MTZ-style exports,
* user-supplied scattering-factor models.

# Parallel CrystFEL Stream Analyser
# OriDyn Stream Parallel Project

Parallel CrystFEL stream analyser for orientation-dependent dynamical-diffraction
risk.

## What This Project Does

This project analyses a CrystFEL `.stream` file frame by frame. It does not sort
or rewrite the stream. Instead, it produces CSV tables that answer:

- which indexed frames are close to low-index zone axes;
- which reflections are excited and land on the detector for each frame;
- which reflections look most vulnerable to dynamical diffraction;
- how much an observed reflection's `sigma(I)` could be inflated by a geometry
  based risk proxy.

It is meant for large SerialED/CrystFEL stream
runs where thousands of independently indexed frames make the original serial
pipeline too slow and too quiet.
It is meant for large SerialED/CrystFEL stream runs where thousands of
independently indexed crystals make a serial analysis slow. The runner
parallelizes over frames, writes checkpoint chunk CSVs during the run, then
concatenates those chunks into final summary tables.

The runner parallelizes over frames, writes checkpoint chunk CSVs while it
runs, and concatenates them at the end.
## Main Entry Point

## Typical Use
Run from inside the project folder:

```bash
python oridyn_stream_parallel_project/run_stream_parallel.py \
cd /Users/xiaodong/Desktop/dynamicity/oridyn_stream_parallel_project

python3 run_stream_parallel.py \
  --stream /path/to/file.stream \
  --composition "62 C, 24 H, 40 O, 8 V" \
  --dmin 0.4 \
  --dmax 20.0 \
  --excitation-tolerance 0.01 \
  --workers 0 \
  --chunk-size 5 \
  --output-dir oridyn_stream_parallel_project/output_stream_dyn
  --output-dir output_stream_dyn \
  --skip-legacy-s-dyn
```

`--workers 0` uses all logical CPUs. Use a smaller value if memory pressure is
too high. `--chunk-size` controls how many frames each worker processes before
writing a checkpoint.
writing a checkpoint CSV.

`--composition` is accepted for compatibility with the older Bloch-wave project,
but this stream-only geometry screen does not use composition for the main
two-channel risk scores.

## Inputs Read From The Stream

The parser reads:

- wavelength;
- detector distance, pixel size, detector size, and beam center;
- unit-cell parameters;
- per-crystal `astar`, `bstar`, and `cstar` reciprocal vectors;
- per-crystal detector shifts and camera length when present;
- measured reflection rows with `h k l I sigma peak background fs ss panel`.

Each indexed crystal block is treated as one independent frame.

## How The Analysis Works

For each frame, the runner:

1. Builds all candidate HKLs in the requested d-spacing range:

   ```text
   --dmin <= d <= --dmax
   ```

2. Uses the frame's `astar/bstar/cstar` matrix to place those HKLs in laboratory
   reciprocal space.

3. Computes excitation error `sg` for each candidate reflection.

4. Projects each candidate reflection to detector pixels.

5. Keeps reflections that are both:

   ```text
   abs(sg) < --excitation-tolerance
   ```

   and on the detector.

6. Finds the nearest low-index zone axis for the frame.

7. Computes reflection-level dynamical-risk proxy columns.

8. Merges in the actually observed stream intensity and sigma when that HKL was
   observed in the frame.

The calculation is intentionally a screening proxy. It is not a full refinement
or a replacement for dynamical diffraction simulation. It is designed to flag
frames/reflections where geometry suggests stronger multi-beam coupling,
self-extinction, or local reciprocal-space crowding.

## Main Risk Columns

The most useful per-reflection columns in `reflections_long.csv` are:

```text
sg_invA
```

Excitation error for the candidate reflection in `1/Angstrom`.

```text
self_extinction_score
attenuation_risk
```

Risk from the target reflection itself being strongly excited. `attenuation_risk`
is a bounded transformed version of the self score.

```text
same_zone_cluster_score_geom
systematic_row_cluster_score_geom
cluster_risk_geom
```

Geometry-only multi-beam risk from nearby excited beams in the same zone/layer
and along systematic reciprocal rows.

```text
cluster_risk_iw
```

Similar to `cluster_risk_geom`, but weighted by an empirical intensity-derived
strength proxy from observed reflections.

```text
total_dynamical_risk_geom
total_dynamical_risk_iw
```

Combined self and cluster risk estimates.

```text
sigma_dyn_rel
sigma_new
weight_new
```

`sigma_dyn_rel` is the relative sigma inflation factor:

```text
sigma_dyn_rel = 1 + alpha * cluster_risk_geom
```

where `alpha` is controlled by `--dynamical-cluster-sigma-alpha`.

For the fastest geometry/two-channel screen, add:
For observed reflections:

```bash
--skip-legacy-s-dyn
```text
sigma_new = sigma_obs * sigma_dyn_rel
weight_new = 1 / sigma_new^2
```

That skips the older `geometry_dynamical_risk()` pathway score and keeps the
newer two-channel columns such as `cluster_risk_geom`, `cluster_risk_iw`,
`attenuation_risk`, `total_dynamical_risk_geom`, and `sigma_dyn_rel`.
These columns are useful if you later want to test uncertainty inflation or
down-weighting outside this project.

## Frame Summary

`frame_summary.csv` has one row per analysed frame. Useful columns include:

```text
frame_number
zone_axis
zone_axis_angle_deg
n_excited
sum_self_extinction_score
mean_attenuation_risk
sum_cluster_score_geom
mean_cluster_risk_geom
p95_cluster_risk_geom
sum_cluster_score_iw
mean_cluster_risk_iw
p95_cluster_risk_iw
mean_sigma_dyn_rel
n_observed_targets
```

Use this table to find whole frames/orientations that look especially
dynamical.

## Outputs

- `frame_summary.csv`
- `reflections_long.csv`
- `two_channel_summary.txt`
- `top_self_extinction_risk.csv`
- `top_cluster_risk_geom.csv`
- `top_cluster_risk_intensity_weighted.csv`
- `top_observed_self_extinction_risk.csv`
- `top_observed_cluster_risk_geom.csv`
- `chunk_manifest.csv`
- `chunks/frame_summary_part_*.csv`
- `chunks/reflections_long_part_*.csv`
The final output directory contains:

```text
frame_summary.csv
reflections_long.csv
two_channel_summary.txt
top_self_extinction_risk.csv
top_cluster_risk_geom.csv
top_cluster_risk_intensity_weighted.csv
top_observed_self_extinction_risk.csv
top_observed_cluster_risk_geom.csv
chunk_manifest.csv
run_metadata.json
chunks/frame_summary_part_*.csv
chunks/reflections_long_part_*.csv
```

The chunk files are kept by default so an interrupted run still leaves useful
partial results. Use `--remove-chunks` to delete them after final concatenation.

## Useful Options

```text
--analysis-frame N
```

Analyse one one-based frame number. Can be repeated.

```text
--analysis-frame-range START END
--analysis-frame-step N
```

Analyse a subset of frames.

```text
--skip-legacy-s-dyn
```

Skip the older `geometry_dynamical_risk()` pathway score. This is usually the
fastest mode and keeps the newer two-channel columns such as
`cluster_risk_geom`, `cluster_risk_iw`, `attenuation_risk`,
`total_dynamical_risk_geom`, and `sigma_dyn_rel`.

```text
--beam-direction plus_z|minus_z
```

Set the beam direction used for excitation and projection.

```text
--stream-det-shift-sign -1|1
--stream-mirror-x-axis
--no-stream-mirror-x-axis
--detector-xy-swapped
```

Adjust how stream detector shifts/projections are mapped into detector
coordinates.

```text
--orientation-sigma-deg FLOAT
```

Set the assumed orientation uncertainty used for orientation excitation
probability columns.

```text
--dynamical-environment-tolerance FLOAT
--dynamical-neighbor-radius FLOAT
--dynamical-zone-sigma FLOAT
--dynamical-row-direction-limit N
--dynamical-row-max-steps N
--dynamical-cluster-sigma-alpha FLOAT
```

Tune the local multi-beam environment and sigma inflation behavior.

## Relationship To The Orientation-Sorting Scripts

The scripts in `sort_by_orientation/` predict a list of risky axes and then sort,
split, filter, or rewrite streams based on closeness to those axes.

This project is more detailed and more diagnostic. It keeps the stream intact
and writes analysis tables instead. It scores individual candidate reflections
inside every frame, including local neighbor beams and systematic-row coupling,
then reports both reflection-level and frame-level risk summaries.
# Sort by Orientation Scripts

This folder contains a small workflow for finding stream frames that are close to problematic zone axes, then either sorting/splitting those frames or modifying their reflection tables.

The usual flow is:

1. Predict risky zone axes from a CrystFEL stream header.
2. Score every indexed frame by its closest risky axis.
3. Sort or split the stream by that frame score.
4. Either remove zone-law reflections from each frame, or divide `sigma(I)` values by the frame score.

Run the examples below from this folder:

```bash
cd /Users/xiaodong/Desktop/dynamicity/sort_by_orientation
```

## 1. Predict Risky Axes

Script:

```bash
python3 predict_risky_axes_scored.py \
  --stream /path/to/input.stream \
  --nrows 150 \
  --thickness-nm 20
```

Output:

```text
/path/to/input_problematic_axes_scored.csv
```

What it does:

- Reads the embedded geometry and unit cell from the `.stream` header.
- Enumerates candidate zone axes `[u v w]` up to `--uvw-max`.
- Counts ZOLZ reflections obeying the zone law `h*u + k*v + l*w = 0`.
- Applies lattice-centering extinction rules.
- Optionally gates reflections by excitation error using `--thickness-nm`.
- Weights low-order reflections more strongly with `--g0` and `--p`.
- Normalizes the weighted reflection density to a `Score` in `[0, 1]`.
- Writes the highest-risk axes to a CSV sorted by score.

Useful options:

```text
--uvw-max N          Maximum absolute u, v, w to enumerate. Default: 5.
--nrows N            Keep only the top N predicted axes.
--thickness-nm T     Use S_max approx 1/(2t), where t is thickness in Angstrom.
--g0 FLOAT           Low-order weighting scale in 1/Angstrom. Default: 0.40.
--p FLOAT            Low-order weighting exponent. Default: 1.5.
```

There is also an example wrapper:

```bash
./predict_risky_axes_scored.sh
```

Edit the hard-coded stream path and parameters in that `.sh` file before using it.

## 2. Score, Sort, or Split Stream Frames

Script:

```bash
python3 split_stream_by_axis_lattice.py \
  --from-csv /path/to/input_problematic_axes_scored.csv \
  --sort-angle \
  --metric angle_over_score
```

Outputs by default:

```text
/path/to/input_report.txt
/path/to/input_sorted.stream
```

What it does:

- Reads risky axes and their `Score` values from the predictor CSV.
- Uses the `# Stream: ...` CSV header to find the original stream, unless a stream path is supplied explicitly.
- Parses each stream chunk as one frame/event.
- Reads `astar`, `bstar`, and `cstar` from each indexed chunk.
- Converts the reciprocal basis to a real-space basis.
- Computes the angle between the beam direction and each risky `[u v w]` axis.
- Assigns each frame the nearest risky axis and that axis score.
- Writes a report line for each frame, for example:

```text
unknown_image //1058 -> [1 0 0] score=0.73 angle=2.41 angle/score=0.058
```

Important metric note:

The report field is called `angle/score` for historical reasons. In the current code, `--metric angle_over_score` is computed as:

```text
angle_norm * (2 - Score)
```

where `angle_norm = min(angle, 90) / 90`. Smaller values are considered closer to a high-risk axis.

Common modes:

```bash
# Write one sorted stream.
python3 split_stream_by_axis_lattice.py \
  --from-csv /path/to/input_problematic_axes_scored.csv \
  --sort-angle \
  --metric angle_over_score

# Split into count bins after sorting by the chosen metric.
python3 split_stream_by_axis_lattice.py \
  --from-csv /path/to/input_problematic_axes_scored.csv \
  --count-bins 0,1000,30000 \
  --metric angle_over_score \
  --also-sorted

# Split by metric value.
python3 split_stream_by_axis_lattice.py \
  --from-csv /path/to/input_problematic_axes_scored.csv \
  --metric-bins 0,0.25,0.5,1.0,2.0 \
  --metric angle_over_score

# Legacy pure-angle splitting.
python3 split_stream_by_axis_lattice.py \
  --from-csv /path/to/input_problematic_axes_scored.csv \
  --angle-bins 0,5,10,20 \
  --metric angle
```

Useful options:

```text
--beam X Y Z                 Beam direction. Default: 0 0 1.
--metric angle               Use only the angle to the nearest risky axis.
--metric angle_over_score    Use the combined closeness/risk metric.
--include-unindexed MODE     end, start, or drop. Default: drop.
--count-split N              Split a sorted stream into parts of N chunks.
--progress / --no-progress   Force progress display on or off.
```

There is also an example wrapper:

```bash
./split_stream_by_axis_lattice.sh
```

Edit the CSV path and any mode options in that `.sh` file before using it.

## 3A. Filter Zone Reflections

Script:

```bash
python3 filter_zone_from_stream.py \
  --in /path/to/input_sorted.stream \
  --out /path/to/input_sorted_noZOLZ_tol0.1.stream \
  --zone-list /path/to/input_report.txt \
  --tolerance 0.1
```

What it does:

- Streams through the input file without loading the whole stream into memory.
- Buffers one crystal block at a time.
- Gets the per-crystal zone axis from `--zone-list`, usually the report written by `split_stream_by_axis_lattice.py`.
- Removes reflection rows satisfying:

```text
abs(h*u + k*v + l*w) <= tolerance
```

- Patches `num_reflections = ...` for each filtered crystal.
- Preserves non-reflection lines as they are.

If no `--zone-list` is supplied, it uses a fixed fallback zone:

```bash
python3 filter_zone_from_stream.py \
  --in /path/to/input.stream \
  --out /path/to/output.stream \
  --zone 0 0 1 \
  --tolerance 0.1
```

The default zone `[0 0 1]` corresponds to classic ZOLZ filtering, where reflections near `l = 0` are removed.

Useful options:

```text
--zone u v w             Fallback fixed zone. Default: 0 0 1.
--zone-list REPORT       Text file containing one bracketed [u v w] per crystal.
--tolerance FLOAT        Drop reflections within this zone-law tolerance. Default: 0.1.
--limit-crystals N       Filter only the first N crystals.
```

There is also a more complete shell wrapper:

```bash
./filter_zone_from_stream.sh
```

Edit the `IN`, `ZONE_LIST`, `NUM_CRYSTALS`, and `TOL` variables at the top of that file.

## 3B. Divide Sigmas by Frame Score

Script:

```bash
python3 divide_sigmas_by_angle_score.py \
  /path/to/input_sorted.stream \
  /path/to/input_report.txt \
  /path/to/input_sigmas_divided.stream
```

What it does:

- Reads the report from `split_stream_by_axis_lattice.py`.
- Extracts the `angle/score=` value for each event.
- Finds the matching `Event: //<frame>` in the stream.
- Rewrites each reflection row with:

```text
new_sigma = old_sigma / angle_score
```

- Preserves the rest of the stream structure.

This script expects every event in the stream to have a matching finite report entry. It raises an error if a report value is missing, duplicated, or zero.

## Current Active Files

```text
predict_risky_axes_scored.py        Predict and score problematic zone axes.
split_stream_by_axis_lattice.py     Score, sort, or split stream chunks by closest risky axis.
filter_zone_from_stream.py          Remove reflections close to each frame's assigned zone law.
divide_sigmas_by_angle_score.py     Divide sigma(I) values by the per-frame report metric.

predict_risky_axes_scored.sh        Example wrapper for the predictor.
split_stream_by_axis_lattice.sh     Example wrapper for the splitter.
filter_zone_from_stream.sh          Example wrapper for zone filtering.
```

`old_predict_scripts/` contains older variants and experiments. Use the top-level scripts above for the current workflow.

## Minimal End-to-End Example

```bash
cd /Users/xiaodong/Desktop/dynamicity/sort_by_orientation

python3 predict_risky_axes_scored.py \
  --stream /data/sample.stream \
  --nrows 150 \
  --thickness-nm 20

python3 split_stream_by_axis_lattice.py \
  --from-csv /data/sample_problematic_axes_scored.csv \
  --sort-angle \
  --metric angle_over_score

python3 filter_zone_from_stream.py \
  --in /data/sample_sorted.stream \
  --out /data/sample_sorted_noZOLZ_tol0.1.stream \
  --zone-list /data/sample_report.txt \
  --tolerance 0.1
```

Alternative final step:

```bash
python3 divide_sigmas_by_angle_score.py \
  /data/sample_sorted.stream \
  /data/sample_report.txt \
  /data/sample_sigmas_divided.stream
```

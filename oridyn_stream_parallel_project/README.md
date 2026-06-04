# Parallel CrystFEL Stream Analyser

This is a stream-only side project copied from the reusable pieces of
`bloch_wave_analyser_project`. It is meant for large SerialED/CrystFEL stream
runs where thousands of independently indexed frames make the original serial
pipeline too slow and too quiet.

The runner parallelizes over frames, writes checkpoint chunk CSVs while it
runs, and concatenates them at the end.

## Typical Use

```bash
python bloch_stream_parallel_project/run_stream_parallel.py \
  --stream /path/to/file.stream \
  --composition "62 C, 24 H, 40 O, 8 V" \
  --dmin 0.4 \
  --dmax 20.0 \
  --excitation-tolerance 0.01 \
  --workers 0 \
  --chunk-size 5 \
  --output-dir bloch_stream_parallel_project/output_stream_dyn
```

`--workers 0` uses all logical CPUs. Use a smaller value if memory pressure is
too high. `--chunk-size` controls how many frames each worker processes before
writing a checkpoint.

For the fastest geometry/two-channel screen, add:

```bash
--skip-legacy-s-dyn
```

That skips the older `geometry_dynamical_risk()` pathway score and keeps the
newer two-channel columns such as `cluster_risk_geom`, `cluster_risk_iw`,
`attenuation_risk`, `total_dynamical_risk_geom`, and `sigma_dyn_rel`.

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

The chunk files are kept by default so an interrupted run still leaves useful
partial results. Use `--remove-chunks` to delete them after final concatenation.


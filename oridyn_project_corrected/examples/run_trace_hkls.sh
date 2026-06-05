#!/usr/bin/env bash
set -euo pipefail

python -m oridyn.cli trace-hkls \
  --stream examples/minimal_example.stream \
  --hkls examples/selected_hkls.csv \
  --output example_trace_out \
  --dmin 0.6 \
  --dmax 20 \
  --uvw-max 5 \
  --normalization rank \
  --frame-normalization rank \
  --workers 1

python -m oridyn.cli reweight \
  --scores example_trace_out/hkl_frame_trajectories.csv \
  --weights examples/weight_presets.json \
  --output example_trace_out/hkl_frame_trajectories_reweighted.csv \
  --alpha 1.0

python -m oridyn.cli plot-hkl-traces \
  --scores example_trace_out/hkl_frame_trajectories_reweighted.csv \
  --output example_trace_out/plots/reweighted_hkl_traces \
  --score-columns S_dyn_geom_default S_dyn_geom_self_only S_dyn_geom_graph_heavy S_dyn_geom_zone_row_heavy

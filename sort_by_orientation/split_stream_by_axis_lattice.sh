# split_stream_by_axis_lattice_from_csv.sh

# python split_stream_by_axis_lattice_from_csv.py \
python split_stream_by_axis_lattice.py \
  --from-csv /home/bubl3932/files/UOX1/xgandalf_iterations_max_radius_0.71_step_0.5/metrics_run_20250922-151741/filtered_metrics_problematic_orientations.csv \
  --report \
  --sort-angle \
  --metric angle_over_score \
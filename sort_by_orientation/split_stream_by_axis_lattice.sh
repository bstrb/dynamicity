# split_stream_by_axis_lattice_from_csv.sh

# python split_stream_by_axis_lattice_from_csv.py \
python split_stream_by_axis_lattice.py \
  --from-csv /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/xgandalf_iterations_max_radius_0.5_step_0.2/metrics_run_20250915-100953/filtered_metrics_problematic_orientations.csv \
  --report \
  --sort-angle \
  --metric angle_over_score \
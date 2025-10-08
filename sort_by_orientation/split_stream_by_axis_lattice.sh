# split_stream_by_axis_lattice.sh

python split_stream_by_axis_lattice.py \
  --from-csv /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/zero_fitering/filtered_metrics_problematic_axes_scored.csv \
  --sort-angle \
  --metric angle_over_score \
  --count-split 5120 \
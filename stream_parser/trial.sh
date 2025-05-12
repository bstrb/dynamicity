#!/usr/bin/env bash

python stream_to_dataframe.py /home/bubl3932/files/MFM300_VIII/simulation-2/xgandalf_iterations_max_radius_1_step_0.5/filtered_metrics/filtered_metrics.stream 

# much looser – see if spikes level out and the 1.0 plateau disappears
scale_serial_ed /home/bubl3932/files/MFM300_VIII/MFM_spot3_streams/filtered_metrics/filtered_metrics.stream --cutoff 3.0 --out merged_cut3.csv

# moderate
scale_serial_ed /home/bubl3932/files/MFM300_VIII/MFM_spot3_streams/filtered_metrics/filtered_metrics.stream --cutoff 2.5 --out merged_cut2p5.csv

# strict 1.0
scale_serial_ed /home/bubl3932/files/MFM300_VIII/MFM_spot3_streams/filtered_metrics/filtered_metrics.stream --cutoff 1.0 --out merged_cut1.csv

# super strict 0.2
scale_serial_ed /home/bubl3932/files/MFM300_VIII/MFM_spot3_streams/filtered_metrics/filtered_metrics.stream --cutoff 0.2 --out merged_cut0p5.csv

# faster Bayesian run with progress bar
scale_serial_ed /home/bubl3932/files/MFM300_VIII/MFM_spot3_streams/filtered_metrics/filtered_metrics.stream \
  --method bayes \
  --mixture-iters 1000 \
  --mixture-draws 150 \
  --subsample 0.7 \
  --n-iter 1 \
  --out merged_bayes_fast.csv

scale_serial_ed /home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics.stream \
  --method bayes \
  --mixture-iters 4000 \
  --mixture-draws 150 \
  --subsample 0.2 \
  --n-iter 3 \
  --out /home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics_merged_bayes.csv

scale_serial_ed /home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics.stream \
  --method bayes \
  --mixture-iters 1500 \
  --mixture-draws 150 \
  --subsample 0.7 \
  --n-iter 2 \
  --out /home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics_merged_bayes_fast_2.csv

  ./csv2hkl.py /home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics_merged_bayes_fast_2.csv \
    /home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot9_20250408_1441/xgandalf_iterations_max_radius_0.18_step_0.1/filtered_metrics-1/filtered_metrics_merged_bayes_fast_2.hkl
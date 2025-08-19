# split_stream_by_axis_lattice.sh
python3 split_stream_by_axis_lattice.py /home/bubl3932/files/simulations/cP_LTA/sim_000/7108314.stream \
  --axes \
  " 0  0  1" " 0  1  0" " 1  0  0" "-1  0  1" "-1  1  0" " 0 -1  1" " 0  1  1" " 1  0  1" " 1  1  0" "-1 -1  1" "-1  1  1" " 1 -1  1" " 1  1  1" \
  --sort-angle --report \
  # --printaxes
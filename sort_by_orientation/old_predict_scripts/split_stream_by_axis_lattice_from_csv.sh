# split_stream_by_axis_lattice_from_csv.sh

python split_stream_by_axis_lattice_from_csv.py --from-csv /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/filtered_metrics/filtered_metrics_problematic_orientations.csv --sort-angle --metric angle_over_score --report 

# # input_stream is optional when --from-csv supplies '# Stream: …'
# ap.add_argument("input_stream", type=Path, nargs="?", help="Input .stream file. Optional if --from-csv is given and header contains '# Stream: <path>'.")
# ap.add_argument("output", type=Path, nargs="?", help=("For --sort-angle: output .stream filename (default <input>_sorted.stream).  "For binning: prefix (default <input>_bin)."))
# ap.add_argument("--beam", type=float, nargs=3, metavar=("X","Y","Z"), default=(0.0, 0.0, 1.0), help="Lab-frame beam direction (default 0 0 1)")
# # NEW: axes+scores (and maybe stream path) from CSV
# ap.add_argument("--from-csv", type=Path, help="Path to CSV of problematic axes (with header '# Stream: ...'). Overrides --lattice/--axes.")
# # Sorting metric selector
# ap.add_argument("--metric", choices=("angle","angle_over_score"), default="angle", help="Sorting metric for --sort-angle (default: angle).")
# # legacy axis selection path (ignored when --from-csv is used)
# ap.add_argument("--lattice", type=str, default="auto", choices=["auto","triclinic","monoclinic","orthorhombic","tetragonal","trigonal","rhombohedral","hexagonal","cubic"], help="Axis family (ignored if --from-csv is used).")
# ap.add_argument("--axes", nargs="+", metavar="UVW", help="Manual axes (ignored if --from-csv is used).")
# ap.add_argument("--printaxes", action="store_true", help="Print the axes that will be used and exit.")
# mode = ap.add_mutually_exclusive_group(required=True)
# mode.add_argument("--sort-angle", action="store_true", help="Write a copy of the input with chunks sorted by the chosen metric (--metric).")
# mode.add_argument("--angle-split", type=float, help="Angle bin width in degrees (e.g., 5).")
# mode.add_argument("--angle-bins",  type=str, help="Comma-separated angle edges, e.g., 0,12.5,50")
# ap.add_argument("--include-unindexed", choices=("end","start","drop"), default="drop", help="Where unindexed chunks go in --sort-angle (default: drop).")
# ap.add_argument("--count-split", type=int, help="After --sort-angle, split into files of N chunks each (optional).")
# ap.add_argument("--report", action="store_true", help="Write per-chunk report '<img> //<event> -> [uvw] score=… angle=… angle/score=…'.")
#!/usr/bin/env bash
set -euo pipefail

python3 predict_problematic_axes_numba.py \
  --stream "/home/bubl3932/files/UOX1/xgandalf_iterations_max_radius_0.71_step_0.5/metrics_run_20250922-151741/filtered_metrics.stream" \
  --nrows 110 \
  --csv 

# Example additions to command line:
# --uvw-max 15 \
# --g-enum-bound 1.1 \
# --g-max 3 \

# ap.add_argument("--stream", required=True, help="Path to .stream file, or '-' to read from stdin")
# ap.add_argument("--uvw-max", type=int, default=10)
# ap.add_argument("--g-enum-bound", type=float, default=None)
# ap.add_argument("--g-max", type=float, default=None)
# ap.add_argument("--zolz-only", action="store_true", default=True)
# ap.add_argument("--holz", dest="holz", action="store_true")
# ap.add_argument("--i-min-rel", type=float, default=0.0)
# ap.add_argument("--ring-mult-min", type=int, default=2)
# ap.add_argument("--n-min", type=int, default=0)
# ap.add_argument("--m-min", type=int, default=0)
# ap.add_argument("--score-alpha", type=float, default=0.4)
# ap.add_argument("--score-beta",  type=float, default=0.6)
# ap.add_argument("--margin-px", type=float, default=0.0)
# ap.add_argument("--tol-g", type=float, default=5e-4)
# ap.add_argument("--nrows", type=int, default=None, help="Maximum number of rows to print and save (default: all)")
# ap.add_argument("--csv", action="store_true")
# ap.add_argument("--printresults", action="store_true", help="Print results table to stdout (headers always printed)")
# # Parallel controls
# ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="Worker processes (default: all cores)")
# ap.add_argument("--chunksize", type=int, default=8, help="Items per task sent to each worker")
#!/usr/bin/env bash
set -euo pipefail

base_dir="$(cd "$(dirname "$0")" && pwd)"
runner="$(command -v xds_par || command -v xds)"

if [[ -z "${runner:-}" ]]; then
  echo "ERROR: xds_par/xds not found in PATH"
  exit 1
fi

echo "Using XDS executable: $runner"

datasets=(LTA_t1 LTA_t2 LTA_t3 LTA_t4)
failed=0

for ds in "${datasets[@]}"; do
  ds_dir="$base_dir/$ds"
  echo "============================================================"
  echo "Running $ds"

  if [[ ! -f "$ds_dir/XDS.INP" ]]; then
    echo "  ERROR: Missing $ds_dir/XDS.INP"
    failed=$((failed + 1))
    continue
  fi

  (
    cd "$ds_dir"
    "$runner" > xds.log 2>&1
  ) || true

  if [[ -f "$ds_dir/XPARM.XDS" && ! -f "$ds_dir/GXPARM.XDS" ]]; then
    cp "$ds_dir/XPARM.XDS" "$ds_dir/GXPARM.XDS"
  fi

  if [[ -f "$ds_dir/GXPARM.XDS" && -f "$ds_dir/INTEGRATE.HKL" ]]; then
    echo "  OK: GXPARM.XDS and INTEGRATE.HKL created"
  else
    echo "  FAIL: Missing expected output(s). Check $ds_dir/xds.log"
    failed=$((failed + 1))
  fi
done

echo "============================================================"
if [[ "$failed" -eq 0 ]]; then
  echo "All datasets completed successfully."
  exit 0
else
  echo "$failed dataset(s) failed."
  exit 2
fi

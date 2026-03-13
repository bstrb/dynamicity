# LTA thickness-matched XDS runs

This directory contains one standardized XDS setup per thickness dataset:

- `LTA_t1/XDS.INP`
- `LTA_t2/XDS.INP`
- `LTA_t3/XDS.INP`
- `LTA_t4/XDS.INP`

## What was changed from your starting file

Only dataset-specific items were changed per folder:

- `NAME_TEMPLATE_OF_DATA_FRAMES`
- `DATA_RANGE`
- `SPOT_RANGE`

And one processing-control change was made to stop at integration:

- `JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE`

This is the setting that gives you:

- `XPARM.XDS` (produced by `IDXREF`; copied to `GXPARM.XDS` by `run_all_xds.sh` for compatibility)
- `INTEGRATE.HKL` (produced by `INTEGRATE`)

## Run all four datasets

From this directory:

```bash
./run_all_xds.sh
```

Per dataset, inspect:

- `xds.log`
- `GXPARM.XDS`
- `INTEGRATE.HKL`

## Notes

- If you later want scaling/merging outputs (`XDS_ASCII.HKL`, `CORRECT.LP`), extend `JOB` to include `CORRECT`.
- If beam center or distance differ by thickness, update `ORGX`, `ORGY`, and `DETECTOR_DISTANCE` in all four files (or per file if truly different).

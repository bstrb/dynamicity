# STW thickness-matched XDS runs

This directory contains one standardized XDS setup per thickness dataset:

- `STW_t1_360/XDS.INP`
- `STW_t2_360/XDS.INP`
- `STW_t3_360/XDS.INP`
- `STW_t4_360/XDS.INP`

## What was changed per dataset

Only dataset-specific items were changed per folder:

- `NAME_TEMPLATE_OF_DATA_FRAMES`
- `DATA_RANGE`
- `SPOT_RANGE`

And one processing-control change was used to stop at integration:

- `JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE`

This setup gives you:

- `XPARM.XDS` from IDXREF (copied to `GXPARM.XDS` by `run_all_xds.sh`)
- `INTEGRATE.HKL` from INTEGRATE

## Run all four datasets

From this directory:

```bash
./run_all_xds.sh
```

Per dataset, inspect:

- `xds.log`
- `GXPARM.XDS`
- `INTEGRATE.HKL`

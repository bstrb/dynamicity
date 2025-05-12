# ────────────────────────────────  file: README.md  ────────────────────────────────

```markdown
# sered_scaler 🔬📈
Light‑weight scaling & merging pipeline for **Serial electron diffraction**
(Serial‑ED) datasets recorded as CrystFEL `*.stream` files.

## Quick install (editable)
```bash
git clone …/sered_scaler.git
cd sered_scaler
pip install -e .[dask,bayes]   # optional extras
```

## Minimal run
```bash
scale_serial_ed run.stream --cutoff 2.5 --out merged_F2.csv
```

## Pipeline stages
1. **ingest**   – stream → DataFrames (`sered_scaler.io`)
2. **provision** – robust log‑scale fit (Huber)
3. **filter**    – Z‑score or Bayesian mixture → weights
4. **merge**     – weighted WLS → nearly kinematic `F²(hkl)`
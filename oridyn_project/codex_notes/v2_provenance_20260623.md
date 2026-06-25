# Geometry-Coupling V2 Provenance

Created UTC: `2026-06-23T13:37:45.812775+00:00`

## Files Created

- `oridyn/coupling_exposure_v2.py` sha256 `203d0ce48fa95038582726ded62eb0c85ca30c640eec7f505fc2cb3670b2d937`
- `tools/compute_geometry_coupling_v2_scores.py` sha256 `276aa2074bfca2145f9ea7ea34aea2406bf2d141273a1a01797a6fdb2a233f01`
- `tools/filter_stream_by_geometry_coupling_v2_keep_fraction.py` sha256 `db4b1a24144a9762a397e872ae068063593a60955bf197132c2df77d00252d80`

## Implemented V2 Equations

For each signed observed target reflection `g`, same-frame observed beams `h` are treated as coupling-exposure sources:

- `delta = h - g`
- `E(h) = exp(-(sg(h) / sg0)^2)` using the existing OriDyn `sg` column.
- `coupling_prior(delta) = 1 / (1 + (q_delta / g0_invA)^p)` when a per-frame reciprocal metric can be fit from `h,k,l,q_invA`.
- Fallback if metric fitting fails: `1 / (1 + (|delta_hkl| / hkl_delta_g0)^p)`.
- `v2_core`: `log1p(sum_top_edges(E(h) * coupling_prior(h-g)))`.
- `v2_core_plus_zone`: v2 core with same assigned-Laue-zone edge boost `(1 + beta_zone)`.
- `v2_core_plus_row`: v2 core with target systematic-row boost `(1 + beta_row * systematic_row_risk_norm(g))`.
- `v2_full`: zone boost plus row boost; `trust_risk_v2_full_norm` additionally uses soft frame-axis boost `(1 + beta_frame * frame_axis_risk_norm)` before global robust p01-p99 normalization.
- Norm columns are robust p01-p99 normalized and clipped to `[0, 1]`.

Important approximation: this first v2 score reads only the existing `reflection_scores.csv`, so it uses observed same-frame reflections as source beams rather than regenerating all unobserved candidate beams from the stream orientation. The metadata reports reciprocal-metric fit/fallback counts.

## Validation Commands Run

```bash
python -m py_compile oridyn/coupling_exposure_v2.py tools/compute_geometry_coupling_v2_scores.py tools/filter_stream_by_geometry_coupling_v2_keep_fraction.py
python tools/compute_geometry_coupling_v2_scores.py --help
python tools/filter_stream_by_geometry_coupling_v2_keep_fraction.py --help
```

A tiny smoke test was also run with `--max-frames 5` and `--max-events 5` under `/tmp/oridyn-v2-coupling-smoke-1782221294`.

## Manual Future Full-Run Commands

```bash
cd /home/bubl3932/projects/dynamicity/oridyn_project

ROOT="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
STREAM="$ROOT/MFM300-VIII_cut_20-0_3.stream"
SCORES="$ROOT/oridyn_large_ABC_min1_mildweights/oridyn_scores_mild/reflection_scores.csv"
OUT="$ROOT/geometry_coupling_v2_filter_keep90_80_20260623"

python tools/compute_geometry_coupling_v2_scores.py \
  --scores-csv "$SCORES" \
  --outdir "$OUT/v2_scores" \
  --workers 0 \
  --progress-every-frames 25

python tools/filter_stream_by_geometry_coupling_v2_keep_fraction.py \
  --stream "$STREAM" \
  --v2-scores-csv "$OUT/v2_scores/geometry_coupling_v2_scores.csv" \
  --output-root "$OUT" \
  --score-column trust_risk_v2_full_norm \
  --keep-fractions 0.90 0.80 \
  --min-obs-per-hkl 10 \
  --keep-low-count-hkls \
  --progress-every 1000000
```

## Status

Full v2 MFM300 scoring/filtering has not yet been run.

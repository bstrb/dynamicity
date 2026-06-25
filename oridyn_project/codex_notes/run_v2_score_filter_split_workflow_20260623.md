# OriDyn V2 Score/Filter/Split Workflow

Created: 2026-06-23

## Purpose

`tools/run_oridyn_v2_score_filter_split_workflow.py` runs the practical OriDyn v2 workflow on a stream that has already been prepared by the user:

1. Run the existing OriDyn base scoring pipeline on the provided stream.
2. Compute v2 coupling-exposure scores from the fresh `base_scores/reflection_scores.csv`.
3. Filter the input stream by selected v2 score columns and keep fractions.
4. Optionally create low/high/random observation-level split streams.

The workflow does not cut streams, merge, run partialator, run QC, or run SHELXL. Those remain separate downstream steps.

Observation matching uses exact signed keys:

```text
source_filename + event + signed h,k,l
```

HKLs are not symmetry-canonicalized.

## V2 Equations

For a signed observed target reflection `g`, same-frame observed beams `h` are used as coupling-exposure sources:

```text
delta = h - g
E(h) = exp(-(sg(h) / sg0)^2)
coupling_prior(delta) = 1 / (1 + (q_delta / g0_invA)^p)
```

When possible, `q_delta` is computed from a per-frame reciprocal metric fit using `h,k,l,q_invA` from the fresh base score table. If that fit fails, the v2 scorer falls back to:

```text
1 / (1 + (|delta_hkl| / hkl_delta_g0)^p)
```

Core score:

```text
v2_core_raw(g) = log1p(sum_top_edges(E(h) * coupling_prior(h-g)))
trust_risk_v2_core_norm = robust_p01_p99_normalize(v2_core_raw)
```

Full score:

```text
same-zone edge boost = 1 + beta_zone
row target boost = 1 + beta_row * systematic_row_risk_norm(g)
frame soft boost = 1 + beta_frame * frame_axis_risk_norm(g)

v2_full_raw(g) = log1p(sum_top_edges(E(h) * coupling_prior(h-g) * same_zone_boost) * row_target_boost)
trust_risk_v2_full_norm = robust_p01_p99_normalize(v2_full_raw * frame_soft_boost)
```

Default score columns used by the workflow:

```text
trust_risk_v2_full_norm
trust_risk_v2_core_norm
```

## Full 0.5 A Workflow Command

```bash
cd /home/bubl3932/projects/dynamicity/oridyn_project

ROOT="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
STREAM="$ROOT/MFM300-VIII_cut_20-0_5.stream"
OUT="$ROOT/oridyn_v2_scoring_filtering_20_0p5_20260623"

python tools/run_oridyn_v2_score_filter_split_workflow.py \
  --stream "$STREAM" \
  --output-root "$OUT" \
  --score-columns trust_risk_v2_full_norm trust_risk_v2_core_norm \
  --keep-fractions 0.90 0.80 0.70 \
  --min-obs-per-hkl 10 \
  --keep-low-count-hkls \
  --make-splits \
  --split-fraction 0.50 \
  --random-seed 1 \
  --workers 0 \
  --progress-every 1000000
```

## Tiny Smoke Test

```bash
cd /home/bubl3932/projects/dynamicity/oridyn_project

ROOT="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
STREAM="$ROOT/MFM300-VIII_cut_20-0_5.stream"
OUT="/tmp/oridyn-v2-workflow-smoke"

python tools/run_oridyn_v2_score_filter_split_workflow.py \
  --stream "$STREAM" \
  --output-root "$OUT" \
  --score-columns trust_risk_v2_full_norm trust_risk_v2_core_norm \
  --keep-fractions 0.90 0.80 \
  --min-obs-per-hkl 2 \
  --keep-low-count-hkls \
  --make-splits \
  --split-fraction 0.50 \
  --random-seed 1 \
  --workers 1 \
  --progress-every 1000 \
  --max-events 3 \
  --force
```

## Downstream Reminder

Merging and QC are intentionally separate. For the 20-0.5 A stream/results, downstream merge/QC should use:

```bash
HIGHRES=0.5
```

Do not reuse a 0.35 A merge/QC cutoff for the 0.5 A workflow.

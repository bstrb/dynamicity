* file created: `/home/bubl3932/projects/dynamicity/oridyn_project/tools/shift_stream_bidirectional_graph_frame_to_lowrisk.py`
* validation status: passed (`python -m py_compile`, `--help`, and `--max-events 5` smoke test)
* exact manual command for the full run:

```bash
ROOT=/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524
PY=/home/bubl3932/anaconda3/envs/pyxem-env/bin/python
SCRIPT=/home/bubl3932/projects/dynamicity/oridyn_project/tools/shift_stream_bidirectional_graph_frame_to_lowrisk.py
OUTROOT=$ROOT/model_free_nonself_diagnostics/bidirectional_graph_frame_aggressive_shift
mkdir -p "$OUTROOT"

"$PY" "$SCRIPT" \
  --stream "$ROOT/MFM300-VIII_cut_20-0_3.stream" \
  --unmerged "$ROOT/model_free_nonself_diagnostics/minimal_unmerged_y1_unity_nopr_n1/unmerged_minimal_y1.hkl" \
  --scores "$ROOT/oridyn_large_ABC_min1_mildweights/oridyn_scores_mild/reflection_scores.csv" \
  --output-stream "$OUTROOT/bidirectional_graph_frame_shift.stream" \
  --baseline-low-risk-fraction 0.20 \
  --shift-tail-fraction 0.10 \
  --correct-top-risk-fraction 0.20 \
  --min-relative-shift 0.05 \
  --lambda-shift 1.0 \
  --min-obs-all 50 \
  --local-neighbor-count 30 \
  --local-min-neighbors 10 \
  --min-denominator 1.0 \
  --exclude-partiality-too-small \
  --progress-every 100000
```

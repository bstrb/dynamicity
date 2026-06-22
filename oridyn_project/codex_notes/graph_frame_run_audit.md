* files created
  * `/home/bubl3932/projects/dynamicity/oridyn_project/tools/audit_graph_frame_runs.py`
  * `/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_run_audit/run_inventory.tsv`
  * `/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_run_audit/correction_comparison.csv`
  * `/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_run_audit/merge_comparison.csv`
  * `/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_run_audit/refinement_comparison.csv`
  * `/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_run_audit/top_changed_hkls.csv`
  * `/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_run_audit/graph_frame_run_audit_summary.md`

* files successfully parsed
  * inventoried 14 runs/artifacts
  * parsed 13 correction summaries
  * wrote 126 merge comparison rows, including shell-wise rows where available
  * parsed 10 SHELXL `.lst` files
  * wrote 600 top-changed-HKL rows from actual full correction runs
  * Python validation passed: `python -m py_compile /home/bubl3932/projects/dynamicity/oridyn_project/tools/audit_graph_frame_runs.py`

* missing files or uncertainties
  * `poc_shift10_scale10_lambda10_minscale00` has correction outputs but no merge/refinement outputs found.
  * old `graph_frame_obskey_scaling_sweep` variants are inventoried but marked pre-patch/zero-scaled; they are not treated as scientific correction results.
  * smoke and provenance folders are inventoried but excluded from interpretation.
  * highest peak/deepest hole were requested, but the parser did not find clear matching SHELXL lines in the parsed `.lst` files.
  * `top_changed_hkls.csv` uses per-HKL median/scale estimates, not exact per-observation stored deltas.
  * bidirectional `default_shelx` has no matched baseline cutoff label, so no baseline delta is computed for that row.

* key findings
  * historical HKL-wide lambda05 remains the strongest 1.5-0.5 refinement result: R1 improves from 0.2175 to 0.2076, but it changed 1,983,082 observations, 18.72% of stream observations, and is the known bugged/proof-of-concept broad target-HKL correction.
  * clean obs-key `poc_shift10_scale20_lambda10_minscale10` is the best proper one-sided result so far: it changes 397,003 observations, 3.75%, and improves 1.5-0.5 R1 by -0.0066 and wR2 by -0.0120.
  * clean obs-key `poc_shift10_scale10_lambda10_minscale10` is gentler, changing 203,765 observations, 1.92%, and is the only parsed corrected merge with both overall CC1/2 gain and Rsplit gain.
  * bidirectional aggressive shift changes 575,242 observations, 5.43%, and gives the best parsed 99-0.35 refinement among corrected full runs, R1 0.2427 vs baseline 0.2504, but merge statistics worsen overall.
  * nonpositive `I_ref` targets remain a real red flag: 20,283 to 39,112 observations in one-sided runs, and 30,494 in the bidirectional run, are associated with nonpositive reference targets.
  * the emerging strategy is not simply "more correction is better": broad correction gives clearer refinement contrast, but clean obs-key variants can improve refinement with much smaller perturbation; the next useful refinement is probably guarding/removing nonpositive-reference targets and tuning scale20-like breadth rather than going fully broad.

* exact command to rerun the audit

```bash
python /home/bubl3932/projects/dynamicity/oridyn_project/tools/audit_graph_frame_runs.py --root /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524 --outdir /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_run_audit
```

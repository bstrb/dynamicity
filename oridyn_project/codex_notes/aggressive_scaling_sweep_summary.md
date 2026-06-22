# Aggressive Graph/Frame Scaling Sweep Summary

All three aggressive proof-of-concept variants completed scaling only. No partialator, merging, or refinement was run as part of this summary.

| variant | output stream path | target signed HKLs | scaled observations | unscaled observations | fraction of stream observations scaled | median scale factor | min scale factor | max scale factor | hitting min scale | median relative shift of target HKLs | notes or warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `poc_shift10_scale10_lambda10_minscale10` | [scaled stream](/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale10_lambda10_minscale10/scaled_graph_frame_poc_shift10_scale10_lambda10_minscale10.stream) | 19,889 | 203,765 | 10,387,978 | 1.924% | 0.329854 | 0.100000 | 0.952342 | HKLs: 3,935/19,889 (19.8%); diagnostic selected obs: 35,737/203,765 (17.5%) | 1.82886 | Outputs present; 0 unmatched observations; 2,339 target HKLs have `I_ref <= 0`, so many scale factors hit the 0.10 floor. |
| `poc_shift10_scale10_lambda10_minscale00` | [scaled stream](/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale10_lambda10_minscale00/scaled_graph_frame_poc_shift10_scale10_lambda10_minscale00.stream) | 19,889 | 203,765 | 10,387,978 | 1.924% | 0.329854 | 0.000000 | 0.952342 | HKLs: 2,339/19,889 (11.8%); diagnostic selected obs: 17,633/203,765 (8.7%) | 1.82886 | Outputs present; 0 unmatched observations; red flag: zero scale factors are allowed, so selected observations for 2,339 target HKLs may be zeroed. |
| `poc_shift10_scale20_lambda10_minscale10` | [scaled stream](/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/model_free_nonself_diagnostics/graph_frame_aggressive_poc_scaling/poc_shift10_scale20_lambda10_minscale10/scaled_graph_frame_poc_shift10_scale20_lambda10_minscale10.stream) | 19,889 | 397,003 | 10,194,740 | 3.748% | 0.329854 | 0.100000 | 0.952342 | HKLs: 3,935/19,889 (19.8%); diagnostic selected obs: 69,365/397,003 (17.5%) | 1.82886 | Outputs present; 0 unmatched observations; same target HKLs and scale factors as `scale10_minscale10`, but about 2x more observations scaled. |

## Cross-Variant Checks

The three variants selected the same number of target weak signed HKLs: 19,889 target HKLs from 20,248 eligible weak signed HKLs.

The `shift10/scale10/minscale10` and `shift10/scale20/minscale10` variants differ only in the top-risk observation fraction and therefore the number of scaled observations. Their target HKLs, median relative shift, and scale-factor summary are identical; scaled stream observations increase from 203,765 to 397,003.

The two `shift10/scale10` variants select the same observations, but differ in the lower scale clamp. `minscale00` allows scale factors of 0.0, while `minscale10` floors them at 0.10.

## Red Flags

No unmatched observations were reported in any variant (`unmatched_observations = 0`), and no duplicate keys, `nan_esd`, nonpositive partiality, or `partiality_too_small` exclusions were reported.

All variants excluded 254,092 observations from flagged crystals.

The main scientific/technical red flag is the aggressive floor behavior: 2,339 target HKLs have `I_ref <= 0`, and the `minscale00` variant can zero selected intensities/sigmas. This is useful as a proof-of-concept stress test, but it is likely too aggressive for a final production correction unless downstream merging/refinement clearly benefits.

Saved to: [aggressive_scaling_sweep_summary.md](/home/bubl3932/projects/dynamicity/oridyn_project/codex_notes/aggressive_scaling_sweep_summary.md)

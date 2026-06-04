# PETS2 Geometry-Only Frame Plotter

This project predicts diffraction-frame spot positions from PETS2 geometry and indexing metadata only.

It does **not** use observed spot-position files such as `.rpl`, `.xyz`, `.cor`, `.clust`, `.diff`.
It also does **not** use SHELX files (`.ins`, `.hkl`).

## Inputs used

- `*.pts2` for:
  - `lambda`
  - `aperpixel`
  - global `omega` / `delta`
  - `ubmatrix`
  - `cell`
  - `imagelist`
- `*_petsdata/*.ptsopt` for refined per-frame:
  - `alpha`, `beta`, `omega` (treated as PETS `domega`)
  - `xcenter`, `ycenter`
- `*_petsdata/*.ptsoptlist` (if present) for per-frame `# Best result` refinements
  - used to further refine `alpha`, `beta`, `omega`, `xcenter`, `ycenter`

## Current model

- PETS reciprocal convention:
  - default (`--ub-convention columns`): `g_cart = UB @ [h, k, l]^T` (`hkls @ UB.T`)
  - alternative (`--ub-convention rows`): `g_cart = hkls @ UB`
- Orientation model:
  - default: PETS-style absolute angles with `pets_ab_xy` (`R = Ry(beta) @ Rx(alpha)`)
  - optional: `pets_ab_yx`, `fixed_x_alpha`, `euler_yxz`, `axis_alpha_legacy`, `none`
  - angle reference is explicit (`--angle-reference absolute|first_frame|zero`)
  - by default, `domega` is **not** used as a lattice `Rz` term to avoid omega double counting
- Beam direction in PETS frame:
  - default incident beam along `-z` (PETS `+z` points to source)
- Reflection selection:
  - geometry-only excitation test from Ewald proximity (`sg`)
- Detector mapping:
  - default full projection using `aperpixel`, `lambda`, and frame center
  - PETS `omega` is applied in detector-plane mapping (not as the default scan axis)
  - default mapping uses frame omega (`--omega-map-mode frame_absolute`)
  - optional constant correction: `--omega-offset-deg` (useful for small global in-plane angle offsets, e.g. ~10°)
- Reference frame:
  - first frame in full sequence is kept as a stable anchor for relative-angle modes
  - absolute-angle mode does not subtract frame-1 angles
- Frame geometry merge:
  - `.ptsopt` overrides are merged by `frame_number` and normalized image basename
  - `.ptsoptlist` `# Best result` rows are merged on frame number (when available)
  - parser fails fast if `xcenter/ycenter` remain unresolved

Center marker note:
- if you apply `--swap-xy` / `--flip-x` / `--flip-y`, the plotted center cross is transformed with the same mapping as predicted spots.

## Usage

```bash
cd /home/bubl3932/projects/dynamicity/plot_frames_from_pets2_data
python pets2_geometry_frame_plotter.py \
  --pets-root /home/bubl3932/files/LTA1_PETS \
  --output-dir /tmp/lta1_pets_geom_only \
  --frame 196 --frame 1130 --frame 2367 \
  --dmin 0.6
```

Optional background images (only for visual overlay):

```bash
python pets2_geometry_frame_plotter.py \
  --pets-root /home/bubl3932/files/LTA1_PETS \
  --images-dir /home/bubl3932/files/3DED-DATA/LTA/LTA_t1/image_16 \
  --output-dir /tmp/lta1_pets_geom_only_with_bg \
  --frame 196 --frame 1130 --frame 2367 \
  --dmin 0.6
```

To force detector size manually, pass `--detector-nx` / `--detector-ny`.
If omitted or set to `<=0`, detector size is auto-detected from loaded images.

## Fast Convention Debugging

When plots look wrong, run a convention sweep instead of guessing sign/axis switches manually.

This tests common combinations of:

- UB convention (`columns` vs `rows`)
- orientation model
- angle reference (`absolute` vs `first_frame`)
- whether `domega` is used in lattice rotation
- beam direction sign
- omega mapping mode/sign
- rotation inversion
- x/y swap and axis flips

and ranks each candidate by image-alignment score using local spot contrast on raw images.

Note: this sweep is intentionally broad and can take several minutes on 3 frames.

```bash
python pets2_geometry_frame_plotter.py \
  --pets-root /home/bubl3932/files/LTA1_PETS \
  --images-dir /home/bubl3932/files/3DED-DATA/LTA/LTA_t1/image_16 \
  --output-dir /tmp/lta1_pets_geom_debug \
  --frame 196 --frame 1130 --frame 2367 \
  --dmin 0.6 \
  --debug-conventions \
  --debug-top-n 8
```

Outputs in debug mode:

- `debug_convention_scores.csv` (ranked candidates)
- `debug_top_candidates/rank_XX__.../detector_frame_XXXX.png` (top overlays)
- `best_convention_flags.txt` (copy/paste CLI flags for the top-ranked convention)

## Outputs

For each requested frame:

- `detector_frame_XXXX.png`
- `predicted_spots_frame_XXXX.csv`
- `centers_used.csv` (raw PETS centers and plotted centers after swap/flip mapping)

And one metadata snapshot:

- `run_metadata.json`

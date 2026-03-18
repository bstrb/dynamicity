# abTEM Single-Orientation Thickness Series

This project simulates diffraction patterns for one fixed orientation at multiple thicknesses using abTEM.

## What it does

- loads one structure file (for example, LTA CIF)
- applies one fixed orientation (Euler rotations)
- runs multislice at multiple thicknesses
- writes one pattern per thickness
- writes quick comparison figures

## Project layout

- `config/simulation.json`: user-editable simulation parameters
- `src/simulate_series.py`: main simulation engine
- `src/plot_series.py`: quick plotting helpers
- `run_simulation.py`: one-command runner
- `results/`: generated outputs

## Setup

```bash
cd /Users/xiaodong/Desktop/dynamicity/abtem_single_orientation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

1. Edit `config/simulation.json` and set `structure_path`.
2. Run:

```bash
python run_simulation.py --config config/simulation.json
```

## Orientation Modes

The config supports two modes:

1. `orientation.mode = "xds_frame"`:
uses a GXPARM rotation axis plus frame number to reproduce orientation from XDS.

2. `orientation.mode = "euler"`:
manual Euler-style `rotation_deg` in x/y/z.

Default config is already set for LTA frame 50 using:

- `structure_path = ../../../SCXRD-DATA/SCXRDLTA/LTA.cif`
- `orientation.gxparm_path = ../../bloch_wave_analyser_project/data/LTA_t1/GXPARM.XDS`
- `orientation.frame_number = 50`
- `thickness_nm = [100, 200, 350, 600]`

## Notes

- Thickness values are in **nm** in the config.
- Internally, thickness is approximated in **A** by repeating the base potential
	along beam direction (`CrystalPotential` z repetitions).
- For `xds_frame` mode, atoms are rotated by XDS axis-angle for the selected frame.
- For `abtem==1.0.x`, use `numpy<2` due compatibility constraints.

## Reproducible Workflow (Current Plots)

The commands below reproduce the latest 10-500 nm results and the HKL-based
variation/stability figure at frame 50.

1. Run the dense thickness simulation (same crystal + orientation):

```bash
/opt/anaconda3/envs/pyxem-env/bin/python run_simulation.py \
	--config config/simulation_10to500_step10.json
```

2. Generate the HKL-based story plot with 20 reflections per group:

```bash
/opt/anaconda3/envs/pyxem-env/bin/python plot_hkl_variation_story_10to500.py \
	--n-per-group 20
```

3. Verify the selection is balanced and non-overlapping:

```bash
/opt/anaconda3/envs/pyxem-env/bin/python - <<'PY'
import pandas as pd
p='results_10to500_step10/figures/thickness_variation_story_hkl_10to500_selected.csv'
df=pd.read_csv(p)
print('total',len(df))
print(df['group'].value_counts().to_string())
print('hkl_overlap',int((df.groupby(['h','k','l'])['group'].nunique()>1).sum()))
print('xy_overlap',int((df.groupby(['x_px','y_px'])['group'].nunique()>1).sum()))
PY
```

Expected checks for step 3:

- `total 40`
- `high_variation 20`
- `stable 20`
- `hkl_overlap 0`
- `xy_overlap 0`

Main generated files:

- `results_10to500_step10/pattern_index.csv`
- `results_10to500_step10/figures/patterns_overview.png`
- `results_10to500_step10/figures/thickness_variation_story_hkl_10to500.png`
- `results_10to500_step10/figures/thickness_variation_story_hkl_10to500_selected.csv`

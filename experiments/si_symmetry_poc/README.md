# Si symmetry POC

This experiment uses py3DED commit `15cb5f8f26a78f9bf78233465c183b61b5bded61`.

## Structure provenance

The structure is diamond Si, space group Fd-3m (No. 227), with the paper's
cell parameter `a = b = c = 5.43053 A`. The symmetry operators and Si site
were taken from public COD entry 9008566, which is crystallographically
equivalent to ICSD collection code 51688 but reports `a = 5.43070 A`.
The local CIF therefore normalizes only the cell parameter to the value stated
by Cabaj et al.; this provenance distinction is intentional and must remain
visible in later results.

ASE validation of the local CIF gives 8 generated Si atoms at the diamond
sites `(0,0,0)`, `(0,1/2,1/2)`, `(1/2,0,1/2)`, `(1/2,1/2,0)`,
`(3/4,3/4,1/4)`, `(3/4,1/4,3/4)`, `(1/4,3/4,3/4)`, and
`(1/4,1/4,1/4)`.

## Configuration provenance

The JSON follows the paper/SI input block: 200 keV, `omega = 25` degrees,
alpha 0 to 45 degrees in 451 points (0.1 degree), maximum thickness 1000 A,
10 exit planes, `g_max = 8.0 A^-1`, `sg_max = 0.5 A^-1`, and
`g_max_store = 2.0 A^-1`. The paper specifies `U_iso = sigma^2 = 0.01 A^2`,
so py3DED receives `thermal_sigmas = 0.1 A`. The mode is changed from the
paper's generic `bw+ms` example to `bw` as required for this experiment.

`slice_thickness`, `sampling`, `box_size_*`, `window_func`, and
`integration_radius` are retained from the paper/SI generic input for
reproducibility, although the BW path does not use the multislice box or
window settings. For BW, the product `slice_thickness * exit_planes` defines
the thickness sampling used by py3DED.

## Environment

The existing `pyxem-env` is used without upgrading abTEM: Python 3.10.16,
abTEM 1.0.6, ASE 3.27.0, NumPy 1.26.4, SciPy 1.15.1, Zarr 2.18.3, and
xarray installed for py3DED's Zarr reader.

## Successful run

The successful output is:

`results/20260924-121554_Si_CollCode51688_paper_cell_1x1x1000_451/bw.zarr`

Run from the py3DED checkout with the user-space CUDA libraries available:

```sh
export CUDA_PATH=/home/bubl3932/anaconda3/envs/pyxem-env/lib/python3.10/site-packages/nvidia/cuda_runtime
export PATH=/home/bubl3932/anaconda3/envs/pyxem-env/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=/home/bubl3932/anaconda3/envs/pyxem-env/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/bubl3932/anaconda3/envs/pyxem-env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/bubl3932/anaconda3/envs/pyxem-env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/bubl3932/anaconda3/envs/pyxem-env/lib/python3.10/site-packages/nvidia/cusolver/lib:/home/bubl3932/anaconda3/envs/pyxem-env/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/bubl3932/anaconda3/envs/pyxem-env/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
python scripts/run_py3DED.py ../dynamicity/experiments/si_symmetry_poc/si_bw.json
```

The run took 3.09 minutes and wrote `array0` with shape `(451, 200, 787)`
and dtype `float32`. The logical dimensions are `(x_rotation, z, hkl)`;
`x_rotation` is 0--45 degrees in 0.1-degree steps and `z` is 5--1000 A in
5 A steps. The raw result contains 787 stored HKLs and all values checked
finite and nontrivial.

The environment versions used were Python 3.10.16, py3DED editable from the
pinned checkout, abTEM from commit `c6cdd78b...` (reported as 1.0.6), ASE
3.27.0, NumPy 1.26.4, SciPy 1.15.1, Zarr 2.18.3, xarray 2025.6.1, CuPy
13.6.0, CUDA runtime/NVRTC/NVCC 12.9, cuBLAS 12.9.0.13, and cuSOLVER
11.7.5.82.

## Extracted outputs

`individual_complete_observations.csv` retains each complete reflection at
each thickness with its Laue family, HKL, Simpson-integrated intensity,
geometrical alpha_B, and local ds_g/dalpha. `candidate_family_summary.csv`
summarizes families with at least two complete members at representative
thickness 505 A. `family_20_diagnostic.png`, `family_52_diagnostic.png`, and
`family_76_diagnostic.png` are display diagnostics only; no family was
preselected scientifically.

The extraction reuses `calculate_integrated_intensities()` from
`../py3DED/scripts/run_py3DED_hkl+Rint.py`. That historical script contains
`mask = mask[0]` immediately after constructing its intended
`(thickness, reflection)` completeness mask. The local extraction preserves
the exact criterion formula but keeps the intended two-dimensional mask so
all 200 thicknesses can be exported; the historical source was not changed.
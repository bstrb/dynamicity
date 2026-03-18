from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read


def _as_numpy(pattern_obj) -> np.ndarray:
    """Best-effort conversion of an abTEM object to a 2D numpy array."""
    obj = pattern_obj
    if hasattr(obj, "intensity"):
        try:
            obj = obj.intensity()
        except Exception:
            # Some abTEM objects are already real intensity measurements.
            pass
    if hasattr(obj, "compute"):
        obj = obj.compute()
    if hasattr(obj, "array"):
        obj = obj.array
    arr = np.asarray(obj)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D diffraction intensity array, got shape {arr.shape}")
    return arr.astype(float)


def _load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def _rotate_atoms(atoms, rotation_deg: dict) -> None:
    rx = float(rotation_deg.get("x", 0.0))
    ry = float(rotation_deg.get("y", 0.0))
    rz = float(rotation_deg.get("z", 0.0))
    if abs(rx) > 0:
        atoms.rotate(rx, "x", rotate_cell=True)
    if abs(ry) > 0:
        atoms.rotate(ry, "y", rotate_cell=True)
    if abs(rz) > 0:
        atoms.rotate(rz, "z", rotate_cell=True)


def _resolve_path(config_path: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = (config_path.parent / p).resolve()
    return p


def _parse_gxparm_orientation(gxparm_path: Path, frame_number: int) -> tuple[np.ndarray, float, float, float]:
    lines = [line.strip() for line in gxparm_path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"GXPARM appears too short: {gxparm_path}")

    line_1 = [float(tok) for tok in lines[1].split()]
    phi0_deg = float(line_1[1])
    dphi_deg = float(line_1[2])
    axis = np.asarray(line_1[3:6], dtype=float)
    norm = np.linalg.norm(axis)
    if norm <= 0:
        raise ValueError("GXPARM rotation axis has zero norm")
    axis = axis / norm

    frame_index = int(frame_number) - 1
    phi_deg = phi0_deg + dphi_deg * frame_index
    return axis, phi0_deg, dphi_deg, phi_deg


def _apply_orientation(atoms, cfg: dict, config_path: Path) -> dict:
    orientation_cfg = cfg.get("orientation", {})
    mode = str(orientation_cfg.get("mode", "euler")).lower()

    meta: dict[str, float | int | str] = {"orientation_mode": mode}

    if mode == "xds_frame":
        gxparm_raw = orientation_cfg.get("gxparm_path")
        if gxparm_raw is None:
            raise ValueError("orientation.mode='xds_frame' requires orientation.gxparm_path")
        gxparm_path = _resolve_path(config_path, str(gxparm_raw))
        if gxparm_path is None or not gxparm_path.exists():
            raise FileNotFoundError(f"GXPARM path not found: {gxparm_path}")

        frame_number = int(orientation_cfg.get("frame_number", 1))
        axis, phi0_deg, dphi_deg, phi_deg = _parse_gxparm_orientation(gxparm_path, frame_number)

        atoms.rotate(phi_deg, v=axis, rotate_cell=True)

        meta.update(
            {
                "gxparm_path": str(gxparm_path),
                "frame_number": int(frame_number),
                "phi0_deg": float(phi0_deg),
                "dphi_deg": float(dphi_deg),
                "phi_deg": float(phi_deg),
                "axis_x": float(axis[0]),
                "axis_y": float(axis[1]),
                "axis_z": float(axis[2]),
            }
        )

        extra_rot = orientation_cfg.get("extra_rotation_deg", {})
        _rotate_atoms(atoms, extra_rot)
    elif mode == "euler":
        _rotate_atoms(atoms, cfg.get("rotation_deg", {}))
    else:
        raise ValueError("orientation.mode must be 'euler' or 'xds_frame'")

    return meta


def run_simulation(config_path: str | Path) -> Path:
    import abtem

    config_path = Path(config_path)
    cfg = _load_config(config_path)

    structure_path = _resolve_path(config_path, cfg["structure_path"])
    if structure_path is None:
        raise ValueError("structure_path is required")
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    output_dir = Path(cfg.get("output_dir", "results"))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent.parent / output_dir).resolve()
    patterns_dir = output_dir / "patterns"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(structure_path)
    orientation_meta = _apply_orientation(atoms, cfg, config_path)

    sampling_A = float(cfg.get("sampling_A", 0.1))
    slice_thickness_A = float(cfg.get("slice_thickness_A", 1.0))
    energy_eV = float(cfg.get("energy_eV", 200000.0))
    max_angle_mrad = float(cfg.get("max_angle_mrad", 35.0))
    gpts = tuple(cfg.get("gpts", [512, 512]))

    potential = abtem.Potential(
        atoms,
        sampling=sampling_A,
        slice_thickness=slice_thickness_A,
        gpts=gpts,
    )
    probe = abtem.PlaneWave(energy=energy_eV)

    # abTEM 1.0.x does not accept a thickness argument in PlaneWave.multislice.
    # We emulate thickness by repeating the unit potential along the beam (z).
    unit_thickness_A = max(float(len(potential)) * slice_thickness_A, 1e-9)

    rows = []
    for thickness_nm in cfg.get("thickness_nm", []):
        thickness_nm = float(thickness_nm)
        requested_thickness_A = 10.0 * thickness_nm
        repetitions_z = int(np.ceil(requested_thickness_A / unit_thickness_A))
        repetitions_z = max(repetitions_z, 1)
        actual_thickness_A = repetitions_z * unit_thickness_A

        crystal_potential = abtem.CrystalPotential(
            potential,
            repetitions=(1, 1, repetitions_z),
        )

        exit_wave = probe.multislice(crystal_potential)
        if hasattr(exit_wave, "diffraction_pattern"):
            pattern = exit_wave.diffraction_pattern(max_angle=max_angle_mrad)
        else:
            pattern = exit_wave.diffraction_patterns(max_angle=max_angle_mrad)
        intensity = _as_numpy(pattern)

        npy_path = patterns_dir / f"pattern_t{int(round(thickness_nm))}nm.npy"
        np.save(npy_path, intensity)

        rows.append(
            {
                "thickness_nm": thickness_nm,
                "requested_thickness_A": requested_thickness_A,
                "actual_thickness_A": actual_thickness_A,
                "repetitions_z": repetitions_z,
                "npy_path": str(npy_path),
                "shape_y": int(intensity.shape[0]),
                "shape_x": int(intensity.shape[1]),
                "intensity_sum": float(np.sum(intensity)),
                "intensity_max": float(np.max(intensity)),
                **orientation_meta,
            }
        )

    index_df = pd.DataFrame(rows).sort_values("thickness_nm").reset_index(drop=True)
    index_csv = output_dir / "pattern_index.csv"
    index_df.to_csv(index_csv, index=False)
    return index_csv

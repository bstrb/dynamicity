#!/usr/bin/env python3
"""Predict PETS2 detector-frame spots from geometry/indexing metadata only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import shlex
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UnitCell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class FrameState:
    frame_index: int
    frame_number: int
    imgname: str
    alpha: float
    beta: float
    domega: float
    xcenter: float
    ycenter: float
    useforcalc: float


@dataclass(frozen=True)
class PetsModel:
    pts2_path: Path
    ptsopt_path: Path | None
    ptsoptlist_path: Path | None
    wavelength_angstrom: float
    aperpixel_invA_per_px: float
    omega_deg: float
    delta_deg: float
    center_x_px: float
    center_y_px: float
    ub_matrix: np.ndarray
    unit_cell: UnitCell
    frames: tuple[FrameState, ...]


FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
PETS_IMAGELIST_DEFAULT_HEADER: tuple[str, ...] = (
    "imgname",
    "alpha",
    "beta",
    "domega",
    "alphaorig",
    "betaorig",
    "domegaorig",
    "xcenter",
    "ycenter",
    "intscale",
    "diffbfac",
    "magcorr",
    "elliamp",
    "elliph",
    "paraamp",
    "paraph",
    "useforcalc",
    "dataset",
)


def _clean_lines(text: str) -> list[str]:
    return [line.rstrip("\n") for line in text.splitlines()]


def _parse_scalar(lines: Sequence[str], key: str) -> float | None:
    key_low = key.lower()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.lower().startswith(key_low):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        try:
            return float(tokens[1])
        except ValueError:
            continue
    return None


def _parse_center(lines: Sequence[str]) -> tuple[float | None, float | None]:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 3 or tokens[0].lower() != "center":
            continue
        if tokens[1].upper() == "AUTO":
            return None, None
        try:
            return float(tokens[1]), float(tokens[2])
        except ValueError:
            continue
    return None, None


def _normalize_imgname(value: object) -> str:
    return str(value).strip().strip('"').strip("'")


def _basename(value: object) -> str:
    return Path(_normalize_imgname(value)).name


def _extract_image_number(value: object) -> int | None:
    name = _basename(value)
    matches = re.findall(r"(\d+)", name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _frame_numbers_from_imgname(series: pd.Series) -> pd.Series:
    image_numbers = series.map(_extract_image_number).astype("float64")
    if image_numbers.notna().any():
        image_numbers = image_numbers.ffill().bfill()
        as_int = image_numbers.astype(int)
        offset = 1 if int(as_int.min()) == 0 else 0
        return as_int + offset
    return pd.Series(np.arange(1, len(series) + 1, dtype=int), index=series.index)


def _parse_cell(lines: Sequence[str]) -> UnitCell:
    for line in lines:
        stripped = line.strip()
        if not stripped.lower().startswith("cell "):
            continue
        tokens = stripped.split()
        if len(tokens) < 7:
            continue
        return UnitCell(
            a=float(tokens[1]),
            b=float(tokens[2]),
            c=float(tokens[3]),
            alpha=float(tokens[4]),
            beta=float(tokens[5]),
            gamma=float(tokens[6]),
        )
    raise ValueError("Could not parse `cell` from .pts2.")


def _parse_ubmatrix(lines: Sequence[str]) -> np.ndarray:
    for index, line in enumerate(lines):
        if line.strip().lower() != "ubmatrix":
            continue
        rows: list[list[float]] = []
        for probe in lines[index + 1 : index + 8]:
            stripped = probe.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                break
            try:
                row = [float(parts[0]), float(parts[1]), float(parts[2])]
            except ValueError:
                break
            rows.append(row)
            if len(rows) == 3:
                return np.asarray(rows, dtype=float)
    raise ValueError("Could not parse `ubmatrix` from .pts2.")


def _parse_imagelist(lines: Sequence[str]) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[dict[str, str | float | None]] = []
    in_block = False
    for raw_line in lines:
        stripped = raw_line.strip()
        lower = stripped.lower()
        if lower.startswith("imagelistheader"):
            parts = stripped.split()
            header = [token.strip().lower() for token in parts[1:]]
            continue
        if lower == "imagelist":
            in_block = True
            continue
        if lower == "endimagelist":
            in_block = False
            continue
        if not in_block or not stripped:
            continue
        values = shlex.split(stripped)
        if not values:
            continue
        if header is None:
            header = list(PETS_IMAGELIST_DEFAULT_HEADER)
        row: dict[str, str | float | None] = {}
        for idx, name in enumerate(header):
            row[name] = values[idx] if idx < len(values) else None
        rows.append(row)
    if not rows:
        raise ValueError("No imagelist rows parsed from .pts2.")

    table = pd.DataFrame.from_records(rows)
    if "imgname" not in table.columns:
        raise ValueError("imagelist does not include `imgname`.")
    table["imgname"] = table["imgname"].map(_normalize_imgname)
    table["imgbase"] = table["imgname"].map(_basename)
    for col in table.columns:
        if col in {"imgname", "imgbase"}:
            continue
        table[col] = pd.to_numeric(table[col], errors="coerce")

    table["frame_number"] = _frame_numbers_from_imgname(table["imgname"]).astype(int)
    table["frame_index"] = table["frame_number"].astype(int) - 1

    for base, fallback in (("alpha", "alphaorig"), ("beta", "betaorig"), ("domega", "domegaorig")):
        if base not in table.columns or not table[base].notna().any():
            if fallback in table.columns:
                table[base] = table[fallback]
            else:
                table[base] = 0.0
    for center_col in ("xcenter", "ycenter"):
        if center_col not in table.columns:
            table[center_col] = np.nan
    if "useforcalc" not in table.columns:
        table["useforcalc"] = 1.0

    return table


def _parse_ptsopt(path: Path) -> pd.DataFrame:
    lines = _clean_lines(path.read_text(errors="ignore"))
    header_line = next((line for line in lines if line.strip()), None)
    if header_line is None:
        return pd.DataFrame()

    header = header_line.split()
    rows: list[dict[str, float | str]] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        raw = stripped.split("|", 1)[0].strip()
        if not raw:
            continue
        tokens = raw.split()
        if len(tokens) < 6:
            continue
        row: dict[str, float | str] = {"imgname": _normalize_imgname(tokens[0])}

        # PETS .ptsopt commonly contains:
        # imgname xcenter ycenter alpha beta omega ...
        numeric_tokens = tokens[1:]
        known = {"xcenter": 0, "ycenter": 1, "alpha": 2, "beta": 3, "omega": 4}
        ok = True
        for key, pos in known.items():
            if pos >= len(numeric_tokens):
                ok = False
                break
            try:
                row[key] = float(numeric_tokens[pos])
            except ValueError:
                ok = False
                break
        if not ok:
            continue

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame.from_records(rows)
    table["imgbase"] = table["imgname"].map(_basename)
    table["frame_number"] = _frame_numbers_from_imgname(table["imgname"]).astype(int)
    return table


def _parse_ptsoptlist_best(path: Path) -> pd.DataFrame:
    lines = _clean_lines(path.read_text(errors="ignore"))
    rows: list[dict[str, float]] = []
    current_frame: int | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match_frame = re.search(r"Analysing frame nr\.\s*(\d+)", stripped, flags=re.IGNORECASE)
        if match_frame is not None:
            current_frame = int(match_frame.group(1))
            continue
        if not stripped.startswith("# Best result:"):
            continue
        numbers = re.findall(FLOAT_RE, stripped)
        if len(numbers) < 5 or current_frame is None:
            continue
        rows.append(
            {
                "frame_number": float(current_frame),
                "alpha": float(numbers[0]),
                "beta": float(numbers[1]),
                "omega": float(numbers[2]),
                "xcenter": float(numbers[3]),
                "ycenter": float(numbers[4]),
            }
        )
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame.from_records(rows)
    table = table.drop_duplicates("frame_number", keep="last")
    return table


def _merge_frame_table(imagelist: pd.DataFrame, ptsopt: pd.DataFrame | None) -> pd.DataFrame:
    table = imagelist.copy()
    if ptsopt is None or ptsopt.empty:
        return table

    merged = table.copy()
    override = ptsopt.copy()

    if "imgname" in merged.columns:
        merged["imgname"] = merged["imgname"].map(_normalize_imgname)
        merged["imgbase"] = merged["imgname"].map(_basename)
    if "imgname" in override.columns:
        override["imgname"] = override["imgname"].map(_normalize_imgname)
        override["imgbase"] = override["imgname"].map(_basename)

    override_by_frame = None
    if "frame_number" in override.columns:
        keyed = override.dropna(subset=["frame_number"]).drop_duplicates("frame_number", keep="last")
        if not keyed.empty:
            override_by_frame = keyed.set_index("frame_number")
    override_by_base = None
    if "imgbase" in override.columns:
        keyed = override.dropna(subset=["imgbase"]).drop_duplicates("imgbase", keep="last")
        if not keyed.empty:
            override_by_base = keyed.set_index("imgbase")

    for col in ("alpha", "beta", "xcenter", "ycenter", "omega"):
        if col not in merged.columns:
            merged[col] = np.nan
        merged_col = merged[col].copy()

        if override_by_frame is not None and col in override_by_frame.columns:
            frame_vals = merged["frame_number"].map(override_by_frame[col])
            merged_col = frame_vals.where(frame_vals.notna(), merged_col)

        if override_by_base is not None and col in override_by_base.columns:
            base_vals = merged["imgbase"].map(override_by_base[col])
            merged_col = base_vals.where(base_vals.notna(), merged_col)

        merged[col] = merged_col

    if "omega" in merged.columns:
        merged["domega"] = merged["omega"].where(merged["omega"].notna(), merged["domega"])
    return merged


def _resolve_paths(
    pets_root: Path,
    pts2_path: Path | None,
    ptsopt_path: Path | None,
) -> tuple[Path, Path | None, Path | None]:
    if pts2_path is None:
        candidates = sorted(pets_root.glob("*.pts2"))
        if not candidates:
            raise FileNotFoundError(f"No .pts2 file found in {pets_root}")
        pts2_path = candidates[0]

    ptsoptlist_path: Path | None = None
    if ptsopt_path is None:
        data_dirs = sorted(pets_root.glob("*_petsdata"))
        for data_dir in data_dirs:
            candidates = sorted(data_dir.glob("*.ptsopt"))
            if candidates:
                ptsopt_path = candidates[0]
            list_candidates = sorted(data_dir.glob("*.ptsoptlist"))
            if list_candidates:
                ptsoptlist_path = list_candidates[0]
            if ptsopt_path is not None or ptsoptlist_path is not None:
                break
    else:
        data_dirs = sorted(pets_root.glob("*_petsdata"))
        for data_dir in data_dirs:
            list_candidates = sorted(data_dir.glob("*.ptsoptlist"))
            if list_candidates:
                ptsoptlist_path = list_candidates[0]
                break
    return pts2_path, ptsopt_path, ptsoptlist_path


def load_pets_model(pets_root: Path, pts2_path: Path | None = None, ptsopt_path: Path | None = None) -> PetsModel:
    pts2_path, ptsopt_path, ptsoptlist_path = _resolve_paths(
        pets_root=pets_root,
        pts2_path=pts2_path,
        ptsopt_path=ptsopt_path,
    )
    lines = _clean_lines(pts2_path.read_text(errors="ignore"))

    wavelength = _parse_scalar(lines, "lambda")
    aperpixel = _parse_scalar(lines, "aperpixel")
    omega = _parse_scalar(lines, "omega")
    delta = _parse_scalar(lines, "delta")
    if wavelength is None or aperpixel is None or omega is None:
        raise ValueError("Missing required `lambda`/`aperpixel`/`omega` in .pts2.")
    if delta is None:
        delta = 0.0
    center_x, center_y = _parse_center(lines)

    ub_matrix = _parse_ubmatrix(lines)
    unit_cell = _parse_cell(lines)
    imagelist = _parse_imagelist(lines)
    ptsopt = _parse_ptsopt(ptsopt_path) if ptsopt_path is not None else None
    ptsoptlist = _parse_ptsoptlist_best(ptsoptlist_path) if ptsoptlist_path is not None else None
    merged = _merge_frame_table(imagelist=imagelist, ptsopt=ptsopt)
    merged = _merge_frame_table(imagelist=merged, ptsopt=ptsoptlist)

    for column in ("alpha", "beta", "domega", "xcenter", "ycenter", "useforcalc"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["alpha"] = merged["alpha"].ffill().bfill().fillna(0.0)
    merged["beta"] = merged["beta"].ffill().bfill().fillna(0.0)
    merged["domega"] = merged["domega"].ffill().bfill().fillna(0.0)

    if center_x is not None:
        merged["xcenter"] = merged["xcenter"].fillna(float(center_x))
    if center_y is not None:
        merged["ycenter"] = merged["ycenter"].fillna(float(center_y))
    x_median = float(np.nanmedian(merged["xcenter"].to_numpy(dtype=float)))
    y_median = float(np.nanmedian(merged["ycenter"].to_numpy(dtype=float)))
    if np.isfinite(x_median):
        merged["xcenter"] = merged["xcenter"].fillna(x_median)
    if np.isfinite(y_median):
        merged["ycenter"] = merged["ycenter"].fillna(y_median)
    if not merged["xcenter"].notna().all() or not merged["ycenter"].notna().all():
        n_missing_x = int(merged["xcenter"].isna().sum())
        n_missing_y = int(merged["ycenter"].isna().sum())
        raise ValueError(
            "Could not determine frame centers for all frames. "
            f"Missing xcenter={n_missing_x}, ycenter={n_missing_y}. "
            "Check .pts2/.ptsopt merge inputs."
        )

    merged["useforcalc"] = merged["useforcalc"].fillna(1.0)
    merged["frame_number"] = merged["frame_number"].astype(int)
    merged["frame_index"] = merged["frame_number"] - 1
    merged = merged.sort_values("frame_number").reset_index(drop=True)

    frames = tuple(
        FrameState(
            frame_index=int(row.frame_index),
            frame_number=int(row.frame_number),
            imgname=str(row.imgname),
            alpha=float(row.alpha),
            beta=float(row.beta),
            domega=float(row.domega),
            xcenter=float(row.xcenter),
            ycenter=float(row.ycenter),
            useforcalc=float(row.useforcalc),
        )
        for row in merged.itertuples(index=False)
    )

    return PetsModel(
        pts2_path=pts2_path,
        ptsopt_path=ptsopt_path,
        ptsoptlist_path=ptsoptlist_path,
        wavelength_angstrom=float(wavelength),
        aperpixel_invA_per_px=float(aperpixel),
        omega_deg=float(omega),
        delta_deg=float(delta),
        center_x_px=float(center_x) if center_x is not None else float(np.nanmedian(merged["xcenter"].to_numpy(dtype=float))),
        center_y_px=float(center_y) if center_y is not None else float(np.nanmedian(merged["ycenter"].to_numpy(dtype=float))),
        ub_matrix=ub_matrix,
        unit_cell=unit_cell,
        frames=frames,
    )


def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rotation_matrix_y(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotation_axis_from_omega_delta(omega_deg: float, delta_deg: float) -> np.ndarray:
    omega = np.deg2rad(float(omega_deg))
    delta = np.deg2rad(float(delta_deg))
    cdelta = float(np.cos(delta))
    return np.asarray(
        [cdelta * float(np.cos(omega)), -cdelta * float(np.sin(omega)), float(np.sin(delta))], dtype=float
    )


def _rodrigues(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("Rotation axis has zero norm.")
    ux, uy, uz = (axis / axis_norm).tolist()
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
        ],
        dtype=float,
    )


def _build_direct_basis(cell: UnitCell) -> np.ndarray:
    alpha = np.deg2rad(cell.alpha)
    beta = np.deg2rad(cell.beta)
    gamma = np.deg2rad(cell.gamma)
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("Cell gamma is too close to singular geometry.")
    cos_alpha = float(np.cos(alpha))
    cos_beta = float(np.cos(beta))
    cos_gamma = float(np.cos(gamma))

    a_vec = np.asarray([cell.a, 0.0, 0.0], dtype=float)
    b_vec = np.asarray([cell.b * cos_gamma, cell.b * sin_gamma, 0.0], dtype=float)
    c_x = cell.c * cos_beta
    c_y = cell.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z_sq = max(cell.c * cell.c - c_x * c_x - c_y * c_y, 0.0)
    c_vec = np.asarray([c_x, c_y, float(np.sqrt(c_z_sq))], dtype=float)
    return np.column_stack([a_vec, b_vec, c_vec])


def generate_hkls(model: PetsModel, dmin: float, dmax: float) -> np.ndarray:
    qmin = 1.0 / float(dmax)
    qmax = 1.0 / float(dmin)

    direct_basis = _build_direct_basis(model.unit_cell)
    reciprocal_basis = np.linalg.inv(direct_basis).T
    a_star, b_star, c_star = [float(np.linalg.norm(reciprocal_basis[:, idx])) for idx in range(3)]
    hmax = max(1, int(np.ceil(qmax / max(a_star, 1.0e-12))) + 2)
    kmax = max(1, int(np.ceil(qmax / max(b_star, 1.0e-12))) + 2)
    lmax = max(1, int(np.ceil(qmax / max(c_star, 1.0e-12))) + 2)

    hkls: list[tuple[int, int, int]] = []
    for h in range(-hmax, hmax + 1):
        for k in range(-kmax, kmax + 1):
            for l in range(-lmax, lmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                g = model.ub_matrix @ np.asarray([h, k, l], dtype=float)
                q = float(np.linalg.norm(g))
                if qmin <= q <= qmax:
                    hkls.append((h, k, l))
    if not hkls:
        raise ValueError("No hkls generated. Check dmin/dmax.")
    return np.asarray(hkls, dtype=int)


def _frame_rotation(
    model: PetsModel,
    frame: FrameState,
    reference_frame: FrameState,
    mode: str,
    angle_reference: str,
    include_domega_in_lattice: bool,
    invert: bool,
) -> np.ndarray:
    if angle_reference in {"absolute", "zero"}:
        alpha_eff = float(frame.alpha)
        beta_eff = float(frame.beta)
        domega_eff = float(frame.domega)
    elif angle_reference == "first_frame":
        alpha_eff = float(frame.alpha - reference_frame.alpha)
        beta_eff = float(frame.beta - reference_frame.beta)
        domega_eff = float(frame.domega - reference_frame.domega)
    else:
        raise ValueError(f"Unknown angle_reference: {angle_reference}")

    rz = _rotation_matrix_z(domega_eff) if include_domega_in_lattice else np.eye(3, dtype=float)

    if mode == "axis_alpha_legacy":
        axis = _rotation_axis_from_omega_delta(model.omega_deg, model.delta_deg)
        rotation = _rodrigues(axis, alpha_eff)
    elif mode == "fixed_x_alpha":
        rotation = _rotation_matrix_x(alpha_eff)
    elif mode == "pets_ab_xy":
        rotation = _rotation_matrix_y(beta_eff) @ _rotation_matrix_x(alpha_eff)
    elif mode == "pets_ab_yx":
        rotation = _rotation_matrix_x(alpha_eff) @ _rotation_matrix_y(beta_eff)
    elif mode == "euler_yxz":
        rotation = _rotation_matrix_y(alpha_eff) @ _rotation_matrix_x(beta_eff)
    elif mode == "euler_xyz":
        rotation = _rotation_matrix_x(alpha_eff) @ _rotation_matrix_y(beta_eff)
    elif mode == "none":
        rotation = np.eye(3, dtype=float)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    rotation = rotation @ rz
    return rotation.T if invert else rotation


def _sg(g_vectors: np.ndarray, wavelength: float, beam_dir: np.ndarray) -> np.ndarray:
    beam_dir = np.asarray(beam_dir, dtype=float)
    beam_dir = beam_dir / np.linalg.norm(beam_dir)
    k0 = beam_dir[None, :] / float(wavelength)
    return np.linalg.norm(g_vectors + k0, axis=1) - (1.0 / float(wavelength))


def _local_alpha_step(frames: Sequence[FrameState], index: int) -> float:
    if len(frames) <= 1:
        return 0.0
    if index == 0:
        return float(frames[1].alpha - frames[0].alpha)
    if index == len(frames) - 1:
        return float(frames[-1].alpha - frames[-2].alpha)
    return 0.5 * float(frames[index + 1].alpha - frames[index - 1].alpha)


def _apply_xy_mapping(
    x: np.ndarray,
    y: np.ndarray,
    detector_nx: int,
    detector_ny: int,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if swap_xy:
        x, y = y.copy(), x.copy()
        detector_nx, detector_ny = detector_ny, detector_nx
    if flip_x:
        x = (detector_nx - 1) - x
    if flip_y:
        y = (detector_ny - 1) - y
    return x, y


def _map_detector_point(
    x: float,
    y: float,
    detector_nx: int,
    detector_ny: int,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
) -> tuple[float, float]:
    x_arr = np.asarray([float(x)], dtype=float)
    y_arr = np.asarray([float(y)], dtype=float)
    x_map, y_map = _apply_xy_mapping(
        x=x_arr,
        y=y_arr,
        detector_nx=detector_nx,
        detector_ny=detector_ny,
        swap_xy=swap_xy,
        flip_x=flip_x,
        flip_y=flip_y,
    )
    return float(x_map[0]), float(y_map[0])


def _reference_g_vectors(hkls: np.ndarray, ub_matrix: np.ndarray, ub_convention: str) -> np.ndarray:
    if ub_convention == "columns":
        return hkls @ ub_matrix.T
    if ub_convention == "rows":
        return hkls @ ub_matrix
    raise ValueError(f"Unknown ub_convention: {ub_convention}")


def _rotate_detector_plane(u: np.ndarray, v: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    x_rot = c * u - s * v
    y_rot = s * u + c * v
    return x_rot, y_rot


def predict_frame(
    model: PetsModel,
    frame: FrameState,
    frames: Sequence[FrameState],
    reference_frame: FrameState,
    hkls: np.ndarray,
    detector_nx: int,
    detector_ny: int,
    mode: str,
    projection: str,
    excitation_tolerance: float,
    beam_direction: str,
    ub_convention: str,
    angle_reference: str,
    include_domega_in_lattice: bool,
    omega_map_mode: str,
    omega_sign: float,
    omega_offset_deg: float,
    invert_rotation: bool,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
) -> pd.DataFrame:
    beam_dir = np.asarray([0.0, 0.0, -1.0 if beam_direction == "minus_z" else 1.0], dtype=float)
    frame_index = int(frame.frame_index)

    g_ref = _reference_g_vectors(hkls, model.ub_matrix, ub_convention=ub_convention)
    rotation_mid = _frame_rotation(
        model,
        frame,
        reference_frame,
        mode=mode,
        angle_reference=angle_reference,
        include_domega_in_lattice=include_domega_in_lattice,
        invert=invert_rotation,
    )
    g_mid = g_ref @ rotation_mid.T
    sg_mid = _sg(g_mid, model.wavelength_angstrom, beam_dir=beam_dir)

    if angle_reference in {"absolute", "zero"}:
        alpha_eff = float(frame.alpha)
    elif angle_reference == "first_frame":
        alpha_eff = float(frame.alpha - reference_frame.alpha)
    else:
        raise ValueError(f"Unknown angle_reference: {angle_reference}")

    if mode in {"axis_alpha_legacy", "fixed_x_alpha", "pets_ab_xy", "pets_ab_yx"}:
        step = _local_alpha_step(frames, frame_index)
        rel_start = alpha_eff - 0.5 * step
        rel_end = alpha_eff + 0.5 * step
        if mode == "axis_alpha_legacy":
            axis = _rotation_axis_from_omega_delta(model.omega_deg, model.delta_deg)
            rot_start = _rodrigues(axis, rel_start)
            rot_end = _rodrigues(axis, rel_end)
        elif mode == "pets_ab_xy":
            if angle_reference in {"absolute", "zero"}:
                beta_eff = float(frame.beta)
            else:
                beta_eff = float(frame.beta - reference_frame.beta)
            rot_start = _rotation_matrix_y(beta_eff) @ _rotation_matrix_x(rel_start)
            rot_end = _rotation_matrix_y(beta_eff) @ _rotation_matrix_x(rel_end)
        elif mode == "pets_ab_yx":
            if angle_reference in {"absolute", "zero"}:
                beta_eff = float(frame.beta)
            else:
                beta_eff = float(frame.beta - reference_frame.beta)
            rot_start = _rotation_matrix_x(rel_start) @ _rotation_matrix_y(beta_eff)
            rot_end = _rotation_matrix_x(rel_end) @ _rotation_matrix_y(beta_eff)
        else:
            rot_start = _rotation_matrix_x(rel_start)
            rot_end = _rotation_matrix_x(rel_end)

        if include_domega_in_lattice:
            if angle_reference in {"absolute", "zero"}:
                domega_eff = float(frame.domega)
            else:
                domega_eff = float(frame.domega - reference_frame.domega)
            rz = _rotation_matrix_z(domega_eff)
            rot_start = rot_start @ rz
            rot_end = rot_end @ rz

        if invert_rotation:
            rot_start = rot_start.T
            rot_end = rot_end.T
        g_start = g_ref @ rot_start.T
        g_end = g_ref @ rot_end.T
        sg_start = _sg(g_start, model.wavelength_angstrom, beam_dir=beam_dir)
        sg_end = _sg(g_end, model.wavelength_angstrom, beam_dir=beam_dir)
        excited = ((sg_start * sg_end) <= 0.0) | (np.abs(sg_mid) < float(excitation_tolerance))
    else:
        excited = np.abs(sg_mid) < float(excitation_tolerance)

    k0 = beam_dir / float(model.wavelength_angstrom)
    k_mid = g_mid + k0[None, :]
    if projection == "full":
        kx = k_mid[:, 0]
        ky = k_mid[:, 1]
        kz = k_mid[:, 2]
        forward = kz < -1.0e-9 if beam_direction == "minus_z" else kz > 1.0e-9
        scale = 1.0 / (float(model.aperpixel_invA_per_px) * float(model.wavelength_angstrom))
        denom = -kz if beam_direction == "minus_z" else kz
        u = (kx / denom) * scale
        v = (ky / denom) * scale
    else:
        forward = np.ones(g_mid.shape[0], dtype=bool)
        u = g_mid[:, 0] / float(model.aperpixel_invA_per_px)
        v = g_mid[:, 1] / float(model.aperpixel_invA_per_px)

    if omega_map_mode == "global":
        omega_map = float(model.omega_deg)
    elif omega_map_mode in {"frame_only", "frame_absolute"}:
        omega_map = float(frame.domega)
    elif omega_map_mode == "global_plus_frame":
        omega_map = float(model.omega_deg + frame.domega)
    elif omega_map_mode == "none":
        omega_map = 0.0
    else:
        raise ValueError(f"Unknown omega_map_mode: {omega_map_mode}")
    omega_total = float(omega_sign) * omega_map + float(omega_offset_deg)
    x_local, y_local = _rotate_detector_plane(u=u, v=v, angle_deg=omega_total)
    x = frame.xcenter + x_local
    y = frame.ycenter + y_local

    x, y = _apply_xy_mapping(
        x=x,
        y=y,
        detector_nx=detector_nx,
        detector_ny=detector_ny,
        swap_xy=swap_xy,
        flip_x=flip_x,
        flip_y=flip_y,
    )
    in_bounds = (x >= 0.0) & (x < detector_nx) & (y >= 0.0) & (y < detector_ny)
    keep = excited & forward & in_bounds

    if not np.any(keep):
        return pd.DataFrame(columns=["frame", "frame_number", "h", "k", "l", "x_px", "y_px", "sg_invA", "score"])

    hkl_keep = hkls[keep, :]
    sg_keep = sg_mid[keep]
    score = np.exp(-np.square(sg_keep / max(float(excitation_tolerance), 1.0e-9)))
    return pd.DataFrame(
        {
            "frame": frame.frame_index,
            "frame_number": frame.frame_number,
            "h": hkl_keep[:, 0],
            "k": hkl_keep[:, 1],
            "l": hkl_keep[:, 2],
            "x_px": x[keep],
            "y_px": y[keep],
            "sg_invA": sg_keep,
            "score": score,
        }
    )


def _try_load_image(images_dir: Path | None, imgname: str) -> np.ndarray | None:
    if images_dir is None:
        return None
    candidates = [images_dir / imgname, images_dir / Path(imgname).name]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            image = plt.imread(candidate)
            return image
        except Exception:
            continue
    return None


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.mean(arr, axis=2)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {arr.shape}")


def _spot_image_alignment_score(
    *,
    spots: pd.DataFrame,
    image: np.ndarray | None,
    detector_nx: int,
    detector_ny: int,
) -> tuple[float, float, float, int]:
    if image is None or spots.empty:
        return float("nan"), float("nan"), float("nan"), int(spots.shape[0])

    gray = _to_grayscale(image)
    if gray.size == 0:
        return float("nan"), float("nan"), float("nan"), int(spots.shape[0])

    med = float(np.median(gray))
    mad = float(np.median(np.abs(gray - med)))
    scale = 1.4826 * mad if mad > 1.0e-12 else float(np.std(gray)) + 1.0e-12
    norm = (gray - med) / scale

    height, width = norm.shape
    sx = (width - 1) / max(detector_nx - 1, 1)
    sy = (height - 1) / max(detector_ny - 1, 1)
    x = np.round(spots["x_px"].to_numpy(dtype=float) * sx).astype(int)
    y = np.round(spots["y_px"].to_numpy(dtype=float) * sy).astype(int)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan"), int(spots.shape[0])

    xv = x[valid]
    yv = y[valid]

    # Local-contrast score is more robust than absolute brightness:
    # true matches land on local peaks, not just globally bright regions.
    contrast_values: list[float] = []
    for xi, yi in zip(xv.tolist(), yv.tolist(), strict=True):
        x0 = max(0, xi - 1)
        x1 = min(width, xi + 2)
        y0 = max(0, yi - 1)
        y1 = min(height, yi + 2)
        patch = norm[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        peak = float(np.max(patch))

        o0 = max(0, xi - 4)
        o1 = min(width, xi + 5)
        p0 = max(0, yi - 4)
        p1 = min(height, yi + 5)
        outer = norm[p0:p1, o0:o1]
        if outer.size == 0:
            contrast_values.append(peak)
            continue
        yy, xx = np.mgrid[p0:p1, o0:o1]
        rr = np.sqrt((xx - xi) ** 2 + (yy - yi) ** 2)
        ring = outer[(rr >= 2.0) & (rr <= 4.0)]
        if ring.size == 0:
            contrast_values.append(peak)
            continue
        contrast_values.append(peak - float(np.median(ring)))

    if not contrast_values:
        return float("nan"), float("nan"), float("nan"), int(spots.shape[0])

    values = np.asarray(contrast_values, dtype=float)
    mean_v = float(np.mean(values))
    p85_v = float(np.quantile(values, 0.85))
    n_spots = int(spots.shape[0])
    score = mean_v + 0.6 * p85_v + 0.04 * float(np.log1p(n_spots))
    return score, mean_v, p85_v, n_spots


def _candidate_signature(candidate: dict[str, object]) -> str:
    ub = str(candidate["ub_convention"])
    mode = str(candidate["orientation_mode"])
    ang_ref = str(candidate["angle_reference"])
    do_lat = "do_lattice" if bool(candidate["include_domega_in_lattice"]) else "no_do_lattice"
    proj = str(candidate["projection"])
    beam = str(candidate["beam_direction"])
    omega_mode = str(candidate["omega_map_mode"])
    omega_sign = "oplus" if float(candidate["omega_sign"]) >= 0.0 else "ominus"
    omega_off = float(candidate.get("omega_offset_deg", 0.0))
    omega_off_label = f"ooff{omega_off:+.1f}".replace("+", "p").replace("-", "m")
    inv = "inv" if bool(candidate["invert_rotation"]) else "fwd"
    swap = "swap" if bool(candidate["swap_xy"]) else "noswap"
    fx = "flipx" if bool(candidate["flip_x"]) else "noflipx"
    fy = "flipy" if bool(candidate["flip_y"]) else "noflipy"
    return (
        f"{ub}__{mode}__{ang_ref}__{do_lat}__{proj}__{beam}__"
        f"{omega_mode}__{omega_sign}__{omega_off_label}__{inv}__{swap}__{fx}__{fy}"
    )


def _candidate_to_cli_flags(candidate: dict[str, object]) -> str:
    parts = [
        f"--ub-convention {candidate['ub_convention']}",
        f"--orientation-mode {candidate['orientation_mode']}",
        f"--angle-reference {candidate['angle_reference']}",
        f"--projection {candidate['projection']}",
        f"--beam-direction {candidate['beam_direction']}",
        f"--omega-map-mode {candidate['omega_map_mode']}",
        f"--omega-sign {candidate['omega_sign']}",
        f"--omega-offset-deg {candidate.get('omega_offset_deg', 0.0)}",
    ]
    if bool(candidate["include_domega_in_lattice"]):
        parts.append("--include-domega-in-lattice")
    if bool(candidate["invert_rotation"]):
        parts.append("--invert-rotation")
    if bool(candidate["swap_xy"]):
        parts.append("--swap-xy")
    if bool(candidate["flip_x"]):
        parts.append("--flip-x")
    if bool(candidate["flip_y"]):
        parts.append("--flip-y")
    return " ".join(parts)


def _build_debug_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    modes = ("pets_ab_xy", "pets_ab_yx", "fixed_x_alpha", "euler_yxz", "axis_alpha_legacy", "none")
    for ub_convention, mode in itertools.product(("columns", "rows"), modes):
        angle_refs = ("absolute",) if mode == "none" else ("absolute", "first_frame")
        # Keep domega-in-lattice mostly for explicit testing; avoid redundant huge sweeps.
        if mode in {"pets_ab_xy", "pets_ab_yx", "euler_yxz"}:
            domega_opts = (False, True)
        else:
            domega_opts = (False,)

        for angle_reference, include_domega_in_lattice, beam, omega_mode, omega_sign, omega_offset_deg, inv, swap, fx, fy in itertools.product(
            angle_refs,
            domega_opts,
            ("minus_z", "plus_z"),
            ("frame_absolute", "global_plus_frame", "global", "none"),
            (-1.0, 1.0),
            (0.0,),
            (False, True),
            (False, True),
            (False, True),
            (False, True),
        ):
            candidates.append(
                {
                    "ub_convention": ub_convention,
                    "orientation_mode": mode,
                    "angle_reference": angle_reference,
                    "include_domega_in_lattice": bool(include_domega_in_lattice),
                    "projection": "full",
                    "beam_direction": beam,
                    "omega_map_mode": omega_mode,
                    "omega_sign": float(omega_sign),
                    "omega_offset_deg": float(omega_offset_deg),
                    "invert_rotation": inv,
                    "swap_xy": swap,
                    "flip_x": fx,
                    "flip_y": fy,
                }
            )
    return candidates


def _run_debug_convention_sweep(
    *,
    model: PetsModel,
    selected_frames: Sequence[FrameState],
    hkls: np.ndarray,
    reference_frame: FrameState,
    images_dir: Path,
    output_dir: Path,
    detector_nx: int,
    detector_ny: int,
    excitation_tolerance: float,
    top_n: int,
) -> None:
    candidates = _build_debug_candidates()
    rows: list[dict[str, object]] = []
    print(f"Debug sweep: testing {len(candidates)} convention candidates")
    progress_stride = max(64, len(candidates) // 40)

    for idx, candidate in enumerate(candidates, start=1):
        per_frame_scores: list[float] = []
        per_frame_mean: list[float] = []
        per_frame_p85: list[float] = []
        per_frame_n: list[int] = []
        for frame in selected_frames:
            spots = predict_frame(
                model=model,
                frame=frame,
                frames=model.frames,
                reference_frame=reference_frame,
                hkls=hkls,
                detector_nx=detector_nx,
                detector_ny=detector_ny,
                mode=str(candidate["orientation_mode"]),
                projection=str(candidate["projection"]),
                excitation_tolerance=float(excitation_tolerance),
                beam_direction=str(candidate["beam_direction"]),
                ub_convention=str(candidate["ub_convention"]),
                angle_reference=str(candidate["angle_reference"]),
                include_domega_in_lattice=bool(candidate["include_domega_in_lattice"]),
                omega_map_mode=str(candidate["omega_map_mode"]),
                omega_sign=float(candidate["omega_sign"]),
                omega_offset_deg=float(candidate.get("omega_offset_deg", 0.0)),
                invert_rotation=bool(candidate["invert_rotation"]),
                swap_xy=bool(candidate["swap_xy"]),
                flip_x=bool(candidate["flip_x"]),
                flip_y=bool(candidate["flip_y"]),
            )
            image = _try_load_image(images_dir, frame.imgname)
            score, mean_v, p85_v, n_spots = _spot_image_alignment_score(
                spots=spots, image=image, detector_nx=detector_nx, detector_ny=detector_ny
            )
            per_frame_scores.append(score)
            per_frame_mean.append(mean_v)
            per_frame_p85.append(p85_v)
            per_frame_n.append(n_spots)

        agg_score = float(np.nanmean(np.asarray(per_frame_scores, dtype=float)))
        agg_mean = float(np.nanmean(np.asarray(per_frame_mean, dtype=float)))
        agg_p85 = float(np.nanmean(np.asarray(per_frame_p85, dtype=float)))
        agg_n = float(np.mean(np.asarray(per_frame_n, dtype=float)))

        row = {
            "rank_placeholder": idx,
            "candidate": _candidate_signature(candidate),
            "score": agg_score,
            "mean_pixel_z": agg_mean,
            "p85_pixel_z": agg_p85,
            "mean_spot_count": agg_n,
            **candidate,
        }
        rows.append(row)
        if idx % progress_stride == 0 or idx == len(candidates):
            print(f"  evaluated {idx}/{len(candidates)} candidates")

    score_table = pd.DataFrame.from_records(rows).sort_values("score", ascending=False).reset_index(drop=True)
    score_table.insert(0, "rank", np.arange(1, score_table.shape[0] + 1, dtype=int))
    score_csv = output_dir / "debug_convention_scores.csv"
    score_table.to_csv(score_csv, index=False)
    print(f"Wrote convention ranking: {score_csv}")

    top_dir = output_dir / "debug_top_candidates"
    top_dir.mkdir(parents=True, exist_ok=True)
    top_rows = score_table.head(max(int(top_n), 1))
    for _, row in top_rows.iterrows():
        rank = int(row["rank"])
        candidate = {
            "ub_convention": str(row["ub_convention"]),
            "orientation_mode": str(row["orientation_mode"]),
            "angle_reference": str(row["angle_reference"]),
            "include_domega_in_lattice": bool(row["include_domega_in_lattice"]),
            "projection": str(row["projection"]),
            "beam_direction": str(row["beam_direction"]),
            "omega_map_mode": str(row["omega_map_mode"]),
            "omega_sign": float(row["omega_sign"]),
            "omega_offset_deg": float(row.get("omega_offset_deg", 0.0)),
            "invert_rotation": bool(row["invert_rotation"]),
            "swap_xy": bool(row["swap_xy"]),
            "flip_x": bool(row["flip_x"]),
            "flip_y": bool(row["flip_y"]),
        }
        label = f"rank_{rank:02d}__{_candidate_signature(candidate)}"
        candidate_dir = top_dir / label
        candidate_dir.mkdir(parents=True, exist_ok=True)

        for frame in selected_frames:
            spots = predict_frame(
                model=model,
                frame=frame,
                frames=model.frames,
                reference_frame=reference_frame,
                hkls=hkls,
                detector_nx=detector_nx,
                detector_ny=detector_ny,
                mode=str(candidate["orientation_mode"]),
                projection=str(candidate["projection"]),
                excitation_tolerance=float(excitation_tolerance),
                beam_direction=str(candidate["beam_direction"]),
                ub_convention=str(candidate["ub_convention"]),
                angle_reference=str(candidate["angle_reference"]),
                include_domega_in_lattice=bool(candidate["include_domega_in_lattice"]),
                omega_map_mode=str(candidate["omega_map_mode"]),
                omega_sign=float(candidate["omega_sign"]),
                omega_offset_deg=float(candidate.get("omega_offset_deg", 0.0)),
                invert_rotation=bool(candidate["invert_rotation"]),
                swap_xy=bool(candidate["swap_xy"]),
                flip_x=bool(candidate["flip_x"]),
                flip_y=bool(candidate["flip_y"]),
            )
            spots.to_csv(candidate_dir / f"predicted_spots_frame_{frame.frame_number:04d}.csv", index=False)
            image = _try_load_image(images_dir, frame.imgname)
            cx_map, cy_map = _map_detector_point(
                x=float(frame.xcenter),
                y=float(frame.ycenter),
                detector_nx=detector_nx,
                detector_ny=detector_ny,
                swap_xy=bool(candidate["swap_xy"]),
                flip_x=bool(candidate["flip_x"]),
                flip_y=bool(candidate["flip_y"]),
            )
            plot_frame(
                frame=frame,
                spots=spots,
                detector_nx=detector_nx,
                detector_ny=detector_ny,
                output_png=candidate_dir / f"detector_frame_{frame.frame_number:04d}.png",
                image=image,
                center_x_px=cx_map,
                center_y_px=cy_map,
            )

    best = top_rows.iloc[0]
    best_candidate = {
        "ub_convention": str(best["ub_convention"]),
        "orientation_mode": str(best["orientation_mode"]),
        "angle_reference": str(best["angle_reference"]),
        "include_domega_in_lattice": bool(best["include_domega_in_lattice"]),
        "projection": str(best["projection"]),
        "beam_direction": str(best["beam_direction"]),
        "omega_map_mode": str(best["omega_map_mode"]),
        "omega_sign": float(best["omega_sign"]),
        "omega_offset_deg": float(best.get("omega_offset_deg", 0.0)),
        "invert_rotation": bool(best["invert_rotation"]),
        "swap_xy": bool(best["swap_xy"]),
        "flip_x": bool(best["flip_x"]),
        "flip_y": bool(best["flip_y"]),
    }
    best_flags = _candidate_to_cli_flags(best_candidate)
    (output_dir / "best_convention_flags.txt").write_text(best_flags + "\n")
    print(
        "Best convention: "
        f"rank=1, candidate={best['candidate']}, score={float(best['score']):.4f}, "
        f"spots≈{float(best['mean_spot_count']):.1f}"
    )
    print(f"Best convention flags: {best_flags}")
    print(f"Wrote best flags: {output_dir / 'best_convention_flags.txt'}")


def plot_frame(
    frame: FrameState,
    spots: pd.DataFrame,
    detector_nx: int,
    detector_ny: int,
    output_png: Path,
    image: np.ndarray | None,
    center_x_px: float | None = None,
    center_y_px: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    if image is not None:
        ax.imshow(image, cmap="gray", origin="upper")
    else:
        ax.set_facecolor("black")

    if not spots.empty:
        sc = ax.scatter(
            spots["x_px"],
            spots["y_px"],
            c=spots["score"],
            s=15.0 + 70.0 * spots["score"],
            cmap="turbo",
            edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label="geometry score")

    cx = float(frame.xcenter) if center_x_px is None else float(center_x_px)
    cy = float(frame.ycenter) if center_y_px is None else float(center_y_px)
    ax.axvline(cx, color="cyan", linewidth=1.0, alpha=0.8, linestyle="--")
    ax.axhline(cy, color="cyan", linewidth=1.0, alpha=0.8, linestyle="--")
    ax.scatter([cx], [cy], s=70, c="none", edgecolors="red", linewidths=1.8, marker="o", zorder=6)
    ax.scatter([cx], [cy], s=70, c="red", marker="+", linewidths=1.8, zorder=7)
    ax.text(
        cx + 6.0,
        cy - 6.0,
        f"center ({cx:.2f}, {cy:.2f})",
        color="red",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
        zorder=8,
    )
    ax.set_xlim(0, detector_nx)
    ax.set_ylim(detector_ny, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Detector x / px")
    ax.set_ylabel("Detector y / px")
    ax.set_title(f"Predicted spots frame {frame.frame_number} ({Path(frame.imgname).name})")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def _resolve_requested_frames(
    frames: Sequence[FrameState],
    requested: Sequence[int] | None,
    use_only_for_calc: bool,
) -> list[FrameState]:
    if use_only_for_calc:
        frames = [frame for frame in frames if frame.useforcalc > 0.0]
    frame_by_number = {frame.frame_number: frame for frame in frames}
    if requested:
        selected: list[FrameState] = []
        missing: list[int] = []
        for number in requested:
            frame = frame_by_number.get(int(number))
            if frame is None:
                missing.append(int(number))
                continue
            selected.append(frame)
        if missing:
            raise ValueError(f"Requested frame numbers not present: {missing}")
        return selected
    return list(frames)


def _infer_detector_shape_from_images(images_dir: Path | None, selected_frames: Sequence[FrameState]) -> tuple[int, int] | None:
    if images_dir is None:
        return None
    for frame in selected_frames:
        image = _try_load_image(images_dir, frame.imgname)
        if image is None:
            continue
        gray = _to_grayscale(image)
        if gray.ndim != 2:
            continue
        ny, nx = gray.shape
        if nx > 0 and ny > 0:
            return int(nx), int(ny)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pets-root", required=True, type=Path, help="PETS root folder containing *.pts2")
    parser.add_argument("--pts2", type=Path, default=None, help="Optional explicit .pts2 path")
    parser.add_argument("--ptsopt", type=Path, default=None, help="Optional explicit .ptsopt path")
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional directory of raw frame images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG/CSV outputs")
    parser.add_argument("--frame", dest="frames", action="append", type=int, default=None, help="Frame number")
    parser.add_argument("--dmin", type=float, default=0.6, help="Minimum d spacing (A)")
    parser.add_argument("--dmax", type=float, default=50.0, help="Maximum d spacing (A)")
    parser.add_argument("--excitation-tolerance", type=float, default=1.5e-3, help="Excitation tolerance (1/A)")
    parser.add_argument(
        "--ub-convention",
        choices=["columns", "rows"],
        default="columns",
        help="Interpretation of UB entries: columns=a*/b*/c* (PETS default) or rows",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=["pets_ab_xy", "pets_ab_yx", "fixed_x_alpha", "euler_yxz", "euler_xyz", "axis_alpha_legacy", "none"],
        default="pets_ab_xy",
        help="Frame-orientation model",
    )
    parser.add_argument(
        "--angle-reference",
        choices=["absolute", "first_frame", "zero"],
        default="absolute",
        help="Interpret alpha/beta/domega as absolute values or offsets from first frame",
    )
    parser.add_argument(
        "--include-domega-in-lattice",
        action="store_true",
        help="Include domega as lattice Rz rotation (disabled by default to avoid omega double-counting)",
    )
    parser.add_argument(
        "--projection",
        choices=["full", "paraxial"],
        default="full",
        help="Detector projection model",
    )
    parser.add_argument(
        "--beam-direction",
        choices=["minus_z", "plus_z"],
        default="minus_z",
        help="Incident beam direction in PETS fixed frame",
    )
    parser.add_argument(
        "--omega-map-mode",
        choices=["frame_absolute", "global_plus_frame", "global", "frame_only", "none"],
        default="frame_absolute",
        help="How PETS omega is applied when mapping reciprocal x/y onto detector x/y",
    )
    parser.add_argument(
        "--omega-sign",
        type=float,
        default=1.0,
        help="Sign applied to omega during detector-plane mapping (+1 or -1)",
    )
    parser.add_argument(
        "--omega-offset-deg",
        type=float,
        default=0.0,
        help="Constant in-plane angle offset (deg) added after omega sign/mode mapping",
    )
    parser.add_argument("--invert-rotation", action="store_true", help="Use inverse rotation convention")
    parser.add_argument("--swap-xy", action="store_true", help="Swap detector x/y for plotting")
    parser.add_argument("--flip-x", action="store_true", help="Flip detector x axis")
    parser.add_argument("--flip-y", action="store_true", help="Flip detector y axis")
    parser.add_argument("--detector-nx", type=int, default=0, help="Detector width in px (<=0: auto from image)")
    parser.add_argument("--detector-ny", type=int, default=0, help="Detector height in px (<=0: auto from image)")
    parser.add_argument(
        "--use-only-for-calc",
        action="store_true",
        help="Use only frames with useforcalc > 0 from imagelist",
    )
    parser.add_argument(
        "--debug-conventions",
        action="store_true",
        help="Sweep common convention combinations and rank by image-alignment score",
    )
    parser.add_argument(
        "--debug-top-n",
        type=int,
        default=8,
        help="Number of top-ranked convention overlays to export in debug mode",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_pets_model(pets_root=args.pets_root, pts2_path=args.pts2, ptsopt_path=args.ptsopt)
    selected_frames = _resolve_requested_frames(
        frames=model.frames,
        requested=args.frames,
        use_only_for_calc=bool(args.use_only_for_calc),
    )
    if not selected_frames:
        raise ValueError("No frames selected.")

    detector_nx = int(args.detector_nx)
    detector_ny = int(args.detector_ny)
    inferred_shape = _infer_detector_shape_from_images(args.images_dir, selected_frames)
    if inferred_shape is not None and (detector_nx <= 0 or detector_ny <= 0):
        detector_nx, detector_ny = inferred_shape
    if detector_nx <= 0:
        detector_nx = 512
    if detector_ny <= 0:
        detector_ny = 512

    hkls = generate_hkls(model=model, dmin=float(args.dmin), dmax=float(args.dmax))
    reference_frame = min(model.frames, key=lambda frame: frame.frame_number)

    metadata = {
        "pets_root": str(args.pets_root.resolve()),
        "pts2_path": str(model.pts2_path.resolve()),
        "ptsopt_path": None if model.ptsopt_path is None else str(model.ptsopt_path.resolve()),
        "ptsoptlist_path": None if model.ptsoptlist_path is None else str(model.ptsoptlist_path.resolve()),
        "n_frames_total": len(model.frames),
        "n_frames_selected": len(selected_frames),
        "wavelength_angstrom": model.wavelength_angstrom,
        "aperpixel_invA_per_px": model.aperpixel_invA_per_px,
        "omega_deg": model.omega_deg,
        "delta_deg": model.delta_deg,
        "center_x_px": model.center_x_px,
        "center_y_px": model.center_y_px,
        "ub_matrix": model.ub_matrix.tolist(),
        "unit_cell": {
            "a": model.unit_cell.a,
            "b": model.unit_cell.b,
            "c": model.unit_cell.c,
            "alpha": model.unit_cell.alpha,
            "beta": model.unit_cell.beta,
            "gamma": model.unit_cell.gamma,
        },
        "hkls_generated": int(hkls.shape[0]),
        "settings": {
            "dmin": float(args.dmin),
            "dmax": float(args.dmax),
            "excitation_tolerance_invA": float(args.excitation_tolerance),
            "ub_convention": args.ub_convention,
            "orientation_mode": args.orientation_mode,
            "angle_reference": args.angle_reference,
            "include_domega_in_lattice": bool(args.include_domega_in_lattice),
            "projection": args.projection,
            "beam_direction": args.beam_direction,
            "omega_map_mode": args.omega_map_mode,
            "omega_sign": float(args.omega_sign),
            "omega_offset_deg": float(args.omega_offset_deg),
            "invert_rotation": bool(args.invert_rotation),
            "swap_xy": bool(args.swap_xy),
            "flip_x": bool(args.flip_x),
            "flip_y": bool(args.flip_y),
            "detector_nx": int(detector_nx),
            "detector_ny": int(detector_ny),
            "debug_conventions": bool(args.debug_conventions),
            "debug_top_n": int(args.debug_top_n),
        },
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(
        "Loaded PETS model: "
        f"{model.pts2_path.name}, frames={len(model.frames)}, selected={len(selected_frames)}, "
        f"hkls={hkls.shape[0]}"
    )
    print(
        "Reference frame: "
        f"{reference_frame.frame_number} ({Path(reference_frame.imgname).name}), "
        f"alpha={reference_frame.alpha:.4f}, beta={reference_frame.beta:.4f}, domega={reference_frame.domega:.4f}"
    )
    print(f"Detector shape used: nx={detector_nx}, ny={detector_ny}")
    print(
        "Model settings: "
        f"ub={args.ub_convention}, mode={args.orientation_mode}, angle_ref={args.angle_reference}, "
        f"domega_in_lattice={bool(args.include_domega_in_lattice)}, "
        f"omega_map={args.omega_map_mode}, omega_sign={float(args.omega_sign):+.1f}, "
        f"omega_offset={float(args.omega_offset_deg):+.2f}"
    )
    if args.debug_conventions:
        if args.images_dir is None:
            raise ValueError("--debug-conventions requires --images-dir for scoring.")
        _run_debug_convention_sweep(
            model=model,
            selected_frames=selected_frames,
            hkls=hkls,
            reference_frame=reference_frame,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            detector_nx=detector_nx,
            detector_ny=detector_ny,
            excitation_tolerance=float(args.excitation_tolerance),
            top_n=int(args.debug_top_n),
        )
    else:
        center_rows: list[dict[str, float | int | str]] = []
        for frame in selected_frames:
            spots = predict_frame(
                model=model,
                frame=frame,
                frames=model.frames,
                reference_frame=reference_frame,
                hkls=hkls,
                detector_nx=detector_nx,
                detector_ny=detector_ny,
                mode=str(args.orientation_mode),
                projection=str(args.projection),
                excitation_tolerance=float(args.excitation_tolerance),
                beam_direction=str(args.beam_direction),
                ub_convention=str(args.ub_convention),
                angle_reference=str(args.angle_reference),
                include_domega_in_lattice=bool(args.include_domega_in_lattice),
                omega_map_mode=str(args.omega_map_mode),
                omega_sign=float(args.omega_sign),
                omega_offset_deg=float(args.omega_offset_deg),
                invert_rotation=bool(args.invert_rotation),
                swap_xy=bool(args.swap_xy),
                flip_x=bool(args.flip_x),
                flip_y=bool(args.flip_y),
            )
            csv_path = args.output_dir / f"predicted_spots_frame_{frame.frame_number:04d}.csv"
            png_path = args.output_dir / f"detector_frame_{frame.frame_number:04d}.png"
            spots.to_csv(csv_path, index=False)
            image = _try_load_image(args.images_dir, frame.imgname)
            cx_map, cy_map = _map_detector_point(
                x=float(frame.xcenter),
                y=float(frame.ycenter),
                detector_nx=detector_nx,
                detector_ny=detector_ny,
                swap_xy=bool(args.swap_xy),
                flip_x=bool(args.flip_x),
                flip_y=bool(args.flip_y),
            )
            center_rows.append(
                {
                    "frame_number": int(frame.frame_number),
                    "imgname": str(Path(frame.imgname).name),
                    "center_raw_x": float(frame.xcenter),
                    "center_raw_y": float(frame.ycenter),
                    "center_plot_x": float(cx_map),
                    "center_plot_y": float(cy_map),
                }
            )
            plot_frame(
                frame=frame,
                spots=spots,
                detector_nx=detector_nx,
                detector_ny=detector_ny,
                output_png=png_path,
                image=image,
                center_x_px=cx_map,
                center_y_px=cy_map,
            )
            print(
                f"Frame {frame.frame_number:4d}: spots={len(spots):5d}, "
                f"csv={csv_path.name}, png={png_path.name}"
            )
            print(
                f"  centers: raw=({float(frame.xcenter):.2f}, {float(frame.ycenter):.2f}) "
                f"plotted=({float(cx_map):.2f}, {float(cy_map):.2f})"
            )
        if center_rows:
            centers_df = pd.DataFrame.from_records(center_rows)
            centers_path = args.output_dir / "centers_used.csv"
            centers_df.to_csv(centers_path, index=False)
            print(f"Wrote centers table: {centers_path.name}")


if __name__ == "__main__":
    main()

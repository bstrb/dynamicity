"""PETS2 geometry bridge—integrates proven detector spot prediction from pets2_geometry_frame_plotter.

This module converts PETS2 project outputs into core analysis inputs:
- GXPARMData-like geometry container
- IntegrateData observations table  
- frame-indexed reciprocal matrices for orientation modeling
- proven point-cloud-based detector spot prediction (no intermediate rasterization)
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
import re
import shlex
from typing import Sequence

import numpy as np
import pandas as pd

from .parsers import GXPARMData, IntegrateData, UnitCell

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


@dataclass(frozen=True)
class PETSFrameState:
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
class PETSModel:
    pts2_path: Path
    ptsopt_path: Path | None
    ptsoptlist_path: Path | None
    rprofall_path: Path
    wavelength_angstrom: float
    aperpixel_invA_per_px: float
    omega_deg: float
    delta_deg: float
    ub_matrix: np.ndarray
    unit_cell: UnitCell
    frames: tuple[PETSFrameState, ...]
    detector_nx: int
    detector_ny: int
    orgx_px: float
    orgy_px: float


@dataclass(frozen=True)
class PETSAlignment:
    reindex_matrix: np.ndarray
    rotation_matrix: np.ndarray
    residual: float


def _signed_permutation_matrices() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=float)
        for row, col in enumerate(perm):
            base[row, col] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mat = base * np.asarray(signs, dtype=float)[:, None]
            matrices.append(mat)
    return matrices


def _closest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def estimate_pets_alignment(
    pets_reciprocal_ref: np.ndarray,
    xds_reciprocal_ref: np.ndarray,
) -> PETSAlignment:
    """Estimate signed-permutation reindex and lab rotation aligning PETS to XDS."""

    pets_ref = np.asarray(pets_reciprocal_ref, dtype=float)
    xds_ref = np.asarray(xds_reciprocal_ref, dtype=float)
    best_residual = np.inf
    best_reindex = None
    best_rotation = None

    for reindex in _signed_permutation_matrices():
        target = xds_ref @ reindex
        rotation = _closest_rotation(pets_ref @ np.linalg.inv(target))
        residual = float(np.linalg.norm(pets_ref - rotation @ target) / np.linalg.norm(pets_ref))
        if residual < best_residual:
            best_residual = residual
            best_reindex = reindex
            best_rotation = rotation

    if best_reindex is None or best_rotation is None:
        raise ValueError("Failed to estimate PETS/XDS alignment.")

    return PETSAlignment(
        reindex_matrix=best_reindex,
        rotation_matrix=best_rotation,
        residual=best_residual,
    )


def _clean_lines(text: str) -> list[str]:
    return [line.rstrip("\n") for line in text.splitlines()]


def _parse_key_scalar(lines: Sequence[str], key: str) -> float | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*(?:=|\s)\s*({FLOAT_RE})", flags=re.IGNORECASE)
    for line in lines:
        match = pattern.match(line.strip())
        if match is None:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return None


def _parse_center(lines: Sequence[str]) -> tuple[float | None, float | None]:
    pattern = re.compile(rf"^\s*center\s+({FLOAT_RE}|AUTO)\s+({FLOAT_RE}|AUTO)", flags=re.IGNORECASE)
    for line in lines:
        match = pattern.match(line.strip())
        if match is None:
            continue
        x_raw = match.group(1)
        y_raw = match.group(2)
        if x_raw.upper() == "AUTO" or y_raw.upper() == "AUTO":
            return None, None
        try:
            return float(x_raw), float(y_raw)
        except ValueError:
            continue
    return None, None


def _parse_badpixel_max(lines: Sequence[str]) -> tuple[int | None, int | None]:
    nx = None
    ny = None
    x_re = re.compile(r"^\s*badpixelx\s*=\s*\d+\s+\d+\s+\d+\s+(\d+)", flags=re.IGNORECASE)
    y_re = re.compile(r"^\s*badpixely\s*=\s*\d+\s+\d+\s+\d+\s+(\d+)", flags=re.IGNORECASE)
    for line in lines:
        if nx is None:
            mx = x_re.match(line)
            if mx is not None:
                nx = int(mx.group(1))
        if ny is None:
            my = y_re.match(line)
            if my is not None:
                ny = int(my.group(1))
    return nx, ny


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
    pattern = re.compile(
        rf"^\s*cell\s+({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})",
        flags=re.IGNORECASE,
    )
    for line in lines:
        match = pattern.match(line.strip())
        if match is None:
            continue
        return UnitCell(
            a=float(match.group(1)),
            b=float(match.group(2)),
            c=float(match.group(3)),
            alpha=float(match.group(4)),
            beta=float(match.group(5)),
            gamma=float(match.group(6)),
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
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                break
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
            header = [token.strip().lower() for token in stripped.split()[1:]]
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
    table["frame_index"] = table["frame_number"] - 1

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


def _parse_ptsopt(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    lines = _clean_lines(path.read_text(errors="ignore"))
    if not lines:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        raw = stripped.split("|", 1)[0].strip()
        tokens = raw.split()
        if len(tokens) < 6:
            continue
        numeric = tokens[1:]
        try:
            rows.append(
                {
                    "imgname": _normalize_imgname(tokens[0]),
                    "xcenter": float(numeric[0]),
                    "ycenter": float(numeric[1]),
                    "alpha": float(numeric[2]),
                    "beta": float(numeric[3]),
                    "omega": float(numeric[4]),
                }
            )
        except (ValueError, IndexError):
            continue
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame.from_records(rows)
    table["imgbase"] = table["imgname"].map(_basename)
    table["frame_number"] = _frame_numbers_from_imgname(table["imgname"]).astype(int)
    return table


def _parse_ptsoptlist(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    lines = _clean_lines(path.read_text(errors="ignore"))
    rows: list[dict[str, float]] = []
    current_frame: int | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        frame_match = re.search(r"Analysing frame nr\.\s*(\d+)", stripped, flags=re.IGNORECASE)
        if frame_match is not None:
            current_frame = int(frame_match.group(1))
            continue
        if not stripped.startswith("# Best result:"):
            continue
        numbers = re.findall(FLOAT_RE, stripped)
        if current_frame is None or len(numbers) < 5:
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
    return table.drop_duplicates("frame_number", keep="last")


def _merge_frame_table(imagelist: pd.DataFrame, override: pd.DataFrame | None) -> pd.DataFrame:
    table = imagelist.copy()
    if override is None or override.empty:
        return table

    merged = table.copy()
    over = override.copy()
    if "imgname" in merged.columns:
        merged["imgname"] = merged["imgname"].map(_normalize_imgname)
        merged["imgbase"] = merged["imgname"].map(_basename)
    if "imgname" in over.columns:
        over["imgname"] = over["imgname"].map(_normalize_imgname)
        over["imgbase"] = over["imgname"].map(_basename)

    over_by_frame = None
    if "frame_number" in over.columns:
        keyed = over.dropna(subset=["frame_number"]).drop_duplicates("frame_number", keep="last")
        if not keyed.empty:
            over_by_frame = keyed.set_index("frame_number")

    over_by_base = None
    if "imgbase" in over.columns:
        keyed = over.dropna(subset=["imgbase"]).drop_duplicates("imgbase", keep="last")
        if not keyed.empty:
            over_by_base = keyed.set_index("imgbase")

    for col in ("alpha", "beta", "xcenter", "ycenter", "omega"):
        if col not in merged.columns:
            merged[col] = np.nan
        values = merged[col].copy()
        if over_by_frame is not None and col in over_by_frame.columns:
            by_frame = merged["frame_number"].map(over_by_frame[col])
            values = by_frame.where(by_frame.notna(), values)
        if over_by_base is not None and col in over_by_base.columns:
            by_base = merged["imgbase"].map(over_by_base[col])
            values = by_base.where(by_base.notna(), values)
        merged[col] = values

    if "omega" in merged.columns:
        merged["domega"] = merged["omega"].where(merged["omega"].notna(), merged["domega"])
    return merged


def _pts_prefix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for marker in (".pts2.backup", ".pts2", ".ptsopt"):
        idx = lower.find(marker)
        if idx >= 0:
            return name[:idx]
    return path.stem


def _candidate_dirs(project: Path, pts2_path: Path | None) -> list[Path]:
    dirs: list[Path] = []

    def _add(path: Path) -> None:
        if path.exists() and path.is_dir() and path not in dirs:
            dirs.append(path)

    if project.is_dir():
        _add(project)
        for child in sorted(project.glob("*_petsdata")):
            _add(child)
        _add(project.parent)
        for child in sorted(project.parent.glob("*_petsdata")):
            _add(child)
    elif project.is_file():
        _add(project.parent)
        _add(project.parent.parent)
        for child in sorted(project.parent.glob("*_petsdata")):
            _add(child)
        for child in sorted(project.parent.parent.glob("*_petsdata")):
            _add(child)

    if pts2_path is not None:
        _add(pts2_path.parent)
        for child in sorted(pts2_path.parent.glob("*_petsdata")):
            _add(child)
    return dirs


def _pick_file(
    dirs: Sequence[Path],
    prefix: str,
    suffixes: Sequence[str],
) -> Path | None:
    for directory in dirs:
        for suffix in suffixes:
            exact = directory / f"{prefix}{suffix}"
            if exact.exists():
                return exact
    for directory in dirs:
        for suffix in suffixes:
            hits = sorted(directory.glob(f"*{suffix}"))
            if hits:
                return hits[0]
    return None


def _resolve_paths(
    pets_project: str | Path,
    *,
    pts2_path: str | Path | None = None,
    ptsopt_path: str | Path | None = None,
    rprofall_path: str | Path | None = None,
) -> tuple[Path, Path | None, Path | None, Path]:
    project = Path(pets_project)
    if not project.exists():
        raise FileNotFoundError(f"PETS project path does not exist: {project}")

    resolved_pts2: Path | None = Path(pts2_path) if pts2_path is not None else None
    if resolved_pts2 is None:
        if project.is_file():
            lower = project.name.lower()
            if ".pts2" in lower:
                resolved_pts2 = project
            elif ".ptsopt" in lower:
                resolved_pts2 = None
        if resolved_pts2 is None:
            search_dirs = _candidate_dirs(project, None)
            for directory in search_dirs:
                candidates = sorted(directory.glob("*.pts2")) + sorted(directory.glob("*.pts2.backup"))
                if candidates:
                    resolved_pts2 = candidates[0]
                    break
    if resolved_pts2 is None:
        raise FileNotFoundError("Could not locate a PETS .pts2/.pts2.backup file.")

    prefix = _pts_prefix(resolved_pts2)
    dirs = _candidate_dirs(project, resolved_pts2)
    resolved_ptsopt: Path | None = Path(ptsopt_path) if ptsopt_path is not None else _pick_file(
        dirs, prefix, (".ptsopt",)
    )
    ptsoptlist_path = _pick_file(
        dirs,
        prefix,
        (".ptsoptlist", ".ptsoptlist.best.txt"),
    )

    resolved_rprofall = Path(rprofall_path) if rprofall_path is not None else _pick_file(
        dirs, prefix, (".rprofall",)
    )
    if resolved_rprofall is None:
        raise FileNotFoundError(
            "Could not locate PETS .rprofall file. Provide --pets-rprofall explicitly."
        )
    return resolved_pts2, resolved_ptsopt, ptsoptlist_path, resolved_rprofall


# ============================================================================
# Proven geometry functions from pets2_geometry_frame_plotter.py
# ============================================================================


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


def _sg(g_vectors: np.ndarray, wavelength: float, beam_dir: np.ndarray) -> np.ndarray:
    beam_dir = np.asarray(beam_dir, dtype=float)
    beam_dir = beam_dir / np.linalg.norm(beam_dir)
    k0 = beam_dir[None, :] / float(wavelength)
    return np.linalg.norm(g_vectors + k0, axis=1) - (1.0 / float(wavelength))


def _local_alpha_step(frames: Sequence[PETSFrameState], index: int) -> float:
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


def _rotate_detector_plane(u: np.ndarray, v: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    x_rot = c * u - s * v
    y_rot = s * u + c * v
    return x_rot, y_rot


def _frame_rotation(
    model: PETSModel,
    frame: PETSFrameState,
    frames: Sequence[PETSFrameState],
    reference_frame: PETSFrameState,
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


def load_pets_model(
    pets_root: str | Path,
    pts2_path: str | Path | None = None,
    ptsopt_path: str | Path | None = None,
    rprofall_path: str | Path | None = None,
    detector_nx: int | None = None,
    detector_ny: int | None = None,
) -> PETSModel:
    """Load PETS model from project directory."""
    pets_root = Path(pets_root)
    resolved_pts2, resolved_ptsopt, resolved_ptsoptlist, resolved_rprofall = _resolve_paths(
        pets_root,
        pts2_path=pts2_path,
        ptsopt_path=ptsopt_path,
        rprofall_path=rprofall_path,
    )

    lines = _clean_lines(resolved_pts2.read_text(errors="ignore"))
    wavelength = _parse_key_scalar(lines, "lambda")
    aperpixel = _parse_key_scalar(lines, "aperpixel")
    omega = _parse_key_scalar(lines, "omega")
    delta = _parse_key_scalar(lines, "delta")
    if wavelength is None or aperpixel is None or omega is None:
        raise ValueError("Missing required `lambda`/`aperpixel`/`omega` in .pts2.")
    if delta is None:
        delta = 0.0
    center_x, center_y = _parse_center(lines)

    ub_matrix = _parse_ubmatrix(lines)
    unit_cell = _parse_cell(lines)
    imagelist = _parse_imagelist(lines)
    ptsopt = _parse_ptsopt(resolved_ptsopt) if resolved_ptsopt is not None else None
    ptsoptlist = _parse_ptsoptlist(resolved_ptsoptlist) if resolved_ptsoptlist is not None else None
    merged = _merge_frame_table(imagelist=imagelist, override=ptsopt)
    merged = _merge_frame_table(imagelist=merged, override=ptsoptlist)

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
        PETSFrameState(
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

    nx = int(detector_nx or 512)
    ny = int(detector_ny or 512)
    det_nx_from_pts, det_ny_from_pts = _parse_badpixel_max(lines)
    if det_nx_from_pts is not None and detector_nx is None:
        nx = int(det_nx_from_pts) + 1
    if det_ny_from_pts is not None and detector_ny is None:
        ny = int(det_ny_from_pts) + 1

    return PETSModel(
        pts2_path=resolved_pts2,
        ptsopt_path=resolved_ptsopt,
        ptsoptlist_path=resolved_ptsoptlist,
        rprofall_path=resolved_rprofall,
        wavelength_angstrom=float(wavelength),
        aperpixel_invA_per_px=float(aperpixel),
        omega_deg=float(omega),
        delta_deg=float(delta),
        ub_matrix=ub_matrix,
        unit_cell=unit_cell,
        frames=frames,
        detector_nx=nx,
        detector_ny=ny,
        orgx_px=float(center_x) if center_x is not None else x_median,
        orgy_px=float(center_y) if center_y is not None else y_median,
    )


def pets_model_to_analysis_inputs(
    pets_model: PETSModel,
    ub_convention: str = "columns",
    orientation_mode: str = "pets_ab_xy",
    angle_reference: str = "absolute",
    include_domega_in_lattice: bool = False,
    invert_rotation: bool = False,
    use_only_for_calc: bool = False,
    hkl_path: str | Path | None = None,
    alignment_rotation: np.ndarray | None = None,
    reindex_matrix: np.ndarray | None = None,
) -> tuple[GXPARMData, IntegrateData, dict[int, np.ndarray], pd.DataFrame]:
    """Convert PETSModel to GXPARMData + IntegrateData + per-frame reciprocal matrices + frame geometry.
    
    Parameters orientation_mode, angle_reference, etc. are stored for consistency but don't affect
    the basic conversion (they're for the caller's orientation model setup).
    
    Parameters:
        hkl_path: Optional path to a SHELX .hkl file (merged reflections) to use as observations.
    """

    # Build minimal GXPARMData stub from PETS model
    # Reciprocal reference is the PETS UB matrix (g_vectors per [hkl])
    if ub_convention == "rows":
        reciprocal_ref = pets_model.ub_matrix.T
    else:
        reciprocal_ref = pets_model.ub_matrix
    
    # Real-space reference (direct basis from unit cell)
    alpha = np.deg2rad(pets_model.unit_cell.alpha)
    beta = np.deg2rad(pets_model.unit_cell.beta)
    gamma = np.deg2rad(pets_model.unit_cell.gamma)
    sin_gamma = float(np.sin(gamma))
    cos_alpha = float(np.cos(alpha))
    cos_beta = float(np.cos(beta))
    cos_gamma = float(np.cos(gamma))
    
    a_vec = np.asarray([pets_model.unit_cell.a, 0.0, 0.0], dtype=float)
    b_vec = np.asarray([pets_model.unit_cell.b * cos_gamma, pets_model.unit_cell.b * sin_gamma, 0.0], dtype=float)
    c_x = pets_model.unit_cell.c * cos_beta
    c_y = pets_model.unit_cell.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z_sq = max(pets_model.unit_cell.c * pets_model.unit_cell.c - c_x * c_x - c_y * c_y, 0.0)
    c_vec = np.asarray([c_x, c_y, float(np.sqrt(c_z_sq))], dtype=float)
    real_space_ref = np.column_stack([a_vec, b_vec, c_vec])

    ub_base = np.asarray(reciprocal_ref, dtype=float)
    if alignment_rotation is not None:
        rotation = np.asarray(alignment_rotation, dtype=float)
        ub_base = rotation.T @ ub_base
        real_space_ref = rotation.T @ real_space_ref
    if reindex_matrix is not None:
        reindex = np.asarray(reindex_matrix, dtype=float)
        ub_base = ub_base @ np.linalg.inv(reindex)
        real_space_ref = reindex @ real_space_ref

    reciprocal_ref = ub_base

    pixel_x_mm = 1.0
    pixel_y_mm = 1.0
    distance_mm = 100.0
    if pets_model.aperpixel_invA_per_px > 0.0 and pets_model.wavelength_angstrom > 0.0:
        scale = 1.0 / (float(pets_model.aperpixel_invA_per_px) * float(pets_model.wavelength_angstrom))
        distance_mm = float(scale) * pixel_x_mm
    
    gxparm = GXPARMData(
        phi0_deg=0.0,
        dphi_deg=0.0,
        rotation_axis=np.asarray([1.0, 0.0, 0.0], dtype=float),
        wavelength_angstrom=pets_model.wavelength_angstrom,
        space_group=1,
        unit_cell=pets_model.unit_cell,
        real_space_reference=real_space_ref,
        reciprocal_reference=reciprocal_ref,
        detector_nx=int(pets_model.detector_nx),
        detector_ny=int(pets_model.detector_ny),
        pixel_x_mm=pixel_x_mm,
        pixel_y_mm=pixel_y_mm,
        orgx_px=pets_model.orgx_px,
        orgy_px=pets_model.orgy_px,
        distance_mm=distance_mm,
    )

    # Load IntegrateData from optional .hkl file, or empty if not provided
    if hkl_path is not None:
        from .parsers import parse_shelx_hkl
        integrate = parse_shelx_hkl(hkl_path)
        # Override estimated_n_frames to actual frame count since aggregated data applies to all frames
        integrate = IntegrateData(
            observations=integrate.observations,
            estimated_n_frames=len(pets_model.frames),
        )
    else:
        integrate = IntegrateData(
            observations=pd.DataFrame(),
            estimated_n_frames=len(pets_model.frames),
        )

    # Per-frame reciprocal matrices with frame-dependent rotations applied
    # This ensures S_orient varies frame-by-frame based on alpha/beta/domega
    reference_frame = min(pets_model.frames, key=lambda f: f.frame_number)
    ub_base = np.asarray(ub_base, dtype=float)
    
    reciprocal_by_frame: dict[int, np.ndarray] = {}
    for frame in pets_model.frames:
        frame_rotation = _frame_rotation(
            pets_model,
            frame,
            pets_model.frames,
            reference_frame,
            mode=orientation_mode,
            angle_reference=angle_reference,
            include_domega_in_lattice=include_domega_in_lattice,
            invert=invert_rotation,
        )
        # Apply the frame rotation to the UB: UB_frame = UB_base @ rotation.T
        ub_frame = ub_base @ frame_rotation.T
        reciprocal_by_frame[int(frame.frame_index)] = ub_frame

    # Frame geometry table
    frame_geom_rows: list[dict[str, object]] = []
    frames_to_use = pets_model.frames
    if use_only_for_calc:
        frames_to_use = tuple(f for f in pets_model.frames if f.useforcalc > 0.0)
    
    for frame in frames_to_use:
        frame_geom_rows.append(
            {
                "frame": int(frame.frame_index),
                "frame_number": int(frame.frame_number),
                "alpha": float(frame.alpha),
                "beta": float(frame.beta),
                "domega": float(frame.domega),
                "xcenter": float(frame.xcenter),
                "ycenter": float(frame.ycenter),
            }
        )
    pets_frame_geometry = pd.DataFrame.from_records(frame_geom_rows) if frame_geom_rows else pd.DataFrame()

    return gxparm, integrate, reciprocal_by_frame, pets_frame_geometry


def predict_pets_detector_spots(
    model: PETSModel,
    *,
    frame_number: int,
    hkls: np.ndarray,
    ub_convention: str = "columns",
    orientation_mode: str = "pets_ab_xy",
    angle_reference: str = "absolute",
    include_domega_in_lattice: bool = False,
    invert_rotation: bool = False,
    excitation_tolerance: float = 1.5e-3,
    projection: str = "full",
    beam_direction: str = "minus_z",
    omega_map_mode: str = "frame_absolute",
    omega_sign: float = 1.0,
    omega_offset_deg: float = 0.0,
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    detector_nx: int | None = None,
    detector_ny: int | None = None,
    alignment_rotation: np.ndarray | None = None,
    reindex_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """Predict PETS detector positions for one frame using proven geometry-only point-cloud method."""

    if ub_convention not in {"columns", "rows"}:
        raise ValueError("ub_convention must be 'columns' or 'rows'.")
    if projection not in {"full", "paraxial"}:
        raise ValueError("projection must be 'full' or 'paraxial'.")

    frames = list(model.frames)
    frame_by_number = {frame.frame_number: frame for frame in frames}
    frame = frame_by_number.get(int(frame_number))
    if frame is None:
        raise KeyError(f"Frame {frame_number} not found in PETS model.")
    reference = min(frames, key=lambda f: f.frame_number)
    frame_pos = next((idx for idx, f in enumerate(frames) if f.frame_number == frame.frame_number), 0)

    # Build reciprocal lattice vectors
    ub_col = np.asarray(model.ub_matrix, dtype=float)
    if ub_convention == "rows":
        ub_col = ub_col.T
    if alignment_rotation is not None:
        rotation = np.asarray(alignment_rotation, dtype=float)
        ub_col = rotation.T @ ub_col
    if reindex_matrix is not None:
        reindex = np.asarray(reindex_matrix, dtype=float)
        ub_col = ub_col @ np.linalg.inv(reindex)
    g_ref = np.asarray(hkls, dtype=int) @ ub_col.T

    # Frame rotation at mid-point
    rotation_mid = _frame_rotation(
        model,
        frame,
        frames,
        reference,
        mode=orientation_mode,
        angle_reference=angle_reference,
        include_domega_in_lattice=include_domega_in_lattice,
        invert=invert_rotation,
    )
    g_mid = g_ref @ rotation_mid.T
    sg_mid = _sg(g_mid, model.wavelength_angstrom, beam_dir=np.asarray([0.0, 0.0, -1.0 if beam_direction == "minus_z" else 1.0], dtype=float))

    # Excitation criterion: check if reflection crosses Ewald sphere within frame step
    if angle_reference in {"absolute", "zero"}:
        alpha_eff = float(frame.alpha)
        beta_eff = float(frame.beta)
        domega_eff = float(frame.domega)
    else:
        alpha_eff = float(frame.alpha - reference.alpha)
        beta_eff = float(frame.beta - reference.beta)
        domega_eff = float(frame.domega - reference.domega)

    step = _local_alpha_step(frames, frame_pos)
    alpha_start = alpha_eff - 0.5 * step
    alpha_end = alpha_eff + 0.5 * step

    if orientation_mode == "axis_alpha_legacy":
        axis = _rotation_axis_from_omega_delta(model.omega_deg, model.delta_deg)
        rot_start = _rodrigues(axis, alpha_start)
        rot_end = _rodrigues(axis, alpha_end)
    elif orientation_mode == "pets_ab_xy":
        rot_start = _rotation_matrix_y(beta_eff) @ _rotation_matrix_x(alpha_start)
        rot_end = _rotation_matrix_y(beta_eff) @ _rotation_matrix_x(alpha_end)
    elif orientation_mode == "pets_ab_yx":
        rot_start = _rotation_matrix_x(alpha_start) @ _rotation_matrix_y(beta_eff)
        rot_end = _rotation_matrix_x(alpha_end) @ _rotation_matrix_y(beta_eff)
    elif orientation_mode in {"fixed_x_alpha", "euler_yxz", "euler_xyz"}:
        rot_start = _rotation_matrix_x(alpha_start)
        rot_end = _rotation_matrix_x(alpha_end)
    else:
        rot_start = np.asarray(rotation_mid, dtype=float)
        rot_end = np.asarray(rotation_mid, dtype=float)

    if include_domega_in_lattice:
        rz = _rotation_matrix_z(domega_eff)
        rot_start = rot_start @ rz
        rot_end = rot_end @ rz
    if invert_rotation:
        rot_start = rot_start.T
        rot_end = rot_end.T

    g_start = g_ref @ rot_start.T
    g_end = g_ref @ rot_end.T
    sg_start = _sg(g_start, model.wavelength_angstrom, beam_dir=np.asarray([0.0, 0.0, -1.0 if beam_direction == "minus_z" else 1.0], dtype=float))
    sg_end = _sg(g_end, model.wavelength_angstrom, beam_dir=np.asarray([0.0, 0.0, -1.0 if beam_direction == "minus_z" else 1.0], dtype=float))
    excited = ((sg_start * sg_end) <= 0.0) | (np.abs(sg_mid) < float(excitation_tolerance))

    # Detector projection
    beam_dir = np.asarray([0.0, 0.0, -1.0 if beam_direction == "minus_z" else 1.0], dtype=float)
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

    # Omega mapping
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

    omega_total = float(omega_sign) * float(omega_map) + float(omega_offset_deg)
    x_local, y_local = _rotate_detector_plane(u=u, v=v, angle_deg=omega_total)
    x = float(frame.xcenter) + x_local
    y = float(frame.ycenter) + y_local

    nx = int(model.detector_nx if detector_nx is None else detector_nx)
    ny = int(model.detector_ny if detector_ny is None else detector_ny)
    x, y = _apply_xy_mapping(
        x=x,
        y=y,
        detector_nx=nx,
        detector_ny=ny,
        swap_xy=swap_xy,
        flip_x=flip_x,
        flip_y=flip_y,
    )

    in_bounds = (x >= 0.0) & (x < nx) & (y >= 0.0) & (y < ny)
    keep = excited & forward & in_bounds

    if not np.any(keep):
        return pd.DataFrame(columns=["frame", "frame_number", "h", "k", "l", "x_px", "y_px", "sg_invA"])

    keep_hkl = np.asarray(hkls, dtype=int)[keep, :]
    return pd.DataFrame(
        {
            "frame": int(frame.frame_index),
            "frame_number": int(frame.frame_number),
            "h": keep_hkl[:, 0],
            "k": keep_hkl[:, 1],
            "l": keep_hkl[:, 2],
            "x_px": x[keep],
            "y_px": y[keep],
            "sg_invA": sg_mid[keep],
        }
    )


def map_detector_point(
    x: float,
    y: float,
    detector_nx: int,
    detector_ny: int,
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple[float, float]:
    """Map a single detector point through swap/flip transformations."""
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

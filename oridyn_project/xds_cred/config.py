"""Configuration loading and provenance helpers for the XDS/cRED workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class ScoreConfig:
    """Geometry-only OriDyn risk parameters."""

    s0: float = 0.005
    sigma_c: float = 0.030
    r_cut: float = 0.20


@dataclass(frozen=True)
class GeometryAuditConfig:
    """Settings for XPARM/GXPARM convention checks."""

    max_observations: int | None = 2000
    prediction_window_frames: float = 30.0
    max_median_abs_z_residual_frames: float = 10.0
    max_median_abs_excitation_invA: float = 0.02


@dataclass(frozen=True)
class WorkflowConfig:
    """Runtime configuration for the first cRED/XDS OriDyn workflow."""

    dataset_name: str
    input_dir: Path
    output_dir: Path
    score: ScoreConfig
    workers: int | str = "auto"
    chunk_size: int = 1000
    min_symmetry_multiplicity: int = 3
    max_observations: int | None = None
    allow_overwrite: bool = False
    random_seed: int = 0
    use_integrate_lp_batches: bool = True
    zcal_angle_offset_frames: float = 1.0
    orientation_audit_count: int = 12
    strongest_neighbors: int = 10
    geometry_audit: GeometryAuditConfig = GeometryAuditConfig()

    def resolved_worker_count(self) -> int:
        """Return the concrete worker count, using all available cores by default."""

        if isinstance(self.workers, int):
            return max(1, self.workers)
        if str(self.workers).lower() != "auto":
            raise ValueError("workers must be an integer or 'auto'.")
        return max(1, os.cpu_count() or 1)

    def to_resolved_dict(self) -> dict[str, Any]:
        """Return JSON-serializable resolved parameters."""

        payload = asdict(self)
        payload["input_dir"] = str(self.input_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["worker_count_resolved"] = self.resolved_worker_count()
        payload["score_formulas"] = {
            "R_env(g)": "sum_{h != g, d_gh <= r_cut} E(h) * C(g,h)",
            "S_risk(g)": "E(g) * R_env(g)",
            "E(q)": "exp[-(s_q / s0)^2]",
            "C(g,h)": "exp[-(d_gh / sigma_C)^2]",
            "d_gh": "||q_h - q_g||",
            "s_q": "||k0 + q|| - ||k0||, ||k0|| = 1/lambda",
        }
        payload["units"] = {
            "reciprocal_space": "A^-1 without 2*pi",
            "s0": "A^-1",
            "sigma_c": "A^-1",
            "r_cut": "A^-1",
        }
        payload["friedel_policy"] = (
            "Friedel mates are not added by assumption; they are grouped only "
            "when related by the actual space-group rotational operations."
        )
        payload["intensity_eligibility"] = "finite INTEGRATE.HKL IOBS; no I/SIGMA, PEAK, CORR, resolution, or risk cutoff"
        payload["xds_orientation_convention"] = {
            "matrix_reference": "XDS unit-cell axes are unrotated-crystal axes at spindle dial 0 degrees; INTEGRATE.LP prints the batch-refined version used during INTEGRATE.",
            "angle_equation": "phi = STARTING_ANGLE + OSCILLATION_RANGE * (ZCAL - STARTING_FRAME + zcal_angle_offset_frames)",
            "zcal_angle_offset_frames": self.zcal_angle_offset_frames,
            "batch_assignment": "nearest integer image to fractional ZCAL",
        }
        payload["analysis_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        return payload


def load_config(path: Path) -> WorkflowConfig:
    """Load a workflow JSON configuration."""

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    score_raw = raw.get("score", {})
    audit_raw = raw.get("geometry_audit", {})
    return WorkflowConfig(
        dataset_name=str(raw["dataset_name"]),
        input_dir=Path(raw["input_dir"]).expanduser(),
        output_dir=Path(raw["output_dir"]).expanduser(),
        score=ScoreConfig(
            s0=float(score_raw.get("s0", 0.005)),
            sigma_c=float(score_raw.get("sigma_c", 0.030)),
            r_cut=float(score_raw.get("r_cut", 0.20)),
        ),
        workers=raw.get("workers", "auto"),
        chunk_size=int(raw.get("chunk_size", 1000)),
        min_symmetry_multiplicity=int(raw.get("min_symmetry_multiplicity", 3)),
        max_observations=raw.get("max_observations"),
        allow_overwrite=bool(raw.get("allow_overwrite", False)),
        random_seed=int(raw.get("random_seed", 0)),
        use_integrate_lp_batches=bool(raw.get("use_integrate_lp_batches", True)),
        zcal_angle_offset_frames=float(raw.get("zcal_angle_offset_frames", 1.0)),
        orientation_audit_count=int(raw.get("orientation_audit_count", 12)),
        strongest_neighbors=int(raw.get("strongest_neighbors", 10)),
        geometry_audit=GeometryAuditConfig(
            max_observations=audit_raw.get("max_observations", 2000),
            prediction_window_frames=float(audit_raw.get("prediction_window_frames", 30.0)),
            max_median_abs_z_residual_frames=float(
                audit_raw.get("max_median_abs_z_residual_frames", 10.0)
            ),
            max_median_abs_excitation_invA=float(audit_raw.get("max_median_abs_excitation_invA", 0.02)),
        ),
    )


def write_json(path: Path, payload: Any) -> None:
    """Write indented deterministic JSON."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def collect_software_versions(project_root: Path) -> dict[str, Any]:
    """Collect Python, dependency, and git provenance."""

    versions: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "matplotlib", "gemmi", "cctbx"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as exc:  # pragma: no cover - depends on local env
            versions[name] = f"not available: {exc}"
    versions.update(_git_info(project_root))
    return versions


def _git_info(project_root: Path) -> dict[str, Any]:
    """Return git commit and dirty-worktree information if available."""

    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(project_root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "git_commit": commit,
        "git_worktree_dirty": bool(status),
        "git_status_short": status,
    }

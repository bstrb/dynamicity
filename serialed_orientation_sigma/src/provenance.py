from __future__ import annotations

import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_info(project_root: Path) -> dict[str, Any]:
    """Best-effort capture of git state for reproducibility."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "commit": rev.stdout.strip(),
            "branch": branch.stdout.strip(),
            "is_dirty": bool(status.stdout.strip()),
        }
    except Exception:
        return {"commit": None, "branch": None, "is_dirty": None}


def build_run_provenance(
    argv: list[str] | None = None,
    *,
    project_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized provenance payload for run outputs."""
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "argv": list(argv) if argv is not None else list(sys.argv[1:]),
        "cwd": str(Path.cwd().resolve()),
        "project_root": str(root),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "git": _git_info(root),
    }
    if extra:
        payload.update(extra)
    return payload

#!/usr/bin/env python3
"""Wrapper for repo-level compute_sres.py."""

from pathlib import Path
import runpy
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "scripts" / "compute_sres.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing script: {target}")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

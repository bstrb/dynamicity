#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gi_util.py
General utilities:
- parse .geom to get 'res' (resolution) for px<->mm conversion
- px_to_mm helpers
- golden-angle direction generator
- .lst writer
- simple JSONL logger
- indexamajig runner
"""

from __future__ import annotations
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional, Dict, Any


# ---------- Geom parsing and units ----------

_RES_PATTERNS = [
    re.compile(r"^\s*res(?:olution)?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE),
    # add more patterns if your geom uses another key
]

def read_geom_res_mm_per_1000px(geom_path: str) -> float:
    """
    Returns 'res' as used in your code, so that mm_per_px = 1000.0 / res.
    """
    with open(geom_path, "r") as f:
        for line in f:
            for pat in _RES_PATTERNS:
                m = pat.match(line)
                if m:
                    return float(m.group(1))
    raise ValueError(f"Could not find 'res' in geom: {geom_path}")


def mm_per_px(geom_res_value: float) -> float:
    """Convert your geom 'res' to mm/px as you specified: 1000/res."""
    return 1000.0 / float(geom_res_value)


def px_to_mm(dx_px: float, dy_px: float, geom_res_value: float) -> Tuple[float, float]:
    s = mm_per_px(geom_res_value)
    return dx_px * s, dy_px * s


# ---------- Golden-angle directions ----------

def golden_angle_radians() -> float:
    return 137.50776405003785 * 3.141592653589793 / 180.0


def ring_directions(K_dir: int, k0: int = 0) -> List[float]:
    """
    Return K_dir angles (radians) in low-discrepancy order using golden-angle.
    k0 is a starting index to rotate sequence run-to-run if desired.
    """
    GA = golden_angle_radians()
    return [((k0 + k) * GA) % (2.0 * 3.141592653589793) for k in range(K_dir)]


# ---------- .lst helpers ----------

def write_lst(lst_path: str, overlay_path: str, indices: Sequence[int]) -> None:
    with open(lst_path, "w") as f:
        for i in indices:
            f.write(f"{overlay_path} //{int(i)}\n")


# ---------- JSONL logging ----------

def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


# ---------- indexamajig runner ----------

@dataclass
class RunResult:
    stream_path: str
    returncode: int
    elapsed_s: float
    stderr_tail: str


def run_indexamajig(
    lst_path: str,
    geom_path: str,
    cell_path: str,
    out_stream_path: str,
    flags_passthrough: Sequence[str],
) -> RunResult:
    """
    Run indexamajig with user's flags as-is.
    We redirect stdout to the stream file and capture only stderr (tail) for logs.
    """
    cmd = ["indexamajig", "-i", lst_path, "-g", geom_path, "-p", cell_path]
    # pass-through flags verbatim (e.g., "-j", "32", "--peaks", "...", ...)
    cmd.extend(flags_passthrough)

    os.makedirs(os.path.dirname(os.path.abspath(out_stream_path)), exist_ok=True)
    t0 = time.time()
    with open(out_stream_path, "w") as stream_out, subprocess.Popen(
        cmd,
        stdout=stream_out,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        stderr = proc.communicate()[1] or ""
        elapsed = time.time() - t0

    # Keep only last few lines of stderr for quick diagnosis
    tail = "\n".join(stderr.strip().splitlines()[-20:])
    return RunResult(stream_path=out_stream_path, returncode=proc.returncode, elapsed_s=elapsed, stderr_tail=tail)

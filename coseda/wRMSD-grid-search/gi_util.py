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
    lst_path,
    geom_path,
    cell_path,
    out_stream_path,
    flags=None,
    log_jsonl_path=None,
    run_script_path=None,
    stderr_path=None,
):
    import os, shlex, subprocess, sys, time, json

    if flags is None:
        flags = []

    # --- Default flags you requested (user flags appended AFTER these to override) ---
    default_flags = [
        "-j 8",
        "--peaks=peakfinder9",
        "--min-snr-biggest-pix=1",
        "--min-snr-peak-pix=6",
        "--min-snr=1",
        "--min-sig=11",
        "--min-peak-over-neighbour=-inf",
        "--local-bg-radius=3",
        "--min-peaks=15",
        "--tolerance=10,10,10,5",
        "--xgandalf-sampling-pitch=5",
        "--xgandalf-grad-desc-iterations=1",
        "--xgandalf-tolerance=0.02",
        "--int-radius=4,5,9",
        "--no-half-pixel-shift",
        "--no-non-hits-in-stream",
        "--no-retry",
        "--fix-profile-radius=70000000",
        "--indexing=xgandalf",
        "--integration=rings",
    ]
    # Split default flags that contain spaces (e.g., "-j 8")
    split_defaults = []
    for df in default_flags:
        split_defaults.extend(shlex.split(df))

    combined_flags = split_defaults + list(flags)

    # Build the EXACT argv we will execute
    cmd = [
        "indexamajig",
        "-g", geom_path,
        "-i", lst_path,
        "-o", out_stream_path,
        "-p", cell_path,
        *combined_flags,
    ]

    # Ensure dirs
    out_dir = os.path.dirname(os.path.abspath(out_stream_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if stderr_path:
        os.makedirs(os.path.dirname(os.path.abspath(stderr_path)), exist_ok=True)
    if run_script_path:
        os.makedirs(os.path.dirname(os.path.abspath(run_script_path)), exist_ok=True)

    # Write .sh BEFORE running; content mirrors `cmd`
    if run_script_path:
        pretty = [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            "# Readable multi-line form (defaults first, then any extra flags)",
            "indexamajig \\",
            f"  -g {shlex.quote(geom_path)} \\",
            f"  -i {shlex.quote(lst_path)} \\",
            f"  -o {shlex.quote(out_stream_path)} \\",
            f"  -p {shlex.quote(cell_path)} \\",
        ]
        # Defaults as separate lines
        for df in default_flags:
            pretty.append(f"  {shlex.quote(df)} \\")
        # Extra flags (if any)
        if flags:
            for k, f in enumerate(flags):
                cont = " \\" if (k < len(flags) - 1) else ""
                pretty.append(f"  {shlex.quote(str(f))}{cont}")
        else:
            # remove trailing backslash from the last defaults line
            if pretty[-1].endswith(" \\"):
                pretty[-1] = pretty[-1][:-2]

        pretty += [
            "",
            "# Exact one-liner actually executed:",
            "# " + " ".join(shlex.quote(c) for c in cmd),
            ""
        ]
        with open(run_script_path, "w") as shf:
            shf.write("\n".join(pretty))
        os.chmod(run_script_path, 0o755)

    # Execute and (optionally) tee stderr to file; stdout/stderr inherited for GUI
    start_ts = time.time()
    if stderr_path:
        rc = 0
        with open(stderr_path, "wb") as ef:
            proc = subprocess.Popen(cmd, stdout=None, stderr=subprocess.PIPE)
            for chunk in iter(lambda: proc.stderr.readline(), b""):
                if not chunk:
                    break
                ef.write(chunk)
                try:
                    sys.stderr.buffer.write(chunk)
                except Exception:
                    sys.stderr.write(chunk.decode(errors="ignore"))
            proc.wait()
            rc = proc.returncode
    else:
        rc = subprocess.call(cmd)
    end_ts = time.time()

    # Log exactly what we executed (argv)
    if log_jsonl_path:
        try:
            with open(log_jsonl_path, "a") as f:
                f.write(json.dumps({
                    "ts_start": start_ts,
                    "ts_end": end_ts,
                    "type": "indexamajig_exec",
                    "exec_cmd": cmd,
                    "stream_out": os.path.abspath(out_stream_path),
                    "sh": run_script_path,
                    "stderr": stderr_path,
                    "returncode": rc,
                }, separators=(",", ":")) + "\n")
        except Exception:
            pass

    return rc

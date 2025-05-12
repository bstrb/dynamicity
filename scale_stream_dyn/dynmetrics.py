#!/usr/bin/env python3
"""
dynmetrics.py – statistics used by the dynamical scaler
=======================================================

Pattern-level:
  • k_p (core-shell log-median scale)
  • R_sysAbs, R_Friedel, P90(log-spread) quality metrics

Reflection-level:
  • shell-based z-score → FLAG_DYN_OUTLIER
"""
from __future__ import annotations
import math, warnings
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

from dynlib import (Chunk, Reflection, FLAG_DYN_OUTLIER,
                    d_spacing, s_in_Ainv, is_forbidden)

###############################################################################
# 1.  Per-pattern scale k_p   (mid-resolution log-median)                     #
###############################################################################
def pattern_scale(chunks: List[Chunk],
                  cell_params, s_min=0.7, s_max=3.0):
    """
    Compute k_p by matching the median(log I) inside 0.7 < s < 3.0 Å⁻¹ shell
    to the global median of that shell.
    """
    a,b,c,al,be,ga = cell_params
    # collect log-intensities of all patterns in the core shell
    core_logs = []
    for ch in chunks:
        logs = []
        for r in ch.reflections:
            d = d_spacing(r.h,r.k,r.l,a,b,c,al,be,ga)
            s = s_in_Ainv(d)
            if s_min < s < s_max and r.I>0:
                logs.append(math.log(r.I))
        if logs:
            ch._core_log_median = np.median(logs)     # stash
            core_logs.extend(logs)
        else:
            ch._core_log_median = None
    global_median = np.median(core_logs)
    # scale
    for ch in chunks:
        if ch._core_log_median is None:
            ch.scale = 1.0
            continue
        ch.scale = math.exp(global_median - ch._core_log_median)

###############################################################################
# 2.  Pattern-quality metrics                                                 #
###############################################################################
def pattern_metrics(chunks: List[Chunk], cell_params, sg_symbol:str|None):
    a,b,c,al,be,ga = cell_params
    for ch in chunks:
        # prepare dicts for fast lookup
        refl_by_index = {(r.h,r.k,r.l):r.I*ch.scale for r in ch.reflections}
        sum_forb = sum_allow = 0.0
        sum_fd_diff = sum_fd_avg = 0.0
        log_vals = []
        for r in ch.reflections:
            I = r.I * ch.scale
            if I<=0: continue
            # sys absences
            if sg_symbol and is_forbidden(r.h,r.k,r.l,sg_symbol):
                sum_forb += I
            else:
                sum_allow += I
            # Friedel mate
            mate = refl_by_index.get((-r.h,-r.k,-r.l))
            if mate:
                sum_fd_diff += abs(I - mate)
                sum_fd_avg  += 0.5*(I+mate)
            # log spread
            log_vals.append(math.log(I))
        # metrics
        ch.R_sysAbs   = (sum_forb / sum_allow) if sum_allow else 0.0
        ch.R_Friedel  = (sum_fd_diff/ sum_fd_avg) if sum_fd_avg else 0.0
        if log_vals:
            log_vals = np.array(log_vals)
            ch.p90_log_spread = np.percentile(
                np.abs(log_vals - np.median(log_vals)), 90)
        else:
            ch.p90_log_spread = 0.0

###############################################################################
# 3.  Pattern selection based on MAD                                          #
###############################################################################
def select_good_patterns(chunks: List[Chunk], n_sigma=2.0):
    """Flag chunks as good/bad based on the three metrics."""
    # stack
    arr = np.array([[ch.R_sysAbs, ch.R_Friedel, ch.p90_log_spread]
                    for ch in chunks])
    med  = np.median(arr, axis=0)
    mad  = np.median(np.abs(arr - med), axis=0) + 1e-12
    for ch, row in zip(chunks, arr):
        ch.good = np.all(np.abs(row - med) < n_sigma*mad)

###############################################################################
# 4.  Shell-wise Wilson mean (clean pool)                                     #
###############################################################################
def shell_means(chunks: List[Chunk],
                cell_params, s_bin=0.1, max_s=4.5) -> Tuple[np.ndarray,np.ndarray]:
    """Return (s_centers, mean_I)  using only good patterns."""
    a,b,c,al,be,ga = cell_params
    nbins = int(max_s / s_bin) + 1
    sums  = np.zeros(nbins);   counts = np.zeros(nbins, dtype=int)
    for ch in chunks:
        if not ch.good: continue
        for r in ch.reflections:
            I = r.I * ch.scale
            if I<=0: continue
            d = d_spacing(r.h,r.k,r.l,a,b,c,al,be,ga)
            s = s_in_Ainv(d)
            idx = int(s / s_bin)
            if idx >= nbins:  continue
            sums[idx]  += I
            counts[idx]+= 1
    s_centers = np.arange(nbins)*s_bin + 0.5*s_bin
    # mean_I    = np.where(counts>0, sums/counts, np.nan)
    # return s_centers, mean_I
    # avoid division by zero:
    mean_I = np.full(nbins, np.nan, dtype=float)
    mask   = counts > 0
    mean_I[mask] = sums[mask] / counts[mask]
    return s_centers, mean_I

###############################################################################
# 5.  Reflection-level outlier flag                                           #
###############################################################################
def flag_dyn_outliers(chunks: List[Chunk],
                      cell_params,
                      s_centers:np.ndarray, mean_I:np.ndarray,
                      s_bin=0.1, z_cut=3.0):
    """Add FLAG_DYN_OUTLIER on reflections that deviate strongly in log(I)."""
    a,b,c,al,be,ga = cell_params
    nbins = len(s_centers)
    # pre-compute per-shell MAD over good observations
    shell_logs = [ [] for _ in range(nbins) ]
    for ch in chunks:
        if not ch.good: continue
        for r in ch.reflections:
            I = r.I * ch.scale
            if I<=0: continue
            d  = d_spacing(r.h,r.k,r.l,a,b,c,al,be,ga)
            idx = int(s_in_Ainv(d)/s_bin)
            if idx>=nbins: continue
            shell_logs[idx].append(math.log(I))
    shell_mad = np.array([np.median(np.abs(np.array(v)-np.median(v))) if v else 1e6
                          for v in shell_logs])

    # flag outliers (all patterns, not only good)
    for ch in chunks:
        for i,r in enumerate(ch.reflections):
            I = r.I * ch.scale
            if I<=0: continue
            d  = d_spacing(r.h,r.k,r.l,a,b,c,al,be,ga)
            idx = int(s_in_Ainv(d)/s_bin)
            if idx>=nbins or np.isnan(mean_I[idx]): continue
            dev = abs(math.log(I) - math.log(mean_I[idx])) / shell_mad[idx]
            if dev > z_cut:
                ch.reflections[i] = r._replace(flag = r.flag|FLAG_DYN_OUTLIER)

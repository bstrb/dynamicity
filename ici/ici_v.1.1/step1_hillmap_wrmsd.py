#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_hillmap_wrmsd.py
Variant of step1_hillmap that scales Gaussian "hill" amplitudes
for successful trials inversely with their normalized wRMSD values.
Lower wRMSD → higher hill amplitude → higher local sampling probability.
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------
class Trial:
    __slots__ = ("x_mm", "y_mm", "indexed", "wrmsd")

    def __init__(self, x_mm: float, y_mm: float, indexed: int, wrmsd: Optional[float]):
        self.x_mm = float(x_mm)
        self.y_mm = float(y_mm)
        self.indexed = int(indexed)
        self.wrmsd = None if wrmsd is None else float(wrmsd)


class Step1Params:
    __slots__ = (
        "radius_mm", "rng_seed", "n_candidates",
        "A0", "hill_amp_frac", "drop_amp_frac",
        "explore_floor", "min_spacing_mm",
        "first_attempt_center_mm", "allow_spacing_relax"
    )

    def __init__(
        self,
        radius_mm: float,
        rng_seed: int,
        n_candidates: int,
        A0: float,
        hill_amp_frac: float,
        drop_amp_frac: float,
        explore_floor: float,
        min_spacing_mm: float,
        first_attempt_center_mm: Tuple[float, float],
        allow_spacing_relax: bool = True,
    ):
        self.radius_mm = float(radius_mm)
        self.rng_seed = int(rng_seed)
        self.n_candidates = int(n_candidates)
        self.A0 = float(A0)
        self.hill_amp_frac = float(hill_amp_frac)
        self.drop_amp_frac = float(drop_amp_frac)
        self.explore_floor = float(explore_floor)
        self.min_spacing_mm = float(min_spacing_mm)
        self.first_attempt_center_mm = (
            float(first_attempt_center_mm[0]), float(first_attempt_center_mm[1])
        )
        self.allow_spacing_relax = bool(allow_spacing_relax)


class Step1Result:
    __slots__ = ("done", "proposal_xy_mm", "reason")

    def __init__(self, done: bool, proposal_xy_mm: Optional[Tuple[float, float]], reason: str):
        self.done = bool(done)
        self.proposal_xy_mm = (
            None if proposal_xy_mm is None else (float(proposal_xy_mm[0]), float(proposal_xy_mm[1]))
        )
        self.reason = reason


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _gauss2d(x, y, cx, cy, sigma):
    dx = x - cx
    dy = y - cy
    return math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))


def _sample_uniform_disk(n, R, rng: random.Random):
    theta = np.array([rng.uniform(0.0, 2 * math.pi) for _ in range(n)], dtype=np.float64)
    r = np.array([R * math.sqrt(rng.random()) for _ in range(n)], dtype=np.float64)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def _filter_min_spacing(cands_xy, tried_xy, min_spacing):
    if tried_xy.size == 0:
        return np.arange(cands_xy.shape[0])
    diffs = cands_xy[:, None, :] - tried_xy[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    ok = np.all(d2 >= (min_spacing * min_spacing), axis=1)
    return np.where(ok)[0]


# ---------------------------------------------------------------------
# Main hillmap with wRMSD-weighted probability
# ---------------------------------------------------------------------
def propose_step1(trials: List[Trial], params: Step1Params, beta = 10.0) -> Step1Result:
    rng = random.Random(params.rng_seed)
    R = params.radius_mm
    sigma = R / 2.0  # 2σ = R
    A0 = params.A0
    A_hill = params.hill_amp_frac * A0
    A_drop = -params.drop_amp_frac * A0
    
    tried_xy = np.array([[t.x_mm, t.y_mm] for t in trials], dtype=np.float64) if trials else np.empty((0, 2), float)

    # successes/failures based only on wrmsd
    successes = [(t.x_mm, t.y_mm, t.wrmsd) for t in trials if t.wrmsd is not None]
    failures = [(t.x_mm, t.y_mm) for t in trials if t.wrmsd is None]

    print("================================")
    for t in trials:
        print(f"(dx,dy) = ({t.x_mm},{t.y_mm}), indexed = {t.indexed}, wrmsd = {t.wrmsd}")


    # Sample candidate points uniformly in disk
    cand_xy = _sample_uniform_disk(params.n_candidates, R, rng)
    keep = _filter_min_spacing(cand_xy, tried_xy, params.min_spacing_mm)
    if keep.size == 0:
        return Step1Result(True, None, "step1_done_exhausted_no_candidates")

    cand_xy = cand_xy[keep, :]

    # Reference center
    c0x, c0y = params.first_attempt_center_mm

    # Base Gaussian field
    w = np.zeros((cand_xy.shape[0],), float)

    n_succ = len(successes)
    if n_succ > 0:
        A0 = A0 / n_succ   # gradually suppress base Gaussian as evidence accumulates
        # A0 = A0 / n_succ**(1/3)    # gradually suppress base Gaussian as evidence accumulates

    g0 = np.array([_gauss2d(x, y, c0x, c0y, sigma) for x, y in cand_xy], float)
    w += A0 * g0

    # wRMSD-based weighting for successful trials (Boltzmann-style)
    if successes:
        wr_vals = np.array([wr for (_, _, wr) in successes if wr is not None], float)
        wmin = float(np.min(wr_vals))
        # exponential weighting relative to the current minimum
        scores = np.exp(-beta * (wr_vals - wmin))
        scores /= np.sum(scores)  # normalize so total contribution stays balanced

        for (cx, cy, wr), score in zip(successes, scores):
            g = np.array([_gauss2d(x, y, cx, cy, sigma) for x, y in cand_xy], float)
            w += A_hill * score * g

    # Penalize failed attempts
    if failures:
        for cx, cy in failures:
            g = np.array([_gauss2d(x, y, cx, cy, sigma) for x, y in cand_xy], float)
            w += A_drop * g
    # Ensure positivity and normalize
    w = np.maximum(0.0, w) + params.explore_floor
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        return Step1Result(True, None, "step1_done_degenerate_weights")

    p = w / s

    # --------------------------------------------------------------
    # NEW: HARD REJECTION OF ANY PREVIOUSLY TRIED CENTER
    # This prevents duplicate (dx,dy) proposals.
    tried_set = {(float(t.x_mm), float(t.y_mm)) for t in trials}

    # Convert candidate positions from local frame to absolute frame
    abs_cand_xy = np.column_stack([
        c0x + cand_xy[:, 0],
        c0y + cand_xy[:, 1]
    ])

    # Mask out any candidates that match a previously tried center
    unique_mask = np.array([
        (cx, cy) not in tried_set
        for cx, cy in abs_cand_xy
    ], dtype=bool)

    if not np.any(unique_mask):
        return Step1Result(True, None, "step1_done_no_unique_candidates")

    # Renormalize over remaining unique candidates
    cand_xy = cand_xy[unique_mask]
    p = p[unique_mask]
    p = p / np.sum(p)
    # --------------------------------------------------------------

    np_rng = np.random.default_rng(params.rng_seed ^ 0xA53E12B4)
    idx = int(np_rng.choice(np.arange(cand_xy.shape[0]), p=p))
    x_mm, y_mm = float(cand_xy[idx, 0]), float(cand_xy[idx, 1])
    return Step1Result(False, (c0x + x_mm, c0y + y_mm), "step1_hillmap_wrmsd_sample")

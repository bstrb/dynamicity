#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gandalfiterator.py
Adaptive per-image center-shift optimization (seed -> ring -> refine) with
wave-based batching via .lst files, overlay HDF5s, and indexamajig.

This module implements:
- Per-image state machine
- Candidate selection (one per active image per wave)
- Overlay writes (absolute shifts in mm for listed images)
- .lst + run_meta.json construction
- (Part 2) Indexamajig execution, stream parsing/scoring, state updates
- (Part 2) Winner chunk extraction, merged stream writing
- (Part 2) gandalf_adaptive() entrypoint callable from runner/GUI

Dependencies: overlay_h5.py, gi_util.py, stream_scoring.py, stream_extract.py
"""

from __future__ import annotations
import os
import json
import time
import math
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field

from overlay_h5 import (
    create_overlay,
    write_shifts_mm,
    get_seed_shifts_mm,
)
from gi_util import (
    read_geom_res_mm_per_1000px,
    mm_per_px,
    px_to_mm,
    ring_directions,
    write_lst,
    jsonl_append,
    run_indexamajig,
    RunResult,
)
from stream_scoring import score_single_chunk_stream_file
from stream_extract import (
    find_chunk_span_for_image,
    extract_winner_stream,
    merge_winner_streams,
)


# ------------------------- Parameters / Defaults -------------------------

@dataclass
class Params:
    R_px: float = 1.0
    s_init_px: float = 0.2
    K_dir: int = 10
    s_refine_px: float = 0.5
    s_min_px: float = 0.1
    eps_rel: float = 0.007        # 0.7%
    N_eval_max: int = 16
    tie_tol_rel: float = 0.01     # 1%
    # behavior knobs
    eight_connected: bool = True  # include diagonals in refinement
    directional_refine: bool = True  # prioritize along descent direction once 2 successes exist


# ------------------------- State Machine Types -------------------------

class ImgState:
    INIT = "INIT"
    SEED_TRY = "SEED_TRY"
    RING_SEARCH = "RING_SEARCH"
    REFINE = "REFINE"
    FINAL = "FINAL"
    UNINDEXED = "UNINDEXED"


@dataclass
class BestPoint:
    dx_px: float
    dy_px: float
    wrmsd: float
    n_reflections: Optional[int] = None
    n_peaks: Optional[int] = None
    cell_dev_pct: Optional[float] = None

    def as_tuple(self):
        return (self.dx_px, self.dy_px, self.wrmsd)


@dataclass
class Candidate:
    h5_src: str
    overlay: str
    image_idx: int
    purpose: str            # "seed" | "ring" | "refine"
    delta_px: Tuple[float, float]  # relative to seed (dx, dy)
    abs_mm: Tuple[float, float]    # absolute shift to write (mm)
    seed_mm: Tuple[float, float]


@dataclass
class ImageController:
    h5_src: str
    overlay: str
    image_idx: int
    geom_res_value: float          # 'res' from .geom
    seed_mm: Tuple[float, float]   # absolute seed shift (mm)
    params: Params

    # dynamic state
    state: str = ImgState.INIT
    best: Optional[BestPoint] = None
    eval_count: int = 0
    ring_r_px: float = 0.0
    refine_s_px: float = 0.0
    have_two_successes: bool = False
    # vector from worse->better (for directional prioritization)
    descent_vec_px: Optional[Tuple[float, float]] = None
    # bookkeeping for refinement sweep
    _sweep_tried: int = 0

    # results
    winner_stream_path: Optional[str] = None

    def activate(self):
        """Initialize to SEED_TRY and set starting radii/step."""
        self.state = ImgState.SEED_TRY
        self.ring_r_px = self.params.s_init_px
        self.refine_s_px = self.params.s_refine_px
        self.eval_count = 0
        self.have_two_successes = False
        self.descent_vec_px = None
        self._sweep_tried = 0

    def mm_per_px(self) -> float:
        return mm_per_px(self.geom_res_value)

    # ---------------- Candidate Proposal ----------------

    def next_candidate(self) -> Optional[Candidate]:
        """
        Propose exactly one candidate for current state, or None if no candidate
        (e.g., finished or unindexed).
        """
        if self.state in (ImgState.FINAL, ImgState.UNINDEXED):
            return None

        if self.eval_count >= self.params.N_eval_max and self.state != ImgState.SEED_TRY:
            # if we haven't even tried seed, allow that; otherwise stop
            self.state = ImgState.FINAL if self.best else ImgState.UNINDEXED
            return None

        if self.state == ImgState.INIT:
            self.activate()

        if self.state == ImgState.SEED_TRY:
            dpx = (0.0, 0.0)
            abs_mm = (self.seed_mm[0], self.seed_mm[1])
            return Candidate(self.h5_src, self.overlay, self.image_idx, "seed", dpx, abs_mm, self.seed_mm)

        if self.state == ImgState.RING_SEARCH:
            cand = self._next_ring_candidate()
            if cand is None:
                # exhausted ring -> UNINDEXED
                self.state = ImgState.UNINDEXED
                return None
            return cand

        if self.state == ImgState.REFINE:
            cand = self._next_refine_candidate()
            if cand is None:
                # halved down below s_min OR no neighbors -> finalize
                self.state = ImgState.FINAL if self.best else ImgState.UNINDEXED
                return None
            return cand

        return None

    def _next_ring_candidate(self) -> Optional[Candidate]:
        """
        Choose the next angular direction at current radius ring_r_px.
        We sample K_dir directions per ring using golden-angle order.
        After proposing K_dir at this radius, move to next radius.
        """
        r = self.ring_r_px
        if r > self.params.R_px + 1e-12:
            return None

        # Determine which angular we are on within the radius
        k = self.eval_count  # local counter; OK to use eval_count as coarse index
        # Derive position within a ring: modulo K_dir
        idx_in_ring = k % self.params.K_dir
        if idx_in_ring == 0 and k > 0:
            # completed a ring -> advance r
            # Note: we advance r only when we return to idx 0 (start of a new ring)
            self.ring_r_px += self.params.s_init_px
            r = self.ring_r_px
            if r > self.params.R_px + 1e-12:
                return None

        # Determine the angle for this index within the ring
        angles = ring_directions(self.params.K_dir, k0=0)
        ang = angles[idx_in_ring]
        dx = r * math.cos(ang)
        dy = r * math.sin(ang)
        dpx = (dx, dy)
        abs_mm = self._abs_mm_from_delta_px(dpx)
        return Candidate(self.h5_src, self.overlay, self.image_idx, "ring", dpx, abs_mm, self.seed_mm)

    def _neighbor_set(self, s: float) -> List[Tuple[float, float]]:
        """Return neighbor offsets around best (dx,dy) at step s (8-connected by default)."""
        moves = [( s, 0.0), (-s, 0.0), (0.0,  s), (0.0, -s)]
        if self.params.eight_connected:
            moves += [( s,  s), ( s, -s), (-s,  s), (-s, -s)]
        return moves

    def _order_neighbors_directional(self, neighbors: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Prioritize neighbors aligning with descent_vec_px (positive dot product).
        If no descent vector yet, return neighbors as-is.
        """
        if not self.params.directional_refine or not self.have_two_successes or self.descent_vec_px is None:
            return neighbors
        vx, vy = self.descent_vec_px
        scored = []
        for dx, dy in neighbors:
            dot = dx * vx + dy * vy
            scored.append((dot, (dx, dy)))
        # Sort descending by dot product (prioritize positive alignment)
        scored.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in scored]

    def _next_refine_candidate(self) -> Optional[Candidate]:
        """
        Propose next neighbor around current best using step s; if all neighbors
        are exhausted without improvement, halve s; stop if s < s_min.
        """
        if self.best is None:
            # Shouldn't happen: refine without a best; fall back to seed try
            self.state = ImgState.SEED_TRY
            return self.next_candidate()

        # If step is too small, stop
        if self.refine_s_px < self.params.s_min_px - 1e-12:
            return None

        # Generate neighbor list centered at best
        bx, by, _ = self.best.as_tuple()
        neighbors = [(bx + dx, by + dy) for (dx, dy) in self._neighbor_set(self.refine_s_px)]
        # Directional prioritization
        moves = [(nx - bx, ny - by) for (nx, ny) in neighbors]
        moves = self._order_neighbors_directional(moves)
        # Convert back to absolute candidate positions
        ordered = [(bx + dx, by + dy) for (dx, dy) in moves]

        # Pick the next neighbor in the sweep
        if self._sweep_tried >= len(ordered):
            # Completed a sweep with no improvement -> halve step and reset sweep
            self.refine_s_px *= 0.5
            self._sweep_tried = 0
            if self.refine_s_px < self.params.s_min_px - 1e-12:
                return None
            # Recompute with new s
            bx, by, _ = self.best.as_tuple()
            neighbors = [(bx + dx, by + dy) for (dx, dy) in self._neighbor_set(self.refine_s_px)]
            moves = [(nx - bx, ny - by) for (nx, ny) in neighbors]
            moves = self._order_neighbors_directional(moves)
            ordered = [(bx + dx, by + dy) for (dx, dy) in moves]

        nx, ny = ordered[self._sweep_tried]
        self._sweep_tried += 1

        # Enforce trust radius relative to seed
        if nx * nx + ny * ny > (self.params.R_px + 1e-12) ** 2:
            # skip this one by recursion (move on to next)
            return self._next_refine_candidate()

        dpx = (nx, ny)
        abs_mm = self._abs_mm_from_delta_px(dpx)
        return Candidate(self.h5_src, self.overlay, self.image_idx, "refine", dpx, abs_mm, self.seed_mm)

    # ---------------- Results Incorporation ----------------

    def incorporate_result(
        self,
        candidate: Candidate,
        indexed: bool,
        metrics: Optional[Dict[str, Any]],
        params: Params,
    ) -> None:
        """
        Update controller state after a run result for this candidate.
        """
        self.eval_count += 1

        if self.state == ImgState.SEED_TRY:
            if indexed:
                self.best = BestPoint(
                    dx_px=0.0, dy_px=0.0, wrmsd=float(metrics["wrmsd"]),
                    n_reflections=metrics.get("n_reflections"),
                    n_peaks=metrics.get("n_peaks"),
                    cell_dev_pct=metrics.get("cell_dev_pct"),
                )
                self.state = ImgState.REFINE
                self._sweep_tried = 0
            else:
                self.state = ImgState.RING_SEARCH
                # keep ring_r_px as initialized

            return

        if self.state == ImgState.RING_SEARCH:
            if indexed:
                self.best = BestPoint(
                    dx_px=candidate.delta_px[0],
                    dy_px=candidate.delta_px[1],
                    wrmsd=float(metrics["wrmsd"]),
                    n_reflections=metrics.get("n_reflections"),
                    n_peaks=metrics.get("n_peaks"),
                    cell_dev_pct=metrics.get("cell_dev_pct"),
                )
                self.state = ImgState.REFINE
                self._sweep_tried = 0
                # First success; have_two_successes remains False until a second success
            else:
                # Not indexed: if this was the last angle at this radius, radius increments
                # Note: _next_ring_candidate handles radius progression based on eval_count.
                pass
            return

        if self.state == ImgState.REFINE:
            if not indexed:
                # No improvement; keep sweeping; when a full sweep has no improvements,
                # _next_refine_candidate will halve step.
                return

            # Compare wrmsd to best with relative threshold
            curr_wr = float(metrics["wrmsd"])
            if self.best is None or (curr_wr < self.best.wrmsd * (1.0 - params.eps_rel)):
                # Track descent direction if we already had a best (this is a second success)
                if self.best is not None and not self.have_two_successes:
                    vx = candidate.delta_px[0] - self.best.dx_px
                    vy = candidate.delta_px[1] - self.best.dy_px
                    if abs(vx) + abs(vy) > 0.0:
                        self.have_two_successes = True
                        self.descent_vec_px = (vx, vy)

                # Accept improvement and reset sweep
                self.best = BestPoint(
                    dx_px=candidate.delta_px[0],
                    dy_px=candidate.delta_px[1],
                    wrmsd=curr_wr,
                    n_reflections=metrics.get("n_reflections"),
                    n_peaks=metrics.get("n_peaks"),
                    cell_dev_pct=metrics.get("cell_dev_pct"),
                )
                self._sweep_tried = 0
            else:
                # Within tie tolerance? we can keep current best using tie-breaks if desired
                tol = params.tie_tol_rel
                if self.best and abs(curr_wr - self.best.wrmsd) <= tol * self.best.wrmsd:
                    # Optional: tie-break policy (prefer smaller shift, more reflections, etc.)
                    pass
                # Else keep sweeping; _next_refine_candidate halves s after a sweep.

            return

        # If FINAL/UNINDEXED, nothing to do (should not be called)

    # ---------------- Utilities ----------------

    def _abs_mm_from_delta_px(self, dpx: Tuple[float, float]) -> Tuple[float, float]:
        dx_mm, dy_mm = px_to_mm(dpx[0], dpx[1], self.geom_res_value)
        return (self.seed_mm[0] + dx_mm, self.seed_mm[1] + dy_mm)


# ------------------------- Engine -------------------------

@dataclass
class SourceContext:
    h5_src: str
    overlay: str
    N_images: int
    controllers: Dict[int, ImageController] = field(default_factory=dict)


@dataclass
class RunWave:
    run_id: int
    lst_path: str
    meta_path: str
    stream_path: str
    # entries in order of .lst lines:
    entries: List[Candidate] = field(default_factory=list)


class AdaptiveEngine:
    """
    Orchestrates multiple source HDF5s, builds waves, writes overlays, runs indexamajig,
    parses results, and drives controllers to FINAL/UNINDEXED.
    """

    def __init__(
        self,
        run_root: str,
        geom_path: str,
        cell_path: str,
        h5_sources: List[str],
        params: Params,
        indexamajig_flags_passthrough: List[str],
    ):
        self.run_root = os.path.abspath(run_root)
        self.geom_path = os.path.abspath(geom_path)
        self.cell_path = os.path.abspath(cell_path)
        self.h5_sources = [os.path.abspath(p) for p in h5_sources]
        self.params = params
        self.idx_flags = list(indexamajig_flags_passthrough)
        self.res_value = read_geom_res_mm_per_1000px(self.geom_path)
        self._mm_per_px = mm_per_px(self.res_value)

        # Paths
        self.inputs_dir = os.path.join(self.run_root, "inputs")
        self.overlays_dir = os.path.join(self.run_root, "overlays")
        self.runs_dir = os.path.join(self.run_root, "runs")
        self.streams_dir = os.path.join(self.run_root, "streams")
        self.merged_dir = os.path.join(self.run_root, "merged")
        self.best_csv = os.path.join(self.run_root, "best_centers.csv")
        self.log_path = os.path.join(self.run_root, "log.jsonl")

        # Sources
        self.sources: Dict[str, SourceContext] = {}

        # Run counter
        self.run_counter = 0

        # Prepare dirs
        os.makedirs(self.inputs_dir, exist_ok=True)
        os.makedirs(self.overlays_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.streams_dir, exist_ok=True)
        os.makedirs(self.merged_dir, exist_ok=True)

        # Symlink/copy inputs for provenance (cheap)
        self._materialize_inputs()

    def _materialize_inputs(self):
        # Keep a copy or symlink to geom and cell for reproducibility
        def _copy(src, dst):
            if os.path.abspath(src) == os.path.abspath(dst):
                return
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
        _copy(self.geom_path, os.path.join(self.inputs_dir, os.path.basename(self.geom_path)))
        _copy(self.cell_path, os.path.join(self.inputs_dir, os.path.basename(self.cell_path)))

    # ----- Source setup -----

    def setup_sources(self) -> None:
        """
        Create overlays for each source and initialize controllers for all images.
        """
        for src in self.h5_sources:
            overlay_path = os.path.join(self.overlays_dir, os.path.basename(src).replace(".h5", ".overlay.h5"))
            N = create_overlay(src, overlay_path, use_vds=False)
            sc = SourceContext(h5_src=src, overlay=overlay_path, N_images=N)
            # read seeds from overlay (copied from src)
            xs, ys = get_seed_shifts_mm(overlay_path)
            controllers: Dict[int, ImageController] = {}
            for idx in range(N):
                ctrl = ImageController(
                    h5_src=src,
                    overlay=overlay_path,
                    image_idx=idx,
                    geom_res_value=self.res_value,
                    seed_mm=(float(xs[idx]), float(ys[idx])),
                    params=self.params,
                )
                controllers[idx] = ctrl
            sc.controllers = controllers
            self.sources[src] = sc

    # ----- Wave construction -----

    def _collect_candidates_for_wave(self) -> Dict[str, List[Candidate]]:
        """
        Produce at most one candidate per active image, grouped by overlay file (source).
        Returns: {overlay_path: [Candidate, ...], ...}
        """
        grouped: Dict[str, List[Candidate]] = {}
        for src, sc in self.sources.items():
            for idx, ctrl in sc.controllers.items():
                if ctrl.state in (ImgState.FINAL, ImgState.UNINDEXED):
                    continue
                cand = ctrl.next_candidate()
                if cand is None:
                    continue
                grouped.setdefault(sc.overlay, []).append(cand)
        return grouped

    def _build_wave_files(self, grouped: Dict[str, List[Candidate]]) -> List[RunWave]:
        """
        For each overlay group, create a run wave:
          - write absolute shifts to overlay for the listed indices
          - create run_X/list, stream path, and meta json
        """
        waves: List[RunWave] = []
        if not grouped:
            return waves

        # new run id
        self.run_counter += 1
        run_id = self.run_counter
        run_dir = os.path.join(self.runs_dir, f"run_{run_id:04d}")
        os.makedirs(run_dir, exist_ok=True)

        # We'll create one .lst per overlay group, but share run_id namespace
        # If you prefer one global .lst across overlays, adapt here.
        for overlay_path, candidates in grouped.items():
            # order candidates deterministically by (h5_src, image_idx)
            candidates.sort(key=lambda c: (c.h5_src, c.image_idx))

            lst_path = os.path.join(run_dir, f"{os.path.basename(overlay_path)}.lst")
            meta_path = os.path.join(run_dir, f"{os.path.basename(overlay_path)}_meta.json")
            stream_path = os.path.join(run_dir, f"{os.path.basename(overlay_path)}.stream")

            indices = [c.image_idx for c in candidates]
            dx_mm = [c.abs_mm[0] for c in candidates]
            dy_mm = [c.abs_mm[1] for c in candidates]

            # Write overlay shifts (absolute)
            write_shifts_mm(overlay_path, indices, dx_mm, dy_mm)
            # Write lst
            write_lst(lst_path, overlay_path, indices)
            # Write meta
            meta = [
                {
                    "h5_path": c.h5_src,
                    "overlay": c.overlay,
                    "image_idx": c.image_idx,
                    "purpose": c.purpose,
                    "seed_mm": [c.seed_mm[0], c.seed_mm[1]],
                    "delta_px": [c.delta_px[0], c.delta_px[1]],
                    "abs_shift_mm": [c.abs_mm[0], c.abs_mm[1]],
                }
                for c in candidates
            ]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            wave = RunWave(
                run_id=run_id,
                lst_path=lst_path,
                meta_path=meta_path,
                stream_path=stream_path,
                entries=candidates,
            )
            waves.append(wave)

            # Log
            jsonl_append(self.log_path, {
                "ts": time.time(), "type": "run_start",
                "run_id": run_id, "overlay": overlay_path,
                "lst_size": len(indices), "lst_path": lst_path
            })

        return waves

from gi_util import (
    jsonl_append,
    run_indexamajig,
    RunResult,
)
from stream_scoring import score_single_chunk_text
from stream_extract import (
    find_chunk_span_for_image,
    extract_winner_stream,
    merge_winner_streams,
)

# NOTE: Part 1/2 defined: Params, ImgState, BestPoint, Candidate, ImageController,
# SourceContext, RunWave, AdaptiveEngine (class skeleton with setup & wave building).

# ------------------------- AdaptiveEngine (continuation) -------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


class AdaptiveEngine(AdaptiveEngine):  # extend the class defined in Part 1
    # ----- Running waves -----

    def _run_wave(self, wave: RunWave) -> RunResult:
        # Execute indexamajig for this overlay-specific .lst
        result = run_indexamajig(
            lst_path=wave.lst_path,
            geom_path=os.path.join(self.inputs_dir, os.path.basename(self.geom_path)),
            cell_path=os.path.join(self.inputs_dir, os.path.basename(self.cell_path)),
            out_stream_path=wave.stream_path,
            flags_passthrough=self.idx_flags,
        )
        # log
        jsonl_append(self.log_path, {
            "ts": time.time(), "type": "run_end",
            "run_id": wave.run_id, "overlay": os.path.basename(wave.lst_path).replace(".lst", ""),
            "lst_size": len(wave.entries),
            "elapsed_s": result.elapsed_s,
            "returncode": result.returncode,
            "stderr_tail": result.stderr_tail,
            "stream_path": wave.stream_path,
        })
        return result

    def _process_wave_results(self, wave: RunWave) -> None:
        """
        For each candidate in the wave, try to find its chunk in the .stream;
        if found, score wRMSD and update controller; else mark as not indexed.
        """
        # Load stream text once
        if not os.path.exists(wave.stream_path):
            # No stream produced (crash); mark all as failed attempt
            for cand in wave.entries:
                ctrl = self.sources[cand.h5_src].controllers[cand.image_idx]
                ctrl.incorporate_result(cand, indexed=False, metrics=None, params=self.params)
                jsonl_append(self.log_path, {
                    "ts": time.time(), "type": "candidate_result",
                    "h5": cand.h5_src, "idx": cand.image_idx, "purpose": cand.purpose,
                    "indexed": False, "reason": "no_stream"
                })
            return

        text = _read_text(wave.stream_path)

        for cand in wave.entries:
            ctrl = self.sources[cand.h5_src].controllers[cand.image_idx]
            span = find_chunk_span_for_image(
                stream_path=wave.stream_path,
                overlay_path=cand.overlay,
                image_idx=cand.image_idx,
            )

            if span is None:
                # Not indexed / not present
                ctrl.incorporate_result(cand, indexed=False, metrics=None, params=self.params)
                jsonl_append(self.log_path, {
                    "ts": time.time(), "type": "candidate_result",
                    "h5": cand.h5_src, "idx": cand.image_idx, "purpose": cand.purpose,
                    "indexed": False
                })
                continue

            # Score the chunk using text slice (avoids temp files)
            chunk_text = text[span[0]:span[1]]
            try:
                metrics = score_single_chunk_text(chunk_text)
                ctrl.incorporate_result(cand, indexed=True, metrics=metrics, params=self.params)
                jsonl_append(self.log_path, {
                    "ts": time.time(), "type": "candidate_result",
                    "h5": cand.h5_src, "idx": cand.image_idx, "purpose": cand.purpose,
                    "indexed": True, "wrmsd": metrics.get("wrmsd"),
                    "n_reflections": metrics.get("n_reflections"),
                    "n_peaks": metrics.get("n_peaks"),
                    "cell_dev_pct": metrics.get("cell_dev_pct"),
                })
            except Exception as e:
                # If scoring failed, treat as not indexed (conservative)
                ctrl.incorporate_result(cand, indexed=False, metrics=None, params=self.params)
                jsonl_append(self.log_path, {
                    "ts": time.time(), "type": "candidate_result",
                    "h5": cand.h5_src, "idx": cand.image_idx, "purpose": cand.purpose,
                    "indexed": False, "reason": f"score_error:{e}"
                })

    # ----- Completion checks & finalization -----

    def _all_images_done(self) -> bool:
        for sc in self.sources.values():
            for ctrl in sc.controllers.values():
                if ctrl.state not in (ImgState.FINAL, ImgState.UNINDEXED):
                    return False
        return True

    def _finalize_winners(self) -> List[str]:
        """
        Extract per-image winner streams for all FINAL images that don't yet
        have a streams/<src>_<idx>.stream file. Return list of paths created.
        """
        winner_paths: List[str] = []

        # Iterate runs newest to oldest to find the chunk where best was achieved
        # Simpler approach: try each run stream until we can extract a chunk at 'best' (overlay, idx).
        # For performance, we could track last-success run, but correctness first.
        run_dirs = sorted(
            [os.path.join(self.runs_dir, d) for d in os.listdir(self.runs_dir) if d.startswith("run_")],
            key=lambda p: p, reverse=True
        )

        for sc in self.sources.values():
            for idx, ctrl in sc.controllers.items():
                if ctrl.state != ImgState.FINAL or ctrl.best is None:
                    continue
                out_name = f"{os.path.basename(sc.h5_src).replace('.h5','')}_{idx}.stream"
                out_path = os.path.join(self.streams_dir, out_name)
                if os.path.exists(out_path):
                    continue  # already extracted

                # Find the stream where this final-best was produced:
                # We search backward over run streams for this overlay that contain this (overlay, idx) chunk.
                found = False
                for run_dir in run_dirs:
                    # Streams are named per overlay in our design
                    candidate_stream = os.path.join(run_dir, f"{os.path.basename(sc.overlay)}.stream")
                    if not os.path.exists(candidate_stream):
                        continue
                    span = find_chunk_span_for_image(candidate_stream, sc.overlay, idx)
                    if span is None:
                        continue
                    try:
                        extract_winner_stream(candidate_stream, out_path, span)
                        ctrl.winner_stream_path = out_path
                        winner_paths.append(out_path)
                        found = True
                        break
                    except Exception as e:
                        # Try older streams if extraction fails
                        continue

                # If not found, we can skip (image may have finalized without a valid chunk, rare)
        return winner_paths

    def _write_best_centers_csv(self) -> None:
        """
        Append/update best_centers.csv with FINAL images' best records.
        We rewrite the file each time from current state to avoid duplicates.
        """
        rows: List[str] = []
        header = "h5_path,image_idx,dx_px,dy_px,dx_mm,dy_mm,wrmsd,n_reflections,n_peaks,cell_dev_pct,evals,time_ms,status"
        rows.append(header)

        for sc in self.sources.values():
            for idx, ctrl in sc.controllers.items():
                if ctrl.best is None:
                    continue
                status = ctrl.state
                dx_px = ctrl.best.dx_px
                dy_px = ctrl.best.dy_px
                dx_mm = ctrl.seed_mm[0] + dx_px * (1.0 * self._mm_per_px)
                dy_mm = ctrl.seed_mm[1] + dy_px * (1.0 * self._mm_per_px)
                wr = ctrl.best.wrmsd
                nrefl = ctrl.best.n_reflections if ctrl.best.n_reflections is not None else ""
                npeaks = ctrl.best.n_peaks if ctrl.best.n_peaks is not None else ""
                celldev = ctrl.best.cell_dev_pct if ctrl.best.cell_dev_pct is not None else ""
                evals = ctrl.eval_count
                # time_ms not tracked per-image here; leave blank for now or compute if you time each attempt
                rows.append(",".join(map(str, [
                    sc.h5_src, idx, f"{dx_px:.6g}", f"{dy_px:.6g}",
                    f"{dx_mm:.6g}", f"{dy_mm:.6g}",
                    f"{wr:.6g}", nrefl, npeaks, celldev, evals, "", status
                ])))

        with open(self.best_csv, "w") as f:
            f.write("\n".join(rows) + "\n")

    def _merge_all_winners(self, out_path: Optional[str] = None) -> Optional[str]:
        # Collect all existing per-image winner streams
        streams = []
        for name in sorted(os.listdir(self.streams_dir)):
            if not name.endswith(".stream"):
                continue
            streams.append(os.path.join(self.streams_dir, name))
        if not streams:
            return None
        if out_path is None:
            out_path = os.path.join(self.merged_dir, "merged_best.stream")
        merge_winner_streams(streams, out_path)
        jsonl_append(self.log_path, {"ts": time.time(), "type": "merge_done",
                                     "images": len(streams), "out_stream": out_path})
        return out_path

    # ----- Public driver -----

    def run(self) -> str:
        """
        Main loop:
          - setup sources (overlays + controllers)
          - iterate waves until all images are FINAL/UNINDEXED
          - extract winners, write CSV, merge
        Returns path to merged stream or empty string if none.
        """
        self.setup_sources()

        # Initialize all controllers to SEED_TRY
        for sc in self.sources.values():
            for ctrl in sc.controllers.values():
                ctrl.activate()

        # Wave loop
        while True:
            grouped = self._collect_candidates_for_wave()
            if not grouped:
                break

            # Build run waves (.lst per overlay) for this wave id
            waves = self._build_wave_files(grouped)

            # Execute each wave (sequential per overlay group)
            for wave in waves:
                result = self._run_wave(wave)
                # Process results even if returncode != 0 (partial outputs may exist)
                self._process_wave_results(wave)

            # Optional pruning: delete old run streams whose images are all finalized
            # (We keep them until the very end for robust extraction.)

            if self._all_images_done():
                break

        # Finalize winners
        self._finalize_winners()
        self._write_best_centers_csv()
        merged_path = self._merge_all_winners()
        return merged_path or ""

# ------------------------- Public API -------------------------

def gandalf_adaptive(
    run_root: str,
    geom_path: str,
    cell_path: str,
    h5_sources: List[str],
    params: Params,
    indexamajig_flags_passthrough: List[str],
) -> str:
    """
    Entry point for runner/GUI.
    Creates the engine and runs the adaptive optimization. Returns path to merged stream.
    """
    eng = AdaptiveEngine(
        run_root=run_root,
        geom_path=geom_path,
        cell_path=cell_path,
        h5_sources=h5_sources,
        params=params,
        indexamajig_flags_passthrough=indexamajig_flags_passthrough,
    )
    merged = eng.run()
    return merged

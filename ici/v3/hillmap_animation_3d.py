# hillmap_animation_3d.py
# 3D-only hillmap animator: builds an evolving probability surface and overlays attempted center shifts.
# Robust log parsing + clear diagnostics if an event has no trials.

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LightSource
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Trial:
    x_mm: float
    y_mm: float
    indexed: int
    wrmsd: Optional[float] = None

    @staticmethod
    def from_values(dx, dy, indexed, wr=None):
        ix = 1 if (str(indexed).strip().lower() in ("1", "true", "yes", "y")) else 0
        try:
            wrf = float(wr) if wr not in (None, "", "nan", "NaN", "None", "none") else None
        except Exception:
            wrf = None
        return Trial(float(dx), float(dy), ix, wrf)

 
# -----------------------------
# Visualization params & field
# -----------------------------

@dataclass
class Step1VizParams:
    radius_mm: float = 0.05            # ±R domain in x/y
    A0: float = 1.0                    # base Gaussian amplitude
    hill_amp_frac: float = 0.2         # success hill amplitude (fraction of A0)
    drop_amp_frac: float = 0.1         # failure dip amplitude (fraction of A0; applied negative)
    explore_floor: float = 1e-6        # keep probabilities positive
    min_spacing_mm: float = 0.003      # mask out attempts within this distance of previous trials
    first_attempt_center_mm: Tuple[float, float] = (0.0, 0.0)
    grid_N: int = 301                  # use odd for symmetric center
    apply_min_spacing_mask: bool = True


def _gauss2d(X, Y, cx, cy, sigma):
    return np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2))

# def probability_grid_from_trials(trials: List[Trial], params: Step1VizParams):
#     """
#     Accumulated Gaussians (same width) for all attempts:
#       - first attempt: amp = +1.0
#       - success:       amp = +0.2
#       - failure:       amp = -0.1
#     No normalization. Returns X, Y, W (height field).
#     """
#     # Domain/grid
#     R = float(params.radius_mm)
#     xs = np.linspace(-R, R, params.grid_N)
#     ys = np.linspace(-R, R, params.grid_N)
#     X, Y = np.meshgrid(xs, ys)
#     disk_mask = (X**2 + Y**2) <= (R**2)

#     # Fixed width for all Gaussians
#     sigma = R / 2.0  # adjust to taste (e.g., R/3.0 for sharper bumps)

#     # Start with zeros, then add Gaussians
#     W = np.zeros_like(X, dtype=float)

#     if trials:
#         # First attempt: +1.0
#         t0 = trials[0]
#         W += 1.0 * _gauss2d(X, Y, t0.x_mm, t0.y_mm, sigma)

#         # Subsequent attempts: +0.2 if indexed==1 else -0.1
#         for t in trials[1:]:
#             amp = 0.2 if t.indexed == 1 else -0.1
#             W += amp * _gauss2d(X, Y, t.x_mm, t.y_mm, sigma)

#     # Mask outside circular region (optional but cleaner)
#     W = np.where(disk_mask, W, np.nan)

#     # Optional: punch small NaN holes near previous trials so points stand out (can disable)
#     if params.apply_min_spacing_mask and trials:
#         ms = float(params.min_spacing_mm)
#         for t in trials:
#             near = ((X - t.x_mm)**2 + (Y - t.y_mm)**2) <= (ms**2)
#             W[near] = np.nan

#     return X, Y, W

def probability_grid_from_trials(trials: List[Trial], params: Step1VizParams,
                                 beta: float = 12.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the evolving probability surface as used in the Boltzmann-weighted hillmap:
      - base Gaussian around first attempt (A0 / n_success)
      - positive hills at successful trials weighted by exp[-β*(wrmsd - wrmsd_min)]
      - negative dips for failures
    """
    R = float(params.radius_mm)
    xs = np.linspace(-R, R, params.grid_N)
    ys = np.linspace(-R, R, params.grid_N)
    X, Y = np.meshgrid(xs, ys)
    disk_mask = (X**2 + Y**2) <= (R**2)

    sigma = R / 1.0  # broader to merge nearby hills
    W = np.zeros_like(X, dtype=float)

    if not trials:
        return X, Y, np.where(disk_mask, np.nan, np.nan)

    successes = [t for t in trials if t.indexed == 1 and t.wrmsd is not None]
    failures = [t for t in trials if t.indexed == 0]

    # Base Gaussian (prior)
    A0 = float(params.A0)
    # if successes:
    #     A0 /= len(successes)  # adaptive prior decay
    t0 = trials[0]
    W += A0 * _gauss2d(X, Y, t0.x_mm, t0.y_mm, sigma)

    # Success hills (Boltzmann-weighted)
    if successes:
        wr_vals = np.array([t.wrmsd for t in successes], float)
        wmin = float(np.min(wr_vals))
        weights = np.exp(-beta * (wr_vals - wmin))
        weights /= np.sum(weights)
        for t, w in zip(successes, weights):
            W += (params.hill_amp_frac * params.A0) * w * _gauss2d(X, Y, t.x_mm, t.y_mm, sigma)

    # Failure dips
    for t in failures:
        W += -(params.drop_amp_frac * params.A0) * _gauss2d(X, Y, t.x_mm, t.y_mm, sigma)

    # Apply mask and floor
    W = np.where(disk_mask, W, np.nan)
    W = np.maximum(W, 0.0) + params.explore_floor

    return X, Y, W

# -----------------------------
# Log parsing (robust)
# -----------------------------

HEADER_RE = re.compile(r"^#.*event\s+(\d+)", flags=re.IGNORECASE)

def read_lines(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Log not found: {path}")
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()


def split_blocks_by_headers(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Split into blocks whenever a line like "# ... event N" appears.
    Returns list of (header_line_or_fallback, block_lines_including_header).
    """
    blocks = []
    cur_header = None
    cur: List[str] = []
    for ln in lines:
        if ln.strip().startswith("#") and "event" in ln.lower():
            if cur:
                blocks.append((cur_header or "#/unknown event -1", cur))
            cur = [ln]
            cur_header = ln
        else:
            cur.append(ln)
    if cur:
        blocks.append((cur_header or "#/unknown event -1", cur))
    return blocks


def parse_event_id_from_header(h: str) -> int:
    m = HEADER_RE.match(h.strip())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Fallback: look for trailing integer
    tail = re.findall(r"(\d+)\s*$", h)
    if tail:
        return int(tail[-1])
    return -1

def parse_trials_from_block(block_lines: List[str]) -> List[Trial]:
    """
    Parses each event block in the hillmap log.
    Compatible with logs where only the first event has the CSV header line
    ('run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,...').
    """
    data_lines = [ln for ln in block_lines if not ln.strip().startswith("#") and ln.strip()]
    if not data_lines:
        return []

    # Only treat this block as headered if its first non-comment line starts with 'run'
    has_header_here = data_lines[0].lower().startswith("run")

    trials: List[Trial] = []

    if has_header_here:
        try:
            reader = csv.DictReader(data_lines)
            header = [h.strip() for h in (reader.fieldnames or [])]

            def find_col(cands: List[str]) -> Optional[str]:
                for c in cands:
                    for h in header:
                        if h.lower().strip() == c:
                            return h
                return None

            # ⬅ supports your column names
            col_dx = find_col(["det_shift_x_mm", "dx_mm", "dx", "x_mm", "x"])
            col_dy = find_col(["det_shift_y_mm", "dy_mm", "dy", "y_mm", "y"])
            col_idx = find_col(["indexed", "success", "ok"])
            col_wr  = find_col(["wrmsd", "w_rmsd", "w-rmsd"])

            if not (col_dx and col_dy and col_idx):
                raise ValueError("Missing required columns in header")

            for row in reader:
                dx = row.get(col_dx, "0")
                dy = row.get(col_dy, "0")
                idx = row.get(col_idx, "0")
                wr = row.get(col_wr, None) if col_wr else None
                trials.append(Trial.from_values(dx, dy, idx, wr))

            if trials:
                return trials
        except Exception:
            pass  # fall through to manual parsing if DictReader fails

    # Manual parsing for header-less blocks (all subsequent events)
    for ln in data_lines:
        if ln.lower().startswith("run"):  # skip the single global header row
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue
        # Layout: run_n, det_shift_x_mm, det_shift_y_mm, indexed, wrmsd, ...
        dx, dy, idx = parts[1], parts[2], parts[3]
        wr = parts[4] if len(parts) >= 5 else None
        try:
            trials.append(Trial.from_values(dx, dy, idx, wr))
        except Exception:
            continue

    return trials


def parse_log_any(path: str) -> Dict[int, List[Trial]]:
    """
    Return dict: event_id -> list of Trial.
    Supports "# ... event N" block logs; falls back to single-block CSV if no headers.
    """
    lines = read_lines(path)
    blocks = split_blocks_by_headers(lines)

    event_map: Dict[int, List[Trial]] = {}

    # If we didn't find any headered blocks with event IDs, try whole-file CSV
    found_any_event_header = any(parse_event_id_from_header(h) != -1 for h, _ in blocks)
    if not found_any_event_header and len(blocks) == 1:
        trials = parse_trials_from_block(blocks[0][1])
        if trials:
            event_map[0] = trials  # single synthetic event
            return event_map

    for h, bl in blocks:
        eid = parse_event_id_from_header(h)
        trials = parse_trials_from_block(bl)
        if trials:
            event_map.setdefault(eid, []).extend(trials)

    return event_map


# -----------------------------
# 3D animation
# -----------------------------


def animate_event_progress_3d(log_abs_path: str,
                              run_root: str,
                              event_id: Optional[int] = None,
                              viz_params: Step1VizParams = None,
                              fps: int = 2,
                              downsample: int = 2,
                              out_path: Optional[str] = None) -> str:
    """
    Build frames incrementally, updating the probability surface as more trials are added.
    """
    if viz_params is None:
        viz_params = Step1VizParams()

    event_map = parse_log_any(log_abs_path)
    if not event_map:
        raise RuntimeError(
            "No trials found in the log. "
            "Check the file path and format, or ensure headers like '# ... event N' are present."
        )
    if getattr(args, "debug", False):
        print("Parsed events with trial counts:",
            {k: len(v) for k, v in event_map.items()})

    # Choose event
    chosen_eid = None
    if event_id is not None and event_id in event_map:
        chosen_eid = event_id
    elif event_id is not None and event_id not in event_map:
        raise RuntimeError(
            f"Event {event_id} not found. Available events: {sorted(event_map.keys())}"
        )
    else:
        # pick the first non-empty
        chosen_eid = sorted(event_map.keys())[0]

    trials_all = event_map[chosen_eid]
    if not trials_all:
        raise RuntimeError(
            f"Selected event {chosen_eid} has no parsed trials. "
            f"Check that rows under '# ... event {chosen_eid}' contain data and are not commented out."
        )

    if out_path is None:
        out_path = str(Path(run_root) / f"event{chosen_eid:03d}_progress_3d.gif")
    
    # --- Precompute frames (uses accumulated W, not normalized) ---
    frames = []
    for k in range(1, len(trials_all) + 1):
        trials_k = trials_all[:k]
        X, Y, W = probability_grid_from_trials(trials_k, viz_params)
        if downsample and downsample > 1:
            X = X[::downsample, ::downsample]
            Y = Y[::downsample, ::downsample]
            W = W[::downsample, ::downsample]

        xs = np.array([t.x_mm for t in trials_k])
        ys = np.array([t.y_mm for t in trials_k])
        idxs = np.array([t.indexed for t in trials_k])  # 0/1 flags

        # robust NN (edges OK)
        ix = np.searchsorted(X[0], xs, side="right") - 1
        iy = np.searchsorted(Y[:, 0], ys, side="right") - 1
        ix = np.clip(ix, 0, X.shape[1] - 1)
        iy = np.clip(iy, 0, Y.shape[0] - 1)

        Wsafe = np.nan_to_num(W, nan=0.0)
        zs = Wsafe[iy, ix]

        frames.append((k, X, Y, W, xs, ys, zs, idxs))


    # --- Plotting (3D) ---


    # --- Figure & initial plot (3D, shaded, with floor contour) ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Global z-limits and normalization for stable colors
    zmin = min(np.nanmin(f[3]) for f in frames)
    zmax = max(np.nanmax(f[3]) for f in frames)
    norm = colors.Normalize(vmin=zmin, vmax=zmax)
    cmap = plt.get_cmap("viridis")
    ls = LightSource(azdeg=315, altdeg=45)

    # First frame
    k0, X0, Y0, W0, xs0, ys0, zs0, idxs0 = frames[0]
    W0m = np.ma.masked_invalid(W0)
    rgb0 = ls.shade(W0m, cmap=cmap, norm=norm, vert_exag=0.7, blend_mode="soft")

    surf = [ax.plot_surface(
        X0, Y0, W0m,
        rstride=2, cstride=2,
        facecolors=rgb0, linewidth=0, antialiased=True, shade=False
    )]

    # Floor contour at zmin
    cont = [ax.contourf(
        X0, Y0, W0m,
        zdir='z', offset=zmin, levels=24, cmap=cmap, norm=norm, alpha=0.85
    )]

    # Boundary circle on the floor (nice frame)
    R = float(viz_params.radius_mm)
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(R*np.cos(theta), R*np.sin(theta), zmin, lw=1.2, alpha=0.8, color="black")

    # First attempt star, then successes/failures
    star = [ax.scatter(xs0[0], ys0[0], zs0[0], marker='*', s=120, edgecolors='k', linewidths=0.6)]
    succ = [ax.scatter(xs0[idxs0==1], ys0[idxs0==1], zs0[idxs0==1], marker='o', s=26)]
    fail = [ax.scatter(xs0[idxs0==0], ys0[idxs0==0], zs0[idxs0==0], marker='x', s=28)]

    # Axis cosmetics
    ax.set_xlabel("dx (mm)")
    ax.set_ylabel("dy (mm)")
    ax.set_zlabel("accumulated surface (a.u.)")
    ax.set_title(f"Event {chosen_eid} — after run {k0}")
    if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
        ax.set_zlim(zmin, zmax)
    ax.view_init(elev=30, azim=-60)
    ax.grid(False)

    # Colorbar for the surface
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])  # required by mpl
    cb = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08)
    cb.set_label("accumulated height (a.u.)")

    # Optional slow camera rotation (set ROTATE=True to enable)
    ROTATE = False
    AZ0 = -60
    AZ_STEP = 1.2

    def update(i):
        k, X, Y, W, xs, ys, zs, idxs = frames[i]
        Wm = np.ma.masked_invalid(W)
        rgb = ls.shade(Wm, cmap=cmap, norm=norm, vert_exag=0.7, blend_mode="soft")

        # Replace our artists cleanly
        try: surf[0].remove()
        except Exception: pass
        surf[0] = ax.plot_surface(
            X, Y, Wm,
            rstride=2, cstride=2,
            facecolors=rgb, linewidth=0, antialiased=True, shade=False
        )

        try:
            for c in cont[0].collections: c.remove()
        except Exception:
            pass
        cont[0] = ax.contourf(
            X, Y, Wm, zdir='z', offset=zmin, levels=24, cmap=cmap, norm=norm, alpha=0.85
        )

        # Update points
        try: succ[0].remove()
        except Exception: pass
        try: fail[0].remove()
        except Exception: pass

        # Star (first attempt) is static; leave it

        succ[0] = ax.scatter(xs[idxs==1], ys[idxs==1], zs[idxs==1], marker='o', s=26)
        fail[0] = ax.scatter(xs[idxs==0], ys[idxs==0], zs[idxs==0], marker='x', s=28)

        ax.set_title(f"Event {chosen_eid} — after run {k}")

        if ROTATE:
            ax.view_init(elev=30, azim=AZ0 + AZ_STEP * i)

        return []


    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // max(1, fps), blit=False)
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=150)
    plt.close(fig)

    manifest = {
        "event_id": chosen_eid,
        "n_runs": len(trials_all),
        "gif_path": out_path,
        "mode": "3d",
        "log_path": str(Path(log_abs_path).resolve()),
    }
    with open(str(Path(out_path).with_suffix(".json")), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_path


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate a 3D HillMap animation from a SerialED run log.")
    ap.add_argument("--log", required=True, help="Path to image_run_log.csv")
    ap.add_argument("--event", type=int, help="Event ID to visualize (if omitted, first event with trials is used)")
    ap.add_argument("--fps", type=int, default=1, help="Frames per second in the GIF")
    ap.add_argument("--radius-mm", type=float, default=0.05)
    ap.add_argument("--A0", type=float, default=2.0)
    ap.add_argument("--hill-amp-frac", type=float, default=5.0)
    ap.add_argument("--drop-amp-frac", type=float, default=0.2)
    ap.add_argument("--min-spacing-mm", type=float, default=0.0001)
    ap.add_argument("--explore-floor", type=float, default=1e-6)
    ap.add_argument("--downsample", type=int, default=2, help="Grid downsample factor for speed (>=1)")
    ap.add_argument("--debug", action="store_true", help="Print parser diagnostics")

    args = ap.parse_args()

    # Infer run_root from the log path (…/runs_timestamp/…/image_run_log.csv → …/runs_timestamp)
    log_path = Path(args.log).resolve()
    run_root = str(log_path.parent.parent) if log_path.parent.name else str(log_path.parent)

    viz = Step1VizParams(
        radius_mm=args.radius_mm,
        A0=args.A0,
        hill_amp_frac=args.hill_amp_frac,
        drop_amp_frac=args.drop_amp_frac,
        explore_floor=args.explore_floor,
        min_spacing_mm=args.min_spacing_mm,
        apply_min_spacing_mask=True,
    )

    try:
        gif_path = animate_event_progress_3d(
            log_abs_path=str(log_path),
            run_root=run_root,
            event_id=args.event,
            viz_params=viz,
            fps=args.fps,
            downsample=max(1, args.downsample),
        )
        print(f"\n✅ Saved animation to: {gif_path}")
    except Exception as e:
        # Helpful diagnostics
        print("\n❌ Could not generate animation.")
        print(f"Reason: {e}")
        print("\nTroubleshooting tips:")
        print("  • Ensure the log has either '# ... event N' headers OR columns like dx_mm, dy_mm, indexed.")
        print("  • If multiple events exist, pass --event <id> shown in the error message.")
        print("  • Open the log and verify that trial rows are not all commented out or empty.")
        sys.exit(1)


import sys, re, json
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import step1_hillmap as s1

@dataclass
class Step1VizParams:
    radius_mm: float = 0.05
    A0: float = 1.0
    hill_amp_frac: float = 1.0
    drop_amp_frac: float = 0.5
    explore_floor: float = 1e-6
    min_spacing_mm: float = 0.003
    first_attempt_center_mm: Tuple[float, float] = (0.0, 0.0)
    grid_N: int = 301
    apply_min_spacing_mask: bool = True

def read_log(run_root: str, log_relpath: str = "runs/image_run_log.csv"):
    p = Path(run_root) / log_relpath
    with p.open("r", encoding="utf-8") as f:
        return f.readlines()

def group_blocks(lines):
    blocks = []
    cur_header = None
    cur = []
    for ln in lines:
        if ln.startswith("#/") and " event " in ln:
            if cur:
                blocks.append((cur_header or "#/unknown event -1", cur))
            cur_header = ln.rstrip("\\n")
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append((cur_header or "#/unknown event -1", cur))
    return blocks

def parse_event_header(header_line: str):
    try:
        prefix, ev = header_line[1:].split(" event ")
        return prefix.strip(), int(ev.strip())
    except Exception:
        return header_line.strip("# \\n"), -1

def parse_trials_from_block(block_lines):
    out = []
    for ln in block_lines:
        s = ln.strip()
        if (not s) or s.startswith("#") or s.startswith("run_n"):
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 5:
            parts += [""] * (5 - len(parts))
        try:
            run_n = int(parts[0])
        except Exception:
            continue
        try:
            dx = float(parts[1]); dy = float(parts[2])
        except Exception:
            dx, dy = 0.0, 0.0
        idx = 1 if parts[3] == "1" else 0
        wr = None
        try:
            wrs = parts[4]
            wr = float(wrs) if wrs not in ("", "nan", "None", "none") else None
        except Exception:
            wr = None
        out.append((run_n, s1.Trial(dx, dy, idx, wr)))
    out.sort(key=lambda t: t[0])
    return [t for _, t in out]

def _gauss2d(X, Y, cx, cy, sigma):
    return np.exp(-0.5 * ((X - cx)**2 + (Y - cy)**2) / (sigma**2))

def probability_grid_from_trials(trials, params: Step1VizParams):
    R = float(params.radius_mm)
    sigma = R / 2.0
    A0 = float(params.A0)
    A_hill = A0 * float(params.hill_amp_frac)
    A_drop = -A0 * float(params.drop_amp_frac)

    xs = np.linspace(-R, R, params.grid_N)
    ys = np.linspace(-R, R, params.grid_N)
    X, Y = np.meshgrid(xs, ys)

    disk_mask = (X**2 + Y**2) <= (R**2)

    if trials:
        c0x, c0y = trials[0].x_mm, trials[0].y_mm
    else:
        c0x, c0y = params.first_attempt_center_mm

    W = np.zeros_like(X)
    W += A0 * _gauss2d(X, Y, c0x, c0y, sigma)

    for t in trials:
        if t.indexed == 1:
            W += A_hill * _gauss2d(X, Y, t.x_mm, t.y_mm, sigma)
        else:
            W += A_drop * _gauss2d(X, Y, t.x_mm, t.y_mm, sigma)

    W = np.maximum(0.0, W) + float(params.explore_floor)
    W = np.where(disk_mask, W, 0.0)

    if params.apply_min_spacing_mask and trials:
        ms = float(params.min_spacing_mm)
        for t in trials:
            near = ((X - t.x_mm)**2 + (Y - t.y_mm)**2) <= (ms**2)
            W[near] = 0.0

    total = W.sum()
    P = W / total if total > 0 else W
    return X, Y, P

def animate_event_progress(run_root: str, event_id: int = None, abs_path_contains: str = None,
                           viz_params: Step1VizParams = None, fps: int = 2, out_path: str = None):
    if viz_params is None:
        viz_params = Step1VizParams()

    lines = read_log(run_root)
    blocks = group_blocks(lines)

    selected = None
    for header, block in blocks:
        abs_path, eid = parse_event_header(header)
        if event_id is not None and eid == event_id:
            selected = (header, block); break
    if selected is None and abs_path_contains:
        for header, block in blocks:
            abs_path, eid = parse_event_header(header)
            if abs_path_contains in abs_path:
                selected = (header, block); break
    if selected is None and blocks:
        selected = blocks[0]

    if selected is None:
        raise RuntimeError("No events found in log.")

    header, block = selected
    abs_path, eid = parse_event_header(header)
    trials_all = parse_trials_from_block(block)

    if out_path is None:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', f"event_{eid}")
        out_path = str(Path(run_root) / f"event{eid:03d}_progress.gif")

    # Precompute frames
    frames = []
    for k in range(1, len(trials_all) + 1):
        trials_k = trials_all[:k]
        X, Y, P = probability_grid_from_trials(trials_k, viz_params)
        xs = [t.x_mm for t in trials_k]
        ys = [t.y_mm for t in trials_k]
        frames.append((k, X, Y, P, xs, ys))

    fig, ax = plt.subplots(figsize=(6,5))
    extent = [-viz_params.radius_mm, viz_params.radius_mm, -viz_params.radius_mm, viz_params.radius_mm]
    im = ax.imshow(frames[0][3], origin='lower', extent=extent)
    scat = ax.scatter(frames[0][4], frames[0][5], s=20, marker='x')
    cb = plt.colorbar(im, ax=ax, label="Probability density (normalized)")
    ax.set_xlabel("dx (mm)"); ax.set_ylabel("dy (mm)")
    title = ax.set_title(f"Event {eid} — after run {frames[0][0]}")

    def update(i):
        k, X, Y, P, xs, ys = frames[i]
        im.set_data(P)
        scat.set_offsets(np.column_stack([xs, ys]))
        title.set_text(f"Event {eid} — after run {k}")
        return [im, scat, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//max(1,fps), blit=False)
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=150)
    plt.close(fig)

    manifest = {"event_id": eid, "abs_path": abs_path, "n_runs": len(trials_all), "gif_path": out_path}
    with open(str(Path(out_path).with_suffix(".json")), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_path

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate HillMap animation from a SerialED run log.")
    ap.add_argument("--log", required=True, help="Path to image_run_log.csv")
    ap.add_argument("--event", type=int, help="Event ID to visualize")
    ap.add_argument("--fps", type=int, default=1, help="Frames per second in the GIF")
    ap.add_argument("--radius-mm", type=float, default=0.05)
    ap.add_argument("--min-spacing-mm", type=float, default=0.001)
    ap.add_argument("--A0", type=float, default=1.0)
    ap.add_argument("--hill-amp-frac", type=float, default=0.4)
    ap.add_argument("--drop-amp-frac", type=float, default=0.2)
    ap.add_argument("--explore-floor", type=float, default=1e-6)
    args = ap.parse_args()

    run_root = str(Path(args.log).resolve().parent.parent)  # infer run_root from log path

    viz = Step1VizParams(
        radius_mm=args.radius_mm,
        A0=args.A0,
        hill_amp_frac=args.hill_amp_frac,
        drop_amp_frac=args.drop_amp_frac,
        explore_floor=args.explore_floor,
        min_spacing_mm=args.min_spacing_mm,
        apply_min_spacing_mask=True,
    )

    gif_path = animate_event_progress(
        run_root=run_root,
        event_id=args.event,
        viz_params=viz,
        fps=args.fps,
    )
    print(f"\n✅ Saved animation to: {gif_path}")

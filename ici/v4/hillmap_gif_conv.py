#!/usr/bin/env python3
# Adaptive hillmap GIF — points sample from the CURRENT surface with temperature,
# explore/exploit control, and sub-pixel jitter.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def smooth2d(A, strength=0.08):
    if strength <= 0: return A
    k = np.array([1,4,6,4,1], float); k /= k.sum()
    Ah = np.pad(A, ((0,0),(2,2)), mode="reflect")
    Ah = (k[0]*Ah[:, :-4] + k[1]*Ah[:, 1:-3] + k[2]*Ah[:, 2:-2] +
          k[3]*Ah[:, 3:-1] + k[4]*Ah[:, 4:])
    Av = np.pad(Ah, ((2,2),(0,0)), mode="reflect")
    Av = (k[0]*Av[:-4,:] + k[1]*Av[1:-3,:] + k[2]*Av[2:-2,:] +
          k[3]*Av[3:-1,:] + k[4]*Av[4:,:])
    return (1-strength)*A + strength*Av

def bump(X, Y, x0, y0, sigma=0.45, amp=1.0):
    r2 = (X-x0)**2 + (Y-y0)**2
    return amp * np.exp(-r2/(2*sigma**2))

def softmax_field(P, temp=0.5, eps=1e-12):
    # temp<1 sharpens, temp>1 flattens
    Q = np.maximum(P, eps)**(1.0/max(temp, 1e-6))
    Qsum = Q.sum()
    return Q/Qsum if Qsum>0 else np.full_like(Q, 1.0/Q.size)

def sample_from_distribution(rng, probs, x, y, jitter=0.3):
    # Discrete index sample + sub-pixel jitter
    idx = rng.choice(probs.size, p=probs.ravel())
    iy, ix = divmod(idx, probs.shape[1])
    # map to coords
    x0, y0 = x[ix], y[iy]
    # sub-pixel gaussian jitter (scaled by grid spacing)
    jx = rng.normal(0, jitter * (x[1]-x[0]))
    jy = rng.normal(0, jitter * (y[1]-y[0]))
    return float(x0 + jx), float(y0 + jy)

def generate_hillmap_gif(
    out="hillmap.gif",
    grid=220, xmin=-5, xmax=5, ymin=-5, ymax=5,
    sigma0=2.2,
    n_frames=80, fps=12,
    succ_amp=0.08, fail_amp=0.05, bump_sigma=0.55,
    smooth_strength=0.08,
    rng_seed=7,
    # success model
    success_mode="field",  # "field" or "distance"
    true_cx=1.2, true_cy=-0.8, prob_sigma=1.0,
    # adaptive sampling controls
    temp=0.6,           # softmax temperature (lower = more greedy)
    epsilon=0.05,       # epsilon-greedy uniform exploration fraction
    p_exploit=0.35,     # after a success, chance to sample near last success
    exploit_sigma=0.6,  # width of local exploit proposal
    jitter=0.35,        # sub-pixel jitter (in grid units)
    # styling
    title="Hillmap search — evolving sampling field",
    subtitle="Non-brute-force sampling adapts to successes (hills) and failures (dips)",
    dpi=120, figsize=(6.8,5.6), cmap="inferno",
    marker_size=30, success_marker="o", fail_marker="x",
    show_true_center=False,
):
    rng = np.random.default_rng(rng_seed)
    x = np.linspace(xmin, xmax, grid)
    y = np.linspace(ymin, ymax, grid)
    X, Y = np.meshgrid(x, y)

    # initial belief
    P = np.exp(-((X**2 + Y**2)/(2*sigma0**2)))

    attempts, successes = [], []
    last_success = None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(P, extent=(xmin,xmax,ymin,ymax), origin="lower", cmap=cmap)
    sc_succ = ax.scatter([], [], s=marker_size, marker=success_marker, linewidths=0.8)
    sc_fail = ax.scatter([], [], s=marker_size, marker=fail_marker, linewidths=0.8)
    t = ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom")
    st = ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Δx (pixels)"); ax.set_ylabel("Δy (pixels)")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    if show_true_center:
        # ax.plot(true_cx, true_cy, marker="+", markersize=10)
        circle = plt.Circle((true_cx, true_cy), 0.4, color='cyan', alpha=0.25, lw=0)
        ax.add_patch(circle)
        ax.plot(true_cx, true_cy, marker='*', markersize=16, color='cyan', markeredgecolor='black', markeredgewidth=1.2, zorder=10)
        ax.text(true_cx + 0.3, true_cy + 0.3, "true center", color='white', fontsize=9, weight='bold')


    def propose_point():
        nonlocal last_success, P
        # With p_exploit, sample locally around last success (if any)
        if (last_success is not None) and (rng.random() < p_exploit):
            lx, ly = last_success
            px = rng.normal(lx, exploit_sigma)
            py = rng.normal(ly, exploit_sigma)
            return float(np.clip(px, xmin, xmax)), float(np.clip(py, ymin, ymax))

        # Else sample from current field with temperature + epsilon-greedy
        probs = softmax_field(P, temp=temp)
        if rng.random() < epsilon:
            # uniform exploration
            px = rng.uniform(xmin, xmax)
            py = rng.uniform(ymin, ymax)
            return px, py
        # adaptive sample with sub-pixel jitter
        return sample_from_distribution(rng, probs, x, y, jitter=jitter)

    def success_probability(px, py):
        if success_mode == "distance":
            return float(np.exp(-((px-true_cx)**2 + (py-true_cy)**2)/(2*prob_sigma**2)))
        # field-driven: normalize local field into [0,1] neighborhood-based prob
        # bilinear interpolate P at (px, py)
        ix = np.searchsorted(x, px) - 1; iy = np.searchsorted(y, py) - 1
        ix = np.clip(ix, 0, len(x)-2); iy = np.clip(iy, 0, len(y)-2)
        tx = (px - x[ix]) / (x[ix+1]-x[ix]); ty = (py - y[iy]) / (y[iy+1]-y[iy])
        Pll = P[iy, ix]; Plr = P[iy, ix+1]; Pul = P[iy+1, ix]; Pur = P[iy+1, ix+1]
        p_loc = (1-tx)*(1-ty)*Pll + tx*(1-ty)*Plr + (1-tx)*ty*Pul + tx*ty*Pur
        # squash to reasonable success rates
        return float(np.clip(0.15 + 0.7*p_loc, 0.0, 1.0))

    def update(frame):
        nonlocal P, last_success
        px, py = propose_point()
        p_succ = success_probability(px, py)
        is_success = (rng.random() < p_succ)

        if is_success:
            P = P + succ_amp * bump(X, Y, px, py, sigma=bump_sigma, amp=1.0)
            last_success = (px, py)
        else:
            P = np.maximum(P - fail_amp * bump(X, Y, px, py, sigma=bump_sigma, amp=1.0), 0.0)

        P = smooth2d(P, strength=smooth_strength)
        # renormalize for display stability
        Pmin, Pmax = P.min(), P.max()
        if Pmax > Pmin: P = (P - Pmin) / (Pmax - Pmin)

        attempts.append((px, py)); successes.append(is_success)

        im.set_data(P)
        succ_pts = np.array([p for p, s in zip(attempts, successes) if s])
        fail_pts = np.array([p for p, s in zip(attempts, successes) if not s])
        if succ_pts.size: sc_succ.set_offsets(succ_pts)
        if fail_pts.size: sc_fail.set_offsets(fail_pts)

        t.set_text(f"{title} (frame {frame+1}/{n_frames})")
        return im, sc_succ, sc_fail, t, st

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/max(fps,1))
    ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="hillmap.gif")
    p.add_argument("--grid", type=int, default=220)
    p.add_argument("--xmin", type=float, default=-5); p.add_argument("--xmax", type=float, default=5)
    p.add_argument("--ymin", type=float, default=-5); p.add_argument("--ymax", type=float, default=5)
    p.add_argument("--sigma0", type=float, default=2.2)
    p.add_argument("--n-frames", type=int, default=200)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--succ-amp", type=float, default=0.1)
    p.add_argument("--fail-amp", type=float, default=0.04)
    p.add_argument("--bump-sigma", type=float, default=0.5)
    p.add_argument("--smooth-strength", type=float, default=0.08)
    p.add_argument("--rng-seed", type=int, default=7)
    p.add_argument("--success-mode", choices=["field","distance"], default="distance")
    p.add_argument("--true-cx", type=float, default=2)
    p.add_argument("--true-cy", type=float, default=2)
    p.add_argument("--prob-sigma", type=float, default=0.7)
    p.add_argument("--temp", type=float, default=0.4)
    p.add_argument("--epsilon", type=float, default=0.02)
    p.add_argument("--p-exploit", type=float, default=0.6)
    p.add_argument("--exploit-sigma", type=float, default=0.45)
    p.add_argument("--jitter", type=float, default=0.35)
    p.add_argument("--title", default="Hillmap search — evolving sampling field")
    p.add_argument("--subtitle", default="")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--figwidth", type=float, default=6.8)
    p.add_argument("--figheight", type=float, default=5.6)
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--marker-size", type=float, default=30)
    p.add_argument("--success-marker", default="o")
    p.add_argument("--fail-marker", default="x")
    p.add_argument("--show-true-center", action="store_true")
    args = p.parse_args()

    path = generate_hillmap_gif(
        out=args.out, grid=args.grid,
        xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax,
        sigma0=args.sigma0, n_frames=args.n_frames, fps=args.fps,
        succ_amp=args.succ_amp, fail_amp=args.fail_amp, bump_sigma=args.bump_sigma,
        smooth_strength=args.smooth_strength, rng_seed=args.rng_seed,
        success_mode=args.success_mode, true_cx=args.true_cx, true_cy=args.true_cy,
        prob_sigma=args.prob_sigma, temp=args.temp, epsilon=args.epsilon,
        p_exploit=args.p_exploit, exploit_sigma=args.exploit_sigma, jitter=args.jitter,
        title=args.title, subtitle=args.subtitle, dpi=args.dpi,
        figsize=(args.figwidth, args.figheight), cmap=args.cmap,
        marker_size=args.marker_size, success_marker=args.success_marker,
        fail_marker=args.fail_marker, show_true_center=True,
    )
    print(f"Saved GIF to: {path}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D wRMSD surface (heatmap + contours) for Keynote
- True center at the global minimum (star).
- HillMap+wRMSD refinements (green) clustered at the minimum.
- CrystFEL refined centers (red) scattered nearby.
- CrystFEL failures (gray X) farther out.
Outputs:
  - wrmsd_2d.png  (transparent, high-DPI)
  - wrmsd_2d.pdf  (vector)
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Parameters (tweak freely)
# --------------------------
np.random.seed(7)
extent = 6.0      # x/y half-range; plot spans [-extent, extent]
N = 501           # grid resolution (higher = smoother)
ax_scale = (2.3, 1.6)  # anisotropy of the bowl
bowl_power = 2.0       # 2.0 = quadratic
ripple_amp = 0.10      # subtle azimuthal ripple (visual depth)
ripple_freq = 2.2

n_wrmsd_points = 18     # HillMap+wRMSD points (green, near minimum)
n_crystfel_points = 14  # CrystFEL refined (red)
n_crystfel_fail = 3     # CrystFEL fails (gray X)

png_path = "wrmsd_2d.png"
pdf_path = "wrmsd_2d.pdf"
dpi = 350

# --------------------------
# Build wRMSD surface
# --------------------------
x = np.linspace(-extent, extent, N)
y = np.linspace(-extent, extent, N)
X, Y = np.meshgrid(x, y)

R = np.sqrt((X/ax_scale[0])**2 + (Y/ax_scale[1])**2)
Z = (R ** bowl_power)

# Gentle ripple for visual interest (keeps center as minimum)
theta = np.arctan2(Y, X)
Z += ripple_amp * np.cos(ripple_freq * theta) * np.exp(-0.25 * R**2)

# Normalize to [0,1]
Z -= Z.min()
Z /= Z.max()

# True center
x_true, y_true = 0.0, 0.0

# --------------------------
# Sample point clouds
# --------------------------
def interp_Z(xp, yp):
    """Fast bilinear interpolation of Z at (xp, yp)."""
    ix = np.interp(xp, x, np.arange(N))
    iy = np.interp(yp, y, np.arange(N))
    ix0 = np.clip(np.floor(ix).astype(int), 0, N-2)
    iy0 = np.clip(np.floor(iy).astype(int), 0, N-2)
    dx = ix - ix0
    dy = iy - iy0
    z00 = Z[iy0, ix0]
    z10 = Z[iy0, ix0+1]
    z01 = Z[iy0+1, ix0]
    z11 = Z[iy0+1, ix0+1]
    return (z00*(1-dx)*(1-dy) + z10*dx*(1-dy) + z01*(1-dx)*dy + z11*dx*dy)

# HillMap + wRMSD minima (tight cluster near true center)
wr_sigma = 0.35
wr_x = np.random.normal(x_true, wr_sigma, n_wrmsd_points)
wr_y = np.random.normal(y_true, wr_sigma, n_wrmsd_points)

# CrystFEL refined: some near, some off
cf_rad_mean, cf_rad_jitter = 2.2, 1.2
cf_theta = 2*np.pi*np.random.rand(n_crystfel_points)
cf_r = np.clip(np.random.normal(cf_rad_mean, cf_rad_jitter, n_crystfel_points), 0.2, extent*0.92)
cf_x = cf_r * np.cos(cf_theta)
cf_y = cf_r * np.sin(cf_theta)

# CrystFEL fails (farther out)
fail_theta = 2*np.pi*np.random.rand(n_crystfel_fail)
fail_r = np.random.uniform(extent*0.65, extent*0.92, n_crystfel_fail)
fail_x = fail_r * np.cos(fail_theta)
fail_y = fail_r * np.sin(fail_theta)

# --------------------------
# Plot
# --------------------------
plt.close("all")
fig, ax = plt.subplots(figsize=(7.8, 7.2))

# Heatmap
im = ax.imshow(
    Z, extent=[-extent, extent, -extent, extent], origin="lower",
    cmap="viridis", interpolation="bilinear", alpha=0.95
)

# Contours for crisp shape perception
levels = np.linspace(0.05, 0.95, 8)
cs = ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.9, alpha=0.65)

# Scatter overlays
# ax.scatter(wr_x, wr_y, s=46, c="#2ca02c", edgecolors="white", linewidths=0.6, label="wRMSD refinement")
ax.scatter(cf_x, cf_y, s=46, c="#d62728", edgecolors="white", linewidths=0.6, label="CrystFEL refined")
ax.scatter(fail_x, fail_y, s=70, c="#7f7f7f", marker="x", linewidths=2.0, label="CrystFEL fail")

# True center
ax.scatter([x_true], [y_true], s=140, c="#222222", marker="*", label="True center", zorder=4)

# Aesthetics for Keynote
ax.set_xlabel(r'$\Delta x$ (pixels)')
ax.set_ylabel(r'$\Delta y$ (pixels)')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-extent, extent)
ax.set_ylim(-extent, extent)

# Minimal axes style
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('wRMSD (arb.)')

leg = ax.legend(loc="upper right", frameon=True, framealpha=0.9)
for txt in leg.get_texts():
    txt.set_fontsize(10)

plt.tight_layout()
fig.savefig(png_path, dpi=dpi, transparent=True)
fig.savefig(pdf_path)
print(f"Saved:\n  - {png_path}\n  - {pdf_path}")

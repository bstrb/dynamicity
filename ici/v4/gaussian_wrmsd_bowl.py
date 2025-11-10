#!/usr/bin/env python3
"""
3D Gaussian "wRMSD bowl" plot with viridis colormap.
Saves as high-resolution PNG for slides.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Parameters ---
cx, cy = 0.0, 0.0      # center of the bowl (optimal wRMSD point)
sigma = 1.5            # width of Gaussian
extent = (-5, 5, -5, 5)  # plotting range in x/y
grid_n = 220           # surface resolution
elev, azim = 28, -55   # viewing angle
dpi = 160              # image quality
outfile = "wrmsd_bowl_viridis.png"

# --- Generate surface ---
x = np.linspace(extent[0], extent[1], grid_n)
y = np.linspace(extent[2], extent[3], grid_n)
X, Y = np.meshgrid(x, y)
R2 = (X - cx)**2 + (Y - cy)**2

# Gaussian "bowl" potential
Z = 1.0 - np.exp(-R2 / (2 * sigma**2))

# Find minimum for marking
min_idx = np.unravel_index(np.argmin(Z), Z.shape)
x_min, y_min, z_min = X[min_idx], Y[min_idx], Z[min_idx]

# --- Plot ---
fig = plt.figure(figsize=(7.5, 5.8), dpi=dpi)
ax = fig.add_subplot(111, projection="3d")

# Use viridis colormap
surf = ax.plot_surface(
    X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
    rstride=2, cstride=2
)

# Highlight the minimum
# ax.scatter([x_min], [y_min], [z_min], color="red", marker="*", s=120, depthshade=False)

# Clean labels for slides
ax.set_xlabel("Δx (pixels)")
ax.set_ylabel("Δy (pixels)")
ax.set_zlabel("wRMSD potential")
ax.view_init(elev=elev, azim=azim)

# Add colorbar if desired
fig.colorbar(surf, shrink=0.6, aspect=10, label="wRMSD value")

plt.tight_layout()
plt.savefig(outfile, bbox_inches="tight", dpi=dpi)
plt.close(fig)

print(f"Saved 3D viridis bowl to: {outfile}")

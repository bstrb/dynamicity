#!/usr/bin/env python3
# Fit Fc^2 vs Fo^2 from SHELX CIF-style .fcf and list outliers wrt the fitted line.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Configuration -----------------------
FCF_PATH = Path("/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/zero_fitering/filtered_metrics_sorted_50000_first_xtals_noZOLZ_tol0.1_merge/shelx/shelx.fcf")  # Path to your .fcf file
TOP_N = 20                    # How many outliers to list/label
FORCE_THROUGH_ORIGIN = False  # If True: fit y = b*x; else y = a + b*x
ROBUST = True                 # Use IRLS with Tukey biweight for robustness
MAX_IRLS_ITERS = 5            # IRLS iterations
C_TUKEY = 4.685               # Tukey biweight tuning constant
POINT_SIZE = 6
ALPHA = 0.5
LABEL_FONT_SIZE = 8
# -------------------------------------------------------------

def parse_fcf_cif_style(fcf_path: Path):
    """
    Parses a CIF-style .fcf with reflection loop:
      _refln_index_h, _refln_index_k, _refln_index_l,
      _refln_F_squared_meas, _refln_F_squared_sigma, _refln_F_calc, _refln_phase_calc
    Returns: hkl (N,3), Fo2 (N,), sigFo2 (N,), Fc2 (N,)
    """
    if not fcf_path.exists():
        raise FileNotFoundError(f"File not found: {fcf_path}")

    hkl, Fo2, sigFo2, Fc2 = [], [], [], []
    in_loop = False
    headers = []

    with fcf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower() == "loop_":
                in_loop = False
                headers = []
                continue
            if line.startswith("_"):
                headers.append(line)
                if ("_refln_index_h" in headers and
                    "_refln_F_squared_meas" in headers and
                    "_refln_F_calc" in headers):
                    in_loop = True
                continue
            if in_loop:
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    h, k, l = map(int, parts[:3])
                    fo2 = float(parts[3])
                    sfo2 = float(parts[4])
                    fc_amp = float(parts[5])     # amplitude
                    fc2 = fc_amp * fc_amp        # Fc^2
                except ValueError:
                    continue
                if not np.isfinite(fo2) or not np.isfinite(fc2):
                    continue
                hkl.append((h, k, l))
                Fo2.append(fo2)
                sigFo2.append(sfo2 if sfo2 > 0 else np.nan)
                Fc2.append(fc2)

    if not Fo2:
        raise RuntimeError("No reflections parsed from .fcf")

    return (np.array(hkl, dtype=int),
            np.array(Fo2, dtype=float),
            np.array(sigFo2, dtype=float),
            np.array(Fc2, dtype=float))

def weighted_linfit(x, y, w=None, through_origin=False):
    """
    Weighted linear regression.
    If through_origin=True: y = b*x
    else: y = a + b*x
    Returns (a, b)
    """
    x = np.asarray(x); y = np.asarray(y)
    if w is None:
        w = np.ones_like(x)
    W = np.diag(w)

    if through_origin:
        # b = sum(w*x*y)/sum(w*x^2)
        denom = np.sum(w * x * x)
        b = np.sum(w * x * y) / denom if denom > 0 else 0.0
        a = 0.0
        return a, b
    else:
        # Solve [1 x] with weights
        X = np.column_stack([np.ones_like(x), x])
        # Normal equations: (X^T W X) beta = X^T W y
        XT_W = X.T * w
        beta = np.linalg.solve(XT_W @ X, XT_W @ y)
        a, b = beta[0], beta[1]
        return a, b

def tukey_weights(r, c=C_TUKEY):
    """
    Tukey's biweight: w = (1 - u^2)^2 for |u|<1, else 0
    where u = r / (c * MAD)
    """
    r = np.asarray(r)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    # Guard against zero MAD:
    scale = c * (mad if mad > 1e-12 else np.std(r) + 1e-12)
    u = (r - med) / scale
    w = np.zeros_like(r)
    mask = np.abs(u) < 1.0
    w[mask] = (1 - u[mask]**2)**2
    return w

def robust_fit(x, y, through_origin=False, max_iters=5):
    """
    Iteratively Reweighted Least Squares with Tukey's biweight.
    Returns (a, b), final weights, residuals
    """
    # Start with unweighted fit
    a, b = weighted_linfit(x, y, w=None, through_origin=through_origin)
    w = np.ones_like(x)
    for _ in range(max_iters):
        y_pred = a + b * x
        r = y - y_pred
        w = tukey_weights(r)
        # If all weights ~0 (extreme), fall back to unweighted
        if np.all(w < 1e-6):
            w = np.ones_like(x)
        a_new, b_new = weighted_linfit(x, y, w=w, through_origin=through_origin)
        if np.allclose([a, b], [a_new, b_new], rtol=0, atol=1e-9):
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    y_pred = a + b * x
    r = y - y_pred
    return (a, b), w, r

def robust_z_scores(residuals):
    """
    Robust z using MAD: z = 0.6745 * |r - med(r)| / MAD
    """
    r = np.asarray(residuals)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad <= 0:
        # Fall back to std if MAD is zero
        s = np.std(r) + 1e-12
        return np.abs(r - med) / s
    return 0.6745 * np.abs(r - med) / mad

def main():
    hkl, Fo2, sigFo2, Fc2 = parse_fcf_cif_style(FCF_PATH)

    x = Fo2
    y = Fc2

    # Fit
    if ROBUST:
        (a, b), weights, residuals = robust_fit(x, y, through_origin=FORCE_THROUGH_ORIGIN, max_iters=MAX_IRLS_ITERS)
    else:
        a, b = weighted_linfit(x, y, w=None, through_origin=FORCE_THROUGH_ORIGIN)
        residuals = y - (a + b * x)
        weights = np.ones_like(x)

    # Goodness-of-fit (R^2) for reporting (unweighted):
    y_pred = a + b * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    r2 = 1.0 - ss_res/ss_tot

    # Outlier scoring
    z = robust_z_scores(residuals)
    order = np.argsort(-z)  # descending by robust z
    top_idx = order[:TOP_N]

    # ---- Plot ----
    xmin, xmax = 0.0, max(1.0, np.max(x))
    ymin, ymax = 0.0, max(1.0, np.max(y))
    diag_max = max(xmax, ymax)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=POINT_SIZE, alpha=ALPHA)
    # Fitted line
    xline = np.linspace(0, diag_max, 200)
    yline = a + b * xline
    plt.plot(xline, yline, linestyle="--", label=f"fit: y = {a:.3f} + {b:.3f} x  (R²={r2:.4f})")

    # Axes start from zero and equal aspect
    plt.xlim(0, 1500)
    plt.ylim(0, 2500)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("Fo² (observed)")
    plt.ylabel("Fc² (calculated)")
    plt.title(f"Fo² vs Fc² fit • {FCF_PATH.name}")

    # Label top outliers
    for i in top_idx:
        hx, hy = x[i], y[i]
        label = f"({hkl[i,0]} {hkl[i,1]} {hkl[i,2]})"
        plt.annotate(label, (hx, hy), textcoords="offset points", xytext=(4, 4),
                     fontsize=LABEL_FONT_SIZE)

    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Print table ----
    print("\nTop {} outliers w.r.t. fitted line (by robust z-score, MAD-based):".format(TOP_N))
    header = (
        "{:>3s} {:>4s} {:>4s} {:>4s} "
        "{:>12s} {:>12s} {:>12s} {:>10s} {:>8s}"
        .format("#", "h", "k", "l", "Fo²", "Fc²", "y_fit", "resid", "z_rob")
    )
    print(header)
    print("-" * len(header))
    for rank, i in enumerate(top_idx, 1):
        yfit_i = a + b * x[i]
        print("{:>3d} {:>4d} {:>4d} {:>4d} {:>12.3f} {:>12.3f} {:>12.3f} {:>10.3f} {:>8.2f}".format(
            rank, hkl[i,0], hkl[i,1], hkl[i,2],
            x[i], y[i], yfit_i, residuals[i], z[i]
        ))

    # Convenience hint
    if len(top_idx) > 0:
        h,k,l = hkl[top_idx[0]]
        print(f"\nTip: In Olex2, omit a reflection via:  omit {h} {k} {l}")

if __name__ == "__main__":
    main()

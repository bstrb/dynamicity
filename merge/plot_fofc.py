#!/usr/bin/env python3
# Fit Fc^2 vs Fo^2 from SHELX CIF-style .fcf and list outliers wrt the fitted line.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Configuration -----------------------
FCF_PATH = Path("/Users/xiaodong/Desktop/MFM300V-III_ZOLZ_ref/MFM300V-III_no_ZOLZ_4-0.4/shelx.fcf")  # Path to your .fcf file
TOP_N = 20                    # How many outliers to list/label
FORCE_THROUGH_ORIGIN = False  # If True: fit y = b*x; else y = a + b*x
ROBUST = True                 # Use IRLS with Tukey biweight for robustness
MAX_IRLS_ITERS = 5            # IRLS iterations
C_TUKEY = 4.685               # Tukey biweight tuning constant
POINT_SIZE = 6
ALPHA = 0.5
LABEL_FONT_SIZE = 8
# -------------------------------------------------------------

def _to_float(tok):
    """Parse CIF numeric token, treating '?' and '.' as NaN."""
    if tok in ("?", "."):
        return np.nan
    return float(tok)

def parse_fcf_cif_style(fcf_path: Path):
    """
    Parses a CIF-style .fcf reflection loop. Accepts both:
      - (_refln_F_squared_calc, _refln_F_squared_meas, _refln_F_squared_sigma, ...)
      - (_refln_F_calc [amplitude], _refln_F_squared_meas, _refln_F_squared_sigma, ...)

    Returns: hkl (N,3), Fo2 (N,), sigFo2 (N,), Fc2 (N,)
    """
    if not fcf_path.exists():
        raise FileNotFoundError(f"File not found: {fcf_path}")

    with fcf_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    hkl, Fo2, sigFo2, Fc2 = [], [], [], []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        # Look for a reflection loop
        if line.lower() == "loop_":
            i += 1
            # Gather header lines that start with "_"
            headers = []
            while i < n and lines[i].lstrip().startswith("_"):
                headers.append(lines[i].strip())
                i += 1

            lower = [h.lower() for h in headers]
            colmap = {lower[j]: j for j in range(len(lower))}

            # Check if this loop contains h,k,l and F columns we need
            required_base = [
                "_refln_index_h",
                "_refln_index_k",
                "_refln_index_l",
                "_refln_f_squared_meas",
            ]
            has_required_base = all(req in colmap for req in required_base)
            has_fc2 = "_refln_f_squared_calc" in colmap
            has_fc_amp = "_refln_f_calc" in colmap  # amplitude

            if has_required_base and (has_fc2 or has_fc_amp):
                # Optional sigma column
                idx_sigma = colmap.get("_refln_f_squared_sigma")
                # Start consuming data rows for THIS loop until a new loop_ or header starts
                while i < n:
                    s = lines[i].strip()
                    if not s or s.startswith("#"):
                        i += 1
                        continue
                    if s.lower() == "loop_" or s.startswith("_") or s.lower().startswith("data_"):
                        # Next section; stop reading rows for this loop
                        break

                    parts = s.split()
                    # Must have at least as many tokens as the last column index referenced
                    max_needed = max(colmap.get("_refln_index_h", 0),
                                     colmap.get("_refln_index_k", 0),
                                     colmap.get("_refln_index_l", 0),
                                     colmap.get("_refln_f_squared_meas", 0),
                                     colmap.get("_refln_f_squared_calc", 0) if has_fc2 else 0,
                                     colmap.get("_refln_f_calc", 0) if has_fc_amp else 0,
                                     idx_sigma if idx_sigma is not None else 0)
                    if len(parts) <= max_needed:
                        i += 1
                        continue  # malformed row (or status flag collapsed); skip safely

                    try:
                        h = int(parts[colmap["_refln_index_h"]])
                        k = int(parts[colmap["_refln_index_k"]])
                        l = int(parts[colmap["_refln_index_l"]])

                        fo2 = _to_float(parts[colmap["_refln_f_squared_meas"]])

                        if has_fc2:
                            fc2 = _to_float(parts[colmap["_refln_f_squared_calc"]])
                        else:
                            fc_amp = _to_float(parts[colmap["_refln_f_calc"]])
                            fc2 = fc_amp * fc_amp if np.isfinite(fc_amp) else np.nan

                        sfo2 = (_to_float(parts[idx_sigma]) if idx_sigma is not None
                                else np.nan)
                    except Exception:
                        # Any parsing hiccup? Skip this line.
                        i += 1
                        continue

                    if np.isfinite(fo2) and np.isfinite(fc2):
                        hkl.append((h, k, l))
                        Fo2.append(float(fo2))
                        sigFo2.append(float(sfo2) if (sfo2 is not None) else np.nan)
                        Fc2.append(float(fc2))
                    i += 1

                # Continue scanning file (there may be more loops, but we already collected compatible rows)
                continue

            # Not the reflection loop we need; move on
            else:
                continue

        i += 1

    if not Fo2:
        raise RuntimeError("No reflections parsed from .fcf (check that your file has a CIF loop with Fo² and Fc or Fc²).")

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
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    if through_origin:
        denom = np.sum(w * x * x)
        b = np.sum(w * x * y) / denom if denom > 0 else 0.0
        a = 0.0
        return a, b
    else:
        X = np.column_stack([np.ones_like(x), x])
        # Use weighted normal equations in a numerically safe way
        WX = X * w[:, None]
        XT_WX = X.T @ WX
        XT_Wy = WX.T @ y
        a, b = np.linalg.solve(XT_WX, XT_Wy)
        return a, b

def tukey_weights(r, c=C_TUKEY):
    """
    Tukey's biweight: w = (1 - u^2)^2 for |u|<1, else 0
    where u = r / (c * MAD)
    """
    r = np.asarray(r, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    # Guard against zero MAD:
    scale = c * (mad if mad > 1e-12 else (np.std(r) + 1e-12))
    if scale <= 0:
        return np.ones_like(r)
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
    a, b = weighted_linfit(x, y, w=None, through_origin=through_origin)
    w = np.ones_like(x, dtype=float)
    for _ in range(max_iters):
        y_pred = a + b * x
        r = y - y_pred
        w_new = tukey_weights(r)
        if np.all(w_new < 1e-6):
            w_new = np.ones_like(x, dtype=float)
        a_new, b_new = weighted_linfit(x, y, w=w_new, through_origin=through_origin)
        if np.allclose([a, b], [a_new, b_new], rtol=0, atol=1e-9):
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
        w = w_new
    y_pred = a + b * x
    r = y - y_pred
    return (a, b), w, r

def robust_z_scores(residuals):
    """
    Robust z using MAD: z = 0.6745 * |r - med(r)| / MAD
    """
    r = np.asarray(residuals, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad <= 0:
        s = np.std(r) + 1e-12
        return np.abs(r - med) / s
    return 0.6745 * np.abs(r - med) / mad

def main():
    hkl, Fo2, sigFo2, Fc2 = parse_fcf_cif_style(FCF_PATH)

    x = Fo2
    y = Fc2

    # Fit
    if ROBUST:
        (a, b), weights, residuals = robust_fit(
            x, y, through_origin=FORCE_THROUGH_ORIGIN, max_iters=MAX_IRLS_ITERS
        )
    else:
        a, b = weighted_linfit(x, y, w=None, through_origin=FORCE_THROUGH_ORIGIN)
        residuals = y - (a + b * x)
        weights = np.ones_like(x, dtype=float)

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
    # plt.xlim(0, 2000)
    # plt.ylim(0, 3500)
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
        
    for rank, i in enumerate(top_idx, 1):
        print("OMIT {:>4d} {:>4d} {:>4d}".format(
            hkl[i,0], hkl[i,1], hkl[i,2],
        ))

if __name__ == "__main__":
    main()

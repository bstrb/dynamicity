
import math
import numpy as np
from math import erf, sqrt

class BO2DConfig:
    def __init__(self, lsx=0.02, lsy=0.02, noise=1e-4, candidates=800, ei_eps=1e-3, rng_seed=1337):
        self.lsx = float(lsx)
        self.lsy = float(lsy)
        self.noise = float(noise)
        self.candidates = int(candidates)
        self.ei_eps = float(ei_eps)
        self.rng = np.random.default_rng(int(rng_seed))

# --------- Gaussian Process (RBF anisotropic) + EI ---------

def _rbf_aniso(X1: np.ndarray, X2: np.ndarray, lsx: float, lsy: float) -> np.ndarray:
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))

def _gp_fit_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, lsx: float, lsy: float, noise: float):
    # Center y for stability
    import numpy as _np
    ymean = float(_np.mean(y))
    yc = y - ymean
    K = _rbf_aniso(X, X, lsx, lsy) + (noise * _np.eye(len(X)))
    # Cholesky with tiny jitter fallback
    jitter = 1e-12
    try:
        L = _np.linalg.cholesky(K)
    except _np.linalg.LinAlgError:
        L = _np.linalg.cholesky(K + jitter * _np.eye(len(X)))
    alpha = _np.linalg.solve(L.T, _np.linalg.solve(L, yc))
    Ks = _rbf_aniso(X, Xstar, lsx, lsy)
    mu = Ks.T @ alpha + ymean
    v = _np.linalg.solve(L, Ks)
    # Kernel amplitude assumed 1.0
    var = _np.maximum(0.0, 1.0 - _np.sum(v * v, axis=0))
    return mu, var

def _phi(z: np.ndarray) -> np.ndarray:
    # standard normal PDF
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * (z * z))

# def _Phi(z: np.ndarray) -> np.ndarray:
#     # standard normal CDF using erf
#     return 0.5 * (1.0 + np.erf(z / math.sqrt(2.0)))

def _Phi(z: np.ndarray) -> np.ndarray:
    # use math.erf element-wise (NumPy lacks np.erf on some builds)
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))

def expected_improvement(mu: np.ndarray, var: np.ndarray, y_best: float, xi: float = 0.0) -> np.ndarray:
    # Minimize: improvement = y_best - Y
    std = np.sqrt(np.maximum(var, 1e-16))
    z = (y_best - mu - xi) / std
    ei = (y_best - mu - xi) * _Phi(z) + std * _phi(z)
    ei[var < 1e-30] = 0.0
    return ei

def bo2d_propose(tried_xy: np.ndarray,
                 tried_wrmsd: np.ndarray,
                 bounds: tuple,
                 config: BO2DConfig,
                 tried_tol: float = 1e-6):
    """
    Args:
        tried_xy: (n,2) past (dx,dy) in mm (finite wrmsd only).
        tried_wrmsd: (n,) past wrmsd values (lower is better).
        bounds: ((xmin,xmax),(ymin,ymax))
        config: BO2DConfig
        tried_tol: distance threshold to consider a point already tried.

    Returns:
        (next_xy, ei_max): tuple, or (None, 0.0) if no improvement expected.
    """
    assert tried_xy.ndim == 2 and tried_xy.shape[1] == 2, "tried_xy must be (n,2)"
    assert tried_wrmsd.ndim == 1 and tried_wrmsd.shape[0] == tried_xy.shape[0], "shape mismatch"

    (xmin, xmax), (ymin, ymax) = bounds
    X = tried_xy.astype(float)
    y = tried_wrmsd.astype(float)

    # Fit GP on tried points
    y_best = float(np.min(y))

    # Sample random candidates
    C = max(10, int(config.candidates))
    xs = config.rng.uniform(xmin, xmax, C)
    ys = config.rng.uniform(ymin, ymax, C)
    Xc = np.column_stack([xs, ys])

    # Drop candidates too close to tried points
    if len(X) > 0:
        d2 = np.sum((Xc[:, None, :] - X[None, :, :]) ** 2, axis=2)
        mind = np.sqrt(np.min(d2, axis=1))
        mask = mind > tried_tol
        Xc = Xc[mask]
        if Xc.shape[0] == 0:
            return None, 0.0

    mu, var = _gp_fit_predict(X, y, Xc, config.lsx, config.lsy, config.noise)
    ei = expected_improvement(mu, var, y_best, xi=0.0)

    if ei.size == 0:
        return None, 0.0

    j = int(np.argmax(ei))
    ei_max = float(ei[j])
    next_xy = (float(Xc[j, 0]), float(Xc[j, 1]))
    return next_xy, ei_max

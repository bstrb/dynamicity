"""Self-excitation and two-beam-like geometry proxy terms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OridynConfig
from .geometry import low_order_prior_from_q


def add_self_risk_terms(scores: pd.DataFrame, config: OridynConfig) -> pd.DataFrame:
    """Add geometry-only self-coupling proxy columns."""

    if scores.empty:
        return scores.copy()
    out = scores.copy()
    coupling = low_order_prior_from_q(out["q_invA"].to_numpy(dtype=float), config.low_order_g0_invA, config.low_order_power)
    sg = out["sg"].to_numpy(dtype=float)
    excitation = out["excitation_weight"].to_numpy(dtype=float)
    xi_proxy = 1.0 / (max(config.sg0, 1e-12) * np.maximum(coupling, 1e-6))
    two_beam = coupling / (1.0 + (sg * xi_proxy) ** 2)
    self_excitation = excitation * coupling
    out["coupling_prior"] = coupling
    out["xi_proxy"] = xi_proxy
    out["self_excitation_score"] = self_excitation
    out["two_beam_proxy_risk"] = two_beam
    out["self_risk_raw"] = 0.5 * (self_excitation + two_beam)
    return out

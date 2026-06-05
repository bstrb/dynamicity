import pandas as pd

from oridyn.config import OridynConfig
from oridyn.self_risk import add_self_risk_terms


def test_self_risk_terms_are_geometry_only_and_bounded_positive():
    scores = pd.DataFrame({"q_invA": [0.1, 1.0], "sg": [0.0, 0.1], "excitation_weight": [1.0, 0.2]})
    out = add_self_risk_terms(scores, OridynConfig(sg0=0.02))

    assert {"coupling_prior", "xi_proxy", "self_excitation_score", "two_beam_proxy_risk", "self_risk_raw"} <= set(
        out.columns
    )
    assert out.loc[0, "coupling_prior"] > out.loc[1, "coupling_prior"]
    assert (out["self_risk_raw"] >= 0.0).all()

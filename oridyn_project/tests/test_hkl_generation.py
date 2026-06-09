from oridyn.hkl_generation import allowed_by_centering, generate_candidate_hkls
from oridyn.stream_parser import UnitCell


def test_candidate_generation_and_centering():
    cell = UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0, centering="P")
    candidates, metadata = generate_candidate_hkls(cell, dmin=2.0, dmax=10.0)

    assert ((candidates["h"] == 1) & (candidates["k"] == 0) & (candidates["l"] == 0)).any()
    assert metadata["n_candidates"] == len(candidates)
    assert allowed_by_centering(1, 1, 0, "I")
    assert not allowed_by_centering(1, 0, 0, "I")

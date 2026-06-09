from pathlib import Path

from oridyn.stream_parser import STREAM_MATRIX_COLUMNS, parse_crystfel_stream


def test_parse_minimal_stream():
    stream = parse_crystfel_stream(Path("examples") / "minimal_example.stream")

    assert stream.wavelength_angstrom == 0.0251
    assert stream.unit_cell.a == 10.0
    assert stream.unit_cell.centering == "P"
    assert len(stream.crystal_table) == 2
    assert len(stream.reflections) == 6
    assert set(STREAM_MATRIX_COLUMNS) <= set(stream.crystal_table.columns)
    assert {"h", "k", "l", "sigma", "fs_px", "ss_px"} <= set(stream.reflections.columns)

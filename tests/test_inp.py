"""Tests for write_inp_file()'s .inp text generation (no CLI binary needed)."""

import os

from pymstm._inp import write_inp_file
from pymstm._inp_parser import parse_inp_text


def _write(tmp_path, **kwargs):
    inp = os.path.join(tmp_path, "test.inp")
    write_inp_file(
        inp,
        radii=[1.0],
        positions=[[0.0, 0.0, 0.0]],
        ref_re=[1.5],
        ref_im=[0.01],
        **kwargs,
    )
    with open(inp) as f:
        return f.read()


def test_scattering_map_model_default_when_dimension_set(tmp_path):
    """scattering_map_dimension alone should auto-enable model 1 -- prior
    to this fix, setting scattering_map_dimension had no effect at all,
    since MSTM only honors it when scattering_map_model != 0."""
    text = _write(tmp_path, scattering_map_dimension=10)
    assert "scattering_map_dimension\n10" in text
    assert "scattering_map_model\n1" in text


def test_scattering_map_model_explicit_zero_kept(tmp_path):
    """Caller's explicit scattering_map_model=0 must be honored, not
    silently overridden to 1 just because scattering_map_dimension is
    also set."""
    text = _write(tmp_path, scattering_map_dimension=10, scattering_map_model=0)
    assert "scattering_map_dimension\n10" in text
    assert "scattering_map_model\n0" in text


def test_scattering_map_model_alone(tmp_path):
    text = _write(tmp_path, scattering_map_model=1)
    assert "scattering_map_model\n1" in text
    assert "scattering_map_dimension" not in text


def test_scattering_map_neither_set_by_default(tmp_path):
    text = _write(tmp_path)
    assert "scattering_map_model" not in text
    assert "scattering_map_dimension" not in text


def test_random_orientation(tmp_path):
    text = _write(tmp_path, random_orientation=True)
    assert "random_orientation\nt" in text


def test_random_orientation_default_off(tmp_path):
    text = _write(tmp_path)
    assert "random_orientation" not in text


def test_incidence_average(tmp_path):
    text = _write(tmp_path, incidence_average=True)
    assert "incidence_average\nt" in text
    assert "number_incident_directions" not in text


def test_incidence_average_with_number_directions(tmp_path):
    text = _write(tmp_path, incidence_average=True, number_incident_directions=32)
    assert "incidence_average\nt" in text
    assert "number_incident_directions\n32" in text


def test_number_incident_directions_ignored_without_incidence_average(tmp_path):
    """number_incident_directions is only meaningful alongside
    incidence_average=True -- confirm it's not written on its own."""
    text = _write(tmp_path, number_incident_directions=32)
    assert "incidence_average" not in text
    assert "number_incident_directions" not in text


# ---------------------------------------------------------------------------
# Round-trip: write_inp_file() -> parse_inp_text() -> to_pymstm_args()
# ---------------------------------------------------------------------------


def test_roundtrip_random_orientation_and_incidence_average(tmp_path):
    inp = os.path.join(tmp_path, "test.inp")
    write_inp_file(
        inp,
        radii=[1.0],
        positions=[[0.0, 0.0, 0.0]],
        ref_re=[1.5],
        ref_im=[0.01],
        random_orientation=True,
        incidence_average=True,
        number_incident_directions=32,
    )
    with open(inp) as f:
        config = parse_inp_text(f.read())

    assert config.random_orientation is True
    assert config.incidence_average is True
    assert config.number_incident_directions == 32

    args = config.to_pymstm_args()
    assert args["random_orientation"] is True
    assert args["incidence_average"] is True
    assert args["number_incident_directions"] == 32


def test_roundtrip_scattering_map(tmp_path):
    inp = os.path.join(tmp_path, "test.inp")
    write_inp_file(
        inp,
        radii=[1.0],
        positions=[[0.0, 0.0, 0.0]],
        ref_re=[1.5],
        ref_im=[0.01],
        scattering_map_dimension=15,
    )
    with open(inp) as f:
        config = parse_inp_text(f.read())

    assert config.scattering_map_dimension == 15
    assert config.scattering_map_model == 1  # auto-enabled by write_inp_file

    args = config.to_pymstm_args()
    assert args["scattering_map_dimension"] == 15
    assert args["scattering_map_model"] == 1


def test_roundtrip_defaults_off(tmp_path):
    inp = os.path.join(tmp_path, "test.inp")
    write_inp_file(
        inp,
        radii=[1.0],
        positions=[[0.0, 0.0, 0.0]],
        ref_re=[1.5],
        ref_im=[0.01],
    )
    with open(inp) as f:
        config = parse_inp_text(f.read())

    assert config.random_orientation is False
    assert config.incidence_average is False
    assert config.scattering_map_dimension is None
    assert config.scattering_map_model == 0

    args = config.to_pymstm_args()
    assert args["random_orientation"] is False
    assert args["incidence_average"] is False
    assert "number_incident_directions" not in args
    assert "scattering_map_dimension" not in args
    assert "scattering_map_model" not in args

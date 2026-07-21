"""Tests for scattering-matrix normalization helpers (pymstm._convert)."""

import numpy as np
import pytest
from pymstm._convert import cli_normalized_s11_to_raw, s11_to_phase_function


def test_s11_to_phase_function_integrates_to_4pi():
    """A properly converted, isotropic-equivalent S11 (constant over the
    sphere) should scale so the solid-angle integral is 4*pi -- check the
    scaling factor directly rather than doing a full angular integral."""
    k = 12.566  # 2*pi/0.5
    c_sca = 5.0
    s11 = np.array([1.0, 2.0, 3.0])
    p = s11_to_phase_function(s11, k, c_sca)
    assert p == pytest.approx(s11 * 4.0 * np.pi / (k**2 * c_sca))


def test_s11_to_phase_function_scalar_and_array():
    k, c_sca = 10.0, 2.0
    scalar = s11_to_phase_function(5.0, k, c_sca)
    array = s11_to_phase_function([5.0], k, c_sca)
    assert float(scalar) == pytest.approx(float(array[0]))


def test_cli_normalized_s11_to_raw_default_correction():
    raw = cli_normalized_s11_to_raw(np.array([2 * np.pi, 4 * np.pi]))
    assert raw == pytest.approx([1.0, 2.0])


def test_cli_normalized_s11_to_raw_custom_correction():
    raw = cli_normalized_s11_to_raw(np.array([10.0]), correction=5.0)
    assert raw == pytest.approx([2.0])


def test_round_trip_cli_to_phase_function():
    """CLI (normalize_s11=False) -> raw -> phase function should match the
    direct binding-path conversion for the same underlying physical S11."""
    k, c_sca = 12.566, 3.0
    raw_s11 = 7.5
    cli_s11 = raw_s11 * 2 * np.pi  # what the CLI would report, pre-correction
    recovered_raw = cli_normalized_s11_to_raw(cli_s11)
    p_from_cli = s11_to_phase_function(recovered_raw, k, c_sca)
    p_direct = s11_to_phase_function(raw_s11, k, c_sca)
    assert float(p_from_cli) == pytest.approx(float(p_direct))

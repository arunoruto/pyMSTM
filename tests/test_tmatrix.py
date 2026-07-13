"""Tests for T-matrix computation and random orientation scattering."""

import os
import numpy as np
import pytest
from pymstm import MSTM


@pytest.fixture
def dimer_mstm():
    m = MSTM()
    m.set_spheres(
        radii=[3.0, 3.0],
        positions=[[0, 0, -3], [0, 0, 3]],
        orders=[5, 5],
        ref_re=[1.5, 1.5],
        ref_im=[0.01, 0.01],
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=500)
    m.prepare()
    return m


def test_compute_tmatrix(dimer_mstm):
    result = dimer_mstm.compute_tmatrix()
    assert result["status"] == 0
    assert result["tmatrix_order"] > 0
    assert result["tmatrix_size"] > 0

    # T-matrix should have non-zero norm
    tdata = result["tmatrix"]
    norm2 = np.sum(tdata[0::2] ** 2 + tdata[1::2] ** 2)
    assert norm2 > 1.0

    # Per-sphere efficiencies should be reasonable
    assert np.all(result["q_ext"] > 0)
    assert np.all(result["q_abs"] >= 0)

    dimer_mstm.finalize()


def test_ranorient_smatrix(dimer_mstm):
    dimer_mstm.compute_tmatrix()

    tmf = "tmatrix_temp.dat"
    if os.path.exists(tmf):
        m2 = MSTM()
        result = m2.ranorient_smatrix(tmf)
        assert result["tmatrix_order"] > 0
        assert len(result["sm_coef"]) == 16 * (2 * result["tmatrix_order"] + 1)

        # Evaluate at a few angles
        for ct in [1.0, 0.5, 0.0, -0.5, -1.0]:
            sm = m2.ranorient_smatrix_at_angle(
                result["sm_coef"], result["tmatrix_order"], ct
            )
            assert sm.shape == (16,)
            assert not np.any(np.isnan(sm))

        m2.finalize()

    dimer_mstm.finalize()

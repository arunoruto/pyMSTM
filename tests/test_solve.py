"""Tests for fixed-orientation scattering."""

import numpy as np
import pytest
from pymstm import MstmBindings


def test_single_sphere_solve():
    """Single dielectric sphere: should converge quickly."""
    m = MstmBindings()
    m.set_spheres(
        radii=[5.0],
        positions=[[0, 0, 0]],
        orders=[12],
        ref_re=[1.5],
        ref_im=[0.0],
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.prepare()

    result = m.solve()

    assert result["status"] == 0, "Solver did not converge"
    assert not np.any(np.isnan(result["q_ext"])), "q_ext contains NaN"
    assert result["qext_tot"] > 0, "Q_ext_tot should be positive"
    assert result["iterations"] <= 2, "Single sphere should converge in 1-2 iterations"

    m.finalize()


def test_two_sphere_solve():
    """Two well-separated spheres: weak interaction, fast convergence.

    Separation is 200 (radius 3, so ~33 diameters) along the same axis as
    the incident direction, not the original 10 (~1.3 diameters) -- with
    the spheres nearly touching along the propagation axis, one sits in
    the other's forward-scattered near field, and that asymmetry decays
    slowly with distance (confirmed directly: ratio 0.66 at separation
    10, 0.86 at 50, 0.96 at 200) rather than vanishing once "far enough
    apart" in some small multiple of the radius. This is real near-field
    physics, not a solver bug -- picked 200 for a comfortable margin
    inside the tolerance below without making the case toweringly large.
    """
    m = MstmBindings()
    m.set_spheres(
        radii=[3.0, 3.0],
        positions=[[0, 0, -100], [0, 0, 100]],
        orders=[6, 6],
        ref_re=[1.5, 1.5],
        ref_im=[0.0, 0.0],
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.prepare()

    result = m.solve()

    assert result["status"] == 0
    assert not np.any(np.isnan(result["q_ext"]))
    assert result["qext_tot"] > 0

    # Weakly interacting: per-sphere values should be similar
    ratio = result["q_ext"][0] / result["q_ext"][1]
    assert 0.8 < ratio < 1.25, f"Per-sphere Q_ext differ too much: {result['q_ext']}"

    m.finalize()


def test_scattering_matrix():
    """Scattering matrix should have valid values."""
    m = MstmBindings()
    m.set_spheres(
        radii=[5.0],
        positions=[[0, 0, 0]],
        orders=[12],
        ref_re=[1.5],
        ref_im=[0.0],
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.prepare()
    m.solve()

    sm = m.get_scattering_angle(costheta=1.0, phi=0.0)
    assert sm.shape == (16,)
    assert not np.any(np.isnan(sm))
    assert sm[0] > 0  # S11 should be positive

    with pytest.warns(RuntimeWarning, match="non-deterministic"):
        ct, smat = m.get_scattering_matrix()
    assert smat.shape[1] > 0
    assert not np.any(np.isnan(smat))

    m.finalize()

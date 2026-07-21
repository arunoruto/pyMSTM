"""Tests for advanced features: layers, lattice."""

import numpy as np
from pymstm import MstmBindings


def test_single_layer():
    """Sphere above a dielectric half-space."""
    m = MstmBindings()
    m.set_spheres(
        radii=[2.0],
        positions=[[0, 0, 10]],
        orders=[5],
        ref_re=[2.0],
        ref_im=[0.0],
    )
    m.set_layers(
        thicknesses=[],
        ref_indices=[(1.0, 0.0), (1.5, 0.0)],  # vacuum above, glass below
    )
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.prepare()
    result = m.solve()
    assert result["status"] == 0
    assert not np.any(np.isnan(result["q_ext"]))
    assert result["qext_tot"] > 0
    m.finalize()


def test_multi_layer():
    """3-layer system: vacuum | film | substrate."""
    m = MstmBindings()
    m.set_spheres(
        radii=[1.5],
        positions=[[0, 0, 1.5]],
        orders=[4],
        ref_re=[1.8],
        ref_im=[0.0],
    )
    m.set_layers(
        thicknesses=[1.0, 3.0],  # film thickness=1, substrate at z=4
        ref_indices=[
            (1.0, 0.0),  # above (vacuum)
            (1.4, 0.0),  # film
            (1.5, 0.0),  # substrate
        ],
    )
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.prepare()
    result = m.solve()
    assert result["status"] == 0
    assert not np.any(np.isnan(result["q_ext"]))
    m.finalize()


def test_periodic_lattice():
    """Single sphere in a 2D periodic lattice."""
    m = MstmBindings()
    m.set_spheres(
        radii=[2.0],
        positions=[[0, 0, 0]],
        orders=[4],
        ref_re=[1.5],
        ref_im=[0.0],
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_lattice(cell_width_x=6.0, cell_width_y=6.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.prepare()
    result = m.solve()
    assert result["status"] == 0
    assert not np.any(np.isnan(result["q_ext"]))
    assert result["qext_tot"] > 0
    m.finalize()


def test_lattice_clear():
    """Enable then disable lattice."""
    m = MstmBindings()
    m.set_spheres(
        radii=[2.0],
        positions=[[0, 0, 0]],
        orders=[4],
        ref_re=[1.5],
        ref_im=[0.0],
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_lattice(cell_width_x=5.0, cell_width_y=5.0)
    m.clear_lattice()
    m.set_incident(0, 0, 1)
    m.prepare()
    result = m.solve()
    assert result["status"] == 0
    assert not np.any(np.isnan(result["q_ext"]))
    m.finalize()

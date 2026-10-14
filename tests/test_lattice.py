"""Periodic-lattice regression tests against the standalone MSTM v4 CLI.

Setup (relative host-medium convention): vacuum medium, sphere index
n_fe / n_host, lengths in nm scaled by 2*pi*n_host/lambda. Reference
values come from the serial CLI built from the same december2023 sources,
run on inputs with these keys (positions/cells in nm, radius 5 nm):

    length_scale_factor  2*pi*1.8476/1000
    sphere ref index     (2.868+4.192j)/1.8476
    periodic_lattice     t
    cell_width           <cell>d0,<cell>d0
    solution_epsilon     1.d-8
    calculate_scattering_matrix  f

A = "unit cell ... absorptance" (unpol), q_abs = per-sphere Qabs table.
"""

import math

import numpy as np
import pytest

from pymstm import MstmBindings

LAM, N_HOST, N_FE = 1000.0, 1.8476, 2.868 + 4.192j
M_REL = N_FE / N_HOST
KS = 2 * math.pi * N_HOST / LAM

# (n_spheres, cell_nm): (unit-cell absorptance A, per-sphere Qabs)
CLI_REF = {
    (1, 20.0): (2.18884e-02, [1.1148e-01]),
    (1, 40.0): (4.88856e-03, [9.9589e-02]),
    (1, 100.0): (7.73168e-04, [9.8443e-02]),
    (3, 80.0): (4.36589e-03, [1.1450e-01, 1.2677e-01, 1.1450e-01]),
    # 97 propagating diffraction orders per side: overflowed the old
    # fixed-size (32, -15:15) scat_mat and corrupted the heap.
    (1, 3000.0): (8.59470e-07, [9.8488e-02]),
}


def _solve(n, cell=None):
    m = MstmBindings()
    m.set_length_scale(KS)
    pos = [[12.0 * i - 6.0 * (n - 1), 3.0 * (i % 2), 0.0] for i in range(n)]
    m.set_spheres([5.0] * n, pos, [4] * n, [M_REL.real] * n, [M_REL.imag] * n)
    m.set_medium_ref(1.0, 0.0)
    if cell:
        m.set_lattice(cell_width_x=cell, cell_width_y=cell)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-8, max_iter=5000)
    m.prepare()
    r = m.solve()
    assert r["status"] == 0
    return r


@pytest.mark.parametrize("case", list(CLI_REF))
def test_lattice_matches_cli(case):
    a_ref, qabs_ref = CLI_REF[case]
    r = _solve(*case)
    assert r["qabs_tot"] == pytest.approx(a_ref, rel=1e-5)
    np.testing.assert_allclose(r["q_abs"], qabs_ref, rtol=1e-4)


def test_repeated_solves_consistent():
    """Lattice and non-lattice solves of different sizes in one process."""
    seq = [(1, None), (1, 3000.0), (5, None), (1, 40.0), (3, 80.0), (2, None)]
    first = {c: _solve(*c) for c in seq}
    for c in seq + seq[::-1]:
        r = _solve(*c)
        assert r["qabs_tot"] == pytest.approx(first[c]["qabs_tot"], rel=1e-12)
        np.testing.assert_allclose(r["q_abs"], first[c]["q_abs"], rtol=1e-12)
    # isolated sphere is independent of what ran before it
    assert first[(1, None)]["qabs_tot"] == pytest.approx(0.0984898, rel=1e-5)


"""OpenMP lattice solves (n_threads): results bit-identical to a serial solve.

The lattice pair matrices are computed in parallel but applied in the
serial order (mstm-modules-33-openmp-lattice.patch), so every output must
match n_threads=1 exactly, not just to round-off.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from pymstm import MstmBindings, MstmTask, solve_task
from pymstm import _mstm_ext
from pymstm._mstm import resolve_n_threads

KS = 2 * math.pi * 1.8476 / 1000  # lengths in nm, lambda = 1000 nm, relative setup
M_REL = (2.868 + 4.192j) / 1.8476
_rng = np.random.default_rng(7)
P10 = np.c_[_rng.uniform(-34, 34, (10, 2)), _rng.uniform(-8, 8, 10)]


def _case(name):
    """(length scale, radii, positions, ref index, cell, alpha, beta, layers)"""
    if name == "lat1":
        return KS, [5.0], [[0, 0, 0]], M_REL, 20.0, 0, 0, None
    if name == "lat3_oblique":
        pos = [[12.0 * i - 12.0, 3.0 * (i % 2), 2.0 * i] for i in range(3)]
        return KS, [5.0] * 3, pos, M_REL, 50.0, 20, 30, None
    if name in ("lat10", "lat10_oblique"):
        beta = 40 if name == "lat10_oblique" else 0
        return KS, [4.0 + 0.2 * i for i in range(10)], P10, M_REL, 80.0, 10, beta, None
    if name == "cluster":
        g = np.array([[i % 3, (i // 3) % 3, i // 9] for i in range(12)], float) * 3.0
        return 1.0, [1.2] * 12, g, 1.6 + 0.01j, None, 10, 25, None
    # spheres in all three media of two plane boundaries (layer_lat: also a
    # lattice -- the plane-boundary lattice kernels and Green's functions)
    pos = [[0.0, 0.0, -0.6], [0.7, 0.3, 0.5], [-0.4, 0.6, 1.2], [0.2, -0.7, 2.6]]
    layers = ([2.0], [(1.0, 0.0), (1.5, 0.4), (1.3, 0.0)])
    cell = 2.5 if name == "layer_lat" else None
    return 1.0, [0.3] * 4, pos, 1.7 + 0.05j, cell, 0, 30, layers


CASES = ["lat1", "lat3_oblique", "lat10", "lat10_oblique", "cluster", "layer", "layer_lat"]


def _solve(name, n_threads):
    ks, radii, pos, m_rel, cell, alpha, beta, layers = _case(name)
    n = len(radii)
    m = MstmBindings(n_threads=n_threads)
    m.set_length_scale(ks)
    m.set_spheres(radii, pos, [4] * n, [m_rel.real] * n, [m_rel.imag] * n)
    m.set_medium_ref(1.0, 0.0)
    if layers:
        m.set_layers(*layers)
    if cell:
        m.set_lattice(cell, cell)
    m.set_incident(alpha, beta)
    m.set_solver_params(eps=1e-6 if n == 10 else 1e-8, max_iter=5000)
    m.prepare()
    r = m.solve()
    assert r["status"] == 0
    if not cell:  # scattering matrix: isolated clusters only
        r["smat"] = np.array([m.get_scattering_angle(math.cos(t), 0.3) for t in (0.2, 1.0, 2.5)])
    return r


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


_serial = {}


def _ref(name):
    if name not in _serial:
        _serial[name] = _solve(name, 1)
    return _serial[name]


@pytest.mark.parametrize("n_threads", [2, 4])
@pytest.mark.parametrize("name", CASES)
def test_threads_bit_identical(name, n_threads):
    _same(_solve(name, n_threads), _ref(name))


@pytest.mark.parametrize("chunk", [1, 300])
def test_small_chunks(chunk):
    """Chunk boundaries (one pair per chunk, a few pairs per chunk)."""
    v = _mstm_ext.periodic_lattice_subroutines
    old = int(v.pl_chunk_elements)
    try:
        v.pl_chunk_elements = chunk
        for name in ("lat3_oblique", "layer_lat"):
            _same(_solve(name, 3), _ref(name))
    finally:
        v.pl_chunk_elements = old


def test_alternating_thread_counts():
    for nt, name in [(4, "lat3_oblique"), (1, "layer_lat"), (2, "lat1"), (4, "layer_lat"),
                     (1, "cluster"), (3, "lat3_oblique"), (1, "lat3_oblique"), (2, "layer_lat")]:
        _same(_solve(name, nt), _ref(name))


def test_n_threads_settable_between_solves():
    m = MstmBindings()
    assert m.n_threads == 1
    m.n_threads = 0
    assert resolve_n_threads(m.n_threads) >= 1


def test_resolve_n_threads():
    import os

    cpus = len(os.sched_getaffinity(0))
    assert resolve_n_threads(1) == 1
    assert resolve_n_threads(4) == 4
    assert resolve_n_threads(0) == cpus
    assert resolve_n_threads(1.0) == cpus
    assert resolve_n_threads(-1) == max(1, cpus - 1)
    assert resolve_n_threads(-10**6) == 1
    assert resolve_n_threads(0.5) == max(1, round(cpus / 2))
    with pytest.raises(TypeError):
        MstmBindings(n_threads="4")


def test_dask_task_isolated_threads():
    """MstmTask(n_threads=...) in-process and in an isolate=True child:
    same numbers as the serial task."""
    pos = [[12.0 * i - 12.0, 3.0 * (i % 2), 2.0 * i] for i in range(3)]
    task = MstmTask(radii=[5.0] * 3, positions=pos, ref_index=M_REL, length_scale=KS,
                    cell=(50.0, 50.0), orders=[4] * 3, solution_eps=1e-8,
                    alpha_deg=20, beta_deg=30)
    serial = solve_task(task)
    for res in (solve_task(replace(task, n_threads=2), isolate=True),
                solve_task(replace(task, n_threads=4))):
        for k in serial:
            if k != "wall_time_s":
                np.testing.assert_array_equal(res[k], serial[k], err_msg=k)

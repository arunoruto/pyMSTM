"""Thread-safety of the bindings: GIL released during solves, MSTM_LOCK
serialises whole solves, solve_task is bit-identical to driving
MstmBindings by hand (in-process and isolated)."""

import math
import threading
import time

import numpy as np
import pytest

from pymstm import MstmBindings, MstmTask, solve_task

N_HOST = 1.8476
KS = 2 * math.pi * N_HOST / 1000  # lengths in nm, lambda = 1000 nm, relative setup
M_REL = (2.868 + 4.192j) / N_HOST  # absorbing spheres


def lattice_task(n, cell, ks=KS):
    pos = [[12.0 * i - 6.0 * (n - 1), 3.0 * (i % 2), 0.0] for i in range(n)]
    return MstmTask(radii=[5.0] * n, positions=pos, ref_index=M_REL, length_scale=ks,
                    cell=cell, orders=[4] * n, solution_eps=1e-8)


def cluster_task(n, eps=1e-8):
    g = np.array([[i % 4, (i // 4) % 4, i // 16] for i in range(n)], float) * 3.0
    return MstmTask(radii=[1.2] * n, positions=g, ref_index=1.6 + 0.01j, solution_eps=eps)


TASKS = [lattice_task(1, (40.0, 40.0)), lattice_task(3, (80.0, 80.0)),
         lattice_task(1, None), cluster_task(6), cluster_task(10), lattice_task(2, None)]


def same(a, b):
    """Bit-identical, ignoring timing."""
    assert a.keys() == b.keys()
    for k in a:
        if k != "wall_time_s":
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_solve_task_bit_identical_to_bindings():
    t = TASKS[1]
    m = MstmBindings()
    m.set_length_scale(t.length_scale)
    m.set_spheres(t.radii, t.positions, t.orders, [M_REL.real] * 3, [M_REL.imag] * 3)
    m.set_medium_ref(1.0, 0.0)
    m.set_lattice(*t.cell)
    m.set_incident(0, 0)
    m.set_solver_params(eps=1e-8, max_iter=5000)
    m.prepare()
    direct = m.solve()
    m.finalize()
    r = solve_task(t)
    for k in direct:
        np.testing.assert_array_equal(r[k], direct[k], err_msg=k)
    assert r["qabs_tot"] == pytest.approx(4.36589e-03, rel=1e-5)  # CLI unit-cell A


def test_isolated_bit_identical():
    for t in TASKS[:2]:
        same(solve_task(t, isolate=True), solve_task(t))


def test_gil_released_during_solve():
    """A background thread keeps ticking while Fortran runs."""
    ticks, stop = [], threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.002)

    th = threading.Thread(target=tick)
    th.start()
    time.sleep(0.02)
    t0 = time.perf_counter()
    solve_task(cluster_task(30))
    t1 = time.perf_counter()
    stop.set()
    th.join()
    assert t1 - t0 > 0.1, "solve too short to tell"
    during = [t0] + [t for t in ticks if t0 < t < t1] + [t1]
    max_gap = max(np.diff(during))
    # GIL held -> one gap as long as prepare() or solve() itself
    assert max_gap < 0.25 * (t1 - t0), (max_gap, t1 - t0)


def test_lock_serialises_threads():
    serial = [solve_task(t) for t in TASKS]
    out = [None] * len(TASKS) * 2
    errors = []

    def work(i):
        try:
            out[i] = solve_task(TASKS[i % len(TASKS)])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(out))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    for i, r in enumerate(out):
        same(r, serial[i % len(TASKS)])


def _sphere(r):
    m = MstmBindings()
    m.set_spheres([r], [[0, 0, 0]], [10], [1.5], [0.01])
    m.prepare()
    return m


def test_only_newest_instance_owns_the_state():
    """Two live instances share MSTM's global Fortran state: the older one
    must refuse to run, and finalizing/collecting it must not free the
    newer one's arrays."""
    m = _sphere(1.0)
    ref = m.solve()["qext_tot"]
    m.finalize()
    with pytest.raises(RuntimeError):
        m.solve()

    a = _sphere(3.0)
    b = _sphere(1.0)
    with pytest.raises(RuntimeError):
        a.solve()
    with pytest.raises(RuntimeError):
        a.set_medium_ref(1.0)
    a.finalize()
    del a
    assert b.solve()["qext_tot"] == ref
    b.finalize()

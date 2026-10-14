"""Independent MSTM solves as picklable tasks -- for Dask, or any pool.

One :class:`MstmTask` = one serial solve (one geometry at one wavelength).
:func:`solve_task` runs it under :data:`~pymstm._mstm.MSTM_LOCK` and
returns a small dict; :func:`run_dask` maps it over a ``distributed``
cluster, mirroring ``pyfastmm._dask.run_resolved_dask``'s client handling.

``distributed`` is only imported inside :func:`run_dask` (install the
``dask`` extra), so ``solve_task`` works without it.

Concurrency model: the Fortran calls release the GIL, so a Dask worker
keeps heartbeating during a long solve, but MSTM's state is process-global,
so a worker process runs one solve at a time (extra threads just queue on
the lock). Parallelism therefore comes from worker *processes*: run workers
with ``--nthreads 1``.

``isolate=True`` runs the solve in a fresh child interpreter (plain
``subprocess``, which works inside Dask's daemonic worker processes, unlike
``multiprocessing``), in its own temporary working directory. A Fortran
crash then becomes a ``RuntimeError`` instead of killing the worker, and no
state (or silent heap damage) can leak from one solve into the next. Cost:
one interpreter start + import per task (well under a second) -- noise for
minute-long solves, significant for tiny ones, hence opt-in.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._mstm import MSTM_LOCK, MstmBindings


@dataclass(frozen=True)
class MstmTask:
    """One fixed-orientation MSTM solve, in MSTM's own conventions.

    Parameters
    ----------
    radii : array-like, shape (n,)
        Sphere radii, in the same units as ``positions``/``cell``.
    positions : array-like, shape (n, 3)
        Sphere centres.
    ref_index : complex or array-like of complex, shape (n,)
        Sphere refractive index *as passed to MSTM* (e.g. relative to the
        host when ``medium_ref_index`` is 1). A scalar broadcasts.
    medium_ref_index : complex
        Medium refractive index (MSTM ``medium_ref_index``).
    length_scale : float
        MSTM ``length_scale_factor``: multiplies radii, positions and the
        lattice cell, e.g. ``2*pi*n_host/wavelength`` for lengths in nm.
    cell : (float, float) or None
        Periodic-lattice unit cell (x, y) in the same units as ``radii``;
        ``None`` for an isolated cluster. With a lattice, ``qabs_tot`` is
        the unit-cell absorptance A.
    orders : sequence of int or None
        Advisory per-sphere Mie orders (allocation size only; truncation
        is set by ``mie_eps``). ``None``: ``max(4, x + 4 x^(1/3) + 2)``
        with ``x = radius * length_scale``.
    """

    radii: Sequence[float]
    positions: Sequence[Sequence[float]]
    ref_index: complex | Sequence[complex]
    medium_ref_index: complex = 1.0
    length_scale: float = 1.0
    cell: tuple[float, float] | None = None
    orders: Sequence[int] | None = None
    alpha_deg: float = 0.0
    beta_deg: float = 0.0
    solution_eps: float = 1e-6
    max_iterations: int = 5000
    mie_eps: float = 1e-6
    translation_eps: float = 1e-5


def solve_task(task: MstmTask, isolate: bool = False) -> dict[str, Any]:
    """Solve one :class:`MstmTask` and return a small, picklable dict.

    Keys: ``qext_tot``/``qabs_tot``/``qsca_tot`` (totals; per unit cell
    with a lattice), ``q_ext``/``q_abs``/``q_sca`` (per-sphere arrays),
    ``solution_error``, ``iterations``, ``status`` (0 = converged), as
    ``MstmBindings.solve()``; plus ``mie_orders`` (per-sphere Mie order
    MSTM actually used), ``t_matrix_order``, ``cross_section_radius`` and
    ``wall_time_s``.

    Same call sequence as driving :class:`~pymstm.MstmBindings` by hand,
    so results are bit-identical to that. ``isolate=True``: run in a
    fresh child process (see the module docstring).
    """
    if isolate:
        return _solve_isolated(task)
    radii = np.asarray(task.radii, dtype=np.float64)
    n = len(radii)
    m_rel = np.broadcast_to(np.asarray(task.ref_index, dtype=np.complex128), (n,))
    orders = task.orders
    if orders is None:
        x = radii * task.length_scale
        orders = np.maximum(4, (x + 4 * np.cbrt(x) + 2).astype(int))
    t0 = time.perf_counter()
    with MSTM_LOCK:
        m = MstmBindings()
        try:
            m.set_length_scale(task.length_scale)
            m.set_spheres(radii, task.positions, orders, m_rel.real, m_rel.imag)
            mr = complex(task.medium_ref_index)
            m.set_medium_ref(mr.real, mr.imag)
            if task.cell is not None:
                m.set_lattice(*task.cell)
            m.set_incident(task.alpha_deg, task.beta_deg)
            m.set_solver_params(eps=task.solution_eps, max_iter=task.max_iterations)
            m.set_mie_eps(task.mie_eps)
            m.set_translation_eps(task.translation_eps)
            m.prepare()
            res = m.solve()
            res["mie_orders"] = np.array(m._ext.spheredata.sphere_order[:n])
            res["t_matrix_order"] = m.get_tmatrix_order()
            res["cross_section_radius"] = m.get_cross_section_radius()
        finally:
            m.finalize()
    for k in ("qext_tot", "qabs_tot", "qsca_tot", "solution_error"):
        res[k] = float(res[k])
    for k in ("iterations", "status"):
        res[k] = int(res[k])
    res["wall_time_s"] = time.perf_counter() - t0
    return res


def _solve_isolated(task: MstmTask) -> dict[str, Any]:
    # Child imports the same pymstm as this process (also when it is only
    # importable via sys.path, e.g. a source checkout under pytest).
    pkg_parent = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [pkg_parent, env.get("PYTHONPATH")]))
    with tempfile.TemporaryDirectory(prefix="pymstm-") as d:
        Path(d, "task.pkl").write_bytes(pickle.dumps(task))
        # Result goes through a file, not stdout: Fortran prints to unit 6.
        p = subprocess.run(
            [sys.executable, "-m", "pymstm._dask"], cwd=d, env=env,
            capture_output=True, text=True,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"isolated MSTM solve failed (exit code {p.returncode}):\n"
                f"{(p.stdout + p.stderr)[-2000:]}"
            )
        return pickle.loads(Path(d, "result.pkl").read_bytes())


def run_dask(
    tasks: Sequence[MstmTask],
    progress_callback: Any = None,
    *,
    client: Any = None,
    scheduler_address: str | None = None,
    isolate: bool = False,
) -> list[dict[str, Any]]:
    """Solve every task on a Dask cluster; results in task order.

    Cluster selection as ``pyfastmm._dask.run_resolved_dask``: *client*
    (used as-is, never closed), else *scheduler_address* (e.g.
    ``"tcp://host:8786"``), else a temporary ``LocalCluster``; a client
    created here is closed on the way out. ``progress_callback(fraction,
    index)`` is called per completed task, from this thread. The first
    failed task raises.
    """
    try:
        import distributed
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pymstm's Dask support needs 'distributed': pip install pymstm[dask]"
        ) from exc

    tasks = list(tasks)
    if not tasks:
        return []
    owns_client = client is None
    if client is None:
        client = distributed.Client(scheduler_address)
    try:
        futures = client.map(solve_task, tasks, isolate=isolate, pure=False)
        index = {f.key: i for i, f in enumerate(futures)}
        out: list[Any] = [None] * len(tasks)
        for done, f in enumerate(distributed.as_completed(futures), 1):
            i = index[f.key]
            out[i] = f.result()
            if progress_callback is not None:
                progress_callback(done / len(tasks), i)
        return out
    finally:
        if owns_client:
            client.close()


if __name__ == "__main__":  # child side of isolate=True; cwd = task's temp dir
    _res = solve_task(pickle.loads(Path("task.pkl").read_bytes()))
    Path("result.pkl").write_bytes(pickle.dumps(_res))

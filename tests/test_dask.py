"""run_dask on a real multi-process LocalCluster (2 workers x 2 threads):
results equal serial solve_task bit for bit, in task order. Skipped
without the `dask` extra."""

import pytest

distributed = pytest.importorskip("distributed")

from pymstm import run_dask, solve_task  # noqa: E402
from test_concurrency import KS, cluster_task, lattice_task, same  # noqa: E402

# 2 geometries per kind x 2 wavelengths (1000 nm, 700 nm)
TASKS = [f(*a, ks=ks) if f is lattice_task else f(*a)
         for ks in (KS, KS * 1000 / 700)
         for f, a in [(lattice_task, (1, (40.0, 40.0))), (lattice_task, (3, (80.0, 80.0))),
                      (lattice_task, (2, None))]] + [cluster_task(4), cluster_task(8)]


@pytest.fixture(scope="module")
def client():
    with distributed.LocalCluster(n_workers=2, threads_per_worker=2, processes=True,
                                  dashboard_address=None) as cluster:
        with distributed.Client(cluster) as c:
            yield c


def test_run_dask_matches_serial(client):
    serial = [solve_task(t) for t in TASKS]
    seen = []
    via_dask = run_dask(TASKS, lambda f, i: seen.append(i), client=client)
    assert sorted(seen) == list(range(len(TASKS)))
    for a, b in zip(via_dask, serial):
        same(a, b)
    assert not client.status == "closed"  # caller's client left open


def test_run_dask_isolated(client):
    for a, t in zip(run_dask(TASKS[:2], client=client, isolate=True), TASKS[:2]):
        same(a, solve_task(t))

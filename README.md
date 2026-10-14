# pyMSTM

Python bindings for the [MSTM](https://github.com/dmckwski/MSTM) (Multiple Sphere T-Matrix) Fortran library.

Compute electromagnetic scattering from clusters of spheres, including layered media and periodic lattices.

## Quick start

```python
import numpy as np
from pymstm import MSTM

# Single dielectric sphere: size parameter x = 5, n = 1.5
m = MSTM()
m.set_spheres(
    radii=[5.0],
    positions=[[0, 0, 0]],
    orders=[12],
    ref_re=[1.5],
    ref_im=[0.0],
)
m.set_medium_ref(1.0, 0.0)
m.set_incident(alpha_deg=0, beta_deg=0)
m.prepare()
result = m.solve()

print(f"Q_ext = {result['qext_tot']:.6f}")
print(f"Q_abs = {result['qabs_tot']:.6f}")
print(f"Q_sca = {result['qsca_tot']:.6f}")
```

## Features

- **Fixed-orientation scattering**: per-sphere and total extinction, absorption, scattering efficiencies; 4x4 Mueller matrix at arbitrary angles
- **T-matrix computation**: full T-matrix output for use with orientation averaging
- **Random orientation scattering**: GSF expansion coefficients and angle-dependent Mueller matrix
- **Layered media**: plane boundaries with arbitrary refractive index profiles
- **Periodic lattices**: 2D periodic arrays with lattice sums

## Parallel solves (threads, Dask)

MSTM keeps all its state in Fortran module variables, so **one process runs
one solve at a time**. The long Fortran calls release the GIL, and
`pymstm.MSTM_LOCK` serialises whole set -> prepare -> solve sequences
(`MSTM.solve()` and `solve_task()` take it; hold it yourself when driving
`MstmBindings` from several threads). Parallelism comes from processes:

```python
import math
from dataclasses import replace
from pymstm import MstmTask, solve_task, run_dask

n_host, n_fe = 1.8476, 2.868 + 4.192j
base = MstmTask(
    radii=[5.0, 5.0], positions=[[-6, 0, 0], [6, 0, 0]],  # nm
    ref_index=n_fe / n_host, cell=(80.0, 80.0),             # None = isolated
    solution_eps=1e-8,
)
tasks = [replace(base, length_scale=2 * math.pi * n_host / wl) for wl in (500.0, 700.0, 1000.0)]

r = solve_task(tasks[0])          # in-process; same numbers as MstmBindings by hand
results = run_dask(tasks, scheduler_address="tcp://scheduler:8786")  # task order
results[0]["qabs_tot"]            # unit-cell absorptance A (lattice); per-sphere in "q_abs"
```

Each result is a small dict: `qext_tot`/`qabs_tot`/`qsca_tot`, per-sphere
`q_ext`/`q_abs`/`q_sca`, `mie_orders`, `t_matrix_order`, `iterations`,
`status` (0 = converged), `solution_error`, `wall_time_s`. `run_dask` takes
`client=` (left open), `scheduler_address=`, or neither (temporary
`LocalCluster`), and `progress_callback(fraction, index)`.

`isolate=True` (on `solve_task` and `run_dask`) runs each solve in a fresh
child interpreter in its own temp directory: a Fortran crash becomes a
`RuntimeError` instead of killing the worker, and nothing can leak from one
solve into the next. It costs one interpreter start per task (< 1 s), so it
is off by default; turn it on for long production runs.

### Running on a remote Dask cluster

1. **Build a portable wheel for the workers' Python** (the extension links
   libgfortran; a local `uv build` on NixOS links the Nix store copy and is not
   portable). Either run the *Wheels* GitHub workflow (`workflow_dispatch`),
   or locally with a container engine:
   `CIBW_CONTAINER_ENGINE=podman uvx cibuildwheel --only cp314-manylinux_x86_64 --output-dir wheelhouse`.
   cibuildwheel builds in manylinux_2_28 and `auditwheel repair` bundles
   libgfortran/libquadmath into the wheel. Check with
   `auditwheel show wheelhouse/pymstm-*.whl`.
2. **Install it on the workers.** Workers need `numpy` and `pydantic`
   already (plus `distributed`). Options:
   - prebuilt environment/image containing the wheel (best);
   - `client.register_plugin(distributed.diagnostics.plugin.PipInstall(["pymstm @ https://.../pymstm-...whl"]))`
     if workers can reach the URL (re-runs on every worker restart);
   - ship the bytes from the client and install once per host, then purge
     `pymstm` from `sys.modules` (the pattern of space-weathering's
     `scripts/dask_fit_runner.py:provision`).
3. **Worker settings:** `dask worker tcp://scheduler:8786 --nworkers <cores> --nthreads 1`.
   Extra threads per worker only queue on `MSTM_LOCK`. No `worker-ttl` change
   is needed: the GIL is released during solves, so heartbeats continue.
   Give each worker enough `--memory-limit` for its largest geometry.
4. Run `run_dask(tasks, client=client, isolate=True)`; results come back
   in task order.

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/arunoruto/pyMSTM.git
cd pyMSTM

# Build the Fortran library (requires gfortran)
make

# Install in development mode
pip install -e .
```

## Requirements

- Python >= 3.11
- NumPy >= 1.25
- gfortran (GCC)

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT. The bundled MSTM Fortran code retains its original license -- see `external/mstm/` for details.

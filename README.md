# pyMSTM

Python bindings for [MSTM](https://github.com/dmckwski/MSTM), the Multiple
Sphere T-Matrix code by Daniel Mackowski: electromagnetic scattering by
clusters of spheres, including spheres in layered media and in periodic
lattices.

pyMSTM compiles MSTM's Fortran sources (the `december2023` version) into a
Python extension with numpy.f2py. If you use it, please cite MSTM (see
[Citing](#citing)).

## Installation

```bash
pip install pymstm            # or: uv add pymstm
pip install "pymstm[dask]"    # with run_dask() for distributed solves
```

Wheels are published for Linux x86_64 and CPython 3.11–3.14. On other
platforms pip builds the source distribution, which needs a Fortran
compiler (gfortran) and a C compiler.

## Quick start

`MSTM` is a validated, self-contained problem description; `solve()` runs it
through the bindings (or, with `cli=True`, through the MSTM command-line
program) and returns an `MstmResult`:

```python
from pymstm import MSTM

# One sphere, size parameter x = 2*pi*r/lambda = 5, refractive index 1.5
result = MSTM(radii=[5.0], positions=[(0, 0, 0)], ref_re=[1.5], ref_im=[0.0]).solve()
print(result.q_ext_total, result.q_abs_total, result.q_sca_total)

# The same sphere in water (absolute refractive indices)
in_water = MSTM(
    radii=[5.0], positions=[(0, 0, 0)], ref_re=[1.5], ref_im=[0.0],
    medium_ref_re=1.33, compute_mueller=True, n_theta=181,
).solve()
in_water.mueller[0].s11   # phase function (integrates to 4*pi) at theta = 0
```

`MstmBindings` is the low-level, stateful interface to everything MSTM
offers (layers, periodic lattices, T-matrices, random orientation):

```python
from pymstm import MstmBindings

m = MstmBindings()
m.set_spheres(radii=[5.0, 5.0], positions=[[0, 0, -6], [0, 0, 6]],
              orders=[12, 12], ref_re=[1.5, 1.5], ref_im=[0.01, 0.01])
m.set_incident(alpha_deg=0, beta_deg=30)
m.prepare()
r = m.solve()               # dict: qext_tot, qabs_tot, qsca_tot, q_ext, q_abs, ...
s = m.get_scattering_angle(costheta=0.5, phi=0.0)   # 16 Mueller elements
m.finalize()
```

MSTM keeps its state in process-global Fortran variables, so only the most
recently created `MstmBindings` can be used; an older instance raises
`RuntimeError`.

## Units and conventions

**Lengths.** MSTM's length unit is `1/k0`, with `k0 = 2*pi/lambda` the
vacuum wavenumber: radii and positions are size parameters (`x = k0 * r`).
To give lengths in physical units instead (say nm), call
`set_length_scale(k0)` (`k0` in nm⁻¹), or set `MstmTask.length_scale`; it
multiplies radii, positions and lattice cell widths. `MstmBindings` applies
the scale when `set_spheres` and `set_lattice` are called, so call
`set_length_scale` first. `set_lattice(wx, wy)` takes the cell widths in
the same units as the radii.

**Medium and relative setups.** For spheres of index `m` in a medium of
index `n`, either set the medium (`medium_ref_re`/`medium_ref_im`,
`set_medium_ref`, `MstmTask.medium_ref_index`) and give absolute sphere
indices, with lengths in units of `1/k0`; or use the relative setup: vacuum
medium, sphere indices `m/n` and lengths scaled by `n` (`length_scale =
2*pi*n/lambda`). For a non-absorbing medium both give the same efficiencies
(with the default convention below). An absorbing medium needs the medium
setup.

**Efficiencies** are cross sections divided by `pi * r**2`: for the totals
`r` is the volume-equivalent radius of the external spheres
(`get_cross_section_radius()`), per sphere it is that sphere's radius. With
a periodic lattice they are per unit-cell area (projected on the incident
direction): `qabs_tot` is the unit-cell absorptance.

**`cross_section_convention`** (`MSTM`, `MstmBindings(...)`, `MstmTask`):

- `"physical"` (default): cross sections per incident irradiance in the
  medium. A single sphere gives the Lorenz–Mie efficiencies in the host,
  and `Q_ext` tends to 2 for large spheres.
- `"mstm"`: MSTM's own numbers, as its command-line program prints them.
  In a medium of index `n`, MSTM normalises by the vacuum irradiance of
  the incident field amplitude, so its extinction, absorption and
  scattering efficiencies (totals, per sphere, per lattice cell) are
  `Re(n)` times the physical ones (`get_mie_efficiencies()`: `Re(n)**2`).
  The unit-cell reflectance and absorptance printed by the CLI carry the
  same factor, and its transmittance is computed as `1 - R - A`.

`Re(n)` is that of the incident medium (layer 0 with plane boundaries), the
quantity MSTM itself divides its scattering matrix by. In vacuum both
conventions agree. pyMSTM 1.x returned the `"mstm"` numbers.

**Scattering matrix.** `MstmBindings.get_scattering_angle(costheta, phi)`
takes angles in the lab frame (the forward direction is
`(theta, phi) = (beta, alpha)` of `set_incident`) and returns MSTM's raw
matrix, flattened row by row. Its S11 integrates over the sphere to
`C_sca = qsca * pi * r**2` (lengths in units of `1/k0`, `qsca` in the
`"mstm"` convention); in vacuum this is the Bohren–Huffman convention
`dC_sca/dOmega = S11 / k**2`. `MstmResult.mueller` and
`s11_to_phase_function` give the phase function, normalised to `4*pi`,
which does not depend on the convention.

## Features

- Fixed orientation: total and per-sphere extinction, absorption and
  scattering efficiencies; the 4x4 Mueller matrix at any angle.
- T-matrix of a cluster (`compute_tmatrix`), and random-orientation
  scattering matrices from it (`ranorient_smatrix`).
- Plane boundaries (layered media) and 2D periodic lattices; lattice solves
  run on several cores (`n_threads`).
- `solve_task`/`run_dask` for many independent solves on a Dask cluster.
- The MSTM command-line program as an alternative backend (`cli=True`,
  `run_mstm`), for cross-checks and MPI runs.

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

n_host = 1.5
base = MstmTask(
    radii=[5.0, 5.0], positions=[[-6, 0, 0], [6, 0, 0]],  # nm
    ref_index=(2.9 + 4.2j) / n_host, cell=(80.0, 80.0),   # relative setup; cell=None: isolated
    solution_eps=1e-8,
)
tasks = [replace(base, length_scale=2 * math.pi * n_host / wl) for wl in (500.0, 700.0, 1000.0)]

r = solve_task(tasks[0])          # in-process; same numbers as MstmBindings by hand
results = run_dask(tasks, scheduler_address="tcp://scheduler:8786")  # task order
results[0]["qabs_tot"]            # unit-cell absorptance A (lattice); per-sphere in "q_abs"
```

Each result is a small dict: `qext_tot`/`qabs_tot`/`qsca_tot`, per-sphere
`q_ext`/`q_abs`/`q_sca`, `mie_orders`, `t_matrix_order`,
`cross_section_radius`, `iterations`, `status` (0 = converged),
`solution_error`, `wall_time_s`. `run_dask` takes `client=` (left open),
`scheduler_address=`, or neither (temporary `LocalCluster`), and
`progress_callback(fraction, index)`.

`isolate=True` (on `solve_task` and `run_dask`) runs each solve in a fresh
child interpreter in its own temp directory: a Fortran crash becomes a
`RuntimeError` instead of killing the worker, and nothing can leak from one
solve into the next. It costs one interpreter start per task (< 1 s), so it
is off by default; turn it on for long production runs.

### Several cores for one lattice solve (`n_threads`)

A periodic-lattice solve spends nearly all its time on the lattice sums of
every sphere pair, and these run in parallel (OpenMP) with
`MstmBindings(n_threads=4)`, `m.n_threads = 4` or `MstmTask(n_threads=4)`.
Results are bit-identical for any thread count; the default is 1. `0` means
all CPUs this process may use, a float in (0, 1] a fraction of them, a
negative number all but that many. `n_threads` overrides `OMP_NUM_THREADS`.
Measured (100 iron spheres in a 160 nm cell, i9-9900K): 152 s serial, 77 s
with 2 threads, 40 s with 4 (3.8x).
Isolated clusters and layered media without a lattice are not parallelised.

Threads x concurrent solves should not exceed a machine's cores. On Dask
workers with 4 cores each, either

- run many small tasks side by side, one core each (default `n_threads=1`,
  `--nthreads 4` with `isolate=True`, or `--nworkers 4 --nthreads 1`), or
- give each task the whole worker: `MstmTask(..., n_threads=4)` and one
  task per worker at a time, i.e. `dask worker ... --nthreads 1`, or keep
  `--nthreads 4` but start the workers with `--resources "cores=4"` and
  call `run_dask(tasks, ..., resources={"cores": 4})`.

The second gets each result up to ~4x sooner (useful when there are fewer
large lattice tasks than workers); for many tasks, one core per task gives
slightly higher total throughput (no parallel overhead). Never combine `n_threads=4` with 4 concurrent
tasks per 4-core worker: 16 threads on 4 cores is slower than either.

### Running on a remote Dask cluster

1. **Install pyMSTM on the workers**, with `distributed`
   (`pip install "pymstm[dask]"`), ideally in the image or environment the
   workers start from. For an unreleased version, build a portable wheel
   with the *Wheels* workflow, or locally with
   `uvx cibuildwheel --only cp314-manylinux_x86_64` (it bundles libgfortran;
   a wheel built with `uv build` on NixOS links the Nix store copy and is
   not portable), and install that.
2. **Worker settings:** `dask worker tcp://scheduler:8786 --nworkers <cores> --nthreads 1`.
   Extra threads per worker only queue on `MSTM_LOCK` (unless `isolate=True`,
   where each task is its own process). For tasks with `n_threads > 1`, see
   above. No `worker-ttl` change is needed: the GIL is released during
   solves, so heartbeats continue. Give each worker enough `--memory-limit`
   for its largest geometry.
3. Run `run_dask(tasks, client=client, isolate=True)`; results come back
   in task order.

## The MSTM command-line program (optional)

`MSTM(..., cli=True)`, `run_mstm()` and the cross-check tests run MSTM's own
program. pyMSTM looks for `mstm` (or `mstm-mpi` with `mpi_processes=`) on
`PATH`, or takes `binary_path=` or the `PYMSTM_MSTM_BIN` /
`PYMSTM_MSTM_MPI_BIN` environment variables. It must be built from the same
`december2023` sources:

```bash
nix build github:arunoruto/pyMSTM#mstm        # or #mstm-mpi
# or, in a checkout with the submodule:
make cli                                      # -> build/mstm
```

Programs built from the older `code/mstm-v4.0` tree of the MSTM repository
normalise S11 differently and write no results for a non-vacuum medium;
pyMSTM does not support them.

## Development

```bash
git clone --recurse-submodules https://github.com/arunoruto/pyMSTM.git
cd pyMSTM
uv sync --extra dask              # builds the extension (needs gfortran)
make cli                           # optional: the CLI for the cross-checks
PYMSTM_MSTM_BIN=build/mstm uv run pytest
```

The editable install rebuilds the extension on import when a source file
changed (`meson` and `ninja` from the virtual environment must be on
`PATH`).

## Citing

pyMSTM is a wrapper; the method and the Fortran code are MSTM's. Please
cite

- D. W. Mackowski and M. I. Mishchenko, "A multiple sphere T-matrix
  Fortran code for use on parallel computer clusters", J. Quant.
  Spectrosc. Radiat. Transfer 112, 2182–2192 (2011),
  [doi:10.1016/j.jqsrt.2011.02.019](https://doi.org/10.1016/j.jqsrt.2011.02.019),
- for plane boundaries and periodic lattices (MSTM version 4): D. W.
  Mackowski, "The extension of the Multiple Sphere T Matrix code to include
  multiple plane boundaries and 2-D periodic systems", J. Quant. Spectrosc.
  Radiat. Transfer 290, 108292 (2022),
  [doi:10.1016/j.jqsrt.2022.108292](https://doi.org/10.1016/j.jqsrt.2022.108292),

and pyMSTM itself if it helped (see `CITATION.cff`).

## License

pyMSTM is MIT-licensed (`LICENSE`). MSTM is MIT-licensed by Daniel
Mackowski; its third-party routines and the GNU Fortran runtime libraries
bundled in the Linux wheels are listed in `LICENSES/`.

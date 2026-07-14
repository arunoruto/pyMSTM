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

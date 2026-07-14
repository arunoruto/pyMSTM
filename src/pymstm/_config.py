"""
Declarative sweep configuration for pyMSTM.

Uses TOML format.  A sweep config specifies a particle cluster, a
wavelength range, solver settings, and output flags.  The config can be
translated into an MSTM ``.inp`` file for the CLI *and* fed directly to
pyMSTM for programmatic iteration.

Example
-------
.. code-block:: toml

    [particles]
    positions_file = "cluster.dat"
    scale = 1e-9
    refractive_index = [1.5, 0.01]

    [wavelengths]
    start = 0.4
    stop = 0.8
    num = 21
    scale = 1e-6

    [medium]
    refractive_index = [1.0, 0.0]

    [solver]
    tolerance = 1e-5
    max_iterations = 5000
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ParticlesConfig(BaseModel):
    """Particle cluster specification."""

    positions_file: str = ""
    scale: float = 1.0
    gap_factor: float = 1.0
    refractive_index: tuple[float, float] = (1.5, 0.0)

    def load_positions(self, base_dir: str | os.PathLike[str] = "") -> np.ndarray:
        """Return (N, 4) array ``[x, y, z, radius]`` in physical units.

        Supports ``.dat``, ``.txt``, ``.csv``, ``.pos`` files.
        Automatically strips comment lines (``#``) for PyFracVAL format.

        *gap_factor* stretches positions (not radii) to separate
        touching spheres, which can crash the Fortran T-matrix solver.
        """
        path = self._resolve(base_dir)
        suffix = Path(path).suffix.lower()
        match suffix:
            case ".csv":
                data = np.loadtxt(str(path), delimiter=",")
            case _:  # .dat, .txt, .pos — whitespace
                data = np.loadtxt(str(path))
        if data.ndim == 1:
            data = data.reshape(-1, 4)
        data = data * self.scale
        if self.gap_factor != 1.0:
            data[:, :3] *= self.gap_factor
        return data

    def _resolve(self, base_dir: str | os.PathLike[str]) -> Path:
        p = Path(self.positions_file)
        if p.is_absolute():
            return p
        return Path(base_dir) / p


class WavelengthsConfig(BaseModel):
    """Wavelength sweep specification.

    Exactly one of *values* or *start / stop / num* must be given.
    *scale* converts the user-specified numbers to meters.
    """

    values: list[float] | None = None
    start: float | None = None
    stop: float | None = None
    num: int | None = None
    scale: float = 1.0  # 1e-6 = µm, 1e-9 = nm

    @model_validator(mode="after")
    def _check_spec(self) -> WavelengthsConfig:
        explicit = self.values is not None
        ranged = (
            self.start is not None and self.stop is not None and self.num is not None
        )
        if not explicit and not ranged:
            raise ValueError(
                "Specify either wavelengths.values or wavelengths.{start,stop,num}"
            )
        return self

    def get_wavelengths_m(self) -> np.ndarray:
        """Wavelengths in meters."""
        if self.values is not None:
            return np.asarray(self.values, dtype=float) * self.scale
        return np.linspace(self.start, self.stop, self.num) * self.scale  # type: ignore[arg-type]

    def get_length_scales(self) -> np.ndarray:
        """Return ``length_scale_factor`` values for each wavelength.

        ``length_scale = λ_ref / λ``  so the size parameter at each λ
        matches the physical wavelength.
        """
        wl = self.get_wavelengths_m()
        ref = wl[0]
        return ref / wl

    def get_linear_length_scales(self) -> np.ndarray:
        """Length scales evenly spaced between first and last wavelength.

        This matches what the MSTM CLI ``loop_variable`` produces when
        given ``start, stop, step`` with a constant step.
        Returns N values linearly interpolated between ``ls[0]`` and ``ls[-1]``.
        """
        ls = self.get_length_scales()
        return np.linspace(ls[0], ls[-1], len(ls))


class MediumConfig(BaseModel):
    """Surrounding medium."""

    refractive_index: tuple[float, float] = (1.0, 0.0)


class IncidentConfig(BaseModel):
    """Incident plane-wave parameters."""

    polar_angle_deg: float = 0.0
    azimuthal_angle_deg: float = 0.0
    direction: int = 1


class SolverConfig(BaseModel):
    """Linear solver and T-matrix settings."""

    method: str = "iterative"
    tolerance: float = 1e-5
    max_iterations: int = 5000
    mie_epsilon: float = 1e-5
    translation_epsilon: float = 1e-5
    max_tmatrix_order: int = 100


class OutputConfig(BaseModel):
    """Output control."""

    calculate_scattering_matrix: bool = True
    print_sphere_data: bool = False
    azimuthal_average: bool = True


class SweepConfig(BaseModel):
    """Complete sweep configuration."""

    particles: ParticlesConfig = Field(default_factory=ParticlesConfig)
    wavelengths: WavelengthsConfig
    medium: MediumConfig = Field(default_factory=MediumConfig)
    incident: IncidentConfig = Field(default_factory=IncidentConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike[str]) -> SweepConfig:
    """Load a sweep TOML configuration file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return SweepConfig(**raw)


def config_to_inp(
    config: SweepConfig,
    base_dir: str | os.PathLike[str] = "",
    output_filename: str = "mstm_output.dat",
) -> str:
    """Translate a ``SweepConfig`` into MSTM ``.inp`` file text.

    The returned string can be written to disk and passed to the
    standalone ``mstm`` CLI.  Cluster coordinates are written inline
    (``sphere_data`` block) because the CLI does not support variable
    refractive index via ``sphere_data_input_file``.
    """
    positions = config.particles.load_positions(base_dir)
    n = positions.shape[0]
    n_re, n_im = config.particles.refractive_index
    ls_linear = config.wavelengths.get_linear_length_scales()
    ls = ls_linear  # use linear spacing for consistency with CLI loop

    fd = _fortran_real

    lines: list[str] = []
    lines.append("output_file")
    lines.append(output_filename)
    lines.append("number_spheres")
    lines.append(str(n))
    lines.append("sphere_data")
    for i in range(n):
        x, y, z, r = positions[i]
        lines.append(f"{fd(x)},{fd(y)},{fd(z)},{fd(r)},({fd(n_re)},{fd(n_im)})")
    lines.append("end_of_sphere_data")

    lines.append("length_scale_factor")
    lines.append(fd(ls[0]))  # reference scale

    med_re, med_im = config.medium.refractive_index
    if med_re != 1.0 or med_im != 0.0:
        lines.append("medium_ref_index")
        lines.append(f"({fd(med_re)},{fd(med_im)})")

    lines.append("incident_alpha_deg")
    lines.append(fd(config.incident.azimuthal_angle_deg))
    lines.append("incident_beta_deg")
    lines.append(fd(config.incident.polar_angle_deg))
    lines.append("incident_direction")
    lines.append(str(config.incident.direction))

    lines.append("solution_epsilon")
    lines.append(fd(config.solver.tolerance))
    lines.append("max_iterations")
    lines.append(str(config.solver.max_iterations))
    lines.append("mie_epsilon")
    lines.append(fd(config.solver.mie_epsilon))
    lines.append("translation_epsilon")
    lines.append(fd(config.solver.translation_epsilon))
    lines.append("max_t_matrix_order")
    lines.append(str(config.solver.max_tmatrix_order))
    lines.append("t_matrix_convergence_epsilon")
    lines.append(fd(config.solver.tolerance))

    if config.output.calculate_scattering_matrix:
        lines.append("calculate_scattering_matrix")
        lines.append("t")

    if config.output.azimuthal_average:
        lines.append("azimuthal_average")
        lines.append("t")

    if config.output.print_sphere_data:
        lines.append("print_sphere_data")
        lines.append("t")

    # Loop
    if len(ls) > 1:
        lines.append("loop_variable")
        lines.append("length_scale_factor")
        step = (float(ls[-1]) - float(ls[0])) / max(len(ls) - 1, 1)
        vals = ", ".join(fd(v) for v in [ls[0], ls[-1], step])
        lines.append(vals)
    else:
        lines.append(f"length_scale_factor")
        lines.append(fd(ls[0]))

    lines.append("end_of_options")
    return "\n".join(lines) + "\n"


def config_to_pymstm_args(config: SweepConfig, base_dir: str = "") -> dict[str, Any]:
    """Return kwargs for ``write_inp_file()``, useful for single-run tests."""
    positions = config.particles.load_positions(base_dir)
    n_re, n_im = config.particles.refractive_index
    wl = config.wavelengths.get_wavelengths_m()
    ls = config.wavelengths.get_length_scales()
    med_re, med_im = config.medium.refractive_index

    return {
        "radii": positions[:, 3].tolist(),
        "positions": positions[:, :3].tolist(),
        "ref_re": [n_re] * len(positions),
        "ref_im": [n_im] * len(positions),
        "medium_ref_re": med_re,
        "medium_ref_im": med_im,
        "alpha_deg": config.incident.azimuthal_angle_deg,
        "beta_deg": config.incident.polar_angle_deg,
        "incident_direction": config.incident.direction,
        "length_scale": float(ls[0]),
        "solution_eps": config.solver.tolerance,
        "max_iterations": config.solver.max_iterations,
        "mie_eps": config.solver.mie_epsilon,
        "translation_eps": config.solver.translation_epsilon,
        "max_tmatrix_order": config.solver.max_tmatrix_order,
        "calculate_scattering_matrix": config.output.calculate_scattering_matrix,
        "print_sphere_data": config.output.print_sphere_data,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fortran_real(val: float) -> str:
    """Format a Python float as a Fortran double-precision literal."""
    if val == 0.0:
        return "0.d0"
    s = f"{val:.14e}"
    mantissa, exp = s.split("e")
    exp = int(exp)
    mantissa = mantissa.rstrip("0")
    if mantissa.endswith("."):
        mantissa += "0"
    if exp == 0:
        return f"{mantissa}d0"
    else:
        return f"{mantissa}d{exp:+d}"

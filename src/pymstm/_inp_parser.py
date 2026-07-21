"""
Parser for MSTM .inp input files.

Reads the keyword-value format used by the MSTM CLI and returns
a Pydantic model suitable for feeding into both pyMSTM and the CLI.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SphereInp(BaseModel):
    """Single sphere entry from ``sphere_data`` block."""

    x: float
    y: float
    z: float
    radius: float
    ref_re: float
    ref_im: float


class LayerRefIndex(BaseModel):
    """Layer refractive index (real + imag)."""

    re: float
    im: float


class MstmInpConfig(BaseModel):
    """Complete MSTM input configuration parsed from a .inp file."""

    output_file: str = "mstm_output.dat"
    spheres: list[SphereInp] = Field(default_factory=list)

    # Medium
    medium_ref_re: float = 1.0
    medium_ref_im: float = 0.0

    # Incident
    alpha_deg: float = 0.0
    beta_deg: float = 0.0
    incident_direction: int = 1

    # Scale
    length_scale: float = 1.0

    # Solver
    solution_eps: float = 1e-6
    max_iterations: int = 5000
    mie_eps: float = 1e-6
    translation_eps: float = 1e-5
    max_tmatrix_order: int = 100
    tmatrix_convergence_eps: float = 1e-6

    # Output flags
    calculate_scattering_matrix: bool = True
    print_sphere_data: bool = True
    scattering_map_model: int = 0
    scattering_map_dimension: int | None = None

    # Orientation / incidence averaging
    random_orientation: bool = False
    incidence_average: bool = False
    number_incident_directions: int = 16

    # Gaussian beam
    gaussian_beam_constant: float = 0.0

    # Layers
    number_plane_boundaries: int = 0
    layer_thicknesses: list[float] = Field(default_factory=list)
    layer_ref_indices: list[LayerRefIndex] = Field(default_factory=list)

    # Lattice
    periodic_lattice: bool = False
    cell_width_x: float = 1.0
    cell_width_y: float = 1.0

    # Loop / sweep
    has_loop: bool = False
    loop_var_name: str = ""
    loop_start: float = 0.0
    loop_stop: float = 0.0
    loop_step: float = 0.0

    @property
    def nspheres(self) -> int:
        return len(self.spheres)

    def to_pymstm_args(self) -> dict[str, Any]:
        """Return keyword arguments suitable for ``write_inp_file()``."""
        spheres = self.spheres
        result: dict[str, Any] = {
            "radii": [s.radius for s in spheres],
            "positions": [[s.x, s.y, s.z] for s in spheres],
            "ref_re": [s.ref_re for s in spheres],
            "ref_im": [s.ref_im for s in spheres],
            "medium_ref_re": self.medium_ref_re,
            "medium_ref_im": self.medium_ref_im,
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
            "incident_direction": self.incident_direction,
            "length_scale": self.length_scale,
            "solution_eps": self.solution_eps,
            "max_iterations": self.max_iterations,
            "mie_eps": self.mie_eps,
            "translation_eps": self.translation_eps,
            "max_tmatrix_order": self.max_tmatrix_order,
            "tmatrix_convergence_eps": self.tmatrix_convergence_eps,
            "calculate_scattering_matrix": self.calculate_scattering_matrix,
            "print_sphere_data": self.print_sphere_data,
            "output_file": self.output_file,
            "gaussian_beam_constant": self.gaussian_beam_constant,
            "random_orientation": self.random_orientation,
            "incidence_average": self.incidence_average,
        }
        if self.scattering_map_dimension is not None:
            result["scattering_map_dimension"] = self.scattering_map_dimension
        if self.scattering_map_model:
            result["scattering_map_model"] = self.scattering_map_model
        if self.incidence_average:
            result["number_incident_directions"] = self.number_incident_directions
        if self.number_plane_boundaries > 0:
            result["layer_thicknesses"] = self.layer_thicknesses
            result["layer_ref_indices"] = [
                (lr.re, lr.im) for lr in self.layer_ref_indices
            ]
        if self.periodic_lattice:
            result["periodic_lattice"] = True
            result["cell_width_x"] = self.cell_width_x
            result["cell_width_y"] = self.cell_width_y
        return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_fortran_real(s: str) -> float:
    """Parse a Fortran double literal like ``1.0d-6`` or ``0.d0``."""
    s = s.strip().replace("d", "e").replace("D", "E")
    # Fortran "0.d0" / "0.D0" becomes "0.e0" which Python rejects.
    # Insert a zero between bare "." and exponent.
    s = re.sub(r"\.([eE])", r".0\1", s)
    return float(s)


def _parse_fortran_complex(s: str) -> tuple[float, float]:
    """Parse a Fortran complex literal like ``(1.5d0,0.0d0)``.

    Returns (real, imag).
    """
    s = s.strip()
    m = re.match(r"\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", s)
    if not m:
        raise ValueError(f"Invalid Fortran complex literal: {s!r}")
    return _parse_fortran_real(m.group(1)), _parse_fortran_real(m.group(2))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_inp_text(text: str) -> MstmInpConfig:
    """Parse an MSTM .inp file from its text content."""
    lines = _preprocess_lines(text)
    if not lines:
        return MstmInpConfig()

    config = MstmInpConfig()

    i = 0
    while i < len(lines):
        keyword = lines[i].lower()
        i += 1

        if keyword == "output_file":
            if i < len(lines):
                config.output_file = lines[i]
                i += 1

        elif keyword == "number_spheres":
            if i < len(lines):
                i += 1  # read but derived from sphere_data

        elif keyword == "sphere_data":
            while i < len(lines) and lines[i].lower() != "end_of_sphere_data":
                config.spheres.append(_parse_sphere_line(lines[i]))
                i += 1
            if i < len(lines):
                i += 1  # skip end_of_sphere_data

        elif keyword in ("medium_ref_index",):
            if i < len(lines):
                rre, rim = _parse_fortran_complex(lines[i])
                config.medium_ref_re = rre
                config.medium_ref_im = rim
                i += 1

        elif keyword == "medium_re_ref_index":
            if i < len(lines):
                config.medium_ref_re = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "medium_im_ref_index":
            if i < len(lines):
                config.medium_ref_im = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "incident_alpha_deg":
            if i < len(lines):
                config.alpha_deg = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "incident_beta_deg":
            if i < len(lines):
                config.beta_deg = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "incident_direction":
            if i < len(lines):
                config.incident_direction = int(lines[i])
                i += 1

        elif keyword == "length_scale_factor":
            if i < len(lines):
                config.length_scale = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "solution_epsilon":
            if i < len(lines):
                config.solution_eps = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "max_iterations":
            if i < len(lines):
                config.max_iterations = int(lines[i])
                i += 1

        elif keyword == "mie_epsilon":
            if i < len(lines):
                config.mie_eps = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "translation_epsilon":
            if i < len(lines):
                config.translation_eps = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "max_t_matrix_order":
            if i < len(lines):
                config.max_tmatrix_order = int(lines[i])
                i += 1

        elif keyword == "t_matrix_convergence_epsilon":
            if i < len(lines):
                config.tmatrix_convergence_eps = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "calculate_scattering_matrix":
            if i < len(lines):
                config.calculate_scattering_matrix = _parse_bool(lines[i])
                i += 1

        elif keyword == "print_sphere_data":
            if i < len(lines):
                config.print_sphere_data = _parse_bool(lines[i])
                i += 1

        elif keyword == "gaussian_beam_constant":
            if i < len(lines):
                config.gaussian_beam_constant = _parse_fortran_real(lines[i])
                i += 1

        elif keyword == "number_plane_boundaries":
            if i < len(lines):
                config.number_plane_boundaries = int(lines[i])
                i += 1

        elif keyword == "layer_thickness":
            if i < len(lines):
                config.layer_thicknesses = [
                    _parse_fortran_real(v) for v in _split_csv(lines[i])
                ]
                i += 1

        elif keyword == "layer_ref_index":
            if i < len(lines):
                raw_parts = re.findall(r"\([^)]+\)", lines[i])
                parsed: list[LayerRefIndex] = []
                for p in raw_parts:
                    m = re.match(r"\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", p)
                    if m is not None:
                        parsed.append(
                            LayerRefIndex(
                                re=_parse_fortran_real(m.group(1)),
                                im=_parse_fortran_real(m.group(2)),
                            )
                        )
                config.layer_ref_indices = parsed
                i += 1

        elif keyword == "periodic_lattice":
            if i < len(lines):
                config.periodic_lattice = _parse_bool(lines[i])
                i += 1

        elif keyword == "cell_width":
            if i < len(lines):
                parts = _split_csv(lines[i])
                if len(parts) >= 2:
                    config.cell_width_x = _parse_fortran_real(parts[0])
                    config.cell_width_y = _parse_fortran_real(parts[1])
                i += 1

        elif keyword == "scattering_map_model":
            if i < len(lines):
                config.scattering_map_model = int(lines[i])
                i += 1

        elif keyword == "scattering_map_dimension":
            if i < len(lines):
                config.scattering_map_dimension = int(lines[i])
                i += 1

        elif keyword == "random_orientation":
            if i < len(lines):
                config.random_orientation = _parse_bool(lines[i])
                i += 1

        elif keyword == "incidence_average":
            if i < len(lines):
                config.incidence_average = _parse_bool(lines[i])
                i += 1

        elif keyword == "number_incident_directions":
            if i < len(lines):
                config.number_incident_directions = int(lines[i])
                i += 1

        elif keyword == "end_of_options":
            break

        # loop_variable: parse and store sweep info
        elif keyword == "loop_variable":
            # format:  loop_variable \n <name> \n <start,stop,step>
            if i < len(lines):
                config.loop_var_name = lines[i].lower()
                i += 1
            if i < len(lines):
                parts = _split_csv(lines[i])
                if len(parts) >= 3:
                    config.has_loop = True
                    config.loop_start = _parse_fortran_real(parts[0])
                    config.loop_stop = _parse_fortran_real(parts[1])
                    config.loop_step = _parse_fortran_real(parts[2])
                i += 1

        elif keyword in (
            "new_run",
            "sphere_data_input_file",
            "append_output_file",
            "ref_index_scale_factor",
            "near_field_minimum_border",
            "near_field_maximum_border",
            "near_field_step_size",
            "near_field_output_file",
            "calculate_near_field",
            "random_configuration",
            "configuration_average",
        ):
            if i < len(lines) and not lines[i].lower() in _KEYWORDS:
                i += 1

        else:
            pass

    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_KEYWORDS = {
    "output_file",
    "number_spheres",
    "sphere_data",
    "end_of_sphere_data",
    "medium_ref_index",
    "medium_re_ref_index",
    "medium_im_ref_index",
    "incident_alpha_deg",
    "incident_beta_deg",
    "incident_direction",
    "length_scale_factor",
    "solution_epsilon",
    "max_iterations",
    "mie_epsilon",
    "translation_epsilon",
    "max_t_matrix_order",
    "t_matrix_convergence_epsilon",
    "calculate_scattering_matrix",
    "print_sphere_data",
    "gaussian_beam_constant",
    "number_plane_boundaries",
    "layer_thickness",
    "layer_ref_index",
    "periodic_lattice",
    "cell_width",
    "end_of_options",
    "loop_variable",
    "new_run",
    "sphere_data_input_file",
    "append_output_file",
    "ref_index_scale_factor",
    "near_field_minimum_border",
    "near_field_maximum_border",
    "near_field_step_size",
    "near_field_output_file",
    "calculate_near_field",
    "random_configuration",
    "configuration_average",
    "incidence_average",
    "random_orientation",
    "number_incident_directions",
    "scattering_map_model",
    "scattering_map_dimension",
}


def _preprocess_lines(text: str) -> list[str]:
    """Strip comments and blank lines, return keyword/value pairs."""
    result: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("!") or stripped.startswith("%"):
            continue
        result.append(raw.strip())
    return result


def _parse_sphere_line(line: str) -> SphereInp:
    """Parse a single sphere_data line like ``0.d0,0.d0,0.d0,5.d0,(1.5d0,0.d0)``."""
    parts = _split_csv(line)
    if len(parts) != 5:
        raise ValueError(f"Invalid sphere_data line: {line!r}")
    ref_re, ref_im = _parse_fortran_complex(parts[4])
    return SphereInp(
        x=_parse_fortran_real(parts[0]),
        y=_parse_fortran_real(parts[1]),
        z=_parse_fortran_real(parts[2]),
        radius=_parse_fortran_real(parts[3]),
        ref_re=ref_re,
        ref_im=ref_im,
    )


def _split_csv(line: str) -> list[str]:
    """Split comma-separated values, respecting parenthesised groups."""
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in line:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current))
    return result


def _parse_bool(s: str) -> bool:
    s = s.strip().lower()
    return s in ("t", "true", "1", ".true.", "yes")

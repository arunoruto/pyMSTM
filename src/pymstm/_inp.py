"""
Generator for MSTM .inp input files.

Provides a function to serialize a pyMSTM configuration into the
keyword-value format understood by the standalone MSTM CLI.
"""

from __future__ import annotations

import os
from typing import Sequence


def _fortran_real(val: float) -> str:
    """Format a Python float as a Fortran double-precision literal.

    Always includes a decimal point, e.g. ``0.d0``, ``1.5d0``, ``1.d-6``.
    """
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


def write_inp_file(
    filepath: str | os.PathLike[str],
    *,
    radii: Sequence[float],
    positions: Sequence[Sequence[float]],
    ref_re: Sequence[float],
    ref_im: Sequence[float],
    medium_ref_re: float = 1.0,
    medium_ref_im: float = 0.0,
    alpha_deg: float = 0.0,
    beta_deg: float = 0.0,
    incident_direction: int = 1,
    length_scale: float = 1.0,
    solution_eps: float = 1e-6,
    max_iterations: int = 5000,
    mie_eps: float = 1e-6,
    translation_eps: float = 1e-5,
    max_tmatrix_order: int = 100,
    tmatrix_convergence_eps: float = 1e-6,
    calculate_scattering_matrix: bool = True,
    scattering_map_dimension: int | None = None,
    normalize_s11: bool = True,
    azimuthal_average: bool = False,
    print_sphere_data: bool = True,
    output_file: str = "mstm_output.dat",
    layer_thicknesses: Sequence[float] | None = None,
    layer_ref_indices: Sequence[tuple[float, float]] | None = None,
    periodic_lattice: bool = False,
    cell_width_x: float = 1.0,
    cell_width_y: float = 1.0,
    gaussian_beam_constant: float = 0.0,
):
    """Write an MSTM .inp file.

    Parameters map directly to the corresponding MSTM input keywords.
    Generated file can be passed to the standalone ``mstm`` CLI binary.
    """
    fd = _fortran_real
    nspheres = len(radii)

    lines: list[str] = []
    lines.append("output_file")
    lines.append(output_file)
    lines.append("number_spheres")
    lines.append(str(nspheres))
    lines.append("sphere_data")

    for i in range(nspheres):
        x, y, z = positions[i]
        re = ref_re[i]
        im = ref_im[i]
        lines.append(f"{fd(x)},{fd(y)},{fd(z)},{fd(radii[i])},({fd(re)},{fd(im)})")

    lines.append("end_of_sphere_data")
    lines.append("length_scale_factor")
    lines.append(fd(length_scale))

    if gaussian_beam_constant != 0.0:
        lines.append("gaussian_beam_constant")
        lines.append(fd(gaussian_beam_constant))

    # Medium / layer configuration
    if layer_ref_indices is not None and layer_thicknesses is not None:
        n_boundaries = len(layer_ref_indices) - 1
        lines.append("number_plane_boundaries")
        lines.append(str(n_boundaries))
        lines.append("layer_ref_index")
        ref_str = ",".join(f"({fd(re)},{fd(im)})" for re, im in layer_ref_indices)
        lines.append(ref_str)
        if layer_thicknesses:
            lines.append("layer_thickness")
            lines.append(",".join(fd(t) for t in layer_thicknesses))
    else:
        if medium_ref_re != 1.0 or medium_ref_im != 0.0:
            lines.append("medium_ref_index")
            lines.append(f"({fd(medium_ref_re)},{fd(medium_ref_im)})")

    # Incident field
    lines.append("incident_alpha_deg")
    lines.append(fd(alpha_deg))
    lines.append("incident_beta_deg")
    lines.append(fd(beta_deg))
    lines.append("incident_direction")
    lines.append(str(incident_direction))

    # Solver
    lines.append("solution_epsilon")
    lines.append(fd(solution_eps))
    lines.append("max_iterations")
    lines.append(str(max_iterations))
    lines.append("mie_epsilon")
    lines.append(fd(mie_eps))
    lines.append("translation_epsilon")
    lines.append(fd(translation_eps))
    lines.append("max_t_matrix_order")
    lines.append(str(max_tmatrix_order))
    lines.append("t_matrix_convergence_epsilon")
    lines.append(fd(tmatrix_convergence_eps))

    # Scattering matrix
    lines.append("calculate_scattering_matrix")
    lines.append("t" if calculate_scattering_matrix else "f")
    if not normalize_s11:
        # Defaults to true in MSTM itself -- S11 normalized so its own
        # angular integral works out to a fixed convention, not a raw
        # cross section. get_scattering_angle() (the f2py binding path)
        # and FaSTMM2 both report *unnormalized* S11 -- confirmed a large
        # magnitude mismatch against get_scattering_angle() on an
        # identical case before setting this to false. Setting it to
        # false brings the *shape* into line at every angle, but leaves a
        # residual, exactly-constant 2*pi factor across the board -- this
        # .inp text path has no further "true unnormalized" mode to
        # request, so callers wanting an exact match to
        # get_scattering_angle()'s convention need to divide that factor
        # back out themselves (see t-bench's mstm_cli.py adapter for
        # where that's done).
        lines.append("normalize_s11")
        lines.append("f")
    if scattering_map_dimension is not None:
        # NOTE: confirmed this has no effect on the "scattering matrix in
        # incident plane" text table parse_mstm_output() reads (fixed
        # -180..180deg, 361 points at 1deg resolution regardless of this
        # value) -- exposed anyway since it's a real .inp keyword that
        # may matter for other output modes (e.g. random-orientation
        # runs) not yet exercised by any caller here.
        lines.append("scattering_map_dimension")
        lines.append(str(scattering_map_dimension))

    if azimuthal_average:
        lines.append("azimuthal_average")
        lines.append("t")

    # Per-sphere output
    lines.append("print_sphere_data")
    lines.append("t" if print_sphere_data else "f")

    # Periodic lattice
    if periodic_lattice:
        lines.append("periodic_lattice")
        lines.append("t")
        lines.append("cell_width")
        lines.append(f"{fd(cell_width_x)},{fd(cell_width_y)}")

    lines.append("end_of_options")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")

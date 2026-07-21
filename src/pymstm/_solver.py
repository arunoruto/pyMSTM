"""
Unified, pydantic-validated interface for solving an MSTM scattering
problem via either the f2py bindings or the standalone CLI binary,
toggled with a single ``cli`` flag.

Built specifically to make it easy to (1) benchmark the two backends
against each other and (2) cross-check the bindings' output against the
CLI's to catch regressions -- both backends are driven from exactly the
same validated input and mapped into the same :class:`MstmResult` shape,
so results are directly comparable regardless of which one produced them.
"""

from __future__ import annotations

import math
import time
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ._cli import run_mstm
from ._convert import s11_to_phase_function
from ._mstm import MSTM as _MstmBindings


class MstmPerSphereResult(BaseModel):
    """Per-sphere efficiencies."""

    q_ext: float
    q_abs: float


class MstmMuellerPoint(BaseModel):
    """A single angle-resolved Mueller-matrix sample.

    ``s11``/``s12`` are normalized to the radiative-transfer phase
    function convention (``integral(S11 dOmega) == 4*pi``), not the raw
    Bohren-Huffman units either backend reports natively -- see
    :func:`pymstm._convert.s11_to_phase_function`.
    """

    theta_deg: float
    s11: float
    s12: float


class MstmResult(BaseModel):
    """Backend-agnostic result of solving an :class:`MstmProblem` --
    the same shape regardless of ``cli=True``/``False``, so bindings and
    CLI results can be compared directly (benchmarking, regression
    testing)."""

    backend: Literal["bindings", "cli"]
    q_ext_total: float
    q_abs_total: float
    q_sca_total: float
    iterations: int
    solution_error: float
    wall_time_seconds: float
    """Includes subprocess spawn + .inp write + output parse overhead for
    the CLI backend, and only the in-process solve (+ Mueller-matrix
    post-processing, if requested) for the bindings backend -- this is
    the real-world cost difference intentionally being measured, not
    something to normalize away."""
    per_sphere: list[MstmPerSphereResult] = Field(default_factory=list)
    mueller: list[MstmMuellerPoint] | None = None


class MstmProblem(BaseModel):
    """A complete, self-contained MSTM scattering problem: cluster
    geometry, medium, incident wave, and solver settings, validated with
    pydantic before ever touching MSTM.

    ``radii``/``positions`` are in MSTM's own native, dimensionless
    size-parameter units (the same convention ``MSTM.set_spheres()`` and
    ``write_inp_file()`` already use -- e.g. ``x = 2*pi*r/lambda`` for a
    single sphere), *not* physical length units. Scale physical
    radii/positions by your wavenumber before constructing this, exactly
    as the existing lower-level APIs already require.

    Set ``cli=True``/``False`` to solve via the standalone CLI binary or
    the f2py bindings respectively -- everything else about the call
    stays identical, which is the whole point: flip one field and get a
    directly comparable :class:`MstmResult` back, for benchmarking the
    two backends' speed or cross-checking their output.

    Examples
    --------
    >>> problem = MstmProblem(
    ...     radii=[5.0], positions=[(0.0, 0.0, 0.0)],
    ...     ref_re=[1.5], ref_im=[0.0],
    ... )
    >>> bindings_result = problem.solve()
    >>> cli_result = problem.model_copy(update={"cli": True}).solve()
    >>> abs(bindings_result.q_ext_total - cli_result.q_ext_total) < 1e-3
    True
    """

    radii: list[float]
    positions: list[tuple[float, float, float]]
    ref_re: list[float]
    ref_im: list[float]

    medium_ref_re: float = 1.0
    medium_ref_im: float = 0.0

    incident_alpha_deg: float = 0.0
    incident_beta_deg: float = 0.0
    incident_direction: int = 1

    solution_eps: float = 1e-6
    max_iterations: int = 5000
    mie_eps: float = 1e-6
    translation_eps: float = 1e-5
    max_tmatrix_order: int = 100

    compute_mueller: bool = False
    n_theta: int = Field(default=19, ge=2)
    """Number of theta samples (0-180deg, uniform) when compute_mueller.
    Both backends return this exact same grid (the CLI backend resamples
    its own native fixed-resolution table onto it), so bindings/CLI
    results are directly, positionally comparable. Note this is a single
    meridian cut (phi_rel=0 relative to the incident direction) -- valid
    as the *complete* angular distribution only for spherically-symmetric
    single-sphere clusters; for a general multi-sphere cluster, S11 also
    depends on phi, so integral(S11 dOmega) computed from this cut alone
    will not exactly equal 4*pi (confirmed empirically: <0.1% error for
    a single sphere, ~10% for an asymmetric two-sphere case at n_theta=181
    -- not a bug, an inherent limitation of a 1D angular cut)."""

    cli: bool = False
    """False (default): solve via the f2py bindings (MSTM class), in
    process, no file I/O. True: solve via the standalone CLI binary
    (run_mstm()) instead -- everything else about the problem is
    unchanged, so the two are directly comparable."""
    binary_path: str | None = None
    """CLI-only: explicit mstm binary path, bypassing PATH discovery."""
    mpi_processes: int | None = None
    """CLI-only: if set, runs via ``mpiexec -n N mstm-mpi`` instead of
    the serial ``mstm`` binary."""

    @model_validator(mode="after")
    def _check_consistency(self) -> MstmProblem:
        n = len(self.radii)
        if n == 0:
            raise ValueError("at least one sphere is required")
        if len(self.positions) != n:
            raise ValueError(
                f"positions has {len(self.positions)} entries, expected {n} "
                "(one per sphere, matching radii)"
            )
        if len(self.ref_re) != n or len(self.ref_im) != n:
            raise ValueError(
                f"ref_re ({len(self.ref_re)}) and ref_im ({len(self.ref_im)}) "
                f"must both have {n} entries (one per sphere, matching radii)"
            )
        if any(r <= 0 for r in self.radii):
            raise ValueError("all radii must be positive")
        if self.compute_mueller and self.cli and self.incident_beta_deg != 0.0:
            # mstm-cli's own scattering-matrix text table is structurally a
            # single fixed lab-frame meridian plane (no per-angle query to
            # rotate the way the bindings' get_scattering_angle() allows --
            # see pymstm._cli's docs and t-bench's mstm_cli.py adapter for
            # the same, independently-confirmed limitation), so it can only
            # correctly report the Mueller matrix for zero-polar-angle
            # incidence. Reject rather than silently return an angularly
            # wrong result.
            raise ValueError(
                "compute_mueller=True with cli=True requires "
                "incident_beta_deg=0.0 -- mstm-cli's scattering-matrix "
                "table can't represent a tilted-incidence angular cut "
                "(use cli=False for tilted-incidence Mueller matrices)"
            )
        return self

    def solve(self) -> MstmResult:
        """Solve via the backend selected by ``cli``."""
        return self._solve_cli() if self.cli else self._solve_bindings()

    # -- bindings backend ----------------------------------------------

    def _orders(self) -> list[int]:
        return [
            max(4, int(x + 4 * x ** (1 / 3) + 2))
            for x in self.radii
        ]

    def _solve_bindings(self) -> MstmResult:
        t0 = time.perf_counter()
        m = _MstmBindings()
        try:
            m.set_spheres(
                radii=self.radii,
                positions=self.positions,
                orders=self._orders(),
                ref_re=self.ref_re,
                ref_im=self.ref_im,
            )
            m.set_medium_ref(self.medium_ref_re, self.medium_ref_im)
            m.set_incident(
                alpha_deg=self.incident_alpha_deg,
                beta_deg=self.incident_beta_deg,
                direction=self.incident_direction,
            )
            m.set_solver_params(eps=self.solution_eps, max_iter=self.max_iterations)
            m.set_mie_eps(self.mie_eps)
            m.set_translation_eps(self.translation_eps)
            m.set_max_tmatrix_order(self.max_tmatrix_order)
            m.set_verbose(False)
            m.prepare()
            raw = m.solve()

            per_sphere = [
                MstmPerSphereResult(q_ext=float(qe), q_abs=float(qa))
                for qe, qa in zip(raw["q_ext"], raw["q_abs"])
            ]

            mueller = None
            if self.compute_mueller:
                r_cs = m.get_cross_section_radius()
                c_sca = float(raw["qsca_tot"]) * math.pi * r_cs**2
                mueller = self._mueller_from_bindings(m, c_sca)

            wall_time = time.perf_counter() - t0
            return MstmResult(
                backend="bindings",
                q_ext_total=float(raw["qext_tot"]),
                q_abs_total=float(raw["qabs_tot"]),
                q_sca_total=float(raw["qsca_tot"]),
                iterations=int(raw["iterations"]),
                solution_error=float(raw["solution_error"]),
                wall_time_seconds=wall_time,
                per_sphere=per_sphere,
                mueller=mueller,
            )
        finally:
            m.finalize()

    def _mueller_from_bindings(
        self, m: _MstmBindings, c_sca: float
    ) -> list[MstmMuellerPoint]:
        # get_scattering_angle()'s (costheta, phi) are lab-frame
        # coordinates, not relative to the incident direction (see its
        # docstring) -- rotate each desired (theta_rel, phi_rel=0) point
        # into the lab frame via the same R = Rz(alpha).Ry(beta) that
        # defines MSTM's own incident-wave convention before calling it.
        # Verified this exact approach against FaSTMM2 to 3-4 significant
        # figures for a tilted-incidence case in a separate project
        # (t-bench) built on this same package.
        alpha_rad = math.radians(self.incident_alpha_deg)
        beta_rad = math.radians(self.incident_beta_deg)
        cos_a, sin_a = math.cos(alpha_rad), math.sin(alpha_rad)
        cos_b, sin_b = math.cos(beta_rad), math.sin(beta_rad)

        # Native MSTM units are already "pre-scaled" by the physical
        # wavenumber (radii/positions are size parameters, e.g.
        # x=2*pi*r/lambda) -- so within this internal unit frame the
        # effective wavenumber is exactly 1.0, and c_sca (computed above
        # from Q_sca and the cross-section radius in these same units)
        # is already in the matching scale for s11_to_phase_function().
        k_internal = 1.0

        points = []
        for j in range(self.n_theta):
            theta_deg = 180.0 * j / (self.n_theta - 1)
            theta_rel = math.radians(theta_deg)
            sin_t, cos_t = math.sin(theta_rel), math.cos(theta_rel)
            x1 = cos_b * sin_t + sin_b * cos_t
            z1 = -sin_b * sin_t + cos_b * cos_t
            theta_lab = math.acos(max(-1.0, min(1.0, z1)))
            phi_lab = math.atan2(sin_a * x1, cos_a * x1)
            sm = m.get_scattering_angle(costheta=math.cos(theta_lab), phi=phi_lab)
            s11 = float(s11_to_phase_function(sm[0], k_internal, c_sca))
            s12 = float(s11_to_phase_function(sm[1], k_internal, c_sca))
            points.append(MstmMuellerPoint(theta_deg=theta_deg, s11=s11, s12=s12))
        return points

    # -- CLI backend ------------------------------------------------------

    def _solve_cli(self) -> MstmResult:
        t0 = time.perf_counter()
        inp_kwargs = dict(
            radii=self.radii,
            positions=self.positions,
            ref_re=self.ref_re,
            ref_im=self.ref_im,
            medium_ref_re=self.medium_ref_re,
            medium_ref_im=self.medium_ref_im,
            alpha_deg=self.incident_alpha_deg,
            beta_deg=self.incident_beta_deg,
            incident_direction=self.incident_direction,
            solution_eps=self.solution_eps,
            max_iterations=self.max_iterations,
            mie_eps=self.mie_eps,
            translation_eps=self.translation_eps,
            max_tmatrix_order=self.max_tmatrix_order,
            calculate_scattering_matrix=self.compute_mueller,
            normalize_s11=False,
            print_sphere_data=True,
        )

        result = run_mstm(
            inp_kwargs=inp_kwargs,
            binary_path=self.binary_path,
            mpi_processes=self.mpi_processes,
        )
        parsed = result.parsed
        total = parsed["total"]

        per_sphere = [
            MstmPerSphereResult(q_ext=float(s["q_ext"]), q_abs=float(s["q_abs"]))
            for s in parsed["per_sphere"]
        ]

        mueller = None
        if self.compute_mueller and parsed.get("scattering_matrix"):
            r_cs = sum(r**3 for r in self.radii) ** (1 / 3)
            c_sca = float(total["q_sca_unpol"]) * math.pi * r_cs**2
            sm = parsed["scattering_matrix"]
            # normalize_s11=False (above) matches the bindings' raw shape
            # but leaves a residual, exactly-constant 2*pi factor -- see
            # write_inp_file()'s own docstring -- undo it, then apply the
            # same 4*pi-phase-function normalization as the bindings path
            # (k=1.0 for the same "already in native MSTM units" reason).
            #
            # mstm-cli's own text table is always a fixed -180..180deg,
            # 1deg-resolution grid (confirmed elsewhere this session to be
            # unaffected by any .inp keyword) -- NOT self.n_theta points.
            # Resample onto the exact same uniform n_theta grid the
            # bindings backend uses (nearest available integer-degree
            # angle) so the two backends' mueller lists are directly,
            # positionally comparable rather than silently different
            # grids that happen to have mismatched lengths/spacing.
            native_theta = [t for t in sm["angles_deg"] if t >= 0]
            native_rows = {
                t: row for t, row in zip(sm["angles_deg"], sm["matrix"]) if t >= 0
            }
            mueller = []
            for j in range(self.n_theta):
                target = 180.0 * j / (self.n_theta - 1)
                nearest = min(native_theta, key=lambda t: abs(t - target))
                row = native_rows[nearest]
                s11_raw = row[0] / (2 * math.pi)
                s12_raw = row[4] / (2 * math.pi)
                s11 = float(s11_to_phase_function(s11_raw, 1.0, c_sca))
                s12 = float(s11_to_phase_function(s12_raw, 1.0, c_sca))
                mueller.append(MstmMuellerPoint(theta_deg=target, s11=s11, s12=s12))

        wall_time = time.perf_counter() - t0
        return MstmResult(
            backend="cli",
            q_ext_total=float(total["q_ext_unpol"]),
            q_abs_total=float(total["q_abs_unpol"]),
            q_sca_total=float(total["q_sca_unpol"]),
            iterations=int(parsed["iterations"]),
            solution_error=float(parsed["solution_error"]),
            wall_time_seconds=wall_time,
            per_sphere=per_sphere,
            mueller=mueller,
        )
